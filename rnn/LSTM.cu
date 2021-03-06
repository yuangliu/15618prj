/* Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
 
/*
 Compile:
  nvcc -arch=sm_52 -O3 -lcublas -lcurand -o LSTM LSTM.cu 
  
  To enable/disable different performance options add the flat -DPERFOPTSx
  Where x is a bitmask defining the options used (see below).
  
 Run:
  ./LSTM
  or
  ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>
  
 Example (run on an NVIDIA M40):
   > ./LSTM
   Running with default settings
   seqLength 100, numLayers 4, hiddenSize 512, miniBatch 64
   i checksum (example 0) 5.113463E+04
   h checksum (example 0) 2.048000E+03
   c checksum (example 0) 2.058137E+05
   i checksum 3.272639E+06     c checksum 1.317278E+07     h checksum 1.310720E+05
   Runtime 27.807743ms
*/

#include <stdio.h>
#include <cublas_v2.h>
#include <curand.h>

#define TRAINING (true)
// #define UPDATE
// #define PEEPHOLES 
// #define CIFG
#define GATE_NUM (4)
#define PEEP_NUM (3)
#define I_INDEX (0)
#define F_INDEX (1)
#define G_INDEX (2)
#define O_INDEX (3)

 #ifndef PERFOPTS
   #define PERFOPTS (31)
#endif

#define GROUP_GEMM ((PERFOPTS & 1))
#define USE_STREAMS ((PERFOPTS & 2))
// #define FUSE_PW ((PERFOPTS & 4))
#define FUSE_PW (1)
// #define PRE_TRANSPOSE ((PERFOPTS & 8))
#define PRE_TRANSPOSE (1)
#define RECUR_BATCH_SIZE (((PERFOPTS & 16) ? 2 : 1))
#define RECUR_BATCH_BP_SIZE (((PERFOPTS & 16) ? 5 : 1))

#define HFUNC tanhf
#define DEHFUNC de_tanhf
#define GFUNC tanhf
#define DEGFUNC de_tanhf
#define LOSSFUNC squaree
#define DELOSSFUNC de_squaree




// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define GetOffsetW(x) (GATE_NUM * inputSize * hiddenSize + GATE_NUM * hiddenSize * hiddenSize + (x - 1) * 2 * GATE_NUM * hiddenSize * hiddenSize)
#define GetOffsetR(x) (GetOffsetW(x) +  GATE_NUM * hiddenSize * hiddenSize)
// (layer - 1) * 8 * hiddenSize * hiddenSize + 8 * hiddenSize * hiddenSize + 4 * hiddenSize * inputSize// #define GetOffsetR(x)
// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

__forceinline__ __device__ float de_sigmoidf(float out) {
   return out * (1-out);
} 

__forceinline__ __device__ float linearf(float in) {
  return in;
}

__forceinline__ __device__ float de_linearf(float out) {
  return 1.f;
}

__forceinline__ __device__ float de_tanhf(float out) {
   return 1.f - pow(out, 2);
}

__forceinline__ __device__ float squaree(float output, float target) {
  return pow(output - target, 2);
}

__forceinline__ __device__ float de_squaree(float output, float target) {
  return 2 * (output - target);
}

__forceinline__ __device__ float entropye(float output, float target) {
  return -(target * logf(output) + (1.f - target) * logf(1.f - output));
}

__forceinline__ __device__ float de_entropye(float output, float target) {
  return -(target / output + (1.f - target) / (output - 1.f));
}

__global__ void pw_de_tanh(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = 1 - pow(a[i], 2);
} 

__global__ void pw_de_sigmoid(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] * (1 - a[i]);
} 
// Pointwise functions
__global__ void pw_biasAdd(float *y, float *bias, int n, int nBias) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] += bias[i % nBias];
}

__global__ void pw_peepsAdd(float *y, float *peeps, float *x, int n, int nPeeps) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] += peeps[i % nPeeps] * x[i];
}

__global__ void pw_vecAdd(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] + b[i];
}

__global__ void pw_vecMul(float *y, float *a,  float *b, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = a[i] * b[i];
}

__global__ void pw_tanh(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = tanh(a[i]);
}

__global__ void pw_sigmoid(float *y, float *a, int n) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) y[i] = sigmoidf(a[i]);
}



// Unfused LSTM (calling many pointwise kernels).
int LSTM_elementwise_unfused( int hiddenSize, 
                               int miniBatch,
                               float * __restrict__ tmp_h, 
                               float * __restrict__ tmp_i, 
                               float * __restrict__ bias,
                               float * __restrict__ peeps,
                               // float * __restrict__ linearGates,
                               // float * __restrict__ h_data,
                               float * __restrict__ i_data,
                               float * __restrict__ c_in,
                               float * __restrict__ c_out,
                               bool training,
                               cudaStream_t stream) {
  dim3 blockDim;
  dim3 gridDim;

  int numElements = hiddenSize * miniBatch;

  blockDim.x = 128;
  gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

               
  for (int i = 0; i < GATE_NUM; i++) {
    if (tmp_h != NULL) {
       pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, tmp_i + i * numElements, 
        tmp_h  + i * numElements, numElements);
       cudaErrCheck(cudaGetLastError());
    }

    pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + i * hiddenSize, numElements, hiddenSize);
    cudaErrCheck(cudaGetLastError());
    
    if (i == 0) {
       pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + i * hiddenSize, numElements, hiddenSize);
       cudaErrCheck(cudaGetLastError());
    }
    
    if (training) {
       printf("LSTM_elementWise_unfused does not support training\n"); 
       return 1;
    }
  }

  float *in_gate     = tmp_i + 
  #ifdef I_INDEX
  I_INDEX * numElements;//i
  #else
  NULL;
  #endif

  float *forget_gate = tmp_i + 
  #ifdef F_INDEX
  F_INDEX * numElements;//f
  #else
  NULL;
  #endif

  float *in_gate2    = tmp_i + 
  #ifdef G_INDEX
  G_INDEX * numElements;//z
  #else
  NULL;
  #endif

  float *out_gate    = tmp_i + 
  #ifdef O_INDEX
  O_INDEX * numElements;//z
  #else
  NULL;
  #endif   


  if (c_in != NULL) {
    //i_t += p_i * c_t-1          

    #ifdef I_INDEX
    pw_peepsAdd <<< gridDim, blockDim, 0, stream >>> (in_gate, peeps + I_INDEX * hiddenSize, c_in, numElements, hiddenSize);
    cudaErrCheck(cudaGetLastError());
    #endif
    //f_t += p_f * c_t-1          
    #ifdef F_INDEX
    pw_peepsAdd <<< gridDim, blockDim, 0, stream >>> (forget_gate, peeps + F_INDEX * hiddenSize, c_in, numElements, hiddenSize);
    cudaErrCheck(cudaGetLastError());
    #endif
    

  }


  pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (in_gate, tmp_i + 0 * numElements, numElements);
  cudaErrCheck(cudaGetLastError());

  pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (forget_gate, tmp_i + 1 * numElements, numElements);
  cudaErrCheck(cudaGetLastError());

  //z'
  pw_tanh    <<< gridDim, blockDim, 0, stream >>> (in_gate2, tmp_i + 2 * numElements, numElements);
  cudaErrCheck(cudaGetLastError());

  if (c_in == NULL) {
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, in_gate2, numElements);
    cudaErrCheck(cudaGetLastError());
  } else {
    //f_t * c_t-1    
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (forget_gate, forget_gate, c_in, numElements);
    cudaErrCheck(cudaGetLastError());
    
    //i_t * z
    pw_vecMul <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, in_gate2, numElements);
    cudaErrCheck(cudaGetLastError());
    
    //c_t = f_t * c_t-1 + i_t * c_t'
    pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, forget_gate, numElements);
    cudaErrCheck(cudaGetLastError());
  }

  //o_t += p_o * c_t  
  pw_peepsAdd <<< gridDim, blockDim, 0, stream >>> (out_gate, peeps + 2 * hiddenSize, in_gate, numElements, hiddenSize);
    cudaErrCheck(cudaGetLastError());

  pw_sigmoid <<< gridDim, blockDim, 0, stream >>> (out_gate, tmp_i + 3 * numElements, numElements);
  cudaErrCheck(cudaGetLastError());



  if (c_out != NULL) {
    cudaErrCheck(cudaMemcpyAsync(c_out, in_gate, numElements * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  }
  

  //tanh(c_t)
  pw_tanh <<< gridDim, blockDim, 0, stream >>> (in_gate, in_gate, numElements);
  cudaErrCheck(cudaGetLastError());

  //y = o_t * tanh(c_t)
  // pw_vecMul <<< gridDim, blockDim, 0, stream >>> (h_data, out_gate, in_gate, numElements);
  // cudaErrCheck(cudaGetLastError());


  pw_vecMul <<< gridDim, blockDim, 0, stream >>> (i_data, out_gate, in_gate, numElements);
  cudaErrCheck(cudaGetLastError());

  return 0;
}

// Fused forward kernel
__global__ void elementWise_fp(int hiddenSize, int miniBatch,
                               float *tmp_h, //hidden_size * mini_batch * 4: W*xt
                               float *tmp_i, //hidden_size * mini_batch * 4: R*yt
                               float *bias, //hidden_size * 4: b*
                               float *peeps,//hidden_size * 3: p*
                               // float *linearGates,// hidden_size * mini_batch * 4
                               float *stateGates,
                               // float *h_out, //h_data
                               float *i_out,
                               float *c_in,
                               float *c_out,
                               float *label,
                               float *loss,
                               float *mask,
                               float *y_diff,
                               bool training) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int numElements = miniBatch * hiddenSize;

  if (index >= numElements) return;

  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + GATE_NUM * batch * hiddenSize;   

  float g[GATE_NUM];

  for (int i = 0; i < GATE_NUM; i++) {
    g[i] = tmp_i[i * hiddenSize + gateIndex] + (tmp_h == NULL? 0:tmp_h[i * hiddenSize + gateIndex]);
    g[i] += bias[i * hiddenSize + index % hiddenSize];
  }  

  #ifdef PEEPHOLES 
    #ifdef I_INDEX
      g[I_INDEX] += peeps[I_INDEX * hiddenSize + index % hiddenSize] * c_in[index];//pi * c[t-1]
    #endif
    #ifdef F_INDEX
      g[F_INDEX] += peeps[F_INDEX * hiddenSize + index % hiddenSize] * c_in[index];//pf * c[t-1]
    #endif
  #endif

  
  float in_gate = 
  #ifdef I_INDEX
    sigmoidf(g[I_INDEX]);//i
  #else
    1.f;
  #endif

  
  float forget_gate = 
  #ifdef F_INDEX
    sigmoidf(g[F_INDEX]);//f
  #else
    #ifdef CIFG
      (1 - in_gate);
    #else
      1.f;
    #endif
  #endif

  
  float in_gate2    = 
  #ifdef G_INDEX
    GFUNC(g[G_INDEX]);//z
  #else
    1.f;
  #endif


  float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);//c[t] = z*i+c[t-1]*f
  c_out[index] = val;

  #ifdef PEEPHOLES
    #ifdef O_INDEX
      g[O_INDEX] += peeps[hiddenSize * (O_INDEX - 1) + index % hiddenSize] * c_out[index];//po * c[t]
    #endif
  #endif

  float out_gate = 
  #ifdef O_INDEX
    sigmoidf(g[O_INDEX]);//o
  #else
    1.f;
  #endif


  // #ifdef TRAINING
  if(training) {
    #ifdef I_INDEX
    stateGates[I_INDEX * hiddenSize + gateIndex] = in_gate;
    #endif

    #ifdef F_INDEX
    stateGates[F_INDEX * hiddenSize + gateIndex] = forget_gate;
    #endif

    #ifdef G_INDEX
    stateGates[G_INDEX*hiddenSize + gateIndex] = in_gate2;
    #endif

    #if O_INDEX
    stateGates[O_INDEX*hiddenSize + gateIndex] = out_gate;
    #endif
  }
  // #endif

  val = out_gate * HFUNC(val); //h                            

  // h_out[index] = val;
  i_out[index] = val;

  if (label != NULL) {
    if (mask[index % hiddenSize] == 1) {

      loss[index] = LOSSFUNC(val, label[index]);
      if (training)
        y_diff[index] = DELOSSFUNC(val, label[index]);
    } else {
      if (training)
        y_diff[index] = 0;
    }
  }
}

// Fused backward kernel
__global__ void elementWise_bp(int hiddenSize, int miniBatch,
                                float *y_diff,
                                float *stateGates_diff_in,// hidden_size *  mini_batch * 4
                                float *stateGates_diff_out,// hidden_size * mini_batch * 4
                                float *stateGates,
                                float *peeps,
                                float *peeps_diff,
                                float *c_in,
                                float *c_out,
                                float *c_diff,
                                bool peeps_update) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int numElements = miniBatch * hiddenSize;

  if (index >= numElements) return;

  int batch = index / hiddenSize;
  int gateIndex = (index % hiddenSize) + GATE_NUM * batch * hiddenSize;   
  

  float in_gate = 
  #ifdef I_INDEX
    stateGates[I_INDEX * hiddenSize + gateIndex];
  #else
    1.f;
  #endif
  
  float forget_gate = 
  #ifdef F_INDEX
    stateGates[F_INDEX * hiddenSize + gateIndex];
  #else    
    #ifdef CIFG
    1 - in_gate;
    #else 
    1.f;
    #endif
  #endif

  float in_gate2 = 
  #ifdef G_INDEX
    stateGates[G_INDEX * hiddenSize + gateIndex];
  #else
    1.f
  #endif

  float out_gate = 
  #ifdef O_INDEX
    stateGates[O_INDEX * hiddenSize + gateIndex];
  #else
    1.f;
  #endif

  
  #ifdef O_INDEX
  float out_diff = y_diff[index]*HFUNC(c_out[index])*de_sigmoidf(out_gate); //do
  #endif
   


  float peep_diff = 0;
  if (stateGates_diff_in != NULL) peep_diff += c_diff[index];

  #ifdef PEEPHOLES

    #ifdef O_INDEX
      peep_diff += peeps[(O_INDEX - 1) * hiddenSize + index % hiddenSize] * out_diff;//po * do
    #endif


    if (stateGates_diff_in != NULL) { 
      #ifdef I_INDEX
        peep_diff += peeps[I_INDEX * hiddenSize + index % hiddenSize] * stateGates_diff_in[I_INDEX * hiddenSize + gateIndex]; //pi * di[t+1]
      #endif

      #ifdef F_INDEX
      peep_diff += peeps[F_INDEX * hiddenSize + index % hiddenSize] * stateGates_diff_in[F_INDEX * hiddenSize + gateIndex];//pf * df[t+1]
      #endif
    }
  #endif


  float local_c_diff = y_diff[index]*out_gate*DEHFUNC(HFUNC(c_out[index])) + peep_diff;
  
  
  float in_diff2 = local_c_diff * in_gate * DEGFUNC(in_gate2);

  #ifdef I_INDEX
    float in_diff = local_c_diff * 
    #ifdef CIFG
    (in_gate2 - c_in[index])
    #else
    in_gate2 
    #endif
    * de_sigmoidf(in_gate);
    stateGates_diff_out[I_INDEX * hiddenSize + gateIndex] = in_diff;
  #endif

  #ifdef F_INDEX
    float forget_diff = local_c_diff * c_in[index] * de_sigmoidf(forget_gate);
    stateGates_diff_out[F_INDEX * hiddenSize + gateIndex] = forget_diff;
  #endif

  #ifdef G_INDEX
    stateGates_diff_out[G_INDEX * hiddenSize + gateIndex] = in_diff2;
  #endif

  #ifdef O_INDEX
    stateGates_diff_out[O_INDEX * hiddenSize + gateIndex] = out_diff;       
  #endif

  #ifdef PEEPHOLES
    int peepIndex = (index % hiddenSize) + PEEP_NUM * batch * hiddenSize;
    if (peeps_update) {

      #ifdef I_INDEX
      peeps_diff[I_INDEX * hiddenSize + peepIndex] = in_diff * c_in[index];//p_i
      #endif 

      #ifdef F_INDEX
      peeps_diff[F_INDEX * hiddenSize + peepIndex] = forget_diff * c_in[index]; //p_f
      #endif 
    }

    #ifdef O_INDEX
    peeps_diff[(O_INDEX - 1) * hiddenSize + peepIndex] = out_diff * c_out[index]; //p_o
    #endif 

  #endif

  c_diff[index] = local_c_diff * forget_gate;
  
}



struct LSTM_scheduler
{
  // float *h_data;//y
  float *i_data;//x

  float *c_data;//c

  float *T;
  float *T_f;

  float *bias;

  

  float *tmp_h;
  float *tmp_i;
  // float *linearGates;
  float *stateGates;

  //diff
  float *stateGates_diff; //di,df,dz,do
  float *y_diff;//dy
  // float *T_diff;//dW, dR
  
  float *diff_helper;

  #ifdef PEEPHOLES
    float *peeps_diff;
    float *peeps;
  #endif

  float *c_diff;//dc*ft

  float *label;
  float *loss;
  float *mask;



  cudaStream_t *stream_i;
  cudaStream_t *stream_h;

  cudaEvent_t **events_i;
  cudaEvent_t **events_h;

  cublasHandle_t handle;

  int hiddenSize;
  int miniBatch;
  int seqLength;
  int numLayers;
  int numElements;
  int inputSize;
  int inputNumElements;
  int inputLayerSize;
  int weightSize; 

  cublasOperation_t transa;
  cublasOperation_t transb;

  curandGenerator_t rng;
    

  void init_helper(float * device_ptr, float data, int size) {
    if (size == 0)
      return;
    float* host_ptr;
    host_ptr = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
      host_ptr[i] = data;
    }
    cudaErrCheck(cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice));
  }


  void set_input(float * input) {

    cudaErrCheck(cudaMemcpy(i_data, input, inputLayerSize * sizeof(float), cudaMemcpyHostToDevice));
  }

  void set_label(float * label_) {
    
    cudaErrCheck(cudaMemcpy(label, label_, numElements * seqLength * sizeof(float), cudaMemcpyHostToDevice));
  }

  void set_mask(float * mask_) {
    if (mask_ == NULL) {
      init_helper(mask, 1, 1);
      init_helper(mask + 1, 0, hiddenSize-1);     
    } else {
      cudaErrCheck(cudaMemcpy(mask, mask_, numElements * seqLength * sizeof(float), cudaMemcpyHostToDevice)); 
    }
  }

  void set_weight(float T_f_=0.1, float bias_=0.1, float peeps_=0.1, int random=0) {
    if (!(random & 1)) {
      init_helper(T_f, T_f_, weightSize);
    }
    else {
      curandErrCheck(curandGenerateUniform(rng, T_f, weightSize));
    }
    
    if (!(random & 2)) {
      init_helper(bias, bias_, hiddenSize * GATE_NUM * numLayers);
    }
    else {
      curandErrCheck(curandGenerateUniform(rng, bias, hiddenSize * GATE_NUM * numLayers));
    }

    #ifdef PEEPHOLES
    if (!(random & 4)) {
      init_helper(peeps, peeps_, hiddenSize * 3 * numLayers);
    }
    else {
      curandErrCheck(curandGenerateUniform(rng, bias, hiddenSize * PEEP_NUM * numLayers));
    }
    #endif 
  }


  LSTM_scheduler(int hiddenSize_, int miniBatch_, int seqLength_, int numLayers_, int inputSize_)
  {
    transa = (PRE_TRANSPOSE && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;
    transb = CUBLAS_OP_N;

    hiddenSize = hiddenSize_;
    miniBatch = miniBatch_;
    seqLength = seqLength_; 
    numLayers = numLayers_; 
    inputSize = inputSize_;
    
    numElements = hiddenSize * miniBatch;
    inputNumElements = inputSize * miniBatch;

    inputLayerSize = inputNumElements * seqLength;
    weightSize = GetOffsetW(numLayers);


    cublasErrCheck(cublasCreate(&handle));

    curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));


    stream_i = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    stream_h = (cudaStream_t*)malloc(numLayers * sizeof(cudaStream_t));
    
    // If we don't want to use streams we can launch everything in to the NULL stream
    for (int i = 0; i < numLayers; i++) {
       if (USE_STREAMS) {
          cudaErrCheck(cudaStreamCreate(&stream_i[i]));
          // Priority is empirical.
          cudaErrCheck(cudaStreamCreateWithPriority(&stream_h[i], 0, -1));   
       }
       else {
          stream_i[i] = NULL;  
          stream_h[i] = NULL;  
       }
    }
    
    
    events_i = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    events_h = (cudaEvent_t**)malloc(numLayers * sizeof(cudaEvent_t*));
    for (int i = 0; i < numLayers; i++) {
       events_i[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
       events_h[i] = (cudaEvent_t*)malloc(seqLength * sizeof(cudaEvent_t));
    }

  }


  

  void init(float* mask_ = NULL) {
    
    // cudaErrCheck(cudaMalloc((void**)&h_data, (seqLength + 1) * (inputNumElements + (numLayers - 1) * numElements) * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&i_data, (seqLength * inputNumElements + seqLength * numLayers  * numElements) * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&c_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));

    

    cudaErrCheck(cudaMalloc((void**)&T, (inputSize * hiddenSize * GATE_NUM + hiddenSize * hiddenSize * GATE_NUM + (numLayers - 1) * hiddenSize * hiddenSize * 2 * GATE_NUM)  * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&T_f, (inputSize * hiddenSize * GATE_NUM + hiddenSize * hiddenSize * GATE_NUM + (numLayers - 1) * hiddenSize * hiddenSize * 2 * GATE_NUM) * sizeof(float)));
    
    
    cudaErrCheck(cudaMalloc((void**)&bias, numLayers * hiddenSize * GATE_NUM * sizeof(float)));

    cudaErrCheck(cudaMalloc((void**)&label, numElements * seqLength * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&loss, numElements * seqLength * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&mask, hiddenSize * sizeof(float)));

    set_mask(NULL);

    init_helper(loss, 0, numElements * seqLength);
    init_helper(label, 1, numElements * seqLength);

    #ifdef PEEPHOLES
      cudaErrCheck(cudaMalloc((void**)&peeps, numLayers * hiddenSize * PEEP_NUM * sizeof(float)));
      init_helper(peeps, 0.1, numLayers * hiddenSize * PEEP_NUM);
      // curandErrCheck(curandGenerateUniform(rng, peeps, numLayers * hiddenSize * 3));
    #endif

    // Workspace
    cudaErrCheck(cudaMalloc((void**)&tmp_h, GATE_NUM * numLayers * numElements * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&tmp_i, GATE_NUM * seqLength * numElements * sizeof(float)));
    
    // // Activations
    if (TRAINING) {
       // cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&stateGates, GATE_NUM * seqLength * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&stateGates_diff, GATE_NUM * seqLength * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&y_diff, seqLength * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&c_diff, numLayers * numElements * seqLength * sizeof(float)));
      
      

      #ifdef PEEPHOLES 
        cudaErrCheck(cudaMalloc((void**)&peeps_diff, numLayers * numElements * seqLength * PEEP_NUM * sizeof(float)));
        init_helper(peeps_diff, 0, numLayers * numElements * seqLength * PEEP_NUM);
      #endif

      cudaErrCheck(cudaMalloc((void**)&diff_helper, miniBatch * seqLength * sizeof(float)));

      init_helper(diff_helper, 1.0, miniBatch * seqLength);
      init_helper(c_diff, 0, numLayers * numElements);
      init_helper(stateGates_diff, 0, GATE_NUM * seqLength * numLayers * numElements);
    }

    // Initialise with random values.
        // curandErrCheck(curandGenerateUniform(rng, h_data, (seqLength + 1) * (numLayers) * numElements));
    // curandErrCheck(curandGenerateUniform(rng, c_data, (seqLength + 1) * (numLayers) * numElements));
    // cudaErrCheck(cudaMemset(c_data, 0,  (seqLength + 1) * (numLayers) * numElements  * sizeof(float)));
    init_helper(c_data, 0, (seqLength + 1) * (numLayers) * numElements);
    // curandErrCheck(curandGenerateUniform(rng, i_data, seqLength * (inputNumElements + numLayers  * numElements)));
    // cudaErrCheck(cudaMemset(i_data, 0, seqLength * (inputNumElements + numLayers  * numElements)  * sizeof(float)));
    init_helper(i_data, 0.2, seqLength * (inputNumElements + numLayers  * numElements));
    // curandErrCheck(curandGenerateUniform(rng, T_f, inputSize * hiddenSize * 4 + hiddenSize * hiddenSize * 4+ (numLayers - 1) * hiddenSize * hiddenSize * 8));
    // cudaErrCheck(cudaMemset(T_f, 0.1, inputSize * hiddenSize * 4 + hiddenSize * hiddenSize * 4+ (numLayers - 1) * hiddenSize * hiddenSize * 8  * sizeof(float)));
    // curandErrCheck(curandGenerateUniform(rng, bias, numLayers * hiddenSize * 4));


    set_weight(0.1,0.1,0.1,0);
    
      
    

    // if (TRAINING) {
    //   init_helper(y_diff, 0, seqLength*numLayers*numElements);
    //   init_helper(y_diff+seqLength*(numLayers-1)*numElements, 1, seqLength * numElements);
    //    // curandErrCheck(curandGenerateUniform(rng, y_diff+seqLength*(numLayers-1)*numElements, seqLength * numElements));
    // }

    // curandErrCheck(curandDestroyGenerator(rng));


    // Make sure everything is done before we start the timers
    cudaErrCheck(cudaDeviceSynchronize());
    // prepare T
    // float alpha = 1.f;
    // float beta = 0.f; 

    // for (int layer = 0; layer < numLayers; layer++) {                     
    //    float *T_i_in = T + layer * hiddenSize * hiddenSize * 8;
    //    float *T_i_out = T_f + layer * hiddenSize * hiddenSize * 8;

    //    float *T_h_in = T + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;
    //    float *T_h_out = T_f + layer * hiddenSize * hiddenSize * 8 + hiddenSize * hiddenSize * 4;

    //    cublasErrCheck(cublasSetStream(handle, stream_i[layer]));
    //    cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_i_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_i_out, 4 * hiddenSize));
     
    //    cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
    //    cublasErrCheck(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, 4 * hiddenSize, hiddenSize, &alpha, T_h_in, hiddenSize, &beta, NULL, 4 * hiddenSize, T_h_out, 4 * hiddenSize));
    // }  
    
  }

  void clearStates(float * input=NULL, float * label_=NULL) {
    if(TRAINING) {

      if(input) {set_input(input);}
      else {init_helper(i_data, 0.2, seqLength * (inputNumElements + numLayers  * numElements));}
      if(label) set_label(label);
      else {init_helper(label, 1, numElements * seqLength);}

    }
      // init_helper(y_diff, 0, seqLength*(numLayers-1)*numElements);
      // init_helper(y_diff+seqLength*(numLayers-1)*numElements, 1, seqLength * numElements);
      // init_helper(peeps_diff, 0, 3 * numElements * numLayers * seqLength);

       // curandErrCheck(curandGenerateUniform(rng, y_diff+seqLength*(numLayers-1)*numElements, seqLength * numElements));
    // init_helper(c_diff, 0, numLayers * numElements );
    // init_helper(loss, 0, numElements * seqLength);
    // cudaErrCheck(cudaDeviceSynchronize());
  }

  float Forward(float* sumLoss) {

    float alpha = 1.f;
    float beta = 0.f; 

    float elapsedTime;
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));


    int lStart = 0;
    int lEnd = 0;
    int rStart = 0;
    int rEnd = 0;
    
    int recurBatchSize = RECUR_BATCH_SIZE;
    
    while (true) {
      // Many layer "scheduling".
      if (lEnd == 0) {
        lStart = 0;
        lEnd = 1;
        rStart = 0;
      }
      else {
        // Move "up" and "left"
        lStart++;
        lEnd++;
        
        rStart -= recurBatchSize;
        
        // Over the top or off the left, reset to layer 0
        if (lEnd > numLayers || rStart < 0) {
            rStart += (lStart + 1) * recurBatchSize;

            lStart = 0;
            lEnd = 1;
        }
        
        // Off the right, step up
        while (rStart >= seqLength && lEnd <= numLayers) {
            lStart++;
            lEnd++;
           
            rStart -= recurBatchSize;
        }
        
        
        // Over the top or off the left, done!
        if (lEnd > numLayers || rStart < 0) {
            break;
        }
      }

      rEnd = rStart + recurBatchSize;
      // printf("lStart %d lEnd %d rStart %d rEnd %d\n", lStart, lEnd,
        // rStart, rEnd);
      if (rEnd > seqLength) rEnd = seqLength;

      for (int layer = lStart; layer < lEnd; layer++) {         
        cublasErrCheck(cublasSetStream(handle, stream_i[layer]));
        
        //wait for xt to be calculated
        for (int i = rStart; i < rEnd; i++) {
            if (layer > 0) {
              cudaErrCheck(cudaStreamWaitEvent(stream_i[layer], events_h[layer - 1][i], 0));
              cudaErrCheck(cudaEventDestroy(events_h[layer - 1][i]));
            }
        }

        // Optimization 1
        if (GROUP_GEMM) {
          //[4N * N] x [N * 2m] = [4N * 2m] 
            cublasErrCheck(cublasSgemm(handle,
                       transa, transb,
                       GATE_NUM * hiddenSize, miniBatch * (rEnd - rStart), layer > 0 ? hiddenSize: inputSize,
                       &alpha,
                       layer > 0 ? &T_f[GetOffsetW(layer)]:T_f,
                       transa == CUBLAS_OP_N ? GATE_NUM * hiddenSize : hiddenSize,
                       layer > 0 ? i_data + rStart * numElements + (layer - 1) * seqLength * numElements + seqLength * inputNumElements: i_data + rStart * inputNumElements,
                       layer > 0 ? hiddenSize:inputSize,
                       &beta,
                       tmp_i + GATE_NUM * rStart * numElements,
                       GATE_NUM * hiddenSize));
        }
        else {
          for (int igemm =0; igemm < GATE_NUM; igemm++) {
            cublasErrCheck(cublasSgemm(handle,
                     transa, transb,
                     hiddenSize, miniBatch * (rEnd - rStart), layer > 0 ? hiddenSize:inputSize,
                     &alpha,
                     layer > 0 ? &T_f[GetOffsetW(layer) + igemm * hiddenSize]:&T_f[igemm * hiddenSize],
                     transa == CUBLAS_OP_N ? GATE_NUM * hiddenSize : hiddenSize,
                     layer > 0 ? i_data + rStart * numElements + (layer - 1) * seqLength * numElements + seqLength * inputNumElements: i_data + rStart * inputNumElements,
                     layer > 0 ? hiddenSize:inputSize,
                     &beta,
                     tmp_i + GATE_NUM * rStart * numElements + igemm * hiddenSize,
                     GATE_NUM * hiddenSize)); 
          }
        }
        
        for (int i = rStart; i < rEnd; i++) {
          cudaErrCheck(cudaEventCreate(&events_i[layer][i], cudaEventDisableTiming));
          cudaErrCheck(cudaEventRecord(events_i[layer][i], stream_i[layer]));  
        }            
        
        for (int i = rStart; i < rEnd; i++) {
          cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
          // Optimization 1
          if (GROUP_GEMM) {
            //[4N * N] x [N * m] = [4N * m] 
            if (i > 0) {
              cublasErrCheck(cublasSgemm(handle,
                          transa, transb,
                          GATE_NUM * hiddenSize, miniBatch, hiddenSize,
                          &alpha,
                          layer > 0? &T_f[GetOffsetR(layer)]:&T_f[GATE_NUM * inputSize * hiddenSize], 
                          transa == CUBLAS_OP_N ? GATE_NUM * hiddenSize : hiddenSize,
                          // h_data + i * numElements + layer * (seqLength + 1) * numElements,
                          i_data + layer * seqLength * numElements + seqLength * inputNumElements + (i - 1) * numElements,
                          hiddenSize,
                          &beta,
                          tmp_h + GATE_NUM * layer * numElements, 
                          GATE_NUM * hiddenSize));
            }
          }
          else {
            if (i > 0) {
              for (int igemm =0; igemm < GATE_NUM; igemm++) {
                cublasErrCheck(cublasSgemm(handle,
                           transa, transb,
                           hiddenSize, miniBatch, hiddenSize,
                           &alpha,
                           layer > 0? &T_f[GetOffsetR(layer) + igemm * hiddenSize]:&T_f[GATE_NUM * inputSize * hiddenSize + igemm * hiddenSize], 
                           transa == CUBLAS_OP_N ? GATE_NUM * hiddenSize : hiddenSize,
                           i_data + layer * seqLength * numElements + seqLength * inputNumElements + (i - 1) * numElements,
                           hiddenSize,
                           &beta,
                           tmp_h + GATE_NUM * layer * numElements + igemm * hiddenSize, 
                           GATE_NUM * hiddenSize));
              }
            }
          }

          cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_i[layer][i], 0));
          cudaErrCheck(cudaEventDestroy(events_i[layer][i]));

          // Optimization 3
          if (FUSE_PW) {
            dim3 blockDim;
            dim3 gridDim;
            
            blockDim.x = 256;
            gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;               
            
            elementWise_fp <<< gridDim, blockDim , 0, stream_h[layer] >>> 
                   (hiddenSize, miniBatch,
                    i > 0 ? tmp_h + GATE_NUM * layer * numElements:NULL, 
                    tmp_i + GATE_NUM * i * numElements, 
                    bias + GATE_NUM * layer * hiddenSize,
                    #ifdef PEEPHOLES 
                      peeps + PEEP_NUM * layer * hiddenSize,
                    #else
                      NULL,
                    #endif
                    // TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                    TRAINING ? stateGates + GATE_NUM * (i * numElements + layer * seqLength * numElements) : NULL,
                    // h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                    i_data + i * numElements + layer  * seqLength * numElements + seqLength * inputNumElements,
                    c_data + i * numElements + layer * (seqLength + 1) * numElements,
                    c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                    layer == numLayers - 1 ? label + i * numElements : NULL,
                    layer == numLayers - 1 ? loss + i * numElements: NULL,
                    mask,
                    y_diff + i * numElements + layer * seqLength * numElements,
                    TRAINING);

            cudaErrCheck(cudaGetLastError());
          }
          else {
            LSTM_elementwise_unfused(hiddenSize, miniBatch,
                    i > 0 ? tmp_h + GATE_NUM * layer * numElements:NULL, 
                    tmp_i + GATE_NUM * i * numElements, 
                    bias + GATE_NUM * layer * hiddenSize,
                    #ifdef PEEPHOLES  
                      peeps + PEEP_NUM * layer * hiddenSize,
                    #else
                      NULL,
                    #endif
                    // TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                    // h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                    i_data + i * numElements + layer * seqLength * numElements + seqLength * inputNumElements,
                    c_data + i * numElements + layer * (seqLength + 1) * numElements,
                    c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                    TRAINING,
                    stream_h[layer]);
          }
          if (layer != numLayers - 1) {
            cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
            cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer]));  
          }
        }
      }
    } 

    cublasErrCheck(cublasSetStream(handle, stream_h[numLayers - 1]));

    cublasErrCheck(cublasSasum(handle, numElements * seqLength, 
      loss, 1, sumLoss));


    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));

    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaEventDestroy(start));
    cudaErrCheck(cudaEventDestroy(stop));

    return elapsedTime;
  }

  float Backward(float learningRate) {

    float elapsedTime;
    cudaEvent_t start_bp, stop_bp;
    cudaErrCheck(cudaEventCreate(&start_bp));
    cudaErrCheck(cudaEventCreate(&stop_bp));

    cudaErrCheck(cudaEventRecord(start_bp));


    int lStart = 0;
    int lEnd = 0;
    int rStart = 0;
    int rEnd = 0;

    int rev_lStart = 0;
    int rev_lEnd = 0;
    int rev_rStart = 0;
    int rev_rEnd = 0;

    int recurBatchSize = RECUR_BATCH_BP_SIZE;
    
    while (true) {
       // Many layer "scheduling".
      if (lEnd == 0) {
        lStart = 0;
        lEnd = 1;
        rStart = 0; 
      } else {
        // Move "up" and "left"
        lStart++;
        lEnd++;
        
        rStart -= recurBatchSize;
        
        // Over the top or off the left, reset to layer 0
        if (lEnd > numLayers || rStart < 0) {
           rStart += (lStart + 1) * recurBatchSize;

           lStart = 0;
           lEnd = 1;
        }
        
        // Off the right, step up
        while (rStart >= seqLength && lEnd <= numLayers) {
           lStart++;
           lEnd++;
           
           rStart -= recurBatchSize;
        }
         
        // Over the top or off the left, done!
        if (lEnd > numLayers || rStart < 0) {
           break;
        }
      }

      rEnd = rStart + recurBatchSize;
      if (rEnd > seqLength) rEnd = seqLength;

      rev_lStart = numLayers - lEnd;
      rev_lEnd = numLayers - lStart;
      rev_rStart = seqLength - rStart - 1;
      rev_rEnd = seqLength - rEnd - 1;
      // printf("rev_lStart %d rev_lEnd %d rev_rStart %d rev_rEnd %d\n", rev_lStart, rev_lEnd, rev_rStart, rev_rEnd);
      

      for (int layer = rev_lStart; layer < rev_lEnd; layer++) {                 
          
        for (int i = rev_rStart; i > rev_rEnd; i--) {
          // printf("level %d row %d\n", layer, i);
          cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
          //wait for the upper layer
          // if (layer < numLayers-1) {
          //   cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_h[layer + 1][i], 0));
          //   cudaErrCheck(cudaEventDestroy(events_h[layer + 1][i]));
          // }
          //pointwise operations get diff
          // cudaErrCheck(cudaDeviceSynchronize());
          dim3 blockDim;
          dim3 gridDim;

          blockDim.x = 256;
          gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;               

          elementWise_bp <<< gridDim, blockDim , 0, stream_h[layer] >>> 
              (hiddenSize, miniBatch,
              y_diff + i * numElements + layer * numElements * seqLength, 
              (i == seqLength - 1) ? NULL : stateGates_diff + GATE_NUM * ((i + 1) * numElements + layer * seqLength * numElements), 
              stateGates_diff + GATE_NUM * (i * numElements + layer * seqLength * numElements), 
              stateGates + GATE_NUM * (i * numElements + layer * seqLength  * numElements),
              #ifdef PEEPHOLES
                peeps + PEEP_NUM * layer * hiddenSize,
                peeps_diff + PEEP_NUM * (i * numElements + layer * seqLength * numElements),
              #else
                NULL,
                NULL,
              #endif
              c_data + i * numElements + layer * (seqLength + 1) * numElements,
              c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
              c_diff + layer * numElements,
              i != 0);

          // printWeight();

          cudaErrCheck(cudaGetLastError());

          if (i == 1) {
            cudaErrCheck(cudaEventCreate(&events_i[layer][i], cudaEventDisableTiming));
            cudaErrCheck(cudaEventRecord(events_i[layer][i], stream_h[layer])); 
          }

          if(layer > 0) {
              cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
              cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer])); 
          }

          if (i > 0) {
            //RT * diff = dy
            float alpha = 1.f;
            float beta = 1.f;

            if (GROUP_GEMM) {
              cublasErrCheck(cublasSgemm(handle,
                    CUBLAS_OP_T, transb,
                    hiddenSize, miniBatch, GATE_NUM * hiddenSize,
                    &alpha,
                    layer > 0 ? &T_f[GetOffsetR(layer)]:&T_f[GATE_NUM * hiddenSize * inputSize], 
                    GATE_NUM * hiddenSize,
                    stateGates_diff + GATE_NUM * (i * numElements + layer * seqLength  * numElements),
                    GATE_NUM * hiddenSize,
                    &beta,
                    y_diff + layer * numElements * seqLength + (i - 1) * numElements, 
                    hiddenSize));
            }
            else {
              for (int igemm = 0; igemm < GATE_NUM; igemm++) {
                
                  cublasErrCheck(cublasSgemm(handle,
                    CUBLAS_OP_T, transb,
                    hiddenSize, miniBatch, hiddenSize,
                    &alpha,
                    layer > 0 ? &T_f[GetOffsetR(layer) + igemm * hiddenSize]:&T_f[GATE_NUM * hiddenSize * inputSize + igemm * hiddenSize], 
                    GATE_NUM * hiddenSize,
                    stateGates_diff + GATE_NUM * (i * numElements + layer * seqLength * numElements) + igemm * hiddenSize,
                    GATE_NUM * hiddenSize,
                    &beta,
                    y_diff + layer * numElements * seqLength + (i - 1) * numElements, 
                    hiddenSize));
                }
              }
            }
            
        
        }
          // transa = (PRE_TRANSPOSE && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;

          //W*diff = dx
        int row = rev_rEnd+1;

        if (layer > 0) {

          

          cublasErrCheck(cublasSetStream(handle, stream_h[layer - 1]));

          for (int i = rev_rStart; i > rev_rEnd; i--) {
            cudaErrCheck(cudaStreamWaitEvent(stream_h[layer-1], events_h[layer][i], 0));
            cudaErrCheck(cudaEventDestroy(events_h[layer][i]));
          }

          float alpha = 1.f;
          float beta = 0.f; 

          if (GROUP_GEMM) {
            cublasErrCheck(cublasSgemm(handle,
                      CUBLAS_OP_T, transb,
                      hiddenSize, miniBatch*(rev_rStart - rev_rEnd), GATE_NUM * hiddenSize,
                      &alpha,
                      &T_f[GetOffsetW(layer)], 
                      GATE_NUM * hiddenSize,
                      stateGates_diff + GATE_NUM * (row * numElements + layer * seqLength  * numElements),
                      GATE_NUM * hiddenSize,
                      &beta,
                      y_diff + (layer - 1) * numElements * seqLength + row * numElements, 
                      hiddenSize));
          }
          else {
            for (int igemm = 0; igemm < GATE_NUM; igemm++) {
              cublasErrCheck(cublasSgemm(handle,
                      CUBLAS_OP_T, transb,
                      hiddenSize, miniBatch*(rev_rStart - rev_rEnd), hiddenSize,
                      &alpha,
                      &T_f[GetOffsetW(layer) + igemm * hiddenSize], 
                      GATE_NUM * hiddenSize,
                      stateGates_diff + GATE_NUM * (row * numElements + layer * seqLength  * numElements) + igemm * hiddenSize,
                      GATE_NUM * hiddenSize,
                      &beta,
                      y_diff + (layer - 1) * numElements * seqLength + row * numElements, 
                      hiddenSize));
              beta = 1.f;
            }
          }
        }

         

        if (row == 0) {
            cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
            float lr = -learningRate;

            float beta = 1.f;
            //update W
            cublasErrCheck(cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        GATE_NUM * hiddenSize, layer > 0 ? hiddenSize : inputSize, miniBatch * seqLength,
                        &lr,
                        stateGates_diff + GATE_NUM * layer * seqLength * numElements, 
                        GATE_NUM * hiddenSize,
                        layer > 0 ? i_data + (layer - 1) * seqLength * numElements + seqLength * inputNumElements : i_data,
                        layer > 0 ? hiddenSize : inputSize,
                        &beta,
                        layer > 0 ? &T_f[GetOffsetW(layer)]:&T_f[0], 
                        GATE_NUM * hiddenSize));


            cudaErrCheck(cudaStreamWaitEvent(stream_i[layer], events_i[layer][1], 0));
            cudaErrCheck(cudaEventDestroy(events_i[layer][1]));

            cublasErrCheck(cublasSetStream(handle, stream_i[layer]));

            //update R
            cublasErrCheck(cublasSgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        GATE_NUM * hiddenSize, hiddenSize, miniBatch * (seqLength - 1),
                        &lr,
                        stateGates_diff + GATE_NUM *  (layer * seqLength  * numElements + numElements), 
                        GATE_NUM * hiddenSize,
                        i_data + layer * seqLength * numElements + seqLength * inputNumElements,
                        hiddenSize,
                        &beta,
                        layer > 0 ? &T_f[GetOffsetR(layer)]:&T_f[GATE_NUM * hiddenSize * inputSize], 
                        GATE_NUM * hiddenSize));

            cublasErrCheck(cublasSetStream(handle, stream_h[layer]));

            //update bias
            cublasErrCheck(cublasSgemv(handle,
                        CUBLAS_OP_N, 
                        GATE_NUM * hiddenSize, miniBatch * seqLength, 
                        &lr,
                        stateGates_diff + GATE_NUM *  (layer * seqLength  * numElements), 
                        GATE_NUM * hiddenSize,
                        diff_helper,
                        1,
                        &beta,
                        &bias[layer * hiddenSize * GATE_NUM], 
                        1));

            #ifdef PEEPHOLES
              //update peeps
            cublasErrCheck(cublasSgemv(handle,
                        CUBLAS_OP_N, 
                        PEEP_NUM * hiddenSize, miniBatch * seqLength, 
                        &lr,
                        peeps_diff + PEEP_NUM *  (layer  * numElements * seqLength), 
                        PEEP_NUM * hiddenSize,
                        diff_helper,
                        1,
                        &beta,
                        &peeps[layer * hiddenSize * PEEP_NUM], 
                        1));
            #endif
            
          
        }
      }
    }
    cudaErrCheck(cudaEventRecord(stop_bp));
    cudaErrCheck(cudaEventSynchronize(stop_bp));
    cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start_bp, stop_bp));
    
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaEventDestroy(start_bp));
    cudaErrCheck(cudaEventDestroy(stop_bp));
    return elapsedTime;
  }

  void printChecksum() {

    float* testOutputi;
    // float* testOutputh;
    float* testOutputc;
    
    
    int numElements = hiddenSize * miniBatch;
    testOutputi = (float*)malloc(numElements * seqLength * sizeof(float));
    // testOutputh = (float*)malloc(numElements * numLayers * sizeof(float));
    testOutputc = (float*)malloc(numElements * numLayers * sizeof(float));

    cudaErrCheck(cudaMemcpy(testOutputi, i_data + seqLength * inputNumElements + (numLayers-1) * seqLength * numElements, seqLength * numElements * sizeof(float), cudaMemcpyDeviceToHost));
    for (int layer = 0; layer < numLayers; layer++) {
       // cudaErrCheck(cudaMemcpy(testOutputh + layer * numElements, h_data + seqLength * numElements + layer * (seqLength + 1) * numElements, numElements * sizeof(float), cudaMemcpyDeviceToHost));
       cudaErrCheck(cudaMemcpy(testOutputc + layer * numElements, c_data + seqLength * numElements + layer * (seqLength + 1) * numElements, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    }
    double checksumi = 0.;
    // double checksumh = 0.;
    double checksumc = 0.;
    
    for (int m = 0; m < miniBatch; m++) {
       for (int j = 0; j < seqLength; j++) {
          for (int i = 0; i < hiddenSize; i++) {
             checksumi += testOutputi[j * numElements + m * hiddenSize + i];
             printf("i: (%d,%d): %f\n", j, i, testOutputi[j * numElements + m * hiddenSize + i]);
          }
       }
       for (int j = 0; j < numLayers; j++) {
          for (int i = 0; i < hiddenSize; i++) {         
             // checksumh += testOutputh[j * numElements + m * hiddenSize + i];
             checksumc += testOutputc[j * numElements + m * hiddenSize + i];
          }
       }
       
       if (m == 0) printf("i checksum (example %d) %E\n", m, checksumi);
       // if (m == 0) printf("h checksum (example %d) %E\n", m, checksumh);
       if (m == 0) printf("c checksum (example %d) %E\n", m, checksumc);
    }
    
    printf("i checksum %f     ", checksumi);
    printf("c checksum %E     \n", checksumc);
    // printf("h checksum %E\n", checksumh);
    
    free(testOutputi);
    free(testOutputc);
    // free(testOutputh);
    cudaErrCheck(cudaDeviceSynchronize());
  }

  void printWeight() {
    int t_size = (GATE_NUM * inputSize * hiddenSize + GATE_NUM * hiddenSize * hiddenSize);
    int bias_size = GATE_NUM * hiddenSize * numLayers;
    int stats_size = GATE_NUM * numElements * seqLength;
    int c_diff_size = numLayers * numElements;
    

    float* t_output, * bias_output, * states_output, 
    * y_output, 
    * c_diff_output;

    t_output = (float*)malloc( t_size * sizeof(float));
    bias_output = (float*)malloc(bias_size * sizeof(float));
    states_output = (float*)malloc(4 * GATE_NUM * sizeof(float));
    y_output = (float*)malloc(seqLength * sizeof(float));
    c_diff_output = (float*)malloc(c_diff_size * sizeof(float));

    cudaErrCheck(cudaMemcpy(t_output, T_f, t_size * sizeof(float),cudaMemcpyDeviceToHost));  
    cudaErrCheck(cudaMemcpy(bias_output,bias,bias_size * sizeof(float),cudaMemcpyDeviceToHost)); 

    cudaErrCheck(cudaMemcpy(states_output,stateGates_diff,stats_size * sizeof(float),cudaMemcpyDeviceToHost));  
    cudaErrCheck(cudaMemcpy(y_output, y_diff, seqLength * sizeof(float), cudaMemcpyDeviceToHost));  
    cudaErrCheck(cudaMemcpy(c_diff_output, c_diff, c_diff_size * sizeof(float), cudaMemcpyDeviceToHost));  

    printf("weights:\t");
    for (int i = 0; i < t_size;i++) {
      printf("%f\t",t_output[i]);
    }
    printf("\nbias:\t");
    for (int i = 0; i < bias_size; i++) {
      printf("%f\t",bias_output[i]);
    }
    printf("\n");

    #ifdef PEEPHOLES
      float * peeps_output;
      int peeps_size = PEEP_NUM * numElements * numLayers * seqLength;
      peeps_output = (float*)malloc(peeps_size * sizeof(float));
      cudaErrCheck(cudaMemcpy(peeps_output,peeps_diff, peeps_size * sizeof(float),cudaMemcpyDeviceToHost)); 
      printf("peeps:\t"); 
      for (int i = 0; i < peeps_size; i++) {
        printf("%f\t",peeps_output[i]);
      }
      printf("\n");
    #endif
      
    
    printf("states:\t");
    for (int i = 0; i < stats_size; i++) {
      printf("%f\t",states_output[i]);
    }
    printf("\ny:\t");
    for (int i = 0; i < seqLength; i++) {
      printf("%f\t",y_output[i]);
    }
    printf("\nc_diff:\t");
    for (int i = 0; i < c_diff_size ; i++) {
      printf("%f\t",c_diff_output[i]);
    }
    printf("\n");
    curandErrCheck(curandDestroyGenerator(rng));

  }

  void freeMemory() {
      // cudaErrCheck(cudaFree(h_data));
    cudaErrCheck(cudaFree(i_data));  
    cudaErrCheck(cudaFree(c_data));  

    if (T != T_f) cudaErrCheck(cudaFree(T)); 
    cudaErrCheck(cudaFree(T_f));

    cudaErrCheck(cudaFree(bias));

    cudaErrCheck(cudaFree(loss));
    cudaErrCheck(cudaFree(label));
    cudaErrCheck(cudaFree(mask));
    

    #ifdef PEEPHOLES
      cudaErrCheck(cudaFree(peeps));
    #endif
    cudaErrCheck(cudaFree(tmp_h));
    cudaErrCheck(cudaFree(tmp_i));


    if (TRAINING) {
      // cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaFree(stateGates));
      cudaErrCheck(cudaFree(stateGates_diff));
      cudaErrCheck(cudaFree(y_diff));
      cudaErrCheck(cudaFree(c_diff));
      
      
      #ifdef PEEPHOLES
        cudaErrCheck(cudaFree(peeps_diff));
      #endif
      cudaErrCheck(cudaFree(diff_helper));

    }

    for (int i = 0; i < numLayers; i++) {
      if (stream_i[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_i[i]));
      if (stream_h[i] != NULL) cudaErrCheck(cudaStreamDestroy(stream_h[i]));
    }

    free(stream_i);
    free(stream_h);

    for (int i = 0; i < numLayers; i++) {
      free(events_i[i]);
      free(events_h[i]);
    }

    free(events_i);
    free(events_h);
  }
   
};




float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers, int inputSize, bool checkF) {

  float loss; 
  float elapsedTime;  

  cudaEvent_t global_start, global_end, run_start, run_end;
  cudaErrCheck(cudaEventCreate(&global_start));
  cudaErrCheck(cudaEventCreate(&global_end));
  cudaErrCheck(cudaEventCreate(&run_start));
  cudaErrCheck(cudaEventCreate(&run_end));
  // cudaErrCheck(cudaEventDestroy(run_start));
  // cudaErrCheck(cudaEventDestroy(run_end));

  cudaErrCheck(cudaEventRecord(global_start));

  LSTM_scheduler scheduler(hiddenSize,miniBatch,seqLength,numLayers,inputSize);


  scheduler.init();
  printf("Initialize success\n");
  


  

  cudaErrCheck(cudaEventRecord(run_start));
  // cudaErrCheck(cudaEventSynchronize(run_start));
  
  // scheduler.Forward(&loss);
  // printf("Forward loss is %f\n", loss);


  // if (checkF) {
  //   scheduler.printChecksum();
  // }


  for (int i = 0; i < 10; i++) {
    elapsedTime = scheduler.Forward(&loss);
    printf("Forward time is %f, loss is %f\n", elapsedTime, loss);

    if (TRAINING) {
      #ifdef UPDATE
      scheduler.clearStates();
      #endif
      elapsedTime = scheduler.Backward(0.2);
      printf("Backward time is %f\n", elapsedTime);
    }

    // scheduler.printWeight();
  }

  // scheduler.Forward(&loss);
  // printf("Forward loss is %f\n", loss);

  cudaErrCheck(cudaEventRecord(run_end));
  // We're done. Print some checksums
  // if (checkF) {
  //   scheduler.printChecksum();
  // }
  scheduler.freeMemory();
  cudaErrCheck(cudaDeviceSynchronize());

  cudaErrCheck(cudaEventRecord(global_end));
  cudaErrCheck(cudaEventSynchronize(global_end));
  
  
  
  
  
  cudaErrCheck(cudaEventElapsedTime(&elapsedTime, run_start, run_end));
  printf("Running time used %f ms, avg %f\n", elapsedTime, elapsedTime/10);


  cudaErrCheck(cudaEventElapsedTime(&elapsedTime, global_start, global_end));
  printf("Total time used %f ms\n", elapsedTime);


  

  // cudaErrCheck(cudaEventDestroy(global_start));
  // cudaErrCheck(cudaEventDestroy(global_end));
  cudaErrCheck(cudaEventElapsedTime(&elapsedTime, global_start, run_start));
  printf("Initialize time used %f ms\n", elapsedTime);

  cudaErrCheck(cudaEventElapsedTime(&elapsedTime, run_end, global_end));
  printf("Memory free time used %f ms\n", elapsedTime);
  

  return 0;
}


int main(int argc, char* argv[]) {

  int seqLength;
  int numLayers;
  int hiddenSize;
  int miniBatch; 
  int inputSize;

  printf("\n");   


  if (argc == 6) {
    seqLength = atoi(argv[1]);
    numLayers =  atoi(argv[2]);
    hiddenSize =  atoi(argv[3]);
    miniBatch =  atoi(argv[4]);   
    inputSize = atoi(argv[5]);
  }
  else if (argc == 1) {
    printf("Running with default settings\n");
    inputSize = 512;
    seqLength = 100;
    numLayers = 4;
    hiddenSize = 512;
    miniBatch = 64;
  }
  else {
    printf("Usage: ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch> <inputSize>\n");
    return 1;      
  }

  printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d inputSize %d\n", seqLength, numLayers, hiddenSize, miniBatch, inputSize);  

  int numRuns = 1;

  float totalTime = 0.f;
  for (int run = 0; run < numRuns; run++) {
    totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers, inputSize, true);
  }

  // printf("Runtime %fms\n", totalTime / numRuns);
  return time < 0;
}


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

// Performance is not significantly different, but false saves memory. 
// False does not work with unfused pointwise ops.
#define TRAINING (true)

#ifndef PERFOPTS
   #define PERFOPTS (15)
#endif

#define GROUP_GEMM ((PERFOPTS & 1))
#define USE_STREAMS ((PERFOPTS & 2))
#define FUSE_PW ((PERFOPTS & 4))
#define PRE_TRANSPOSE ((PERFOPTS & 8))
#define RECUR_BATCH_SIZE (((PERFOPTS & 16) ? 2 : 1))

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



// Device functions
__forceinline__ __device__ float sigmoidf(float in) {
   return 1.f / (1.f + expf(-in));  
}

__forceinline__ __device__ float de_sigmoidf(float out) {
   return out * (1-out);
} 

__forceinline__ __device__ float de_tanhf(float out) {
   return 1.f - pow(out, 2);
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
                               float * __restrict__ h_data,
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

                 
   for (int i = 0; i < 4; i++) {
      if (tmp_h != NULL) {
         pw_vecAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, tmp_i  + i * numElements, tmp_h  + i * numElements, numElements);
         cudaErrCheck(cudaGetLastError());
      }

      pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + i       * hiddenSize, numElements, hiddenSize);
      cudaErrCheck(cudaGetLastError());
      
      if (i == 0) {
         pw_biasAdd <<< gridDim, blockDim, 0, stream >>> (tmp_i + i * numElements, bias + i       * hiddenSize, numElements, hiddenSize);
         cudaErrCheck(cudaGetLastError());
      }
      
      if (training) {
         printf("LSTM_elementWise_unfused does not support training\n"); 
         return 1;
      }
   }

   float *in_gate     = tmp_i + 0 * numElements;//i
   float *forget_gate = tmp_i + 1 * numElements;//f
   float *in_gate2    = tmp_i + 2 * numElements;//z
   float *out_gate    = tmp_i + 3 * numElements;//o   

 
   if (c_in != NULL) {
      //i_t += p_i + c_t-1          
      pw_peepsAdd <<< gridDim, blockDim, 0, stream >>> (in_gate, peeps, c_in, numElements, hiddenSize);
      cudaErrCheck(cudaGetLastError());
      //f_t += p_f + c_t-1          
      pw_peepsAdd <<< gridDim, blockDim, 0, stream >>> (forget_gate, peeps + 1 * hiddenSize, c_in, numElements, hiddenSize);
      cudaErrCheck(cudaGetLastError());

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
   }
   else {
      

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
   pw_vecMul <<< gridDim, blockDim, 0, stream >>> (h_data, out_gate, in_gate, numElements);
   cudaErrCheck(cudaGetLastError());
   
   
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
                               float *h_out, //h_data
                               float *i_out,
                               float *c_in,
                               float *c_out,
                               bool training) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int numElements = miniBatch * hiddenSize;
   
   if (index >= numElements) return;
   
   int batch = index / hiddenSize;
   int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;   
   
   float g[4];

   for (int i = 0; i < 4; i++) {
      g[i] = tmp_i[i * hiddenSize + gateIndex] + tmp_h[i * hiddenSize + gateIndex];
      g[i] += bias[i * hiddenSize + index % hiddenSize];
   }   
   
   g[0] += peeps[index % hiddenSize] * c_in[index];
   g[1] += peeps[hiddenSize + index % hiddenSize] * c_in[index];
   
   float in_gate     = sigmoidf(g[0]);//i
   float forget_gate = sigmoidf(g[1]);//f
   float in_gate2    = tanhf(g[2]);//z
   
   float val = (forget_gate * c_in[index]) + (in_gate * in_gate2);
   c_out[index] = val;

   g[3] += peeps[hiddenSize*2 + index % hiddenSize] * c_out[index];
   float out_gate    = sigmoidf(g[3]);//o
   
   if(training) {
      
      stateGates[gateIndex] = in_gate;
      stateGates[hiddenSize + gateIndex] = forget_gate;
      stateGates[2*hiddenSize + gateIndex] = in_gate2;
      stateGates[3*hiddenSize + gateIndex] = out_gate;
   }

   val = out_gate * tanhf(val); //h                            

   h_out[index] = val;
   i_out[index] = val;
}

// Fused forward kernel
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
   int gateIndex = (index % hiddenSize) + 4 * batch * hiddenSize;   
   

   float in_gate = stateGates[gateIndex];
   float forget_gate = stateGates[hiddenSize + gateIndex];
   float in_gate2 = stateGates[2 * hiddenSize + gateIndex];
   float out_gate = stateGates[3 * hiddenSize + gateIndex];

   float out_diff = y_diff[index]*tanhf(c_out[index])*de_sigmoidf(out_gate); //do
   float peep_diff = 0;
   if (stateGates_diff_in != NULL) { 
     peep_diff = peeps[index % hiddenSize] * out_diff + peeps[hiddenSize + index % hiddenSize] * stateGates_diff_in[gateIndex] + peeps[2 * hiddenSize + index % hiddenSize] * stateGates_diff_in[hiddenSize + gateIndex];
  }
   float local_c_diff = y_diff[index]*out_gate*de_tanhf(tanhf(c_out[index])) + peep_diff + c_diff[index];
   float forget_diff = local_c_diff * c_in[index] * de_sigmoidf(forget_gate);
   float in_diff = local_c_diff * tanhf(in_gate2) * de_sigmoidf(in_gate);
   float in_diff2 = local_c_diff * in_gate * de_tanhf(in_gate2);

   stateGates_diff_out[gateIndex] = in_diff;
   stateGates_diff_out[hiddenSize + gateIndex] = forget_diff;
   stateGates_diff_out[2 * hiddenSize + gateIndex] = in_diff2;
   stateGates_diff_out[3 * hiddenSize + gateIndex] = out_diff;       

   if (peeps_update) {
      peeps_diff[gateIndex] += in_diff * c_in[index];//p_i
      peeps_diff[hiddenSize + gateIndex] += forget_diff * c_in[index]; //p_f
      peeps_diff[2 * hiddenSize + gateIndex] += out_diff * c_out[index]; //p_o
   }
   c_diff[index] = local_c_diff * forget_diff;
}

struct LSTM_scheduler
{
   float *h_data;//y
   float *i_data;//x

   float *c_data;//c
   
   float *T;
   float *T_f;
   
   float *bias;
   float *peeps;

   float *tmp_h;
   float *tmp_i;
   // float *linearGates;
   float *stateGates;

   //diff
   float *stateGates_diff; //di,df,dz,do
   float *y_diff;//dy
   float *T_diff;//dW, dR
   float *bias_diff;
   float *diff_helper;
   float *peeps_diff;
   float *c_diff;//dc*ft

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

    

   cublasOperation_t transa;
   cublasOperation_t transb;

   LSTM_scheduler(int hiddenSize_, int miniBatch_, int seqLength_, int numLayers_)
   {
      transa = (PRE_TRANSPOSE && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;
      transb = CUBLAS_OP_N;

      hiddenSize = hiddenSize_;
      miniBatch = miniBatch_;
      seqLength = seqLength_; 
      numLayers = numLayers_; 
      
      
      numElements = hiddenSize * miniBatch;
      cublasErrCheck(cublasCreate(&handle));
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

   void init() {
      
      cudaErrCheck(cudaMalloc((void**)&h_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&i_data, (seqLength) * (numLayers + 1) * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&c_data, (seqLength + 1) * (numLayers) * numElements * sizeof(float)));

      

      cudaErrCheck(cudaMalloc((void**)&T, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&T_f, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
      
      
      cudaErrCheck(cudaMalloc((void**)&bias, numLayers * hiddenSize * 4 * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&peeps, numLayers * hiddenSize * 3 * sizeof(float)));
      
      // Workspace
      cudaErrCheck(cudaMalloc((void**)&tmp_h, 4 * numLayers * numElements * sizeof(float)));
      cudaErrCheck(cudaMalloc((void**)&tmp_i, 4 * seqLength * numElements * sizeof(float)));
      
      // // Activations
      if (TRAINING) {
         // cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&stateGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&stateGates_diff, 4 * seqLength * numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&y_diff, seqLength * numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&c_diff, numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&T_diff, numLayers * hiddenSize * hiddenSize * 8 * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&bias_diff, numLayers * hiddenSize * 4 * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&peeps_diff, numLayers * numElements * 3 * sizeof(float)));
         cudaErrCheck(cudaMalloc((void**)&diff_helper, miniBatch * seqLength * sizeof(float)));
         cudaErrCheck(cudaMemset(diff_helper, 1,  miniBatch * seqLength * sizeof(float)));
      }

      // Initialise with random values.
      curandGenerator_t rng;
      curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
      curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 1337ull));
      curandErrCheck(curandGenerateUniform(rng, h_data, (seqLength + 1) * (numLayers) * numElements));
      curandErrCheck(curandGenerateUniform(rng, c_data, (seqLength + 1) * (numLayers) * numElements));
      curandErrCheck(curandGenerateUniform(rng, i_data, (seqLength) * (numLayers + 1) * numElements));
      
      curandErrCheck(curandGenerateUniform(rng, T_f, numLayers * hiddenSize * hiddenSize * 8));
      curandErrCheck(curandGenerateUniform(rng, bias, numLayers * hiddenSize * 4));
      curandErrCheck(curandGenerateUniform(rng, peeps, numLayers * hiddenSize * 3));

      if (TRAINING) {
         curandErrCheck(curandGenerateUniform(rng, y_diff+seqLength*(numLayers-1)*numElements, seqLength * numElements));
      }
      curandErrCheck(curandDestroyGenerator(rng));

      
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

   float Forward() {

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
                           4 * hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                           &alpha,
                           &T_f[layer * 8 * hiddenSize * hiddenSize],
                           transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                           i_data + rStart * numElements + layer * seqLength * numElements,
                           hiddenSize,
                           &beta,
                           tmp_i + 4 * rStart * numElements,
                           4 * hiddenSize));
            }
            else {
               for (int igemm =0; igemm < 4; igemm++) {
                  cublasErrCheck(cublasSgemm(handle,
                           transa, transb,
                           hiddenSize, miniBatch * (rEnd - rStart), hiddenSize,
                           &alpha,
                           &T_f[layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize],
                           transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                           i_data + rStart * numElements + layer * seqLength * numElements,
                           hiddenSize,
                           &beta,
                           tmp_i + 4 * rStart * numElements + igemm * hiddenSize,
                           4 * hiddenSize)); 
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
                  cublasErrCheck(cublasSgemm(handle,
                              transa, transb,
                              4 * hiddenSize, miniBatch, hiddenSize,
                              &alpha,
                              &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize], 
                              transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                              h_data + i * numElements + layer * (seqLength + 1) * numElements,
                              hiddenSize,
                              &beta,
                              tmp_h + 4 * layer * numElements, 
                              4 * hiddenSize));
               }
               else {
                  for (int igemm =0; igemm < 4; igemm++) {
                     cublasErrCheck(cublasSgemm(handle,
                                 transa, transb,
                                 hiddenSize, miniBatch, hiddenSize,
                                 &alpha,
                                 &T_f[4 * hiddenSize * hiddenSize + layer * 8 * hiddenSize * hiddenSize + igemm * hiddenSize], 
                                 transa == CUBLAS_OP_N ? 4 * hiddenSize : hiddenSize,
                                 h_data + i * numElements + layer * (seqLength + 1) * numElements,
                                 hiddenSize,
                                 &beta,
                                 tmp_h + 4 * layer * numElements + igemm * hiddenSize, 
                                 4 * hiddenSize));
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
                          tmp_h + 4 * layer * numElements, 
                          tmp_i + 4 * i * numElements, 
                          bias + 4 * layer * hiddenSize,
                          peeps + 3 * layer * hiddenSize,
                          // TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                          TRAINING ? stateGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                          h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                          i_data + i * numElements + (layer + 1) * seqLength * numElements,
                          c_data + i * numElements + layer * (seqLength + 1) * numElements,
                          c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                          TRAINING);
                  cudaErrCheck(cudaGetLastError());
               }
               else {
                  LSTM_elementwise_unfused(hiddenSize, miniBatch,
                          tmp_h + 4 * layer * numElements, 
                          tmp_i + 4 * i * numElements, 
                          bias + 4 * layer * hiddenSize,
                          peeps + 3 * layer * hiddenSize,
                          // TRAINING ? linearGates + 4 * (i * numElements + layer * seqLength * numElements) : NULL,
                          h_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                          i_data + i * numElements + (layer + 1) * seqLength * numElements,
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
      cudaErrCheck(cudaEventRecord(stop));
      cudaErrCheck(cudaEventSynchronize(stop));
      cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start, stop));
      
      cudaErrCheck(cudaDeviceSynchronize());
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

      int recurBatchSize = 1;
      
      
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

         rev_lStart = numLayers - lEnd;
         rev_lEnd = numLayers - lStart;
         rev_rStart = seqLength - rStart - 1;
         rev_rEnd = seqLength - rEnd - 1;
         // printf("lStart %d lEnd %d rStart %d rEnd %d\n", rev_lStart, rev_lEnd,
         //    rev_rStart, rev_rEnd);
         if (rEnd > seqLength) rEnd = seqLength;
         
         for (int layer = rev_lStart; layer < rev_lEnd; layer++) {         
            
            

            //wait for the upper layer
            for (int i = rev_rStart; i > rev_rEnd; i--) {
               // printf("level %d row %d\n", layer, i);

               cublasErrCheck(cublasSetStream(handle, stream_h[layer]));
               if (layer < numLayers-1) {
                  cudaErrCheck(cudaStreamWaitEvent(stream_h[layer], events_h[layer + 1][i], 0));
                  cudaErrCheck(cudaEventDestroy(events_h[layer + 1][i]));
               }
               //pointwise operations get diff
               dim3 blockDim;
               dim3 gridDim;
               
               blockDim.x = 256;
               gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;               
               
               elementWise_bp <<< gridDim, blockDim , 0, stream_h[layer] >>> 
                      (hiddenSize, miniBatch,
                       y_diff + i * numElements + layer * numElements * seqLength, 
                       ((i + 1) >= numLayers) ? NULL : stateGates_diff + 4 * ((i + 1) * numElements + layer * seqLength  * numElements), 
                       stateGates_diff + 4 * (i * numElements + layer * seqLength  * numElements), 
                       stateGates + 4 * (i * numElements + layer * seqLength  * numElements),
                       peeps + 3 * layer * hiddenSize,
                       peeps_diff + 3 * layer * numElements,
                       c_data + i * numElements + layer * (seqLength + 1) * numElements,
                       c_data + (i + 1) * numElements + layer * (seqLength + 1) * numElements,
                       c_diff + layer * numElements,
                       i == 0);


               cudaErrCheck(cudaGetLastError());
               // transa = (PRE_TRANSPOSE && (seqLength > 1)) ? CUBLAS_OP_N : CUBLAS_OP_T;

               
               //W*diff = dx
               if (layer >= 1) {
                  float alpha = 1.f;
                  float beta = 0.f; 
                  cublasErrCheck(cublasSgemm(handle,
                              CUBLAS_OP_T, transb,
                              hiddenSize, miniBatch, 4 * hiddenSize,
                              &alpha,
                              &T_f[layer * 8 * hiddenSize * hiddenSize], 
                              4 * hiddenSize,
                              stateGates_diff + 4 * (i * numElements + layer * seqLength  * numElements),
                              4 * hiddenSize,
                              &beta,
                              y_diff + (layer - 1) * numElements * seqLength + i * numElements, 
                              hiddenSize));
               }

               if(layer != 0) {
                  cudaErrCheck(cudaEventCreate(&events_h[layer][i], cudaEventDisableTiming));
                  cudaErrCheck(cudaEventRecord(events_h[layer][i], stream_h[layer])); 
               } 


               
               if (i >= 1) {
                  //RT * diff = dy
                  float alpha = 1.f;
                  float beta = 1.f;
                  cublasErrCheck(cublasSgemm(handle,
                                 CUBLAS_OP_T, transb,
                                 hiddenSize, miniBatch, 4 * hiddenSize,
                                 &alpha,
                                 &T_f[layer * 8 * hiddenSize * hiddenSize + 4 * hiddenSize * hiddenSize], 
                                 4 * hiddenSize,
                                 stateGates_diff + 4 * (i * numElements + layer * seqLength  * numElements),
                                 4 * hiddenSize,
                                 &beta,
                                 y_diff + layer * numElements * seqLength + (i - 1) * numElements, 
                                 hiddenSize));
                  cudaErrCheck(cudaEventCreate(&events_i[layer][i], cudaEventDisableTiming));
                  cudaErrCheck(cudaEventRecord(events_i[layer][i], stream_h[layer])); 
              } 
              else {
                  
                  
                  float lr = -learningRate;

                  float beta = 1.f;
                  //update W
                  cublasErrCheck(cublasSgemm(handle,
                                 CUBLAS_OP_N, transb,
                                 4 * hiddenSize, miniBatch * seqLength, hiddenSize,
                                 &lr,
                                 stateGates_diff + 4 *  layer * seqLength  * numElements, 
                                 4 * hiddenSize,
                                 i_data + layer * seqLength * numElements,
                                 hiddenSize,
                                 &beta,
                                 &T_f[layer * 8 * hiddenSize * hiddenSize], 
                                 4 * hiddenSize));

                  
                  cudaErrCheck(cudaStreamWaitEvent(stream_i[layer], events_i[layer][i+1], 0));
                  cudaErrCheck(cudaEventDestroy(events_i[layer][i+1]));

                  cublasErrCheck(cublasSetStream(handle, stream_i[layer]));

                  //update R
                  cublasErrCheck(cublasSgemm(handle,
                                 CUBLAS_OP_N, transb,
                                 4 * hiddenSize, miniBatch * (seqLength - 1), hiddenSize,
                                 &lr,
                                 stateGates_diff + 4 *  (layer * seqLength  * numElements + numElements), 
                                 4 * hiddenSize,
                                 i_data + (layer + 1) * seqLength * numElements,
                                 hiddenSize,
                                 &beta,
                                 &T_f[layer * 8 * hiddenSize * hiddenSize + 4 * hiddenSize * hiddenSize], 
                                 4 * hiddenSize));
                  //update bias
                  cublasErrCheck(cublasSgemv(handle,
                                 CUBLAS_OP_N, 
                                 4 * hiddenSize, miniBatch * seqLength, 
                                 &lr,
                                 stateGates_diff + 4 *  (layer * seqLength  * numElements), 
                                 4 * hiddenSize,
                                 diff_helper,
                                 1,
                                 &beta,
                                 &bias[layer * hiddenSize * 4], 
                                 4 * hiddenSize));

                  //update peeps
                  cublasErrCheck(cublasSgemv(handle,
                                 CUBLAS_OP_N, 
                                 3 * hiddenSize, miniBatch * seqLength, 
                                 &lr,
                                 peeps_diff + 3 *  (layer  * numElements), 
                                 3 * hiddenSize,
                                 diff_helper,
                                 1,
                                 &beta,
                                 &peeps[layer * hiddenSize * 3], 
                                 3 * hiddenSize));

              }


            }
            

         }
      }
      cudaErrCheck(cudaEventRecord(stop_bp));
      cudaErrCheck(cudaEventSynchronize(stop_bp));
      cudaErrCheck(cudaEventElapsedTime(&elapsedTime, start_bp, stop_bp));
      
      cudaErrCheck(cudaDeviceSynchronize());
      return elapsedTime;
     
   }

   void printChecksum() {

      float* testOutputi;
      float* testOutputh;
      float* testOutputc;
      
      
      int numElements = hiddenSize * miniBatch;
      testOutputi = (float*)malloc(numElements * seqLength * sizeof(float));
      testOutputh = (float*)malloc(numElements * numLayers * sizeof(float));
      testOutputc = (float*)malloc(numElements * numLayers * sizeof(float));
   
      cudaErrCheck(cudaMemcpy(testOutputi, i_data + numLayers * seqLength * numElements, seqLength * numElements * sizeof(float), cudaMemcpyDeviceToHost));
      for (int layer = 0; layer < numLayers; layer++) {
         cudaErrCheck(cudaMemcpy(testOutputh + layer * numElements, h_data + seqLength * numElements + layer * (seqLength + 1) * numElements, numElements * sizeof(float), cudaMemcpyDeviceToHost));
         cudaErrCheck(cudaMemcpy(testOutputc + layer * numElements, c_data + seqLength * numElements + layer * (seqLength + 1) * numElements, numElements * sizeof(float), cudaMemcpyDeviceToHost));
      }
      double checksumi = 0.;
      double checksumh = 0.;
      double checksumc = 0.;
      
      for (int m = 0; m < miniBatch; m++) {
         for (int j = 0; j < seqLength; j++) {
            for (int i = 0; i < hiddenSize; i++) {
               checksumi += testOutputi[j * numElements + m * hiddenSize + i];
               if (hiddenSize <= 8) printf("i: (%d,%d): %E\n", j, i, testOutputi[j * numElements + m * hiddenSize + i]);
            }
         }
         for (int j = 0; j < numLayers; j++) {
            for (int i = 0; i < hiddenSize; i++) {         
               checksumh += testOutputh[j * numElements + m * hiddenSize + i];
               checksumc += testOutputc[j * numElements + m * hiddenSize + i];
            }
         }
         
         if (m == 0) printf("i checksum (example %d) %E\n", m, checksumi);
         if (m == 0) printf("h checksum (example %d) %E\n", m, checksumh);
         if (m == 0) printf("c checksum (example %d) %E\n", m, checksumc);
      }
      
      printf("i checksum %E     ", checksumi);
      printf("c checksum %E     ", checksumc);
      printf("h checksum %E\n", checksumh);
      
      free(testOutputi);
      free(testOutputc);
      free(testOutputh);
      cudaErrCheck(cudaDeviceSynchronize());
   }

   void freeMemory() {
      cudaErrCheck(cudaFree(h_data));
      cudaErrCheck(cudaFree(i_data));  
      cudaErrCheck(cudaFree(c_data));  

      if (T != T_f) cudaErrCheck(cudaFree(T)); 
      cudaErrCheck(cudaFree(T_f));
      
      cudaErrCheck(cudaFree(bias));
      cudaErrCheck(cudaFree(peeps));
      cudaErrCheck(cudaFree(tmp_h));
      cudaErrCheck(cudaFree(tmp_i));
      

      if (TRAINING) {
         // cudaErrCheck(cudaMalloc((void**)&linearGates, 4 * seqLength * numLayers * numElements * sizeof(float)));
         cudaErrCheck(cudaFree(stateGates));
         cudaErrCheck(cudaFree(stateGates_diff));
         cudaErrCheck(cudaFree(y_diff));
         cudaErrCheck(cudaFree(c_diff));
         cudaErrCheck(cudaFree(T_diff));
         cudaErrCheck(cudaFree(bias_diff));
         cudaErrCheck(cudaFree(peeps_diff));
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




float LSTMTest(int hiddenSize, int miniBatch, int seqLength, int numLayers, bool checkF) {
   
   LSTM_scheduler scheduler(hiddenSize,miniBatch,seqLength,numLayers);
   

   scheduler.init();

  
   // Timing starts here
   float elapsedTime = scheduler.Forward();
   printf("Forward time is %f\n", elapsedTime);
   
   // We're done. Print some checksums
   if (checkF) {
      scheduler.printChecksum();
   }

   if(TRAINING) {
      elapsedTime = scheduler.Backward(0.2);
      printf("Backward time is %f\n", elapsedTime);
   }
   
   scheduler.freeMemory();
   
   return 0;
}


int main(int argc, char* argv[]) {
   int seqLength;
   int numLayers;
   int hiddenSize;
   int miniBatch; 
   
   if (argc == 5) {
      seqLength = atoi(argv[1]);
      numLayers =  atoi(argv[2]);
      hiddenSize =  atoi(argv[3]);
      miniBatch =  atoi(argv[4]);   
   }
   else if (argc == 1) {
      printf("Running with default settings\n");
      seqLength = 100;
      numLayers = 4;
      hiddenSize = 512;
      miniBatch = 64;
   }
   else {
      printf("Usage: ./LSTM <seqLength> <numLayers> <hiddenSize> <miniBatch>\n");
      return 1;      
   }

   printf("seqLength %d, numLayers %d, hiddenSize %d, miniBatch %d\n", seqLength, numLayers, hiddenSize, miniBatch);  
   
   int numRuns = 1;
   
   float totalTime = 0.f;
   for (int run = 0; run < numRuns; run++) {
      totalTime += LSTMTest(hiddenSize, miniBatch, seqLength, numLayers, true);
   }
   
   // printf("Runtime %fms\n", totalTime / numRuns);


   
   return time < 0;
}


#include <cublas_v2.h>   
// #include <helper_cuda.h>  
#include <stdio.h> 
#include <curand.h>


#define checkCudaErrors(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


int main(void)  
{  
    // float alpha=1.0;  
    // float beta=0.0;  
    float h_A[6]={1,1,2,2,3,3};  
    float h_B[4]={1,2,3,4};  
    float h_C[6];  
    
    float *d_a,*d_b,*d_c;  

    curandGenerator_t rng;
    curandErrCheck(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(rng, 13ull));
    checkCudaErrors(cudaMalloc((void**)&d_a,6*sizeof(float)));  
    checkCudaErrors(cudaMalloc((void**)&d_b,4*sizeof(float)));  
    checkCudaErrors(cudaMalloc((void**)&d_c,6*sizeof(float)));  
    checkCudaErrors(cudaMemcpy(d_a,&h_A,6*sizeof(float),cudaMemcpyHostToDevice));  
    checkCudaErrors(cudaMemcpy(d_b,&h_B,4*sizeof(float),cudaMemcpyHostToDevice));  
    // checkCudaErrors(cudaMemset(d_c,0,6*sizeof(float)));  
    curandErrCheck(curandGenerateUniform(rng, d_c, 6));
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,3,2,&alpha,d_b,1,d_a,2,&beta,d_c,1);  
    // cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,3,2,2,&alpha,d_a,2,d_b,2,&beta,d_c,3);  
    checkCudaErrors(cudaMemcpy(h_C,d_c,6*sizeof(float),cudaMemcpyDeviceToHost));  
    for(int i=0;i<6;i++)  
    {  
        printf("%f\n",h_C[i]);  
    }  
    printf("\n");  
    return 0;  
}  

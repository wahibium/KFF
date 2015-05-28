#include "cutil_temp.h"
#include "DebugOutput.h"

float* debug_init_memory(void) {
	// allocate device memory
	float *DEBUG_OUTPUT_D;


	CUDA_SAFE_CALL(cudaMalloc((void**) &DEBUG_OUTPUT_D, \
				  sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol("DEBUG_OUTPUT_C", \
					  &DEBUG_OUTPUT_D , \
					  sizeof(float*), 0, \
					  cudaMemcpyHostToDevice));

	// Initialize memory to zero
	debug_reset(DEBUG_OUTPUT_D);

	return DEBUG_OUTPUT_D;
	
}



float debug_print(float* DEBUG_OUTPUT_D) {
	float DEBUG_OUTPUT;
 	CUDA_SAFE_CALL(cudaMemcpy(&DEBUG_OUTPUT, \
				  DEBUG_OUTPUT_D, \
				  sizeof (float), \
				  cudaMemcpyDeviceToHost) );

	return DEBUG_OUTPUT;
}


void debug_reset(float* DEBUG_OUTPUT_D) {
	// Reset memory to zero
	float zero = 0;
 	CUDA_SAFE_CALL(cudaMemcpy(DEBUG_OUTPUT_D, \
				  &zero, \
				  sizeof (float), \
				  cudaMemcpyHostToDevice) );

}

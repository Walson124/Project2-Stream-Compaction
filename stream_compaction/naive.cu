#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int thisIndex = blockDim.x * blockIdx.x + threadIdx.x;
            if (thisIndex >= n) {
                return;
            }
            if (thisIndex >= (1 << (d-1))) {
                odata[thisIndex] = idata[thisIndex - (1 << (d - 1))] + idata[thisIndex];
            }
            else {
                odata[thisIndex] = idata[thisIndex];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* bufferA;
            int* bufferB;
            cudaMalloc((void**) &bufferA, n * sizeof(int));
            cudaMalloc((void**) &bufferB, n * sizeof(int));
            cudaMemcpy(bufferB, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan<<<blocks, blockSize>>>(n, d, bufferA, bufferB);
                int* temp = bufferA;
                bufferA = bufferB;
                bufferB = temp;
            }
            cudaMemset(bufferA, 0, sizeof(int));
            cudaMemcpy(bufferA + 1, bufferB, (n - 1) * sizeof(int), cudaMemcpyDeviceToDevice);
            timer().endGpuTimer();
            
            cudaMemcpy(odata, bufferA, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bufferA);
            cudaFree(bufferB);
        }
    }
}

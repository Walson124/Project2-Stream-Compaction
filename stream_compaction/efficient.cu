#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* idata) {
            int thisIndex = threadIdx.x + blockDim.x * blockIdx.x;
            if (thisIndex >= n) {
                return;
            }
            if (thisIndex % (1 << (d + 1)) == 0) {
                idata[thisIndex + (1 << (d + 1)) - 1] = idata[thisIndex + (1 << d) - 1] + idata[thisIndex + (1 << (d + 1)) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* idata) {
            int thisIndex = threadIdx.x + blockDim.x * blockIdx.x;
            if (thisIndex >= n) {
                return;
            }
            if (thisIndex % (1 << (d + 1)) == 0) {
                int t = idata[thisIndex + (1 << d) - 1];
                idata[thisIndex + (1 << d) - 1] = idata[thisIndex + (1 << (d + 1)) - 1];
                idata[thisIndex + (1 << (d + 1)) - 1] = t + idata[thisIndex + (1 << (d + 1)) - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // TODO
            int* idataBuffer;

            // extra pad for 2^ilog2ceil(n)
            int padding = 1 << ilog2ceil(n);
            cudaMalloc((void**) &idataBuffer, padding * sizeof(int));
            cudaMemset(idataBuffer, 0, padding * sizeof(int));
            cudaMemcpy(idataBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (padding + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernUpSweep<<<blocks, blockSize>>>(padding, d, idataBuffer);
            }

            cudaMemset(idataBuffer + padding - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDownSweep<<<blocks, blockSize>>>(padding, d, idataBuffer);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, idataBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(idataBuffer);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int* boolsBuffer;
            int* idataBuffer;
            int* odataBuffer;
            cudaMalloc((void**) &boolsBuffer, n * sizeof(int));
            cudaMalloc((void**) &idataBuffer, n * sizeof(int));
            cudaMalloc((void**) &odataBuffer, n * sizeof(int));
            cudaMemcpy(idataBuffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            // for scan later
            int* scanResults;
            int padding = 1 << ilog2ceil(n);
            int scanBlocks = (padding + blockSize - 1) / blockSize;
            cudaMalloc((void**) &scanResults, padding * sizeof(int));

            timer().startGpuTimer();
            StreamCompaction::Common::kernMapToBoolean<<<blocks, blockSize>>>(n, boolsBuffer, idataBuffer);

            // scan part
            cudaMemset(scanResults, 0, padding * sizeof(int));
            cudaMemcpy(scanResults, boolsBuffer, n * sizeof(int), cudaMemcpyDeviceToDevice);
            for (int d = 0; d < ilog2ceil(n); d++) {
                kernUpSweep<<<scanBlocks, blockSize>>>(padding, d, scanResults);
            }

            cudaMemset(scanResults + padding - 1, 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                kernDownSweep<<<scanBlocks, blockSize>>>(padding, d, scanResults);
            }
            
            int temp1;
            int temp2;
            cudaMemcpy(&temp1, scanResults + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&temp2, boolsBuffer + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = temp1 + temp2;
            
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, odataBuffer, idataBuffer, boolsBuffer, scanResults);
            timer().endGpuTimer();

            cudaMemcpy(odata, odataBuffer, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(boolsBuffer);
            cudaFree(idataBuffer);
            cudaFree(odataBuffer);
            cudaFree(scanResults);

            return count;
            return -1;
        }
    }
}

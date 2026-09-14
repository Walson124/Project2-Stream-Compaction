#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int temp = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = temp;
                temp += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int temp = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[temp] = idata[i];
                    temp++;
                }
            }
            timer().endCpuTimer();
            return temp;
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            // build into booleans
            // then build sums with boolean to get index
            // then insert idata[i] into odata[index] when boolean[i] == 1
            int* booleans = new int[n];
            int* sums = new int[n];
            int count = 0;
            int tempsum = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    booleans[i] = 1;
                    sums[i] = tempsum;
                    tempsum++;
                    count++;
                }
                else {
                    booleans[i] = 0;
                    sums[i] = tempsum;
                }
            }
            for (int i = 0; i < n; i++) {
                if (booleans[i] == 1) {
                    odata[sums[i]] = idata[i];
                }
            }
            delete[] booleans;
            delete[] sums;
            timer().endCpuTimer();
            return count;
            return -1;
        }
    }
}

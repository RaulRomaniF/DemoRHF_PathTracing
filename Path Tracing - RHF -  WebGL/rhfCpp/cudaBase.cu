
#include <cuda.h>
#include "cudaBase.h"
#include <stdio.h>
#include <iostream>
#include <cstdio>


#include <cuda_runtime.h>

using namespace std;

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n", msg, file_name, line_number, cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call, msg) _safe_cuda_call((call),(msg),__FILE__, __LINE__)



class CudaTime {
	cudaEvent_t start, stop;

public:
	CudaTime() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}
	void record() {
		cudaEventRecord( start, 0 );
	}
	void stopAndPrint(const char* msg) {
		cudaEventRecord( stop, 0 );
		cudaEventSynchronize( stop );
		float elapsedTime;
		cudaEventElapsedTime( &elapsedTime, start, stop);
		printf( "Elapsed time on %s: %3.1f ms\n", msg , elapsedTime );
	}
};

#include "math.h"
#include <climits>

#define MAX(i,j) ( (i)<(j) ? (j):(i) )
#define MIN(i,j) ( (i)<(j) ? (i):(j) )

#define dTiny 1e-10
#define fTiny 0.00000001f
#define fLarge 100000000.0f

//small value
#define EPSILON 0.0001

// Gaussian subsampling blur
#define SIGMASCALE 0.55


__host__ __device__ void fpClear_cu(float *fpI, float fValue, int iLength) {
	for (int ii = 0; ii < iLength; ii++) fpI[ii] = fValue;
}
// total sample contibution   //image in one channel
__host__ __device__ float fiChiSquareNDfFloatDist_cu(int *df, float *k0, float *k1,      float *u0, float *u1, int i0, int j0, int i1, int j1, int radius,
    int width0, int width1) {

	float dist = 0.0;
	for (int s = -radius; s <= radius; s++) {

		int l = (j0 + s) * width0 + (i0 - radius);
		float *ptr0 = &u0[l];
		float *ptrK0 = &k0[l];

		l = (j1 + s) * width1 + (i1 - radius);
		float *ptr1 = &u1[l];
		float *ptrK1 = &k1[l];


		for (int r = -radius; r <= radius; r++, ptr0++, ++ptrK0, ptr1++, ++ptrK1) {

			float sum = (*ptr0 + *ptr1);

			if (sum > 1.00f && //to avoid problems due to little values.
			    *ptrK0 != 0.f && // to avoid inf in empty pixels
			    *ptrK1 != 0.f)
			{
				float dif =  (*ptr0) * (*ptrK1) - (*ptr1) * (*ptrK0);
				dist += (dif * dif) / ((*ptrK0) * (*ptrK1) * sum);
				(*df)++;
			}

		}

	}

	return dist;
}



__host__ __device__ float fiChiSquareNDfFloatDist_cu(int *df, float **u0, float **u1, int i0, int j0,
    int i1, int j1, int radius, int channels, int width0, int width1) {

	float dif = 0.0f;

	for (int ii = 0; ii < channels - 1; ii++) {

		dif += fiChiSquareNDfFloatDist_cu(df, u0[channels - 1], u1[channels - 1],
		                                  u0[ii], u1[ii], i0, j0, i1, j1, radius,
		                                  width0, width1);
	}

	return dif;
}


__host__ __device__ void compute_knn_index_cu(int k, float *ivect_dist, int *ovect_ind, int n)
{
    int *ind = new int[n];

    for (int i = 0; i < n; i++)
        ind[i] = i;

    float minv;
    int minind;

    /*Outer Loop*/
    for (int i = 0; i < k; i++)
    {

        minv = ivect_dist[i]; /*Big Number*/
        minind = i;

        /*inner loop: find minimum value*/
        for (int j = i + 1; j < n; j++)
        {
            if (ivect_dist[j] < minv)
            {
                minv = ivect_dist[j];
                minind = j;
            }
        }

        /*Swap index*/
        int ind_aux = ind[i];
        ind[i] = ind[minind];
        ind[minind] = ind_aux;

        /*Swap values*/
        float val_aux = ivect_dist[i];
        ivect_dist[i] = ivect_dist[minind];
        ivect_dist[minind] = val_aux;

    }
    //Return all the indices also the part no sorted
    for (int i = 0; i < n; i++)
        ovect_ind[i] = ind[i];

    delete[] ind;

}


__global__ void rhf_knn_kernel(int iDWin,       // Half size of patch
             int iDBloc,      // Half size of search window
             float fDistance, // Max-Distance parameter
             int knnT,         // Minimum k-nearest neighbours
             float **fhI,     // Histogram Image
             float **fpI,     // Input image
             float **fpO,     // Output image
             int iChannels,   // Number of channels
             int iWidth,      // Image width
             int iHeight,     // Image height
             int iBins, int ihwl, int iwl, float *fpCount )      // Number of bins Histogram image
{

	// 2D Index of current thread
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // int xVal = 0, yVal = 0;
    // if (x == xVal && y == yVal)
    // {
    // 	printf("fhI[%d][%d]: %f\n", xVal, yVal, fhI[xVal][yVal]);
    // 	printf("fpI[%d][%d]: %f\n", xVal, yVal, fpI[xVal][yVal]);
    // 	printf("fpO[%d][%d]: %f\n", xVal, yVal, fpO[xVal][yVal]);
    // 	printf("iDWin: %d\n", iDWin);
    // 	printf("iDBloc: %d\n", iDBloc);
    // 	printf("fDistance: %f\n", fDistance);
    // 	printf("ihwl: %d\n", ihwl);
    // 	printf("iwl: %d\n", iwl);
    // }

    //Only valid threads perform memory I/O
    if(x < iWidth && y < iHeight)
    {

								//Reduce the size of comparison window  near the boundary
                int iDWin0 = MIN(iDWin, MIN(iWidth - 1 - x,
                                            MIN(iHeight - 1 - y, MIN(x, y))));

                //Research zone depending on the boundary/size of the window
                int imin = MAX(x - iDBloc, iDWin0);
                int jmin = MAX(y - iDBloc, iDWin0);

                int imax = MIN(x + iDBloc, iWidth - 1 - iDWin0);
                int jmax = MIN(y + iDBloc, iHeight - 1 - iDWin0);


                // auxiliary variable
		            // denoised patch centered at a certain pixel
		            // float **fpODenoised = new float*[iChannels];
		            // for (int ii = 0; ii < iChannels; ii++)
		            //     fpODenoised[ii] = new float[iwl];

              //   //  clear current denoised patch
              //   for (int ii = 0; ii < iChannels; ii++)
              //       fpClear_cu(fpODenoised[ii], 0.0f, iwl);
                float fpODenoised[3][9] = {0};


                /*Check if we need to denoise this pixel!!*/
                // sum of weights
                float fTotalWeight = 0.0f;

                // weights
                float fWeight = 1.0f;

                int dj = jmax - jmin + 1;
                int di = imax - imin + 1;

                // int *ovect_ind = new int[dj * di];
                // float *fDif_all = new float[dj * di];
                int* ovect_ind = (int*) malloc (sizeof(int) * dj * di);
                float* fDif_all = (float*) malloc (sizeof(float) * dj * di);

                for (int j = jmin; j <= jmax; j++)
                    for (int i = imin ; i <= imax; i++)
                    {
                        int df = 0;
                        float fDifHist = fiChiSquareNDfFloatDist_cu(&df, fhI, fhI,
                                         x, y,
                                         i, j, iDWin0,
                                         iBins,
                                         iWidth,
                                         iWidth);
                        fDif_all[(j - jmin) + (i - imin)*dj] = fDifHist / (df + EPSILON);
                    }

                compute_knn_index_cu(knnT, fDif_all, ovect_ind, dj * di);

                //ALWAYS: select at least KNN similar patchs.
                int kk;
                for (kk = 0; kk < knnT; kk++)
                {

                    fTotalWeight += fWeight;

                    //Reconvert index
                    int i = ovect_ind[kk] / dj + imin;
                    int j = ovect_ind[kk] % dj + jmin;

                    for (int is = -iDWin0; is <= iDWin0; is++) {
                        int aiindex = (iDWin + is) * ihwl + iDWin;
                        int ail = (j + is) * iWidth + i;

                        for (int ir = -iDWin0; ir <= iDWin0; ir++) {

                            int iindex = aiindex + ir;
                            int il = ail + ir;

                            for (int ii = 0; ii < iChannels; ii++)
                                fpODenoised[ii][iindex] +=  fWeight * fpI[ii][il];
                        }
                    }
                }

                /*SOMETIMES: select those patchs at distance < dmax*/
                for (kk = knnT; kk < dj * di; kk++)
                {
                    if (fDif_all[kk] < fDistance)
                    {

                        fTotalWeight += fWeight;

                        int i = ovect_ind[kk] / dj + imin;
                        int j = ovect_ind[kk] % dj + jmin;

                        for (int is = -iDWin0; is <= iDWin0; is++) {
                            int aiindex = (iDWin + is) * ihwl + iDWin;
                            int ail = (j + is) * iWidth + i;

                            for (int ir = -iDWin0; ir <= iDWin0; ir++) {

                                int iindex = aiindex + ir;
                                int il = ail + ir;

                                for (int ii = 0; ii < iChannels; ii++)
                                    fpODenoised[ii][iindex] +=
                                        fWeight * fpI[ii][il];
                            }
                        }
                    }
                }

                //Normalize average value when fTotalweight is not near zero

                for (int is = -iDWin0; is <= iDWin0; is++) {
                    int aiindex = (iDWin + is) * ihwl + iDWin;
                    int ail = (y + is) * iWidth + x;

                    for (int ir = -iDWin0; ir <= iDWin0; ir++) {
                        int iindex = aiindex + ir;
                        int il = ail + ir;

                        // #pragma omp atomic
                        // fpCount[il]++;
                        // __syncthreads();
                        atomicAdd( &(fpCount[il]), 1 );

                        for (int ii = 0; ii < iChannels; ii++) {
                            // #pragma omp atomic

                            // fpO[ii][il] += fpODenoised[ii][iindex] / fTotalWeight;

                            // __syncthreads();
														atomicAdd( &(fpO[ii][il]), fpODenoised[ii][iindex] / fTotalWeight );

                        }
                    }
                }

                free(ovect_ind);
                free(fDif_all);

                // delete[] ovect_ind;
                // delete[] fDif_all;
    }
}



void rhf_knn_cu(int iDWin,       // Half size of patch
             int iDBloc,      // Half size of search window
             float fDistance, // Max-Distance parameter
             int knn,         // Minimum k-nearest neighbours
             float **fhI,     // Histogram Image
             float **fpI,     // Input image
             float **fpO,     // Output image
             int iChannels,   // Number of channels
             int iWidth,      // Image width
             int iHeight,     // Image height
             int iBins)       // Number of bins Histogram image
{

    printf("---->rhf_knn: dmax = %f, k = %d\n", fDistance, knn);

    //k nearest neighbors + the current patch
    int knnT = knn + 1;

    // length of each channel
    int iwxh = iWidth * iHeight;

    //  length of comparison window
    int ihwl = (2 * iDWin + 1);
    int iwl = (2 * iDWin + 1) * (2 * iDWin + 1);

    // auxiliary variable
    // number of denoised values per pixel
    float *fpCount = new float[iwxh];
    // fpClear(fpCount, 0.0f, iwxh);

    // clear output
    // for (int ii = 0; ii < iChannels; ii++) fpClear(fpO[ii], 0.0f, iwxh);

    // PROCESS STARTS
	const int bytesWxH = sizeof(float)*iwxh;

	/*** Copy histogram HOST TO DEVICE***/

		float** dev_fhI = 0;
		float*  dev_temp_fhI[iBins];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fhI,  sizeof(float*)  * iBins), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iBins; i++)
		{
			cudaMalloc(&dev_temp_fhI[i], bytesWxH );
			cudaMemcpy(&(dev_fhI[i]), &(dev_temp_fhI[i]), sizeof(float *), cudaMemcpyHostToDevice);//copy child pointer to device
			cudaMemcpy(dev_temp_fhI[i], fhI[i], bytesWxH, cudaMemcpyHostToDevice); // copy image to device
		}
	/*** end Copy histogram  ***/

	/*** Copy input HOST TO DEVICE ***/
		float** dev_fpI = 0;
		float*  dev_temp_fpI[iChannels];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fpI,  sizeof(float*)  * iChannels), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iChannels; i++)
		{
			cudaMalloc(&dev_temp_fpI[i], bytesWxH );
			cudaMemcpy(&(dev_fpI[i]), &(dev_temp_fpI[i]), sizeof(float *), cudaMemcpyHostToDevice);//copy child pointer to device
			cudaMemcpy(dev_temp_fpI[i], fpI[i], bytesWxH, cudaMemcpyHostToDevice); // copy image to device
		}
	/*** Copy input ***/

  /*** Copy Output HOST TO DEVICE ***/
		float** dev_fpO = 0;
		float*  dev_temp_fpO[iChannels];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fpO,  sizeof(float*)  * iChannels), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iChannels; i++)
		{
			SAFE_CALL(cudaMalloc(&dev_temp_fpO[i], bytesWxH ), "CUDA Memset Failed");
			SAFE_CALL(cudaMemcpy(&(dev_fpO[i]), &(dev_temp_fpO[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
			SAFE_CALL(cudaMemset(dev_temp_fpO[i], 0.0, bytesWxH), "CUDA Memset Failed");
			// SAFE_CALL(cudaMemcpy(dev_temp_fpO[i], fpO[i], bytesWxH, cudaMemcpyHostToDevice), "CUDA Memset Failed"); // copy image to device
		}
	/*** Copy Output ***/


	// float linearImage[iwxh * iChannels];

	float *dev_fpCount;
	SAFE_CALL(cudaMalloc(&dev_fpCount, bytesWxH ), "CUDA Malloc Failed");
	SAFE_CALL(cudaMemset(dev_fpCount, 0.0, bytesWxH), "CUDA Memset Failed");
	// SAFE_CALL(cudaMemcpy(dev_fpCount, fpCount, bytesWxH, cudaMemcpyHostToDevice), "CUDA Memset Failed"); // copy image to device

	    // cudaMemset(dataGPU, 0, 1000*sizeof(int));


	 //Specify a reasonable block size
  const dim3 block(16, 16);

  //Calculate grid size to cover the whole image
  const dim3 grid((iHeight + block.x - 1) / block.x, 
                  (iWidth  + block.y - 1) / block.y);  //implicit cast to int ceil 

    //Launch the color conversion kernel
    // histogram_kernelSM<<<grid, block>>>(d_input, d_histogram, output.step, output.cols, output.rows);

	// test2Dkernel <<< 1, block>>>(dev_images, dev_linearImage, height, width);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, block.x * bytesWxH );
	rhf_knn_kernel<<<grid, block>>>(iDWin,        // Half size of patch
								              iDBloc,       // Half size of research window
								              fDistance,  // Max-Distance parameter
								              knnT,
								              dev_fhI,      // Histogram
								              dev_fpI,      // Input
								              dev_fpO,      // Output
								              iChannels,    // Number of channels
								              iWidth,       // Image width
								              iHeight,      // Image height
								              iBins, ihwl, iwl, dev_fpCount);        // Number of bins Histogram image

	//COPY DATA DEVICE TO HOST
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(fpCount, dev_fpCount, bytesWxH, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");


	for (int i = 0; i < iChannels; i++)
		SAFE_CALL(cudaMemcpy(fpO[i], dev_temp_fpO[i], bytesWxH, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
	
  // int xVal = 0, yVal = 0;
	// printf("CPU values\n");
	// printf("fhI[%d][%d]: %f\n", xVal, yVal, fhI[xVal][yVal]);
 //  printf("fpI[%d][%d]: %f\n", xVal, yVal, fpI[xVal][yVal]);
 //  printf("fpO[%d][%d]: %f\n", xVal, yVal, fpO[xVal][yVal]);
 //  printf("iDWin: %d\n", iDWin);
 //  printf("iDBloc: %d\n", iDBloc);
 //  printf("fDistance: %f\n", fDistance);
 //  printf("ihwl: %d\n", ihwl);
 //  printf("iwl: %d\n", iwl);

	//FREEING MEMORY
	// Histogram
	for (int k = 1; k < iBins; k++)
		SAFE_CALL(cudaFree(dev_temp_fhI[k]	), " cudaFree Error" );

	//END PROCESS



    for (int ii = 0; ii < iwxh; ii++)
        if (fpCount[ii] > 0.0) {
          for (int jj = 0; jj < iChannels; jj++)  fpO[jj][ii] /= fpCount[ii];
        } else {
          for (int jj = 0; jj < iChannels; jj++)  fpO[jj][ii] = fpI[jj][ii];
        }

    // delete memory
    delete[] fpCount;
}





//constant iDWin: 1
__global__ void rhf_kernel(int iDWin,        // Half size of patch
                           int iDBloc,       // Half size of research window
                           float fDistance,  // Max-Distance parameter
                           float **fhI,      // Histogram
                           float **fpI,      // Input
                           float **fpO,      // Output
                           int iChannels,    // Number of channels
                           int iWidth,       // Image width
                           int iHeight,      // Image height
                           int iBins, int ihwl, int iwl, float *fpCount ) {

	  // 2D Index of current thread
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // int xVal = 0, yVal = 0;
    // if (x == xVal && y == yVal)
    // {
    // 	printf("fhI[%d][%d]: %f\n", xVal, yVal, fhI[xVal][yVal]);
    // 	printf("fpI[%d][%d]: %f\n", xVal, yVal, fpI[xVal][yVal]);
    // 	printf("fpO[%d][%d]: %f\n", xVal, yVal, fpO[xVal][yVal]);
    // 	printf("iDWin: %d\n", iDWin);
    // 	printf("iDBloc: %d\n", iDBloc);
    // 	printf("fDistance: %f\n", fDistance);
    // 	printf("ihwl: %d\n", ihwl);
    // 	printf("iwl: %d\n", iwl);
    // }

    //Only valid threads perform memory I/O
    if(x < iWidth && y < iHeight)
    {
			/*Check if we need to denoise this pixel!!*/
			// sum of weights
			float fTotalWeight = 0.0f;

			// weights
			float fWeight = 1.0f;

			//Reduce the size of comparison window near the boundary
			int iDWin0 = MIN(iDWin, MIN(iWidth - 1 - x, MIN(iHeight - 1 - y, MIN(x, y))));

			float fpODenoised[3][9] = {0};

			//Research zone depending on the boundary/size of the window
			int imin = MAX(x - iDBloc, iDWin0);
			int jmin = MAX(y - iDBloc, iDWin0);

			int imax = MIN(x + iDBloc, iWidth  - 1 - iDWin0);
			int jmax = MIN(y + iDBloc, iHeight - 1 - iDWin0);

			for (int j = jmin; j <= jmax; j++)
				for (int i = imin ; i <= imax; i++)
					if (i != x || j != y)
					{

						int df = 0;
						float fDifHist = fiChiSquareNDfFloatDist_cu(&df, fhI,
						                 fhI, x, y,
						                 i, j, iDWin0,
						                 iBins,
						                 iWidth,
						                 iWidth);

						if (fDifHist <  fDistance * df)
						{
							fTotalWeight += fWeight;

							for (int is = -iDWin0; is <= iDWin0; is++) {
								int aiindex = (iDWin + is) * ihwl + iDWin;
								int ail = (j + is) * iWidth + i;

								for (int ir = -iDWin0; ir <= iDWin0; ir++) {

									int iindex = aiindex + ir;
									int il = ail + ir;

									for (int ii = 0; ii < iChannels; ii++)
										fpODenoised[ii][iindex] += fWeight * fpI[ii][il];
								}
							}
						}
					}

			// current patch with fMaxWeight
			for (int is = -iDWin0; is <= iDWin0; is++) {
				int aiindex = (iDWin + is) * ihwl + iDWin;
				int ail = (y + is) * iWidth + x;

				for (int ir = -iDWin0; ir <= iDWin0; ir++) {

					int iindex = aiindex + ir;
					int il = ail + ir;

					for (int ii = 0; ii < iChannels; ii++)
						fpODenoised[ii][iindex] += fWeight * fpI[ii][il];
				}
			}

			fTotalWeight += fWeight;

			// normalize average value when fTotalweight is not near zero
			if (fTotalWeight > fTiny) {

				for (int is = -iDWin0; is <= iDWin0; is++) {
					int aiindex = (iDWin + is) * ihwl + iDWin;
					int ail = (y + is) * iWidth + x;

					for (int ir = -iDWin0; ir <= iDWin0; ir++) {
						int iindex = aiindex + ir;
						int il = ail + ir;

						// #pragma omp atomic
						// __syncthreads();
				    atomicAdd( &(fpCount[il]), 1 );


						for (int ii = 0; ii < iChannels; ii++) {
							// #pragma omp atomic
							// fpO[ii][il] += fpODenoised[ii][iindex] / fTotalWeight;
							// __syncthreads();
							atomicAdd( &(fpO[ii][il]), fpODenoised[ii][iindex] / fTotalWeight );
						}
					}
				}
			}//end if Tiny

			// const int tid = y*iWidth + x; //linearizar
			// __syncthreads();
			// if (fpCount[tid] > 0.0) {
			// 	for (int jj = 0; jj < iChannels; jj++)  fpO[jj][tid] /= fpCount[tid];
			// } else {
			// 	for (int jj = 0; jj < iChannels; jj++)  fpO[jj][tid] = fpI[jj][tid];
			// }

	  }
}



void rhf_cu(int iDWin,        // Half size of patch
            int iDBloc,       // Half size of research window
            float fDistance,  // Max-Distance parameter
            float **fhI,      // Histogram
            float **fpI,      // Input
            float **fpO,      // Output
            int iChannels,    // Number of channels
            int iWidth,       // Image width
            int iHeight,      // Image height
            int iBins)        // Number of bins Histogram image
{
	printf("---->rhf: dmax = %f\n", fDistance);

	printf("iChannels: %d\n", iChannels);
	printf("iBins: %d\n", iBins);
	printf("iDWin: %d\n", iDWin);

	// length of each channel
	int iwxh = iWidth * iHeight;

	//  length of comparison window
	int ihwl = (2 * iDWin + 1);
	int iwl  = (2 * iDWin + 1) * (2 * iDWin + 1);  //kernel

	// auxiliary variable
	// number of denoised values per pixel
	float *fpCount = new float[iwxh];
	// fpClear_cu(fpCount, 0.0f, iwxh);

	// clear output
	// for (int ii = 0; ii < iChannels; ii++) fpClear_cu(fpO[ii], 0.0f, iwxh);





	// PROCESS STARTS
	const int bytesWxH = sizeof(float)*iwxh;

	/*** Copy histogram HOST TO DEVICE***/

		float** dev_fhI = 0;
		float*  dev_temp_fhI[iBins];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fhI,  sizeof(float*)  * iBins), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iBins; i++)
		{
			cudaMalloc(&dev_temp_fhI[i], bytesWxH );
			cudaMemcpy(&(dev_fhI[i]), &(dev_temp_fhI[i]), sizeof(float *), cudaMemcpyHostToDevice);//copy child pointer to device
			cudaMemcpy(dev_temp_fhI[i], fhI[i], bytesWxH, cudaMemcpyHostToDevice); // copy image to device
		}
	/*** end Copy histogram  ***/

	/*** Copy input HOST TO DEVICE ***/
		float** dev_fpI = 0;
		float*  dev_temp_fpI[iChannels];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fpI,  sizeof(float*)  * iChannels), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iChannels; i++)
		{
			cudaMalloc(&dev_temp_fpI[i], bytesWxH );
			cudaMemcpy(&(dev_fpI[i]), &(dev_temp_fpI[i]), sizeof(float *), cudaMemcpyHostToDevice);//copy child pointer to device
			cudaMemcpy(dev_temp_fpI[i], fpI[i], bytesWxH, cudaMemcpyHostToDevice); // copy image to device
		}
	/*** Copy input ***/

  /*** Copy Output HOST TO DEVICE ***/
		float** dev_fpO = 0;
		float*  dev_temp_fpO[iChannels];

		// first create top level pointer
		SAFE_CALL(cudaMalloc(&dev_fpO,  sizeof(float*)  * iChannels), "CUDA Malloc Failed");

		// then create child pointers on host, and copy to device, then copy image
		for (int i = 0; i < iChannels; i++)
		{
			SAFE_CALL(cudaMalloc(&dev_temp_fpO[i], bytesWxH ), "CUDA Memset Failed");
			SAFE_CALL(cudaMemcpy(&(dev_fpO[i]), &(dev_temp_fpO[i]), sizeof(float *), cudaMemcpyHostToDevice), "CUDA Memset Failed");//copy child pointer to device
			SAFE_CALL(cudaMemset(dev_temp_fpO[i], 0.0, bytesWxH), "CUDA Memset Failed");
			// SAFE_CALL(cudaMemcpy(dev_temp_fpO[i], fpO[i], bytesWxH, cudaMemcpyHostToDevice), "CUDA Memset Failed"); // copy image to device
		}
	/*** Copy Output ***/


	// float linearImage[iwxh * iChannels];

	float *dev_fpCount;
	SAFE_CALL(cudaMalloc(&dev_fpCount, bytesWxH ), "CUDA Malloc Failed");
	SAFE_CALL(cudaMemset(dev_fpCount, 0.0, bytesWxH), "CUDA Memset Failed");
	// SAFE_CALL(cudaMemcpy(dev_fpCount, fpCount, bytesWxH, cudaMemcpyHostToDevice), "CUDA Memset Failed"); // copy image to device

	    // cudaMemset(dataGPU, 0, 1000*sizeof(int));


	 //Specify a reasonable block size
  const dim3 block(16, 16);

  //Calculate grid size to cover the whole image
  const dim3 grid((iHeight + block.x - 1) / block.x, 
                  (iWidth  + block.y - 1) / block.y);  //implicit cast to int ceil 

    //Launch the color conversion kernel
    // histogram_kernelSM<<<grid, block>>>(d_input, d_histogram, output.step, output.cols, output.rows);

	// test2Dkernel <<< 1, block>>>(dev_images, dev_linearImage, height, width);

	rhf_kernel<<<grid, block>>>(iDWin,        // Half size of patch
								              iDBloc,       // Half size of research window
								              fDistance,  // Max-Distance parameter
								              dev_fhI,      // Histogram
								              dev_fpI,      // Input
								              dev_fpO,      // Output
								              iChannels,    // Number of channels
								              iWidth,       // Image width
								              iHeight,      // Image height
								              iBins, ihwl, iwl, dev_fpCount);        // Number of bins Histogram image

	//COPY DATA DEVICE TO HOST
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(fpCount, dev_fpCount, bytesWxH, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");


	for (int i = 0; i < iChannels; i++)
		SAFE_CALL(cudaMemcpy(fpO[i], dev_temp_fpO[i], bytesWxH, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");
	
  // int xVal = 0, yVal = 0;
	// printf("CPU values\n");
	// printf("fhI[%d][%d]: %f\n", xVal, yVal, fhI[xVal][yVal]);
 //  printf("fpI[%d][%d]: %f\n", xVal, yVal, fpI[xVal][yVal]);
 //  printf("fpO[%d][%d]: %f\n", xVal, yVal, fpO[xVal][yVal]);
 //  printf("iDWin: %d\n", iDWin);
 //  printf("iDBloc: %d\n", iDBloc);
 //  printf("fDistance: %f\n", fDistance);
 //  printf("ihwl: %d\n", ihwl);
 //  printf("iwl: %d\n", iwl);

	//FREEING MEMORY
	// Histogram
	for (int k = 1; k < iBins; k++)
		SAFE_CALL(cudaFree(dev_temp_fhI[k]	), " cudaFree Error" );

	//END PROCESS
	for (int ii = 0; ii < iwxh; ii++)
		if (fpCount[ii] > 0.0) {
			for (int jj = 0; jj < iChannels; jj++)  fpO[jj][ii] /= fpCount[ii];
		} else {
			for (int jj = 0; jj < iChannels; jj++)  fpO[jj][ii] = fpI[jj][ii];
		}

	// delete memory
	delete[] fpCount;
}

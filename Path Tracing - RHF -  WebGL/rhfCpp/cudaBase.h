

#pragma once


void dummyCalc(  );

void rhf_cu(int iDWin,        // Half size of patch
         int iDBloc,       // Half size of research window
         float fDistance,  // Max-Distance parameter
         float **fhI,      // Histogram
         float **fpI,      // Input
         float **fpO,      // Output
         int iChannels,    // Number of channels
         int iWidth,       // Image width
         int iHeight,      // Image height
         int iBins)  ;      // Number of bins Histogram image


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
             int iBins) ;      // Number of bins Histogram image

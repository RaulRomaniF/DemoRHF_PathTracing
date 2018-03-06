/*----------------------------------------------------------------------------

 RHF - Ray Histogram Fusion

 Copyright (c) 2013, A. Buades <toni.buades@uib.es>,
 M. Delbracio <mdelbra@gmail.com>,
 J-M. Morel <morel@cmla.ens-cachan.fr>,
 P. Muse <muse@fing.edu.uy>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program. If not, see <http://www.gnu.org/licenses/>.

 ----------------------------------------------------------------------------*/


/**
 * @file rhf.cpp
 * @brief RHF filter.
 * @author Mauricio Delbracio  (mdelbra@gmail.com)
 */


/** @mainpage Accelerating Monte Carlo Renderers by Ray Histogram Fusion
 *
 * The following code is an implementation of the Ray Histogram Fusion (RHF)
 * filter presented in
 *
 *
 * \li Delbracio, M., Musé, P., Buades, A., Chauvier, J., Phelps, N. & Morel, J.M. <br>
 *  "Boosting Monte Carlo rendering by ray histogram fusion" <br>
 *  ACM Transactions on Graphics (TOG), 33(1), 8. 2014
 *
 * and in more detail described on the online journal IPOL (www.ipol.im)
 * where there is a more precise algorithm description, including this code and an
 * online demo version.
 *
 *
 * The source code consists of:
 *
 * \li  COPYING
 * \li  Makefile
 * \li  README.txt
 * \li  VERSION
 * \li  exrcrop.cpp
 * \li  exrdiff.cpp
 * \li  exrtopng.cpp
 * \li  io_exr.cpp
 * \li  io_exr.h
 * \li  io_png.c
 * \li  io_png.h
 * \li  libauxiliar.cpp
 * \li  libauxiliar.h
 * \li  libdenoising.cpp
 * \li  libdenoising.h
 * \li  rhf.cpp
 * \li  extras/pbrt-v2-rhf (A modified version of PBRT-v2)
 *
 *
 *
 * The core of the filtering algorithm is in libdenoising.cpp.
 *
 * HISTORY:
 * - Version 1.2 - January 10, 2015
 * - Version 1.1 - June 09, 2014
 *
 *
 * @author mauricio delbracio (mdelbra@gmail.com)
 * @date jan 2015
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "libdenoising.h"
#include "io_exr.h"

#include "io_png.h"
// #include "pathTracer.h"
#include <curl/curl.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <SOIL.h>

using namespace cv;

#include <vector>

using namespace std;

//output screen resolution
const int WIDTH  = 800;
const int HEIGHT = 600;



void flipVertically(unsigned char* pData, int texture_width, int texture_height, int channels) {
    //vertically flip the image on Y axis since it is inverted
    // int i, j;
    // for ( j = 0; j * 2 < texture_height; ++j )
    // {
    //     int index1 = j * texture_width * channels;
    //     int index2 = (texture_height - 1 - j) * texture_width * channels;
    //     for ( i = texture_width * channels; i > 0; --i )
    //     {
    //         GLubyte temp = pData[index1];
    //         pData[index1] = pData[index2];
    //         pData[index2] = temp;
    //         ++index1;
    //         ++index2;
    //     }
    // }
}
// utility function for loading a 2D texture from file
// ---------------------------------------------------
// unsigned int loadTexture(char const * path)
// {
//     unsigned int textureID;
//     glGenTextures(1, &textureID);

//     int width, height, nrComponents;
//     // unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
//     unsigned char *data = SOIL_load_image(path, &width, &height, &nrComponents, SOIL_LOAD_AUTO);
//     // flipVertically(data,width,height,nrComponents);

//     // glBindTexture(GL_TEXTURE_2D, specularMap);
//     // glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

//     if (data)
//     {
//         GLenum format;
//         if (nrComponents == 1)
//             format = GL_RED;
//         else if (nrComponents == 3)
//             format = GL_RGB;
//         else if (nrComponents == 4)
//             format = GL_RGBA;

//         glBindTexture(GL_TEXTURE_2D, textureID);
//             glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
//             glGenerateMipmap(GL_TEXTURE_2D);

//             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
//             glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

//             SOIL_free_image_data(data);
//         glBindTexture(GL_TEXTURE_2D, 0); //unbind : deactivate
//     }
//     else
//     {
//         std::cout << "Texture failed to load at path: " << path << std::endl;
//         SOIL_free_image_data(data);
//     }

//     return textureID;
// }


//curl writefunction to be passed as a parameter
// we can't ever expect to get the whole image in one piece,
// every router / hub is entitled to fragment it into parts
// (like 1-8k at a time),
// so insert the part at the end of our stream.
// size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
// {
//     vector<uchar> *stream = (vector<uchar>*)userdata;
//     size_t count = size * nmemb;
//     stream->insert(stream->end(), ptr, ptr + count);
//     return count;
// }

// //function to retrieve the image as cv::Mat data type
// cv::Mat curlImg(const char *img_url, int timeout=10)
// {
//     vector<uchar> stream;
//     CURL *curl = curl_easy_init();
//     curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
//     curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
//     curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
//     curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs, 
//     CURLcode res = curl_easy_perform(curl); // start curl
//     curl_easy_cleanup(curl); // cleanup
//     return imdecode(stream, -1); // 'keep-as-is'
// }


//curl writefunction to be passed as a parameter
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata) {
    std::ostringstream *stream = (std::ostringstream*)userdata;
    size_t count = size * nmemb;
    stream->write(ptr, count);
    return count;
}

//function to retrieve the image as Cv::Mat data type
cv::Mat curlImg(string url)
{
    CURL *curl;
    CURLcode res;
    std::ostringstream stream;
    curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, url); //the img url
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr when the writefunction is called
    res = curl_easy_perform(curl); // start curl
    std::string output = stream.str(); // convert the stream into a string
    curl_easy_cleanup(curl); // cleanup
    std::vector<char> data = std::vector<char>( output.begin(), output.end() ); //convert string into a vector
    cv::Mat data_mat = cv::Mat(data); // create the cv::Mat datatype from the vector
    cv::Mat image = cv::imdecode(data_mat,1); //read an image from memory buffer
    return image;
}
// int main(void)
// {
//     cv::Mat image = curlImg();
//     cv::namedWindow( "Image output", CV_WINDOW_AUTOSIZE );
//     cv::imshow("Image output",image); //display image
//     cvWaitKey(0); // press any key to exit
//     cv::destroyWindow("Image output");
// }

//display callback function
void generateInputs( float **fpI, float **fpHisto) {

    
    int nc_h = 61;

    // initialize histogram
    for (int i = 0; i < nc_h; i++)
        for (int j = 0; j < WIDTH * HEIGHT; j++)
            fpHisto[i][j] = 0.0;

    float spp = 20.0;
    for (int s = 1; s <= spp; ++s)
    {
        // Mat sample = imread("../static/rhf/sample" + to_string(s) + ".png", 1);
        Mat sample = imread("static/rhf/sample" + to_string(s) + ".png", 1);
        unsigned char* samplePtr = sample.ptr();

        for (unsigned int y = 0; y < HEIGHT; ++y)
            for (unsigned int x = 0; x < WIDTH; ++x) {

                int offset = y * WIDTH  + x ;
                int offset2 = y * WIDTH * 3 + x * 3;

                //RGB
                for (int c = 0; c < 3; ++c) {
                    unsigned char channelValue = samplePtr[offset2 + c];

                    for (int bin = 0; bin < 20; ++bin) //bins
                    {
                        if (channelValue >= 12.75 * bin && channelValue <= 12.75 * (bin + 1)) {
                            fpHisto[bin + 20 * c][offset] += 1.0;
                            break;
                        }
                    }
                }
                fpHisto[60][offset] = spp; //total sample contribution
            }
    }

    // Mat image = imread("/home/raul/imssage.png", 1);

    // cout<<"starting"<<endl;
    // printf("starting\n");

    // // curlImg7
    // Mat image = curlImg("http://localhost:5000/static/rhf/image.png");
    // if (image.empty())
    //     cout<<"not found"<<endl;


    // int width, height, nrComponents;
    // // unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    // unsigned char *data = SOIL_load_image("http://localhost:5000/static/rhf/image.png", &width, &height, &nrComponents, SOIL_LOAD_AUTO);

    // printf("%d  %d  %d\n", int(data[0]), int(data[1]), int(data[2]) );
    Mat image = imread("static/rhf/image.png", 1);

    unsigned char* imagePtr = image.ptr();

    for (unsigned int y = 0; y < HEIGHT; ++y)
        for (unsigned int x = 0; x < WIDTH; ++x) {

            int offset = y * WIDTH  + x ;
            int offset1 = y * WIDTH*3  + x*3 ;
            fpI[0][offset] = float(imagePtr[offset1 + 2])/255.0;
            fpI[1][offset] = float(imagePtr[offset1 + 1])/255.0;
            fpI[2][offset] = float(imagePtr[offset1 + 0])/255.0;
        }

}








/** @brief Struct of program parameters */
typedef struct
{
    int t;
    float max_distance;
    int knn;
    char *hist_file;
    char *input;
    char *output;
    int win;
    int bloc;
    int nscales;
} program_argums;


/** @brief Error/Exit print a message and exit.
 *  @param msg
 */
static void error(const char *msg)
{
    fprintf(stderr, "nlmeans Error: %s\n", msg);
    exit(EXIT_FAILURE);
}


/** @brief Print program's usage and exit.
 *  @param args
 */
static void usage(const char* name)
{
    printf("RHF: Ray Histogram Fusion Filter v1.1 Jun 2014\n");
    printf("Copyright (c) 2014 M.Delbracio, P.Muse, A.Buades and JM.Morel\n\n");
    printf("Usage: %s [options] <input file> <output file>\n"
           "Only EXR images are supported.\n\n", name);
    printf("Options:\n");
    printf("   -h <hist>   The filename with the histogram\n");
    printf("   -d <float>  Max-distance between patchs\n");
    printf("   -k <int>    Minimum number of similar patchs (default: 2)\n");
    printf("   -b <int>    Half the block size  (default: 6)\n");
    printf("   -w <int>    Half the windows size (default: 1)\n");
    printf("   -s <int>    Number of Scales - Multi-Scale (default: 2)\n");
}

static void parse_arguments(program_argums *param, int argc, char *argv[])
{
    char *OptionString;
    char OptionChar;
    int i;


    if (argc < 4)
    {
        usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    /* loop to read parameters*/
    for (i = 1; i < argc;)
    {
        if (argv[i] && argv[i][0] == '-')
        {
            if ((OptionChar = argv[i][1]) == 0)
            {
                error("Invalid parameter format.\n");
            }

            if (argv[i][2])
                OptionString = &argv[i][2];
            else if (++i < argc)
                OptionString = argv[i];
            else
            {
                error("Invalid parameter format.\n");
            }

            switch (OptionChar)
            {
            case 's':
                param->nscales = atoi(OptionString);
                if (param->nscales < 0 || param->nscales > 6)
                {
                    error("s must be  0-3.\n");
                }
                break;


            case 'd':
                param->max_distance = (float) atof(OptionString);
                if (param->max_distance < 0)
                {
                    error("Invalid parameter d (max_distance).\n");
                }
                break;

            case 'k':
                param->knn =  atoi(OptionString);
                if (param->knn < 0)
                {
                    error("Invalid parameter k.\n");
                }
                break;


            case 'b':
                param->bloc =  atoi(OptionString);
                if (param->bloc < 0)
                {
                    error("Invalid parameter b.\n");
                }
                break;

            case 'w':
                param->win =  atoi(OptionString);
                if (param->win < 0)
                {
                    error("Invalid parameter w.\n");
                }
                break;

            case 'h':
                param->hist_file = OptionString;
                break;

            case '-':
                usage(argv[0]);
                exit(EXIT_FAILURE);

            default:
                if (isprint(OptionChar))
                {
                    fprintf(stderr, "Unknown option \"-%c\".\n",
                            OptionChar);
                    exit(EXIT_FAILURE);
                } else
                    error("Unknown option.\n");
            }

        }
        else
        {
            if (!param->input)
                param->input = argv[i];
            else
                param->output = argv[i];

        }

        i++;
    }

    if (!param->input || !param->output)
    {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }


    /* If parameters weren't set, set deafult parameters*/
    param->bloc = param->bloc >= 0 ? param->bloc : 6;
    param->win  = param->win >= 0 ? param->win  : 1;
    param->knn  = param->knn >= 0 ? param->knn  : 2;
    param->nscales = param->nscales > 0 ? param->nscales : 2;

    /*Check parameters are consistent*/
    if (param->max_distance < 0)  error("Parameter max_distance not set.\n");

    printf("Loaded Parameters\n");
    printf("-----------------\n");
    printf("Number of scales: %d\n", param->nscales);
    printf("      block size: %d\n", param->bloc);
    printf("      patch size: %d\n", param->win);
    printf("      dmax      : %f\n", param->max_distance);
    printf("      knn       : %d\n\n", param->knn);

    /* Print parameters*/
}


int main(int argc, char **argv) {


    /*Initialize the structure param->* to -1 or null */
    program_argums param = { -1, -1, -1, NULL, NULL, NULL, -1, -1, 0};

    /*Parse command-line arguments*/
    parse_arguments(&param, argc, argv);


    float **fpHisto = new float*[61];
    for (int i = 0; i < 61; i++) {
        fpHisto[i] = new float[WIDTH * HEIGHT]; // &fpH[i * nx_h*ny_h];
    }

    float **fpI = new float*[3];

    for (int ii = 0; ii < 3; ii++) {
        fpI[ii] = new float[WIDTH * HEIGHT];
    }
    generateInputs(fpI, fpHisto);

    cout<<"ALl good!"<<endl;


    // cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    // unsigned char* data = img.ptr();

    // for (unsigned int y = 0; y < HEIGHT; ++y)
    //     for (unsigned int x = 0; x < WIDTH; ++x) {

    //         int offset = y * WIDTH  + x ;
    //         int offset2 = y * WIDTH * 3 + x * 3;

    //         //RGB to BRG
    //         data[offset2 + 0] = fpI[2][offset]*255.0;
    //         data[offset2 + 1] = fpI[1][offset]*255.0;
    //         data[offset2 + 2] = fpI[0][offset]*255.0;
    //     }

    // imshow("pathTraceImage", img) ;
    // waitKey(0);


    

    int nx = WIDTH, ny = HEIGHT, nc;
    // float *d_v = NULL;

    // d_v = ReadImageEXR(param.input, &nx, &ny);
    nc = 3; //Assume 3 color channels

    // cv::Mat img = cv::Mat::zeros(nx, ny, CV_8UC3);

    // unsigned char* data = img.ptr();

    // for (int i = 0; i < nx*ny; ++i){
    //     data[i*3 + 0] = static_cast<unsigned char>(d_v[i + 2*nx*ny]*255);
    //     data[i*3 + 1] = static_cast<unsigned char>(d_v[i + nx*ny   ]*255);
    //     data[i*3 + 2] = static_cast<unsigned char>(d_v[i           ]*255);
    // }

    // namedWindow( "hello", CV_WINDOW_AUTOSIZE );
    // imshow("hello",img) ;
    // waitKey(0);

    // Mat img2 = imread("cornell-path_00256_test2.exr", 1);
    // // imwrite("cornell-path_00256_test.exr", img2);

    // img2.convertTo( img2, CV_8U, 40. );
    // namedWindow( "hello2", CV_WINDOW_AUTOSIZE );
    // imshow("hello2",img2) ;
    // waitKey(0);


    // data[i]                      =  pixelsR[i];
    // data[i + width * height]     =  pixelsG[i];
    // data[i + 2 * width * height] =  pixelsB[i];


    // for (int i = 0; i < 100; ++i)
    // {
    //     printf("Pixel %d: %f %f %f\n", i, d_v[200*512 + i*3 +0], d_v[200*512 + i*3 +1], d_v[200*512 + i*3 +2]);
    // }





    // if (!d_v) {
    //     printf("error :: %s not found  or not a correct exr image \n", argv[1]);
    //     exit(-1);
    // }

    // variables
    int d_w = (int) nx;
    int d_h = (int) ny;
    int d_c = (int) nc;

    // if (d_c == 2) {
    //     d_c = 1;    // we do not use the alpha channel
    // }
    // if (d_c > 3) {
    //     d_c = 3;    // we do not use the alpha channel
    // }

    int d_wh  = d_w * d_h;
    int d_whc = d_c * d_w * d_h;

    // test if image is really a color image even if it has more than one channel
    // if (d_c > 1) {
    //     // dc equals 3
    //     int i = 0;
    //     while (i < d_wh && d_v[i] == d_v[d_wh + i] && d_v[i] == d_v[2 * d_wh + i ])  {
    //         i++;
    //     }

    //     if (i == d_wh) d_c = 1;
    //     printf("i = %d \n", i);  //21158

    // }
    printf("nx = %d \n", nx);
    printf("ny = %d \n", ny);
    printf("nc = %d \n", nc);
    printf("d_c = %d \n", d_c);


    // denoise
    // float **fpI     = new float*[d_c];
    float **fpO     = new float*[d_c];
    float *denoised = new float[d_whc];

    for (int ii = 0; ii < d_c; ii++) {

        // fpI[ii] = &d_v[ii * d_wh];
        fpO[ii] = &denoised[ii * d_wh];

    }

    //-Read Histogram image----------------------------------------------------
    int nx_h = WIDTH, ny_h = HEIGHT, nc_h = 61;
    // float *fpH = NULL;

    // fpH = readMultiImageEXR(param.hist_file,
    //                         &nx_h, &ny_h, &nc_h);

    printf("nx_h = %d \n", nx_h);
    printf("ny_h = %d \n", ny_h);
    printf("nc_h = bins =  %d \n", nc_h);

    // for (int ii= nx_h*ny_h + nx_h*120; ii < nx_h*ny_h +nx_h*121; ii++){
    //     printf("%f ", fpH[ii ]);
    // }
    // printf("\n");


    // float **fpHisto = new float*[nc_h];
    // for (int ii=0; ii < nc_h; ii++){
    //     fpHisto[ii] = &fpH[ii * nx_h*ny_h];
    // }

    printf("\n\n");
    float sum = 0.0;
    for (int ii = 0; ii < 61; ii++) {
        printf("%f  ", fpHisto[ii][nx_h * 3 * 120  ]);
        // if (ii >=0 && ii <20 )
        // {
        //     sum += fpHisto[ii][nx_h*3*120 ];
        // }
        // if (ii ==19)
        //     printf("\n");
    }
    printf("sum: %f\n", sum);
    printf("\n");

    // Total number of samples in the whole image
    // double dtotal = 0.0f;

    // for (int ii = 0; ii < nx_h*ny_h; ii++)
    // {
    //     dtotal += fpHisto[61 - 1][ii];
    // }

    // printf("total: %f\n", dtotal);


    // Measure Filtering time
    struct timeval tim;
    gettimeofday(&tim, NULL);
    double t1 = tim.tv_sec + (tim.tv_usec / 1000000.0);

    printf("Running...\n");
    rhf_multiscale(param.win,
                   param.bloc,
                   param.max_distance,
                   param.knn,
                   param.nscales,
                   fpHisto,
                   fpI, fpO, d_c, d_w, d_h, nc_h);

    gettimeofday(&tim, NULL);
    double t2 = tim.tv_sec + (tim.tv_usec / 1000000.0);
    printf("Filtering Time: %.2lf seconds\n", t2 - t1);


    // save EXR denoised image
    WriteImageEXR(param.output, denoised, d_w, d_h);















    // cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    // unsigned char* data = img.ptr();

    // for (unsigned int y = 0; y < HEIGHT; ++y)
    //     for (unsigned int x = 0; x < WIDTH; ++x) {

    //         int offset = y * WIDTH  + x ;
    //         int offset2 = y * WIDTH * 3 + x * 3;

    //         //RGB to BRG
    //         data[offset2 + 0] = fpO[2][offset]*255.0;
    //         data[offset2 + 1] = fpO[1][offset]*255.0;
    //         data[offset2 + 2] = fpO[0][offset]*255.0;
    //     }

    // imwrite("static/rhf/image_filt.png", img) ;
    // waitKey(0);





    //EXR TO PNG 
    int nrow, ncol;
    float *data;
    int i;

    data = ReadImageEXR("static/rhf/image_filt.exr", &ncol, &nrow);

    /*Rescale image to 0-255*/
    for (i = 0; i < nrow * ncol * 3; i++)
        data[i] = data[i] * 255;

    // cv::Mat img = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
    // unsigned char* dataPtr = img.ptr();
    // for (int i = 0; i < ncol*nrow; ++i){
    //     dataPtr[i*3 + 0] = static_cast<unsigned char>(data[i + 2*ncol*nrow]*255);
    //     dataPtr[i*3 + 1] = static_cast<unsigned char>(data[i + ncol*nrow   ]*255);
    //     dataPtr[i*3 + 2] = static_cast<unsigned char>(data[i           ]*255);
    // }
    // imwrite("static/rhf/image_filt.png", img) ;



    io_png_write_f32("static/rhf/image_filt.png", data, (size_t) ncol, (size_t) nrow, 3);
    free(data);


    delete[] fpHisto;
    // delete[] fpH;
    delete[] fpI;
    delete[] fpO;
    // delete[] d_v;
    delete[] denoised;

    return 0;

}







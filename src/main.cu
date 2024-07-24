#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

typedef struct Pixel{
    unsigned char r, g, b, a;
} Pixel;

void ConvertImageToGrayCPU(char* imageRGBA, int width, int height){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            Pixel* pixel = (Pixel*)&imageRGBA[(i*width)*4+4*j];
            float pixelValue = pixel->r*0.299f + pixel->g*0.587f + pixel->b*0.114f;
            unsigned char gray = (unsigned char)pixelValue;
            pixel->r = gray;
            pixel->g = gray;
            pixel->b = gray;
            pixel->a = 255;
        }
    }
} 

__global__ void ConvertImageToGrayGPU(unsigned char* imageRGBA, int width, int height) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uint32_t idx = (y * width + x) * 4;
        Pixel* pixel = (Pixel*)&imageRGBA[idx];
        float pixelValue = pixel->r * 0.299f + pixel->g * 0.587f + pixel->b * 0.114f;
        unsigned char gray = (unsigned char)pixelValue;
        pixel->r = gray;
        pixel->g = gray;
        pixel->b = gray;
        pixel->a = 255;
    }
}

int main(){
    // Hardcoded image filename
    const char* filename = "data/sample_data_color.jpg";

    // Open image
    int width, height, componentCount;
    unsigned char* imageData = stbi_load(filename, &width, &height, &componentCount, 0);
    if(!imageData){
        cout << "Failed to open \"" << filename << "\"\n";
        return -1;
    }

    // Validate image size
    if(width % 32 != 0 || height % 32 != 0){
        // Image size must be a multiple of 32
        cout << "Image size must be a multiple of 32\n";
        stbi_image_free(imageData);
        return -1;
    }

    /*
    cout << "Image size: " << width << "x" << height << "\n";
    cout << "Loading....." << "\n";
    cout << "Processing the image....." << "\n";
    ConvertImageToGrayCPU((char*)imageData, width, height);
    cout << "DONE" << "\n"; 
    */
    //copy data to device

    cout << "Copying Data to GPU" << "\n";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width*height*4)== cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice)==cudaSuccess);
    cout << "DONE" << "\n";


    cout << "Running CUDA Kernel.." << "\n";
    dim3 blockSize(32,32);
    dim3 gridSize(width/blockSize.x, height/blockSize.y);
    ConvertImageToGrayGPU<<<gridSize, blockSize>>>(ptrImageDataGpu, width, height);
    cout << "DONE" << "\n";

    //copying data from gpu to host
    cout << "Copying Data from GPY.." << "\n";
    assert(cudaMemcpy(imageData,ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost)==cudaSuccess);

    // Building output filename
    string outputFilename = filename;
    size_t dotPos = outputFilename.find_last_of('.');
    if (dotPos != string::npos) {
        outputFilename = "data/" + outputFilename.substr(outputFilename.find_last_of('/') + 1, dotPos) + "_gray.jpg";
    } else {
        outputFilename = "data/" + outputFilename.substr(outputFilename.find_last_of('/') + 1) + "_gray.jpg";
    }

    // Write image back
    stbi_write_jpg(outputFilename.c_str(), width, height, 4, imageData, 100);

    // Close 
    stbi_image_free(imageData);

    cout << "Converted image saved as \"" << outputFilename << "\"\n";
    cout << "DONE" << "\n";

    return 0;
}

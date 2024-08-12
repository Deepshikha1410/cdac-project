#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel to compute the histogram
__global__ void computeHistogram(const unsigned char *data, int width, int height, int *histogram) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * 3 + x * 3;
        unsigned char gray = (unsigned char)((data[idx] * 0.299) + (data[idx + 1] * 0.587) + (data[idx + 2] * 0.114));
        atomicAdd(&histogram[gray], 1);
    }
}

// CUDA kernel to perform histogram equalization
__global__ void equalizeHistogram(unsigned char *data, int width, int height, const int *cumulativeHistogram, int totalPixels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width * 3 + x * 3;
        unsigned char gray = (unsigned char)((data[idx] * 0.299) + (data[idx + 1] * 0.587) + (data[idx + 2] * 0.114));
        unsigned char equalizedValue = (unsigned char)(((float)cumulativeHistogram[gray] / (totalPixels - 1)) * 255);
        data[idx] = equalizedValue;
        data[idx + 1] = equalizedValue;
        data[idx + 2] = equalizedValue;
    }
}

// Function to save histogram as an image
void saveHistogramImage(const int *histogram, int width = 256, int height = 200) {
    unsigned char *histogramImage = (unsigned char *)malloc(width * height * 3);
    memset(histogramImage, 255, width * height * 3); // Initialize with white

    int maxCount = 0;
    for (int i = 0; i < 256; ++i) {
        if (histogram[i] > maxCount) {
            maxCount = histogram[i];
        }
    }

    for (int i = 0; i < 256; ++i) {
        int barHeight = (histogram[i] * height) / maxCount;
        for (int y = 0; y < barHeight; ++y) {
            int idx = ((height - y - 1) * width + i) * 3;
            histogramImage[idx] = 0; // Red
            histogramImage[idx + 1] = 0; // Green
            histogramImage[idx + 2] = 0; // Blue
        }
    }

    stbi_write_jpg("histogram.jpg", width, height, 3, histogramImage, 90);
    free(histogramImage);
}

void histogramEqualization(unsigned char *data, int width, int height) {
    unsigned char *d_data;
    int *d_histogram, *d_cumulativeHistogram;
    int *histogram = (int *)malloc(256 * sizeof(int));
    int *cumulativeHistogram = (int *)malloc(256 * sizeof(int));
    int totalPixels = width * height;

    // Allocate memory on the GPU
    CHECK_CUDA(cudaMalloc(&d_data, totalPixels * 3 * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_histogram, 256 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cumulativeHistogram, 256 * sizeof(int)));
    
    // Copy data to the GPU
    CHECK_CUDA(cudaMemcpy(d_data, data, totalPixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_histogram, 0, 256 * sizeof(int)));

    // Launch kernel to compute histogram
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    clock_t start = clock();
    computeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, width, height, d_histogram);
    CHECK_CUDA(cudaDeviceSynchronize());
    clock_t end = clock();

    double timeTaken = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken to compute histogram: %f seconds\n", timeTaken);

    // Copy histogram back to the CPU
    CHECK_CUDA(cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // Calculate cumulative histogram
    cumulativeHistogram[0] = histogram[0];
    for (int i = 1; i < 256; i++) {
        cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
    }

    // Copy cumulative histogram to the GPU
    CHECK_CUDA(cudaMemcpy(d_cumulativeHistogram, cumulativeHistogram, 256 * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel to equalize histogram
    start = clock();
    equalizeHistogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, width, height, d_cumulativeHistogram, totalPixels);
    CHECK_CUDA(cudaDeviceSynchronize());
    end = clock();

    timeTaken = double(end - start) / CLOCKS_PER_SEC;
    printf("Time taken for histogram equalization: %f seconds\n", timeTaken);

    // Copy data back to the CPU
    CHECK_CUDA(cudaMemcpy(data, d_data, totalPixels * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_cumulativeHistogram);
    
    // Free CPU memory
    free(histogram);
    free(cumulativeHistogram);

    // Save histogram image
    saveHistogramImage(histogram);
}

int main() {
    char filename[256];
    printf("Enter the image file name: ");
    scanf("%255s", filename);

    int width, height, channels;
    unsigned char *imageData = stbi_load(filename, &width, &height, &channels, 3);

    if (imageData == NULL) {
        fprintf(stderr, "Error opening image file\n");
        return EXIT_FAILURE;
    }

    histogramEqualization(imageData, width, height);

    stbi_write_jpg("equalized_image.jpg", width, height, 3, imageData, 90);

    stbi_image_free(imageData);

    printf("Equalized image saved as 'equalized_image.jpg'\n");

    return 0;
}

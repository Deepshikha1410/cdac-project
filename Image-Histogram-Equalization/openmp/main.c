#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>
#include <string.h>
#include <omp.h>  // Include OpenMP header

// Function prototypes
void readJPEG(const char *filename, unsigned char **data, int *width, int *height, int *color_space);
void writeJPEG(const char *filename, unsigned char *data, int width, int height, int color_space);
void histogramEqualization(unsigned char *data, int width, int height, int color_space);
void saveHistogramImageJPEG(const int histogram[], const char *filename);

void readJPEG(const char *filename, unsigned char **data, int *width, int *height, int *color_space) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    *width = cinfo.output_width;
    *height = cinfo.output_height;
    *color_space = cinfo.out_color_space;
    int row_stride = cinfo.output_width * cinfo.output_components;

    *data = (unsigned char *)malloc(row_stride * cinfo.output_height);
    if (*data == NULL) {
        perror("Memory allocation failed");
        jpeg_destroy_decompress(&cinfo);
        fclose(file);
        exit(EXIT_FAILURE);
    }

    unsigned char *row_pointer[1];
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = *data + (cinfo.output_scanline) * row_stride;
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(file);
}

void writeJPEG(const char *filename, unsigned char *data, int width, int height, int color_space) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = (color_space == JCS_GRAYSCALE) ? 1 : 3;
    cinfo.in_color_space = color_space;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    jpeg_start_compress(&cinfo, TRUE);
    int row_stride = width * cinfo.input_components;
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned char *row_pointer[1];
        row_pointer[0] = data + (cinfo.next_scanline) * row_stride;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);
}

void saveHistogramImageJPEG(const int histogram[], const char *filename) {
    int width = 800;
    int height = 400;
    int barWidth = width / 256;
    int maxHeight = height - 20;
    int maxValue = 0;

    // Find the maximum value in the histogram for scaling
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > maxValue) {
            maxValue = histogram[i];
        }
    }

    // Create a buffer for the image (all white initially)
    unsigned char *image_buffer = (unsigned char *)malloc(width * height * 3);
    if (image_buffer == NULL) {
        perror("Error allocating memory for histogram image");
        exit(EXIT_FAILURE);
    }

    memset(image_buffer, 255, width * height * 3); // White background

    // Draw the histogram bars
    for (int i = 0; i < 256; i++) {
        int barHeight = (maxValue > 0) ? ((double)histogram[i] / maxValue) * maxHeight : 0;
        int y_offset = height - barHeight;
        for (int y = y_offset; y < height; y++) {
            for (int x = i * barWidth; x < (i + 1) * barWidth; x++) {
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    int index = (y * width + x) * 3;
                    image_buffer[index + 0] = 0;   // Red
                    image_buffer[index + 1] = 0;   // Green
                    image_buffer[index + 2] = 0;   // Blue
                }
            }
        }
    }

    // Create JPEG image
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        free(image_buffer);
        exit(EXIT_FAILURE);
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE);

    jpeg_start_compress(&cinfo, TRUE);
    unsigned char *row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &image_buffer[cinfo.next_scanline * width * 3];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);
    free(image_buffer);
}

void histogramEqualization(unsigned char *data, int width, int height, int color_space) {
    if (color_space == JCS_GRAYSCALE) {
        int histogram[256] = {0};
        int cumulativeHistogram[256] = {0};
        unsigned char newPixelValue[256];

        // Calculate histogram
        int rowSize = width;
        #pragma omp parallel
        {
            int local_histogram[256] = {0};
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j++) {
                    unsigned char pixel = data[i * rowSize + j];
                    local_histogram[pixel]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    histogram[i] += local_histogram[i];
                }
            }
        }

        // Save histogram before equalization
        saveHistogramImageJPEG(histogram, "histogram_before.jpg");

        // Calculate cumulative histogram
        cumulativeHistogram[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
            newPixelValue[i] = (unsigned char)(((float)cumulativeHistogram[i] - cumulativeHistogram[0]) / (width * height - 1) * 255);
        }

        // Apply histogram equalization
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j++) {
                    data[i * rowSize + j] = newPixelValue[data[i * rowSize + j]];
                }
            }
        }

        // Save histogram after equalization
        int newHistogram[256] = {0};
        #pragma omp parallel
        {
            int local_new_histogram[256] = {0};
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j++) {
                    unsigned char pixel = data[i * rowSize + j];
                    local_new_histogram[pixel]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    newHistogram[i] += local_new_histogram[i];
                }
            }
        }
        saveHistogramImageJPEG(newHistogram, "histogram_after.jpg");
    } else if (color_space == JCS_RGB) {
        int histogram[256] = {0};
        int cumulativeHistogram[256] = {0};
        unsigned char newPixelValue[256];

        // Calculate histogram for grayscale image
        int rowSize = width * 3;
        #pragma omp parallel
        {
            int local_histogram[256] = {0};
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j += 3) {
                    unsigned char gray = (data[i * rowSize + j] * 0.299) + (data[i * rowSize + j + 1] * 0.587) + (data[i * rowSize + j + 2] * 0.114);
                    local_histogram[gray]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    histogram[i] += local_histogram[i];
                }
            }
        }

        // Save histogram before equalization
        saveHistogramImageJPEG(histogram, "histogram_before.jpg");

        // Calculate cumulative histogram
        cumulativeHistogram[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
            newPixelValue[i] = (unsigned char)(((float)cumulativeHistogram[i] - cumulativeHistogram[0]) / (width * height - 1) * 255);
        }

        // Apply histogram equalization
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j += 3) {
                    unsigned char gray = (data[i * rowSize + j] * 0.299) + (data[i * rowSize + j + 1] * 0.587) + (data[i * rowSize + j + 2] * 0.114);
                    unsigned char equalizedValue = newPixelValue[gray];
                    data[i * rowSize + j] = equalizedValue;         // Red
                    data[i * rowSize + j + 1] = equalizedValue;     // Green
                    data[i * rowSize + j + 2] = equalizedValue;     // Blue
                }
            }
        }

        // Save histogram after equalization
        int newHistogram[256] = {0};
        #pragma omp parallel
        {
            int local_new_histogram[256] = {0};
            #pragma omp for
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < rowSize; j += 3) {
                    unsigned char gray = (data[i * rowSize + j] * 0.299) + (data[i * rowSize + j + 1] * 0.587) + (data[i * rowSize + j + 2] * 0.114);
                    local_new_histogram[gray]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    newHistogram[i] += local_new_histogram[i];
                }
            }
        }
        saveHistogramImageJPEG(newHistogram, "histogram_after.jpg");
    }
}

int main() {
    char filename[256];
    printf("Enter the image file name (with extension): ");
    scanf("%255s", filename);

    unsigned char *data = NULL;
    int width, height;
    int color_space;
    char *extension = strrchr(filename, '.');

    if (extension != NULL) {
        if (strcmp(extension, ".jpg") == 0 || strcmp(extension, ".jpeg") == 0) {
            readJPEG(filename, &data, &width, &height, &color_space);

            histogramEqualization(data, width, height, color_space);

            char output_filename[256];
            snprintf(output_filename, sizeof(output_filename), "equalized_image%s", extension);
            writeJPEG(output_filename, data, width, height, color_space);

            free(data);
            printf("Equalized image saved as '%s'\n", output_filename);
        } else {
            fprintf(stderr, "Unsupported file extension\n");
            return EXIT_FAILURE;
        }
    } else {
        fprintf(stderr, "Invalid file name\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

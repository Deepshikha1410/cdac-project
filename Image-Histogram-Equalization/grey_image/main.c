#include <stdio.h>
#include <stdlib.h>
#include <jpeglib.h>

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
    *color_space = cinfo.out_color_space; // Save the color space
    int row_stride = cinfo.output_width * cinfo.output_components;

    *data = (unsigned char *)malloc(row_stride * cinfo.output_height);
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
    cinfo.input_components = (color_space == JCS_GRAYSCALE) ? 1 : 3; // 1 for grayscale, 3 for RGB
    cinfo.in_color_space = color_space;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 75, TRUE); // Quality 75

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

void histogramEqualization(unsigned char *data, int width, int height, int color_space) {
    if (color_space == JCS_GRAYSCALE) {
        int histogram[256] = {0};
        int cumulativeHistogram[256] = {0};
        unsigned char newPixelValue[256];

        // Calculate histogram
        int rowSize = width;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < rowSize; j++) {
                histogram[data[i * rowSize + j]]++;
            }
        }

        // Calculate cumulative histogram
        cumulativeHistogram[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
            newPixelValue[i] = (unsigned char)(((float)cumulativeHistogram[i] - cumulativeHistogram[0]) / (width * height - 1) * 255);
        }

        // Apply histogram equalization
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < rowSize; j++) {
                data[i * rowSize + j] = newPixelValue[data[i * rowSize + j]];
            }
        }
    } else if (color_space == JCS_RGB) {
        int histogram[256] = {0};
        int cumulativeHistogram[256] = {0};
        unsigned char newPixelValue[256];

        // Calculate histogram for grayscale image
        int rowSize = width * 3;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < rowSize; j += 3) {
                unsigned char gray = (data[i * rowSize + j] * 0.299) + (data[i * rowSize + j + 1] * 0.587) + (data[i * rowSize + j + 2] * 0.114);
                histogram[gray]++;
            }
        }

        // Calculate cumulative histogram
        cumulativeHistogram[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cumulativeHistogram[i] = cumulativeHistogram[i - 1] + histogram[i];
            newPixelValue[i] = (unsigned char)(((float)cumulativeHistogram[i] - cumulativeHistogram[0]) / (width * height - 1) * 255);
        }

        // Apply histogram equalization
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
}

int main() {
    char filename[256];
    printf("Enter the JPEG image file name: ");
    scanf("%255s", filename);

    unsigned char *data = NULL;
    int width, height;
    int color_space;

    readJPEG(filename, &data, &width, &height, &color_space);

    histogramEqualization(data, width, height, color_space);

    writeJPEG("equalized_image.jpg", data, width, height, color_space);

    free(data);
    printf("Equalized image saved as 'equalized_image.jpg'\n");

    return 0;
}

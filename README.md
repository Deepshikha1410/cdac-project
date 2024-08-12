**Image Processing: Histogram Equalization**

#Overview
This repository contains implementations of histogram equalization, a technique used to enhance the contrast of images, using OpenMP, CUDA, and C.

#Features
Histogram equalization using:
OpenMP (shared memory parallelization)
CUDA (GPU acceleration)
C (sequential implementation)
Supports grayscale images and colour images
Example images provided for testing

Getting Started

**Prerequisites**
OpenMP-enabled compiler (e.g., GCC)
CUDA Toolkit (for CUDA implementation)
C compiler (e.g., GCC)
Building

*Commands to Compiile and Execute*
OpenMP: gcc -fopenmp <filename>.c -o <executable_name>
CUDA: nvcc <filename>.cu -o <executable_name>
C: gcc <filename>.c -o <executable_name>

Usage
./<executable_name>
./<executable_name>
./<executable_name>

Example Images
jpeg and jpg extension are accepetable
example_image.jpg(grayscale image)

**Benchmarks**
Coming soon!
Contributing
Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

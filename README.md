# JPEG Image Compression

## 1. Description of the JPEG Compression Algorithm

JPEG is a widely used lossy image compression algorithm in electronic devices and applications. It reduces image file size at the cost of some image quality.

**Input:** Digital image (represented as a matrix of pixel values)
**Output:** Compressed image file (e.g., JPEG, JPG)

### Common Applications:

1. Image storage on electronic devices and websites
2. High-speed image processing applications
3. Medical image compression (X-rays, MRIs)

### Main Steps of JPEG Algorithm:

1. Divide the image into 8x8 pixel blocks
2. Apply Discrete Cosine Transform (DCT) to each block
3. Use filters to remove less important frequency information
4. Apply entropy coding (Huffman coding) for data compression
5. Store compressed 8x8 pixel blocks sequentially

## 2. Need for Acceleration

Accelerating the JPEG algorithm is crucial for applications requiring fast image processing, such as video processing, continuous shooting in digital cameras, and real-time applications.

## 3. Potential Challenges

- JPEG is a sequential algorithm, limiting GPU acceleration to only certain parts
- Uncertainty about JPEG being the best compression algorithm for GPU processing

## 4. Resources

The project will be implemented in Python and executed on Google Colab. We will research JPEG compression from online resources (viblo, GitHub, etc.) and gradually build a parallelized and optimized version.

Useful resources:

- [Cornell University JPEG Algorithm Overview](http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/JPEG.html)
- [JPEG Compression on CUDA](https://www.eecg.toronto.edu/~moshovos/CUDA08/arx/JPEG_report.pdf)

## 5. Objectives

### Plan to Achieve:

- Implement JPEG compression algorithm
- Produce lower quality but smaller file size images while preserving important features
- Achieve at least 2x speedup using GPU parallelization compared to sequential algorithm

### Hope to Achieve:

- Develop an improved second version with better results
- Create a user-friendly application interface

### Minimum Goal (75% completion):

- Implement sequential version of JPEG compression
- Parallelize some steps of the algorithm
- Produce compressed images with slightly reduced quality and smaller file size

## Links

- [Google Colab Notebook](https://colab.research.google.com/drive/1hASbTgy0KDWVjUZCzC_-opx6PiAa4Zg3?usp=sharing)
- [Web Application](https://nguyenkhanh.pythonanywhere.com/)

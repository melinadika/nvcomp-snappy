#include "nvcomp/snappy.h"
#include <assert.h>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

using namespace std::chrono;

void checkCudaError(cudaError_t err, const char* msg)
{
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}
void execute_example(char* compressed_data, const size_t compressed_size)
{
  auto start_setup = high_resolution_clock::now();

  // allows for concurrent cuda processes
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Compute chunk sizes
  size_t chunk_size = 65536; // Adjust if needed
  size_t batch_size = (compressed_size + chunk_size - 1) / chunk_size;

  // Allocate device memory for compressed data
  char* device_compressed_data;
  cudaMalloc((void**)&device_compressed_data, compressed_size);
  cudaMemcpyAsync(
      device_compressed_data,
      compressed_data,
      compressed_size,
      cudaMemcpyHostToDevice,
      stream);

  // Set up data structures
  size_t* device_compressed_bytes;
  void** device_compressed_ptrs;
  cudaMalloc((void**)&device_compressed_bytes, sizeof(size_t) * batch_size);
  cudaMalloc((void**)&device_compressed_ptrs, sizeof(size_t) * batch_size);

  // Calculate the size of each chunk and set up pointers
  size_t* host_compressed_bytes;
  void** host_compressed_ptrs;
  cudaMallocHost((void**)&host_compressed_bytes, sizeof(size_t) * batch_size);
  cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    host_compressed_bytes[i] = (i + 1 < batch_size)
                                   ? chunk_size
                                   : compressed_size - (chunk_size * i);
    host_compressed_ptrs[i] = device_compressed_data + chunk_size * i;
  }

  cudaMemcpyAsync(
      device_compressed_bytes,
      host_compressed_bytes,
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice,
      stream);
  cudaMemcpyAsync(
      device_compressed_ptrs,
      host_compressed_ptrs,
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice,
      stream);

  // Allocate temporary workspace for decompression
  size_t decomp_temp_bytes;
  nvcompBatchedSnappyDecompressGetTempSize(
      batch_size, chunk_size, &decomp_temp_bytes);
  void* device_decomp_temp;
  cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

  // Allocate space for decompressed data on device
  size_t* device_uncompressed_bytes;
  void** device_uncompressed_ptrs;
  cudaMalloc((void**)&device_uncompressed_bytes, sizeof(size_t) * batch_size);
  cudaMalloc((void**)&device_uncompressed_ptrs, sizeof(size_t) * batch_size);

  // Allocate space for decompressed data on host
  size_t* host_uncompressed_bytes;
  char** host_uncompressed_ptrs;
  cudaMallocHost((void**)&host_uncompressed_bytes, sizeof(size_t) * batch_size);
  cudaMallocHost((void**)&host_uncompressed_ptrs, sizeof(char*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_uncompressed_ptrs[i] = (char*)malloc(chunk_size);
  }

  cudaMemcpyAsync(
      device_uncompressed_ptrs,
      host_uncompressed_ptrs,
      sizeof(size_t) * batch_size,
      cudaMemcpyHostToDevice,
      stream);

  // Allocate statuses
  nvcompStatus_t* device_statuses;
  cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

//   auto stop_setup = high_resolution_clock::now();
// duration = duration_cast<microseconds>(stop_setup - start_setup);
//   std::cout << "Setup time: " << duration.count() << "µs" << std::endl;

  // Perform decompression
  auto start_decompress = high_resolution_clock::now();

  nvcompStatus_t decomp_res = nvcompBatchedSnappyDecompressAsync(
      device_compressed_ptrs,
      device_compressed_bytes,
      device_uncompressed_bytes,
      nullptr, // No need to store actual uncompressed sizes here
      batch_size,
      device_decomp_temp,
      decomp_temp_bytes,
      device_uncompressed_ptrs,
      device_statuses,
      stream);

  checkCudaError(
      cudaStreamSynchronize(stream),
      "Error synchronizing stream after decompression");

//   auto stop_decompress = high_resolution_clock::now();
//   duration = duration_cast<microseconds>(stop_decompress - start_decompress);
//   std::cout << "Decompression time: " << duration.count() << "µs" << std::endl;

  if (decomp_res != nvcompSuccess) {
    std::cerr << "Failed decompression!" << std::endl;
    assert(decomp_res == nvcompSuccess);
  }

  std::cout << "Starting data copy..." << std::endl;

  // Copy decompressed data back to host
  auto start_copy = high_resolution_clock::now();
  for (size_t i = 0; i < batch_size; ++i) {
    std::cout << "Copy iteration: " << i << std::endl;

    cudaError_t copy_result = cudaMemcpy(
        host_uncompressed_ptrs[i],
        device_uncompressed_ptrs[i],
        chunk_size,
        cudaMemcpyDeviceToHost);

    if (copy_result != cudaSuccess) {
      std::cerr << "Error copying data from device to host at iteration " << i
                << ": " << cudaGetErrorString(copy_result) << std::endl;
      exit(EXIT_FAILURE);
    }

    std::cout << "Completed copy iteration: " << i << std::endl;
  }

  checkCudaError(
      cudaStreamSynchronize(stream), "Error synchronizing stream after copy");
//   auto stop_copy = high_resolution_clock::now();
//   duration = duration_cast<microseconds>(stop_copy - start_copy);
//   std::cout << "Copy time: " << duration.count() << "µs" << std::endl;

  std::cout << "Data copy completed." << std::endl;

  // Print the first 25 characters of the decompressed data
  std::cout << "First 25 characters of the decompressed data: ";
  for (size_t i = 0; i < batch_size; ++i) {
    std::cout.write(
        host_uncompressed_ptrs[i], std::min(chunk_size, size_t(25)));
    if (25 <= chunk_size) {
      break; // Exit loop once the first 25 characters have been printed
    }
  }
  std::cout << std::endl;

  // Cleanup
  cudaFree(device_compressed_data);
  cudaFree(device_compressed_bytes);
  cudaFree(device_compressed_ptrs);
  cudaFree(device_decomp_temp);
  cudaFree(device_uncompressed_bytes);
  cudaFree(device_uncompressed_ptrs);
  cudaFree(device_statuses);
  cudaFreeHost(host_compressed_bytes);
  cudaFreeHost(host_compressed_ptrs);
  for (size_t i = 0; i < batch_size; ++i) {
    free(host_uncompressed_ptrs[i]);
  }
  cudaFreeHost(host_uncompressed_bytes);
  cudaFreeHost(host_uncompressed_ptrs);

  cudaStreamDestroy(stream);
}

int main(int argc, char** argv)
{
  auto file = argv[1]; // Replace with your file path

  // Read compressed file
  std::ifstream inputFile(file, std::ios::binary | std::ios::ate);
  if (!inputFile) {
    std::cerr << "Error opening file" << std::endl;
    return 1;
  }

  std::streamsize compressed_size = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);

  char* compressed_data = (char*)malloc(compressed_size);
  if (!inputFile.read(compressed_data, compressed_size)) {
    std::cerr << "Error reading file" << std::endl;
    return 1;
  }

  std::cout << "Compressed size: " << compressed_size << std::endl;

  execute_example(compressed_data, compressed_size);

  free(compressed_data);
  return 0;
}

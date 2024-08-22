#include <random>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include "nvcomp/snappy.h"
#include <chrono>
using namespace std::chrono;

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
  cudaMemcpyAsync(device_compressed_data, compressed_data, compressed_size, cudaMemcpyHostToDevice, stream);

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
    host_compressed_bytes[i] = (i + 1 < batch_size) ? chunk_size : compressed_size - (chunk_size * i);
    host_compressed_ptrs[i] = device_compressed_data + chunk_size * i;
  }
  
  cudaMemcpyAsync(device_compressed_bytes, host_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

  // Allocate temporary workspace for decompression
  size_t decomp_temp_bytes;
  nvcompBatchedSnappyDecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
  void* device_decomp_temp;
  cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

  // Allocate space for decompressed data
  size_t* device_uncompressed_bytes;
  void** device_uncompressed_ptrs;
  cudaMalloc((void**)&device_uncompressed_bytes, sizeof(size_t) * batch_size);
  cudaMalloc((void**)&device_uncompressed_ptrs, sizeof(size_t) * batch_size);

  cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
      cudaMalloc(&host_compressed_ptrs[i], chunk_size);
  }

  cudaMemcpyAsync(device_uncompressed_ptrs, host_compressed_ptrs, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);

  // Allocate statuses
  nvcompStatus_t* device_statuses;
  cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

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

  cudaStreamSynchronize(stream);

  auto stop_decompress = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop_decompress - start_decompress);
  std::cout << "Decompression time: " << duration.count() << "µs" << std::endl;

  if (decomp_res != nvcompSuccess)
  {
    std::cerr << "Failed decompression!" << std::endl;
    assert(decomp_res == nvcompSuccess);
  }

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
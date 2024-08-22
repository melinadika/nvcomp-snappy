file(GLOB GDEFLATE_CPU_SOURCES *gdeflate_cpu*.cpp *gdeflate_cpu*.cu)

# Add quickstart example
add_executable(low_level_quickstart_example low_level_quickstart_example.cpp)
target_link_libraries(low_level_quickstart_example PRIVATE nvcomp CUDA::cudart)
target_include_directories(low_level_quickstart_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add decomp
add_executable(decomp decomp.cpp)
target_link_libraries(decomp PRIVATE nvcomp CUDA::cudart)
target_include_directories(decomp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

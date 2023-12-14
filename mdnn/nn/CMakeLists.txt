cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(atomic_nn)

find_package(Torch REQUIRED)

# Define our library target
add_library(atomic_nn SHARED atomic_nn.cpp)
set(CMAKE_CXX_STANDARD 14)
# Link against LibTorch
target_link_libraries(atomic_nn "${TORCH_LIBRARIES}")
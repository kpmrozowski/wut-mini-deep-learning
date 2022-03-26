cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cnn VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
option(WITH_CUDA "should be built for CUDA" OFF)
option(MEASURE_TIME "measure time too?" ON)

if (${WITH_CUDA})
    include_directories(/usr/local/cuda/targets/x86_64-linux/include)
    set(CUDA_ARCH "-arch=sm_80" CACHE STRING "Value of the NVCC -arch option.")
    if(${CMAKE_VERSION} VERSION_EQUAL 3.7 OR ${CMAKE_VERSION} VERSION_GREATER 3.7)
        option(AUTO_DETECT_ARCH "Detect local GPU compute arch automatically" ON)
    endif()
    message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")
    SET(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17 )

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")# -allow-unsupported-compiler")

    SET(CUDA_PROPAGATE_HOST_FLAGS ON)
    SET(CUDA_SEPARABLE_COMPILATION ON)
    SET(CUDA_VERBOSE_BUILD ON)

    find_package(CUDA REQUIRED)
    add_definitions(-DWITH_CUDA)
    add_subdirectory(src/Eden_resources)
    add_subdirectory(src/cudatest)
endif()

if(${MEASURE_TIME})
    add_definitions(-DMEASURE_TIME)
endif()

find_package(Torch REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

set(EXECUTABLE_NAME cnn)

add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/convnet.cpp
                                          src/imagefolder_dataset.cpp
                                          include/convnet.h
                                          include/imagefolder_dataset.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

add_subdirectory(utils/image_io)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} image-io fmt spdlog)

if (${WITH_CUDA})
   target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES} eden_resources)
endif()

# External dependencies
add_subdirectory("extern")

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)
if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} imagenette)
endif()


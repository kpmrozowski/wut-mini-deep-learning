# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/wut-mini-deep-learning/cnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/wut-mini-deep-learning/cnn/out/build

# Include any dependencies generated for this target.
include CMakeFiles/convolutional-neural-network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/convolutional-neural-network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/convolutional-neural-network.dir/flags.make

CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o: CMakeFiles/convolutional-neural-network.dir/flags.make
CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o: ../../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o -c /workspace/wut-mini-deep-learning/cnn/src/main.cpp

CMakeFiles/convolutional-neural-network.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convolutional-neural-network.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/wut-mini-deep-learning/cnn/src/main.cpp > CMakeFiles/convolutional-neural-network.dir/src/main.cpp.i

CMakeFiles/convolutional-neural-network.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convolutional-neural-network.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/wut-mini-deep-learning/cnn/src/main.cpp -o CMakeFiles/convolutional-neural-network.dir/src/main.cpp.s

CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o: CMakeFiles/convolutional-neural-network.dir/flags.make
CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o: ../../src/convnet.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o -c /workspace/wut-mini-deep-learning/cnn/src/convnet.cpp

CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/wut-mini-deep-learning/cnn/src/convnet.cpp > CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.i

CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/wut-mini-deep-learning/cnn/src/convnet.cpp -o CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.s

CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o: CMakeFiles/convolutional-neural-network.dir/flags.make
CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o: ../../src/imagefolder_dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o -c /workspace/wut-mini-deep-learning/cnn/src/imagefolder_dataset.cpp

CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/wut-mini-deep-learning/cnn/src/imagefolder_dataset.cpp > CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.i

CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/wut-mini-deep-learning/cnn/src/imagefolder_dataset.cpp -o CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.s

CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o: CMakeFiles/convolutional-neural-network.dir/flags.make
CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o: ../../utils/image_io/src/image_io.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o -c /workspace/wut-mini-deep-learning/cnn/utils/image_io/src/image_io.cpp

CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/wut-mini-deep-learning/cnn/utils/image_io/src/image_io.cpp > CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.i

CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/wut-mini-deep-learning/cnn/utils/image_io/src/image_io.cpp -o CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.s

# Object files for target convolutional-neural-network
convolutional__neural__network_OBJECTS = \
"CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o" \
"CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o" \
"CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o" \
"CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o"

# External object files for target convolutional-neural-network
convolutional__neural__network_EXTERNAL_OBJECTS =

convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/src/main.cpp.o
convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/src/convnet.cpp.o
convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/src/imagefolder_dataset.cpp.o
convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/utils/image_io/src/image_io.cpp.o
convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/build.make
convolutional-neural-network: /usr/lib/libc10.so
convolutional-neural-network: /usr/lib/libkineto.a
convolutional-neural-network: /usr/local/cuda/lib64/stubs/libcuda.so
convolutional-neural-network: /usr/local/cuda/lib64/libnvrtc.so
convolutional-neural-network: /usr/local/cuda/lib64/libnvToolsExt.so
convolutional-neural-network: /usr/local/cuda/lib64/libcudart.so
convolutional-neural-network: /usr/lib/libc10_cuda.so
convolutional-neural-network: utils/image_io/libimage-io.so
convolutional-neural-network: /usr/lib/libtorch.so
convolutional-neural-network: /usr/lib/libc10_cuda.so
convolutional-neural-network: /usr/lib/libc10.so
convolutional-neural-network: /usr/local/cuda/lib64/libcufft.so
convolutional-neural-network: /usr/local/cuda/lib64/libcurand.so
convolutional-neural-network: /usr/local/cuda/lib64/libcublas.so
convolutional-neural-network: /usr/local/cuda/lib64/libcudnn.so
convolutional-neural-network: /usr/lib/libc10.so
convolutional-neural-network: /usr/lib/libkineto.a
convolutional-neural-network: /usr/local/cuda/lib64/stubs/libcuda.so
convolutional-neural-network: /usr/local/cuda/lib64/libnvrtc.so
convolutional-neural-network: /usr/local/cuda/lib64/libnvToolsExt.so
convolutional-neural-network: /usr/local/cuda/lib64/libcudart.so
convolutional-neural-network: /usr/lib/libc10_cuda.so
convolutional-neural-network: CMakeFiles/convolutional-neural-network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable convolutional-neural-network"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convolutional-neural-network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/convolutional-neural-network.dir/build: convolutional-neural-network

.PHONY : CMakeFiles/convolutional-neural-network.dir/build

CMakeFiles/convolutional-neural-network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/convolutional-neural-network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/convolutional-neural-network.dir/clean

CMakeFiles/convolutional-neural-network.dir/depend:
	cd /workspace/wut-mini-deep-learning/cnn/out/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/wut-mini-deep-learning/cnn /workspace/wut-mini-deep-learning/cnn /workspace/wut-mini-deep-learning/cnn/out/build /workspace/wut-mini-deep-learning/cnn/out/build /workspace/wut-mini-deep-learning/cnn/out/build/CMakeFiles/convolutional-neural-network.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/convolutional-neural-network.dir/depend


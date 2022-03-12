# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.18.4)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(extern_stb_image_resize-populate NONE)

include(ExternalProject)
ExternalProject_Add(extern_stb_image_resize-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://raw.githubusercontent.com/nothings/stb/master/stb_image_resize.h" "DOWNLOAD_DIR" "/workspace/wut-mini-deep-learning/cnn/extern/stb_image_resize" DOWNLOAD_NO_EXTRACT YES
                    SOURCE_DIR          "/workspace/wut-mini-deep-learning/cnn/out/build/_deps/extern_stb_image_resize-src"
                    BINARY_DIR          "/workspace/wut-mini-deep-learning/cnn/out/build/_deps/extern_stb_image_resize-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


ExternalProject_Get_Property(extern_stb_image_resize-populate DOWNLOADED_FILE)
get_filename_component(dlFileName "${DOWNLOADED_FILE}" NAME)

ExternalProject_Add_Step(extern_stb_image_resize-populate copyfile
  COMMAND    "/usr/bin/cmake" -E copy_if_different
             "<DOWNLOADED_FILE>" "/workspace/wut-mini-deep-learning/cnn/out/build/_deps/extern_stb_image_resize-src"
  DEPENDEES  patch
  DEPENDERS  configure
  BYPRODUCTS "/workspace/wut-mini-deep-learning/cnn/out/build/_deps/extern_stb_image_resize-src/${dlFileName}"
  COMMENT    "Copying file to SOURCE_DIR"
)


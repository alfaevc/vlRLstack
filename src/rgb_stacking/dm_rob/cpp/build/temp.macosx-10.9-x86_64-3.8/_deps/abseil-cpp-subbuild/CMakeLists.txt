# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.18.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(abseil-cpp-populate NONE)

include(ExternalProject)
ExternalProject_Add(abseil-cpp-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/abseil/abseil-cpp.git" "GIT_TAG" "origin/master"
                    SOURCE_DIR          "/Users/inflalfa/Desktop/Columbia/robotics/final project/vlRLstack/src/rgb_stacking/dm_rob/cpp/build/temp.macosx-10.9-x86_64-3.8/_deps/abseil-cpp-src"
                    BINARY_DIR          "/Users/inflalfa/Desktop/Columbia/robotics/final project/vlRLstack/src/rgb_stacking/dm_rob/cpp/build/temp.macosx-10.9-x86_64-3.8/_deps/abseil-cpp-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)



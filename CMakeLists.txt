cmake_minimum_required (VERSION 3.10)

project (test VERSION 0.1)

set (CMAKE_CXX_STANDARD 11)

# enable with: cmake .. -DCMAKE_BUILD_TYPE=Release
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(SndFile REQUIRED)
#find_package(ZLIB REQUIRED)

find_package(PkgConfig  REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

include_directories (
    include
    test
)

add_executable (test
    src/main.cc
    src/weights.cc
    src/signal.cc
    src/samples.cc
    test/signal_test.cc
)

target_compile_options (test PRIVATE
    #-Wall
    #-Werror
)
    #-std=c++17

#set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
#set (CMAKE_CXX_STANDARD 17)
#set (CMAKE_CXX_STANDARD_REQUIRED True)


target_link_libraries (test PRIVATE
    SndFile::sndfile
    PkgConfig::FFTW
    #ZLIB::ZLIB
)

#install (TARGETS test RUNTIME DESTINATION bin)

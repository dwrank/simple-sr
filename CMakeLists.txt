cmake_minimum_required (VERSION 3.10)

project (simple-sr VERSION 0.1)

set (CMAKE_CXX_STANDARD 11)

# enable with: cmake .. -DCMAKE_BUILD_TYPE=Release
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(PkgConfig  REQUIRED)
pkg_search_module(SNDFILE REQUIRED sndfile IMPORTED_TARGET)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

include_directories (
    include
)

add_executable (simple-sr
    src/main.cc
    src/utils.cc
    src/weights.cc
    src/signal.cc
    src/samples.cc
)

target_compile_options (simple-sr PRIVATE
    #-Wall
    #-Werror
)

target_link_libraries (simple-sr PRIVATE
    PkgConfig::SNDFILE
    PkgConfig::FFTW
)

install (TARGETS simple-sr
    RUNTIME DESTINATION bin
)

if(TESTS)
    add_subdirectory(test)
endif()

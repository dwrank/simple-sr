cmake_minimum_required (VERSION 3.10)

find_package(PkgConfig  REQUIRED)
pkg_search_module(SNDFILE REQUIRED sndfile IMPORTED_TARGET)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

add_executable (signal-test
    signal_test.cc
    ../src/signal.cc
)

target_link_libraries (signal-test PRIVATE
    PkgConfig::SNDFILE
    PkgConfig::FFTW
)


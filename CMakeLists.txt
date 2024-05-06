cmake_minimum_required (VERSION 3.10)

project (simple-sr VERSION 0.1)

set (CMAKE_CXX_STANDARD 11)

# enable with: cmake .. -DCMAKE_BUILD_TYPE=Release
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package (PkgConfig  REQUIRED)
pkg_search_module (SNDFILE REQUIRED sndfile IMPORTED_TARGET)
pkg_search_module (FFTW REQUIRED fftw3 IMPORTED_TARGET)

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set (CMAKE_INSTALL_INCLUDEDIR /usr/local/lib)
endif()

if (NOT DEFINED CMAKE_INSTALL_INCLUDEDIR)
    set (CMAKE_INSTALL_INCLUDEDIR /usr/local/include)
endif()

include_directories (
    include
)

add_executable (simple-sr
    src/main.cc
)

add_library (simplesr SHARED
    src/utils.cc
    src/weights.cc
    src/signal.cc
    src/samples.cc
    src/model.cc
)

set (PUBLIC_INCLUDES
    include/model.h
    include/tensor.h
)

target_compile_options (simple-sr PRIVATE
    -Wall
    -Werror
)

target_link_libraries (simple-sr PRIVATE
    simplesr
)

target_link_libraries (simplesr PRIVATE
    PkgConfig::SNDFILE
    PkgConfig::FFTW
)

install (TARGETS simple-sr
    RUNTIME DESTINATION bin
)

set_target_properties (simplesr PROPERTIES
    PUBLIC_HEADER
    "${PUBLIC_INCLUDES}"
)

install (TARGETS simplesr
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libsimplesr
)

if(TESTS)
    add_subdirectory(test)
endif()

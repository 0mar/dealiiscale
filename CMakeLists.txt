cmake_minimum_required(VERSION 3.1)
project(dealiiscale)
set(CMAKE_CXX_STANDARD 14)

if(APPLE)
    set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)
else()
    set(deal.II_DIR /usr/share/cmake/deal.II)
endif()

find_package(deal.II 9.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
    message(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
endif()

deal_II_initialize_cached_variables()

add_library(manufactured manufactured.cpp manufactured.h)
deal_II_setup_target(manufactured)
add_executable(typical manager.cpp manager.h)
deal_II_setup_target(typical)
target_link_libraries(typical manufactured)

add_executable(demo demo.cpp)
deal_II_setup_target(demo)

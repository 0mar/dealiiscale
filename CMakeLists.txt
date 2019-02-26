cmake_minimum_required(VERSION 3.1)
project(twoscale)

set(CMAKE_CXX_STANDARD 14)
set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)


add_executable(templater templater.cpp)

SET(TARGET "manufactured")
SET(TARGET_SRC ${TARGET}.cpp)
FIND_PACKAGE(deal.II 9.0.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
    MESSAGE(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
ENDIF()
PROJECT(${TARGET})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
DEAL_II_INVOKE_AUTOPILOT()

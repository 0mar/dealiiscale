cmake_minimum_required(VERSION 3.1)
project(dealiiscale)
set(CMAKE_CXX_STANDARD 14)

if (APPLE)
    set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)
else () # todo Add switch for linux
    set(deal.II_DIR /usr/share/cmake/deal.II)
endif ()

find_package(deal.II 9.0 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
if (NOT ${deal.II_FOUND})
    message(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
endif ()

deal_II_initialize_cached_variables()

add_library(multiscale src/manager.h src/manager.cpp src/micro.h src/micro.cpp src/macro.h src/macro.cpp)
deal_II_setup_target(multiscale)

add_library(single_scale src/micro_manager.h src/micro_manager.cpp src/micro.h src/micro.cpp src/macro.h src/macro.cpp src/micro_manager.h src/macro_manager.cpp)
deal_II_setup_target(single_scale)


add_executable(manufactured src/manufactured.cpp)
deal_II_setup_target(manufactured)
target_link_libraries(manufactured multiscale)

add_executable(separate src/separate_scales.cpp)
deal_II_setup_target(separate)
target_link_libraries(separate single_scale)

add_executable(demo src/demo.cpp)
deal_II_setup_target(demo)

set(PLOT_NOTEBOOK plot_convergence.ipynb)

# Create result directory
add_custom_command(TARGET manufactured POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        ${CMAKE_CURRENT_BINARY_DIR}/results)
# Run everytime we make
add_custom_command(TARGET manufactured POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/results
        ${CMAKE_CURRENT_BINARY_DIR}/results)
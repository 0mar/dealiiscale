cmake_minimum_required(VERSION 3.1)
project(dealiiscale)
set(CMAKE_CXX_STANDARD 14)

if (APPLE)
    set(deal.II_DIR /Applications/deal.II-9.0.0.app/Contents/Resources/lib/cmake/deal.II)
elseif (UNIX)
    set(deal.II_DIR /usr/share/cmake/deal.II)
else ()
    message(FATAL_ERROR "Runs only on Mac or Linux")
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

set(ELLIPTIC_DIR src/elliptic-elliptic) # Todo: Make a CMakeLists for each subdirectory
set(PARABOLIC_DIR src/elliptic-parabolic)
set(DEMO_DIR src/playground)
add_library(elliptic ${ELLIPTIC_DIR}/manager.h ${ELLIPTIC_DIR}/manager.cpp ${ELLIPTIC_DIR}/micro.h ${ELLIPTIC_DIR}/micro.cpp ${ELLIPTIC_DIR}/macro.h ${ELLIPTIC_DIR}/macro.cpp)
deal_II_setup_target(elliptic)

add_library(single_scale ${ELLIPTIC_DIR}/micro_manager.h ${ELLIPTIC_DIR}/micro_manager.cpp ${ELLIPTIC_DIR}/micro.h ${ELLIPTIC_DIR}/micro.cpp ${ELLIPTIC_DIR}/macro.h ${ELLIPTIC_DIR}/macro.cpp ${ELLIPTIC_DIR}/micro_manager.h ${ELLIPTIC_DIR}/macro_manager.cpp)
deal_II_setup_target(single_scale)

add_library(parabolic ${PARABOLIC_DIR}/manager.h ${PARABOLIC_DIR}/manager.cpp ${PARABOLIC_DIR}/rho_solver.h ${PARABOLIC_DIR}/rho_solver.cpp ${PARABOLIC_DIR}/pi_solver.h ${PARABOLIC_DIR}/pi_solver.cpp)
deal_II_setup_target(parabolic)

add_executable(manufactured ${ELLIPTIC_DIR}/manufactured.cpp)
deal_II_setup_target(manufactured)
target_link_libraries(manufactured multiscale)

add_executable(solve_two_scale ${PARABOLIC_DIR}/two_scale.cpp)
deal_II_setup_target(solve_two_scale)
target_link_libraries(solve_two_scale parabolic)

add_executable(separate ${ELLIPTIC_DIR}/separate_scales.cpp)
deal_II_setup_target(separate)
target_link_libraries(separate single_scale)

add_executable(simple ${DEMO_DIR}/simple.cpp)
deal_II_setup_target(simple)

add_executable(robin ${DEMO_DIR}/robin.cpp)
deal_II_setup_target(robin)

add_executable(demo ${DEMO_DIR}/demo.cpp)
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

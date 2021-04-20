cmake_minimum_required(VERSION 3.1)
project(dealiiscale)
set(CMAKE_CXX_STANDARD 14)
include(ProcessorCount)
ProcessorCount(N)

message(STATUS "Number of cores ${N}")
if (EXISTS "/.dockerenv")
    message(STATUS "Detected running in Docker, loading config for dealii/dealii:v9.1.1-gcc-mpi-fulldepscandi-debugrelease")
    set(deal.II_DIR /home/dealii/dealii-v9.1.1/lib/cmake/deal.II)
    set(GCC_FLAGS "-fext-numeric-literals")
    set(CMAKE_CXX_FLAGS "${GCC_FLAGS}")
    include_directories("/home/dealii/dealii-v9.1.1-src/bundled/muparser_v2_2_4/include")
elseif (APPLE)
    set(deal.II_DIR /Applications/deal.II-9.1.1.app/Contents/Resources/lib/cmake/deal.II)
elseif (UNIX)
    message(STATUS "Running native Linux build.")
    set(deal.II_DIR /usr/share/cmake/deal.II)
    set(GCC_FLAGS "-fext-numeric-literals")
    set(CMAKE_CXX_FLAGS "${GCC_FLAGS}")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
else ()
    message(FATAL_ERROR "Runs only on Mac or Linux")
endif ()

find_package(deal.II 9.1 QUIET HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
if (NOT ${deal.II_FOUND})
    message(FATAL_ERROR "\n"
            "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
            "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
            "or set an environment variable \"DEAL_II_DIR\" that contains this path."
            )
endif ()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS}) # Todo: Generalize this for all architectures.

deal_II_initialize_cached_variables()

set(VESICLE_DIR src/vesicles) # Todo: Make a CMakeLists for each subdirectory
set(TOOLS_DIR src/tools)
set(TEST_DIR tests)
add_executable(test_tools ${TEST_DIR}/test_tools.cpp ${TOOLS_DIR}/multiscale_function_parser.cpp ${TOOLS_DIR}/multiscale_function_parser.h ${TOOLS_DIR}/pde_data.cpp ${TOOLS_DIR}/pde_data.h ${TOOLS_DIR}/mapping.h ${TOOLS_DIR}/mapping.cpp)
deal_II_setup_target(test_tools)
target_link_libraries(test_tools ${Boost_LIBRARIES})

add_library(tools ${TOOLS_DIR}/multiscale_function_parser.cpp ${TOOLS_DIR}/multiscale_function_parser.h ${TOOLS_DIR}/pde_data.cpp ${TOOLS_DIR}/pde_data.h ${TOOLS_DIR}/mapping.cpp ${TOOLS_DIR}/mapping.h)
deal_II_setup_target(tools)

add_library(vesicles ${VESICLE_DIR}/manager.h ${VESICLE_DIR}/manager.cpp ${VESICLE_DIR}/micro.h ${VESICLE_DIR}/micro.cpp ${VESICLE_DIR}/macro.h ${VESICLE_DIR}/macro.cpp)
deal_II_setup_target(vesicles)
target_link_libraries(vesicles tools)

set(PLOT_NOTEBOOK plot_convergence.ipynb)
add_executable(solve_vesicles ${VESICLE_DIR}/vesicles.cpp)
deal_II_setup_target(solve_vesicles)
target_link_libraries(solve_vesicles vesicles)

enable_testing()
add_definitions(-DBOOST_TEST_DYN_LINK)
add_test(test_tools test_tools)

# Symlink input and result directory into build directory
execute_process(COMMAND ${CMAKE_SOURCE_DIR}/populate.sh "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

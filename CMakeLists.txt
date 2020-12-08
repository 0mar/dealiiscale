cmake_minimum_required(VERSION 3.1)
project(dealiiscale)
set(CMAKE_CXX_STANDARD 14)
include(ProcessorCount)
ProcessorCount(N)
set(APOLLO $ENV{HOSTNAME})

message(STATUS "Number of cores ${N}")
if (EXISTS "/.dockerenv")
    message(STATUS "Detected running in Docker, loading config for dealii/dealii:v9.1.1-gcc-mpi-fulldepscandi-debugrelease")
    set(deal.II_DIR /home/dealii/dealii-v9.1.1/lib/cmake/deal.II)
    set(GCC_FLAGS "-fext-numeric-literals")
    set(CMAKE_CXX_FLAGS "${GCC_FLAGS}")
    # set(MPI_FLAGS "-fopenmp")
    include_directories("/home/dealii/dealii-v9.1.1-src/bundled/muparser_v2_2_4/include")
elseif (APPLE)
    set(deal.II_DIR /Applications/deal.II-9.1.1.app/Contents/Resources/lib/cmake/deal.II)
elseif (APOLLO MATCHES "apollo.+")
    message(STATUS "Running build for Apollo2")
    set(deal.II_DIR /mnt/ebinstall/software/deal.II/9.1.1-intel-2019a)
    set(GCC_FLAGS "-fext-numeric-literals")
    set(CMAKE_CXX_FLAGS "${GCC_FLAGS}")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    include_directories(${CMAKE_SOURCE_DIR}/include)
    set(CMAKE_C_COMPILER icc)
    set(CMAKE_CXX_COMPILER icpc)
    find_package(Threads REQUIRED)
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


deal_II_initialize_cached_variables()

set(ELLIPTIC_DIR src/elliptic-elliptic) # Todo: Make a CMakeLists for each subdirectory
set(PARABOLIC_DIR src/elliptic-parabolic)
set(TOOLS_DIR src/tools)
set(BIOMATH_DIR src/bio-multiscale)
set(DEMO_DIR src/playground)

add_library(tools ${TOOLS_DIR}/multiscale_function_parser.cpp ${TOOLS_DIR}/multiscale_function_parser.h ${TOOLS_DIR}/pde_data.cpp ${TOOLS_DIR}/pde_data.h ${TOOLS_DIR}/mapping.cpp ${TOOLS_DIR}/mapping.h 
    ${TOOLS_DIR}/muParser.cpp ${TOOLS_DIR}/muParserBase.cpp ${TOOLS_DIR}/muParserBytecode.cpp ${TOOLS_DIR}/muParserCallback.cpp ${TOOLS_DIR}/muParserError.cpp ${TOOLS_DIR}/muParserTokenReader.cpp
    )
deal_II_setup_target(tools)

add_library(elliptic ${ELLIPTIC_DIR}/manager.h ${ELLIPTIC_DIR}/manager.cpp ${ELLIPTIC_DIR}/micro.h ${ELLIPTIC_DIR}/micro.cpp ${ELLIPTIC_DIR}/macro.h ${ELLIPTIC_DIR}/macro.cpp)
deal_II_setup_target(elliptic)
target_link_libraries(elliptic tools)

add_library(parabolic ${PARABOLIC_DIR}/time_manager.h ${PARABOLIC_DIR}/time_manager.cpp ${PARABOLIC_DIR}/rho_solver.h ${PARABOLIC_DIR}/rho_solver.cpp ${PARABOLIC_DIR}/pi_solver.h ${PARABOLIC_DIR}/pi_solver.cpp)
deal_II_setup_target(parabolic)
target_link_libraries(parabolic tools)

add_library(biomath ${BIOMATH_DIR}/manager.h ${BIOMATH_DIR}/manager.cpp ${BIOMATH_DIR}/micro.h ${BIOMATH_DIR}/micro.cpp ${BIOMATH_DIR}/macro.h ${BIOMATH_DIR}/macro.cpp)
deal_II_setup_target(biomath)
target_link_libraries(biomath tools)

add_executable(solve_elliptic ${ELLIPTIC_DIR}/manufactured.cpp)
deal_II_setup_target(solve_elliptic)
target_link_libraries(solve_elliptic elliptic)

add_executable(solve_biomath ${BIOMATH_DIR}/manufactured.cpp)
deal_II_setup_target(solve_biomath)
target_link_libraries(solve_biomath biomath)

set(PLOT_NOTEBOOK plot_convergence.ipynb)

add_executable(solve_parabolic ${PARABOLIC_DIR}/two_scale.cpp)
deal_II_setup_target(solve_parabolic)
target_link_libraries(solve_parabolic parabolic)

add_executable(simple ${DEMO_DIR}/simple.cpp)
deal_II_setup_target(simple)

add_executable(robin ${DEMO_DIR}/robin.cpp)
deal_II_setup_target(robin)

add_executable(transform ${DEMO_DIR}/transform.cpp)
deal_II_setup_target(transform)

add_executable(example ${DEMO_DIR}/example.cpp)
deal_II_setup_target(example)

add_executable(demo ${DEMO_DIR}/demo.cpp)
deal_II_setup_target(demo)

add_executable(integration ${DEMO_DIR}/integration.cpp)
deal_II_setup_target(integration)

add_executable(dof_mapper ${DEMO_DIR}/dof_mapper.cpp)
deal_II_setup_target(dof_mapper)

add_executable(compare_solutions ${DEMO_DIR}/compare_solutions.cpp)
deal_II_setup_target(compare_solutions)

add_executable(parsing ${DEMO_DIR}/parsing.cpp)
deal_II_setup_target(parsing)
target_link_libraries(parsing tools)

# Symlink input and result directory into build directory
execute_process(COMMAND ${CMAKE_SOURCE_DIR}/populate.sh "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}")

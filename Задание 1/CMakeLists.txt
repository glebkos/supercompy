cmake_minimum_required(VERSION 3.16)
project(Variant14 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(MPI REQUIRED COMPONENTS CXX)

foreach(mode IN ITEMS seq omp mpi hybrid)
    add_executable(poisson_${mode} solver.cpp)
    target_compile_options(poisson_${mode} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()
target_link_libraries(poisson_omp PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(poisson_hybrid PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
target_link_libraries(poisson_mpi PRIVATE MPI::MPI_CXX)
target_compile_definitions(poisson_mpi PRIVATE USE_MPI)
target_compile_definitions(poisson_hybrid PRIVATE USE_MPI)

enable_testing()
add_executable(test_geometry test_geometry.cpp)
add_test(NAME geometry_and_partition COMMAND test_geometry)
find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
add_test(NAME numerical_and_parallel
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check.py
            --build $<TARGET_FILE_DIR:poisson_seq> --mpiexec ${MPIEXEC_EXECUTABLE})
set_tests_properties(numerical_and_parallel PROPERTIES TIMEOUT 180)

add_executable(sift-qg-test "sift-qg-test.cpp")


find_package(OpenMP)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(sift-qg-test
            PUBLIC
            OpenMP::OpenMP_CXX
            /usr/local/lib/libngt.a
            ${BLAS_LIBRARIES}
            ${LAPACK_LIBRARIES}
    )
else()
    target_link_libraries(sift-qg-test
            PUBLIC
            /usr/local/lib/libngt.a
    )
endif()

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")



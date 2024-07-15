add_executable(sift-panng-test "sift-panng-test.cpp"
        sift-panng-test.cpp)


find_package(OpenMP)
# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(sift-panng-test
            PUBLIC
            OpenMP::OpenMP_CXX
            /usr/local/lib/libngt.a
    )
else()
    target_link_libraries(sift-panng-test
            PUBLIC
            /usr/local/lib/libngt.a
    )
endif()

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")



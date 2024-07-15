add_executable(trevi-panng-test "trevi-panng-test.cpp"
        trevi-panng-test.cpp)


find_package(OpenMP)
# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(trevi-panng-test
            PUBLIC
            OpenMP::OpenMP_CXX
            /usr/local/lib/libngt.a
    )
else()
    target_link_libraries(trevi-panng-test
            PUBLIC
            /usr/local/lib/libngt.a
    )
endif()

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")



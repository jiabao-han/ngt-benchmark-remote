add_executable(enron-panng-test "enron-panng-test.cpp")


find_package(OpenMP)
# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(enron-panng-test
            PUBLIC
            OpenMP::OpenMP_CXX
            /usr/local/lib/libngt.a
    )
else()
    target_link_libraries(enron-panng-test
            PUBLIC
            /usr/local/lib/libngt.a
    )
endif()

# Specify the directories for NGT include files
include_directories("${PROJECT_BINARY_DIR}/lib/")
include_directories("${CMAKE_BINARY_DIR}/configured_files/include")



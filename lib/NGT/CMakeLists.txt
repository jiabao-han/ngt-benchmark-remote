project(ngt)

set(OpenMP_C "${CMAKE_C_COMPILER}")
set(OpenMP_C_FLAGS ${OpenMP_C_FLAGS})
set(OpenMP_C_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "omp")

set(OpenMP_CXX "${CMAKE_C_COMPILER}")
set(OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS})
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "omp")

#link_directories("/usr/local/Cellar/libomp/18.1.8/lib/")
#target_link_directories(${PROJECT_NAME} PRIVATE "/usr/local/Cellar/libomp/18.1.8/lib/")


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)
string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_BUILD_TYPE_LOWER: ${CMAKE_BUILD_TYPE_LOWER}")

if(${NGT_SHARED_MEMORY_ALLOCATOR})
	set(NGT_QBG_DISABLED TRUE)
endif(${NGT_SHARED_MEMORY_ALLOCATOR})

if(${UNIX})
	set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

	if(CMAKE_VERSION VERSION_LESS 3.1)
		set(BASE_OPTIONS "-Wall -std=gnu++0x -lrt")

		if(${NGT_AVX_DISABLED})
			message(STATUS "AVX will not be used to compute distances.")
		endif()

		if(${NGT_OPENMP_DISABLED})
			message(STATUS "OpenMP is disabled.")
		else()
			set(BASE_OPTIONS "${BASE_OPTIONS} -fopenmp")
		endif()

		if(${NGT_ASSERT_DISABLED})
			message(STATUS "assert() is disabled.")
			set(BASE_OPTIONS "${BASE_OPTIONS} -DNDEBUG")
		endif()

		set(CMAKE_CXX_FLAGS_DEBUG "-g ${BASE_OPTIONS}")

		if(${NGT_MARCH_NATIVE_DISABLED})
			message(STATUS "Compile option -march=native is disabled.")
			set(CMAKE_CXX_FLAGS_RELEASE "-O2 ${BASE_OPTIONS}")
		else()
			set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native ${BASE_OPTIONS}")
		endif()
	else()
		if (CMAKE_BUILD_TYPE_LOWER STREQUAL "release")
			set(CMAKE_CXX_FLAGS_RELEASE "")
			if(${NGT_MARCH_NATIVE_DISABLED})
				message(STATUS "Compile option -march=native is disabled.")
				add_compile_options(-O2 -DNDEBUG)
			elseif(${NGT_AVX2})
				add_compile_options(-Ofast -march=haswell -DNDEBUG)
			elseif(${NGT_AVX_DISABLED})
				message(STATUS "AVX will not be used to compute distances.")
				add_compile_options(-Ofast -march=x86-64 -DNDEBUG)
			else()
				add_compile_options(-Ofast -march=native -DNDEBUG)
			endif()
		endif()
		add_compile_options(-Wall)
		if(${NGT_OPENMP_DISABLED})
			message(STATUS "OpenMP is disabled.")
		else()
			if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
				if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1.0")
					message(FATAL_ERROR "Insufficient AppleClang version")
				endif()
				cmake_minimum_required(VERSION 3.16)
			endif()
			find_package(OpenMP REQUIRED)
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
		endif()
		set(CMAKE_CXX_STANDARD 11) # for std::unordered_set, std::unique_ptr
		set(CMAKE_CXX_STANDARD_REQUIRED ON)
		find_package(Threads REQUIRED)
		if(${NGT_QBG_DISABLED})
			message(STATUS "QBG is disabled.")
		else()
			find_package(BLAS REQUIRED)
			find_package(LAPACK REQUIRED)
		endif()
	endif()

	if(${NGT_BFLOAT_DISABLED})
		message(STATUS "bfloat is disabled.")
	endif()


endif(${UNIX})


if( ${UNIX} )
	option(NGT_SHARED_MEMORY_ALLOCATOR "enable shared memory" OFF)
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in ${CMAKE_CURRENT_BINARY_DIR}/defines.h)
	include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/lib" "${PROJECT_BINARY_DIR}/lib/")

	if(${NGT_QBG_DISABLED})
		file(GLOB NGT_SOURCES *.cpp)
	else()
		file(GLOB NGT_SOURCES *.cpp NGTQ/*.cpp)
	endif()
	file(GLOB HEADER_FILES *.h *.hpp)
	file(GLOB NGTQ_HEADER_FILES NGTQ/*.h NGTQ/*.hpp)

	add_library(ngtstatic STATIC ${NGT_SOURCES})
	set(LIBRARY_INCLUDES "./" "${CMAKE_BINARY_DIR}/configured_files/include")
	target_include_directories(ngtstatic PUBLIC ${LIBRARY_INCLUDES})



	target_link_libraries(ngtstatic)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
		target_link_libraries(ngtstatic OpenMP::OpenMP_CXX)
	endif()

	add_library(ngt SHARED ${NGT_SOURCES})


	add_dependencies(ngt ngtstatic)
        if(NOT DEFINED NGT_QBG_DISABLED OR (NOT ${NGT_QBG_DISABLED}))
		target_link_libraries(ngt lapack blas)
	endif()
	if(${APPLE})
		if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
			target_link_libraries(ngt OpenMP::OpenMP_CXX)
		else()
			target_link_libraries(ngt gomp)
		endif()
	else(${APPLE})
		target_link_libraries(ngt gomp rt)
	endif(${APPLE})



	install(TARGETS
		ngt
		ngtstatic
		RUNTIME DESTINATION bin
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib)
	install(FILES ${NGTQ_HEADER_FILES} DESTINATION include/NGT/NGTQ)

endif()

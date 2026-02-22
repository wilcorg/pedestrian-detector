cmake_minimum_required(VERSION 3.16)
project(koller)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

add_executable(svm
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_include_directories(svm PUBLIC ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR})

target_compile_options(svm PUBLIC
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-O3 -march=native -pipe>
)

target_link_libraries(svm ${OpenCV_LIBS})

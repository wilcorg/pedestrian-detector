cmake_minimum_required(VERSION 3.16)
project(koller1)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(koller1 src/main.cpp src/Koller1.cpp)

target_compile_options(koller1 PUBLIC
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-O3 -march=native -pipe>
)

target_link_libraries(koller1 ${OpenCV_LIBS})

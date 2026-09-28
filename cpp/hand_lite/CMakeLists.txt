cmake_minimum_required(VERSION 3.16)
project(hand_lite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HAND_LITE_WITH_NCNN "Build NcnnGraphs (find_package(ncnn): set ncnn_DIR to <ncnn>/lib/cmake/ncnn)" OFF)
option(HAND_LITE_BUILD_TESTS "Build golden_test" ON)

add_library(hand_lite src/geometry.cpp src/runtime.cpp)
target_include_directories(hand_lite PUBLIC include PRIVATE src)
if(MSVC)
    target_compile_options(hand_lite PRIVATE /W4)
else()
    target_compile_options(hand_lite PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(HAND_LITE_WITH_NCNN)
    find_package(ncnn REQUIRED)
    target_sources(hand_lite PRIVATE src/ncnn_graphs.cpp)
    target_link_libraries(hand_lite PUBLIC ncnn)
    target_compile_definitions(hand_lite PUBLIC HAND_LITE_WITH_NCNN)
endif()

if(HAND_LITE_BUILD_TESTS)
    enable_testing()
    add_executable(golden_test tests/golden_test.cpp)
    target_link_libraries(golden_test PRIVATE hand_lite)
    set(GOLDEN_DATA ${CMAKE_CURRENT_SOURCE_DIR}/tests/data)
    # Logic around the graphs, with the recorded network outputs.
    add_test(NAME golden_replay COMMAND golden_test ${GOLDEN_DATA})
    if(HAND_LITE_WITH_NCNN)
        # The same scenario through the real ncnn graphs (poses within 2e-3).
        add_test(NAME golden_ncnn COMMAND golden_test ${GOLDEN_DATA} --ncnn)
    endif()
endif()

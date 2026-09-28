cmake_minimum_required(VERSION 3.24)
project(hand_tracker_master LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The slave project provides the hand landmark models and code (hand_detector.hpp) and the UV2
# protocol (uv_sender.hpp, cli.hpp); the master Pi needs both folders side by side.
set(HAND_TRACKER_SLAVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hand_tracker_slave" CACHE PATH "hand_tracker_slave source folder")
if(NOT EXISTS "${HAND_TRACKER_SLAVE_DIR}/hand_detector.hpp")
  message(FATAL_ERROR "hand_tracker_slave not found at ${HAND_TRACKER_SLAVE_DIR}; set HAND_TRACKER_SLAVE_DIR")
endif()
option(MASTER_BUILD_APP "Build the camera app (needs OpenCV); OFF builds the model library and tests only" ON)

find_package(Threads REQUIRED)
option(USE_SYSTEM_NCNN "Use a preinstalled ncnn via ncnn_DIR" OFF)
if(USE_SYSTEM_NCNN)
  find_package(ncnn CONFIG REQUIRED)
else()
  include(FetchContent)
  set(NCNN_VULKAN OFF CACHE BOOL "" FORCE)
  set(NCNN_OPENMP ON CACHE BOOL "" FORCE)
  set(NCNN_RUNTIME_CPU ON CACHE BOOL "" FORCE)
  set(NCNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(ncnn
    URL https://github.com/Tencent/ncnn/archive/refs/tags/20260526.zip
    URL_HASH SHA256=5aedf25e22d4a9ab7785d33774affee96b55b546e85318cc351a48a365d15ced)
  FetchContent_MakeAvailable(ncnn)
endif()

# HandDirect runtime, rig features and ncnn graphs (no OpenCV).
add_library(master_core STATIC src/direct_runtime.cpp src/rig.cpp src/ncnn_graphs.cpp)
target_include_directories(master_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(master_core PUBLIC ncnn)
if(NOT MSVC)
  target_compile_options(master_core PRIVATE -Wall -Wextra)
endif()

set(MASTER_BIN "${CMAKE_BINARY_DIR}/bin")
if(MASTER_BUILD_APP)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
  add_executable(hand_tracker_master src/main.cpp)
  target_include_directories(hand_tracker_master PRIVATE ${OpenCV_INCLUDE_DIRS} "${HAND_TRACKER_SLAVE_DIR}")
  target_link_libraries(hand_tracker_master PRIVATE master_core ${OpenCV_LIBS} Threads::Threads)
  if(WIN32)
    target_link_libraries(hand_tracker_master PRIVATE ws2_32)
  endif()
  if(NOT MSVC)
    target_compile_options(hand_tracker_master PRIVATE -Wall -Wextra)
  endif()
  set_target_properties(hand_tracker_master PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${MASTER_BIN}")
  # models/: this project's HandDirect export and rig, plus the slave's landmark models.
  add_custom_target(hand_tracker_master_models
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/models" "${MASTER_BIN}/models"
    COMMAND ${CMAKE_COMMAND} -E copy "${HAND_TRACKER_SLAVE_DIR}/models/hand_detector.param"
            "${HAND_TRACKER_SLAVE_DIR}/models/hand_detector.bin" "${HAND_TRACKER_SLAVE_DIR}/models/hand_landmarks_detector.param"
            "${HAND_TRACKER_SLAVE_DIR}/models/hand_landmarks_detector.bin" "${MASTER_BIN}/models/")
  add_dependencies(hand_tracker_master hand_tracker_master_models)
endif()

enable_testing()
add_executable(runtime_test tests/runtime_test.cpp)
target_link_libraries(runtime_test PRIVATE master_core)
add_test(NAME runtime_golden COMMAND runtime_test "${CMAKE_CURRENT_SOURCE_DIR}/models/hand_direct_wrist"
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/runtime_golden.txt")
add_executable(features_test tests/features_test.cpp)
target_link_libraries(features_test PRIVATE master_core)
add_test(NAME rig_features COMMAND features_test "${CMAKE_CURRENT_SOURCE_DIR}/models/rig.json"
         "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/features_golden.txt")
add_executable(packet_test tests/packet_test.cpp)
target_include_directories(packet_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${HAND_TRACKER_SLAVE_DIR}")
if(WIN32)
  target_link_libraries(packet_test PRIVATE ws2_32)
endif()
add_test(NAME packets COMMAND packet_test)
if(MASTER_BUILD_APP)
  add_test(NAME model_smoke COMMAND hand_tracker_master --check-model)
endif()

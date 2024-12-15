cmake_minimum_required(VERSION 3.10)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable (Ensure `main.cpp` is listed here)
add_executable(face_recognition faceRecognition.cpp)

# Link OpenCV libraries
target_link_libraries(face_recognition ${OpenCV_LIBS})

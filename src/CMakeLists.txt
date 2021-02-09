project(opencv_test)

find_package(OpenCV REQUIRED)

# Set Path
set(OPENCV_LIBS_PATH ~/opencv-3.4.1/build/lib)

# Include directory
include_directories(../thirdparty/include)

# Link to library
link_directories(../thirdparty/lib)

add_executable(openimg openimg.cpp)
target_link_libraries(openimg ${OpenCV_LIBS})

add_executable(face_detection face_detection.cpp)
target_link_libraries(face_detection ${OpenCV_LIBS})

add_executable(face_detect_webcam face_detect_webcam.cpp)
target_link_libraries(face_detect_webcam ${OpenCV_LIBS})

add_executable(circle_gaussian circle_gaussian.cpp)
target_link_libraries(circle_gaussian ${OpenCV_LIBS})
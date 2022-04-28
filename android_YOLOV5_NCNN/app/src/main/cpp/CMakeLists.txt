cmake_minimum_required(VERSION 3.4.1)
project(yolov5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
if (DEFINED ANDROID_NDK_MAJOR AND ${ANDROID_NDK_MAJOR} GREATER 20)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-openmp")
endif ()

# disable rtti and exceptions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
# enable rtti and exceptions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
set_target_properties(
        ncnn PROPERTIES
        INTERFACE_COMPILE_OPTIONS "-frtti;-fexceptions"
        # ncnn.cmake 里面是关的，把它重新打开防止跟opencv2冲突，如果是重新编译ncnn的请自己尝试要开还是关
)

# 搜索当前目录下的所有.cpp文件
aux_source_directory(. SRC_LIST)
aux_source_directory(./ocr OCR_SRC_LIST)
add_library(
        yolov5
        SHARED
        ${SRC_LIST}
        ${OCR_SRC_LIST}
)

include_directories(
        ocr
        # ./ncnnvulkan  # 使用这种写法需要把代码引入头文件全部改为 "net.h" 而不是 "ncnn/net.h" 形式
        ${CMAKE_SOURCE_DIR}/ncnnvulkan/${ANDROID_ABI}/include/  # 懒的改代码了需要用这种方式引入 "ncnn/net.h"
        ${CMAKE_SOURCE_DIR}/opencv/include/
)

add_library(libopencv_java4 STATIC IMPORTED)
set_target_properties(
        libopencv_java4
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/opencv/${ANDROID_ABI}/libopencv_java4.so
)

target_link_libraries(
        yolov5
        libopencv_java4
        ncnn
)

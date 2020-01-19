cmake_minimum_required(VERSION 2.8)

project( Imagic )

find_package( OpenCV REQUIRED )

add_executable( video src/video.cpp )

target_link_libraries( video ${OpenCV_LIBS} )
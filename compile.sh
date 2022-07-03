#!/bin/sh
echo $file Compiling Shaders
for file in /Users/alexander/Desktop/Apps/MazeDayTwo/MazeDayTwo/shaders_src/*
do
    if [ "${file: -4}" != ".spv" ]
    then
        echo $file Compiling
        filename=${file##*/}
        # echo ${filename%.*}
        /Applications/Development/VulkanSDK/macOS/bin/glslc $file -o /Users/alexander/Desktop/Apps/MazeDayTwo/MazeDayTwo/shaders/$filename.spv
    fi
done


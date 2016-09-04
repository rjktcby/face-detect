#!/bin/bash

current_dir=$(pwd)
script_dir=$(dirname "$0")

cd $script_dir
mkdir -p build
cd build

command="clang++-3.5 -ObjC++ --std=c++11 ../src/*.mm ../tests/*.mm -I../src -I../tests\
`gnustep-config --objc-flags` -lgnustep-base -lobjc -ldlib -llapack `pkg-config opencv --libs` -o face-detect"
echo $command
exec $command

cd $current_dir

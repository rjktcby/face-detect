#!/bin/bash

current_dir=$(pwd)
script_dir=$(dirname "$0")

cd $script_dir
mkdir -p build
cd build

echo "clang++-3.5 -ObjC++ ../src/*.mm -I../src `gnustep-config --objc-flags` -lgnustep-base -lobjc `pkg-config opencv --libs` -o face-detect"
clang++-3.5 -ObjC++ ../src/*.mm ../tests/*.mm -I../src -I../tests -Wall `gnustep-config --objc-flags` -lgnustep-base -lobjc `pkg-config opencv --libs` -o face-detect

cd $current_dir

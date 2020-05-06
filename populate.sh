#!/usr/bin/env bash
# This script is called by make with two arguments: source directory and build directory (in that order)
# Because it is a pain to spell out the copy command in cmake for each file.
if [[ "$#" -ne 2 ]]
then
    echo "Illegal number of parameters"
    echo "Provide (1) source directory and (2) build directory"
    exit 1
fi
SOURCE="$1"
BUILD="$2"
for script in "$SOURCE"/{input,results}
do
    ln -fs $(readlink -f "$script") "$BUILD/$(basename $script)"
done

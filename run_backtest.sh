#!/bin/bash

# runs the backtest script given and then moves all the test output 
# to the target directory given 
# make it easier to run and package up indivdual tests

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_script> <target_directory>"
    exit 1
fi


# Assign arguments to variables
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
test_output_file=$(date +"$2.%H%M.out")
test_script=$1
target_directory=$2

# Check if target directory exists, create if not
if [ ! -f "$test_script" ]; then
   echo "Cannot locate test script: $test_script"
   exit 1 
fi

# Check if target directory exists, create if not
if [ ! -d "$target_directory" ]; then
    mkdir -p "$target_directory"
fi

# copy over test script
cp *.py $target_directory

# run test
python3 $test_script | tee $test_output_file

# Find and move all other files
find . -type f ! -name "*.py" -newermt "$timestamp" -exec mv {} "$target_directory" \;


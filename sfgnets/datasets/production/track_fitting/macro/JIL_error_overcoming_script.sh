#!/bin/bash 

# Script used to run convert_root_to_numpy.py to overcome the frequently occuring JIL error
# by restarting the conversion process from the last processed entry when encountering the error
# Usage: bash JIL_error_overcoming_script.sh [particle name]

part=${1}

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ ! -f "${SCRIPT_DIR}/prod_numpy_${part}.txt" ]
then
    echo "0" > "${SCRIPT_DIR}/prod_numpy_${part}.txt"
fi
count=0

while [ -f "${SCRIPT_DIR}/prod_numpy_${part}.txt" ] && [ $count -lt 30 ];
do 
    start=$(head -n 1 "${SCRIPT_DIR}/prod_numpy_${part}.txt")
    echo "Restarting for particle ${part} at entry ${start}..."
    rm "${SCRIPT_DIR}/prod_numpy_${part}.txt"
    python  ${SCRIPT_DIR}/convert_root_to_numpy.py --aux ${SCRIPT_DIR}/../output/pgun_output ${SCRIPT_DIR}/../prod_numpy/0_test/ -p ${part} -s ${start}

    (( count ++ ))
done
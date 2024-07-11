#!/bin/bash 
# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


# Submit the jobs to condor
# Usage: bash condor_submit_prod.sh [imin to start from] [imax] [j]

# ${1} is the starting i (subrun)
# ${2} is the ending i
# ${3} is the j tag
j=${3}


for i in $(seq $1 $2)
do
    rm -f ${SCRIPT_DIR}/condor/${i}_${j}.submit   
    echo "Executable = /bin/bash" > ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "getenv = True" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "Arguments = ${SCRIPT_DIR}/prod_start_with_neut_to_eventana.sh ${i} ${j}" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "Output = ${SCRIPT_DIR}/condor/${i}_${j}.out" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "Error = ${SCRIPT_DIR}/condor/${i}_${j}.err" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "Log = ${SCRIPT_DIR}/condor/${i}_${j}.log" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit
    echo "Queue" >> ${SCRIPT_DIR}/condor/${i}_${j}.submit   
    #submit the job
    condor_submit ${SCRIPT_DIR}/condor/${i}_${j}.submit
done
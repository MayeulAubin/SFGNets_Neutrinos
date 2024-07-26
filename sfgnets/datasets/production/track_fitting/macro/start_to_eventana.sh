#!/bin/bash 


# Script that controls the production of the dataset. 
# It runs the whole pipeline of ND280 Software by calling successively the individual packages through ND280_execute.sh (and GEANT4_execute).
# Usage: bash start_to_eventana.sh [file #] [div #] [Geant4 macro file]


# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Source the packages to setup ND280 Software
source $ND280_PILOT/nd280SoftwarePilot.profile
source $ND280_ROOT/eventAnalysis-feature-upgrade_highland/bin/setup.sh
source $SOFTWARE_MASTER_PATH/bin/setup.sh

# Max duration of the sub scripts
max_duration="4h"

# Set the timed_out variable to 0 (keeping track of whether or not some sub script has timed out)
timed_out=0

# Time 
start_time=`date "+%Y-%m-%d %H:%M:%S"`
SECONDS=0

# Get input parameters
i=${1}
j=${2}
macro=${3}

# Remove previous log
rm -rf ${SCRIPT_DIR}/../prod/log/log_neut_execute_${i}_${j}.txt

# Run GEANT4_execute.sh
start_exec=`date +%s`
timeout ${max_duration} bash ${SCRIPT_DIR}/GEANT4_execute.sh ${i} ${j} ${macro} &> ${SCRIPT_DIR}/../prod/geant4_output/log_${i}_${j}.txt || timed_out=$?
end_exec=`date +%s`
runt=$((end_exec-start_exec))

if [ "$timed_out" -eq "0" ]
then
    echo "Executed geant4 ("${runt}" s)"
    echo "geant4, "${i}", "${j}", "${runt} >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
    
else
    echo "Failed to execute geant4 after ${runt} s with exit code ${timed_out} (worry if different from 124)"
    echo "geant4, ${i}, ${j}, -1" >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
    exit 1
fi

# Run the rest of the jobs
for job_type in detres eventcalib eventrecon eventana
do
    
    start_exec=`date +%s`
    timeout ${max_duration} bash ${SCRIPT_DIR}/ND280_execute.sh ${job_type} ${i} ${j} &>> ${SCRIPT_DIR}/../prod/log/log_neut_execute_${i}_${j}.txt || timed_out=$?
    end_exec=`date +%s`
    runt=$((end_exec-start_exec))
    if [ "$timed_out" -eq "0" ]
    then  
        echo "Executed "${job_type}" ("${runt}" s)"
        echo ${job_type}", "${i}", "${j}", "${runt} &>> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
    else
        echo "Failed to execute "${job_type}" after ${runt} s with exit code ${timed_out} (worry if different from 124)"
        echo "${job_type}, ${i}, ${j}, -1" >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
        exit 2
    fi
done 


echo "Deleting the intermediary root files..."
for job_type in detres eventcalib eventrecon geant4
do
    rm -f ${SCRIPT_DIR}/../prod/${job_type}_output/${i}_${j}.root
done

#Time 
end_time=`date "+%Y-%m-%d %H:%M:%S"`
run_time=$SECONDS
echo ""
echo ""
echo "==== start_to_eventana.sh "${i}" "${j}" ===="
echo "======================================="
echo "Summary : "
echo " - Start time : ${start_time}"
echo " - End   time : ${end_time}"
echo " - The Total  : $run_time sec"
echo "======================================="
echo ""
echo "start_to_eventana, "${i}", "${j}", "${run_time} >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
#!/bin/bash 

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Script that controls the production of the dataset. 
# It runs the whole pipeline of ND280 Software by calling successively the individual packages through neut_execute.sh (and rooTracker.sh).
# Usage: bash start_to_eventana.sh [file #] [div #] [Geant4 macro file]

# Source the packages to setup ND280 Software
source $ND280_PILOT/nd280SoftwarePilot.profile
source $ND280_ROOT/eventAnalysis-feature-upgrade_highland/bin/setup.sh
source $SOFTWARE_MASTER_PATH/bin/setup.sh

# Time 
start_time=`date "+%Y-%m-%d %H:%M:%S"`
SECONDS=0

# Get input parameters
i=${1}
j=${2}

# Remove previous logs
rm -rf ${SCRIPT_DIR}/../prod/log/log_rooTracker_${i}_${j}.txt
rm -rf ${SCRIPT_DIR}/../prod/log/log_neut_execute_${i}_${j}.txt

# Run rooTracker.sh (selectEventSim and Geant4Sim)
start_exec=`date +%s`
bash ${SCRIPT_DIR}/rooTracker.sh ${i} ${j} &> ${SCRIPT_DIR}/../prod/log/log_rooTracker_${i}_${j}.txt
end_exec=`date +%s`
runt=$((end_exec-start_exec))
echo "Executed geant4&select ("${runt}" s)"
echo "geant4&select, "${i}", "${j}", "${runt} >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv

# Extract the number of events processed and selected by selectEventSim
number_of_events=$( cat ${SCRIPT_DIR}/../prod/log/log_rooTracker_${i}_${j}.txt | grep "events processed" )
number_of_events=${number_of_events//" events processed"/}
number_of_events=${number_of_events//" events selected."/}
echo "${i}, ${j}, ${number_of_events}" >> ${SCRIPT_DIR}/../prod/log/number_of_events_selected.csv


# Run the rest of the jobs
for job_type in detres eventcalib eventrecon eventana
do
    start_exec=`date +%s`
    bash ${SCRIPT_DIR}/neut_execute.sh ${job_type} ${i} ${j} &> ${SCRIPT_DIR}/../prod/log/log_neut_execute_${i}_${j}.txt
    end_exec=`date +%s`
    runt=$((end_exec-start_exec))
    echo "Executed "${job_type}" ("${runt}" s)"
    echo ${job_type}", "${i}", "${j}", "${runt} &>> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
done 


echo "Deleting the intermediary root files..."
for job_type in detres eventcalib eventrecon geant4 select_output
do
    rm -f ${SCRIPT_DIR}/../prod/${job_type}_output/${i}_${j}.root
done

#Time 
end_time=`date "+%Y-%m-%d %H:%M:%S"`
run_time=$SECONDS
echo ""
echo ""
echo "==== neut_to_eventana.sh "${i}" "${j}" ===="
echo "======================================="
echo "Summary : "
echo " - Start time : ${start_time}"
echo " - End   time : ${end_time}"
echo " - The Total  : $run_time sec"
echo "======================================="
echo ""
echo "neut_to_eventana, "${i}", "${j}", "${run_time} >> ${SCRIPT_DIR}/../prod/log/execution_durations.csv
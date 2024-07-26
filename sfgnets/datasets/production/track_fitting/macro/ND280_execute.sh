#!/bin/bash 


# Script used to control the different packages of ND280 Software (except Geant4) for dataset production for SFG
# Usage: bash ND280_execute.sh [job type] [file #] [div #]

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Time 
start_time=`date "+%Y-%m-%d %H:%M:%S"`
SECONDS=0

job_type=${1}
ifile=${2}
jdiv=${3}


MINI_PRO=${SCRIPT_DIR}/../prod
G4MC_PATH=${MINI_PRO}/geant4_output
ELMC_PATH=${MINI_PRO}/detres_output
CALI_PATH=${MINI_PRO}/eventcalib_output
RECO_PATH=${MINI_PRO}/eventrecon_output
ANAL_PATH=${MINI_PRO}/eventana_output
SELE_PATH=${MINI_PRO}/select_output

# output file
g4mc=${G4MC_PATH}/${ifile}_${jdiv}.root
elmc=${ELMC_PATH}/${ifile}_${jdiv}.root
cali=${CALI_PATH}/${ifile}_${jdiv}.root
reco=${RECO_PATH}/${ifile}_${jdiv}.root
anal=${ANAL_PATH}/${ifile}_${jdiv}.root
sele=${SELE_PATH}/${ifile}_${jdiv}.root
 

# log file
logelmc=${ELMC_PATH}/log_${ifile}_${jdiv}.txt
logcali=${CALI_PATH}/log_${ifile}_${jdiv}.txt
logreco=${RECO_PATH}/log_${ifile}_${jdiv}.txt
loganal=${ANAL_PATH}/log_${ifile}_${jdiv}.txt
logsele=${SELE_PATH}/log_${ifile}_${jdiv}.txt


if [ "${job_type}" = "detres" ]; then
    rm -rf ${elmc}; rm -rf ${logelmc}
    disable="-O disable-fgd -O disable-ecal -O disable-smrd -O disable-hat -O disable-tpc -O disable-tracker"  # to do only SFG
    DETRESPONSESIM.exe ${disable} -o ${elmc} ${g4mc} &> ${logelmc}
elif [ "${job_type}" = "selectEventSim" ]; then
    rm -rf ${sele}; rm -rf ${logsele}
    RCHERRYPICK.exe -l "SFG" -o ${sele} ${4} &> ${logsele} 
elif [ "${job_type}" = "eventcalib" ]; then
    rm -rf ${cali}; rm -rf ${logcali}
    RunEventCalib.exe -o ${cali} ${elmc} &> ${logcali} 
elif [ "${job_type}" = "eventrecon" ]; then
    rm -rf ${reco}; rm -rf ${logreco}
    RunEventRecon.exe -O par_override=${SCRIPT_DIR}/eventrecon_parameters.dat  -o ${reco} ${cali} &> ${logreco}
elif [ "${job_type}" = "eventana" ]; then
    rm -rf ${anal}; rm -rf ${loganal}
    RunEventAnalysis.exe -o ${anal} ${reco} &> ${loganal} 
    number_of_events=$( cat ${loganal} | grep "Total Events Read:" | cut -c 20-)
    echo ${ifile}", "${jdiv}", "${number_of_events} >> ${SCRIPT_DIR}/number_of_events.csv
else 
    echo "ERROR: unrecognised jobtype: "${job_type}
fi


#Time 
end_time=`date "+%Y-%m-%d %H:%M:%S"`
run_time=$SECONDS
echo " "
echo "== ND280_execute.sh ${job_type} ${ifile} ${jdiv} =="
echo "======================================="
echo "Summary : "
echo " - Start time : ${start_time}"
echo " - End   time : ${end_time}"
echo " - The Total  : $run_time sec"
echo "======================================="
echo " "


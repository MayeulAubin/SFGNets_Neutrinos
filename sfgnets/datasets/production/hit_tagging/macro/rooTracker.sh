#!/bin/sh
# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Script to run the CHERRYPICKER selection and the Geant4 simulation
# Usage: bash rooTracker.sh [file #] [div #] 

# Time 
start_time=`date "+%Y-%m-%d %H:%M:%S"`
SECONDS=0

i=${1}
j=${2}

input_dir=${SCRIPT_DIR}/../vector

# Run selectEventSim (select events only in SFG)
input=${input_dir}/nu.13a_nom_ND6_250ka_flukain.all_${i}.root
output=${SCRIPT_DIR}/../prod/select_output/${i}_${j}.root
CHERRYPICK.exe  -g ${SCRIPT_DIR}/selectEvent.geo.root -l "SFG" -o ${output} ${input}

## FHC 
input=${SCRIPT_DIR}/../prod/select_output/${i}_${j}.root
count=3000 # FHC nd6

## RHC 
# input=${input_dir}/nu.13a_nom_ND6_m250ka_flukain.all_${i}.root
# count=2000 # RHC nd6

# first=`expr ${j} \* ${count}`

# Instead of job file, use macro file
macro=${SCRIPT_DIR}/${i}_${j}.mac

echo "/t2k/control baseline-2022 1.0"                      >> ${macro}
echo "/t2k/update"                                         >> ${macro}
echo "/generator/kinematics/set rooTracker"                >> ${macro}
echo "/generator/kinematics/rooTracker/input ${input}"     >> ${macro}
echo "/generator/kinematics/rooTracker/tree nRooTracker"   >> ${macro}
echo "/generator/kinematics/rooTracker/generator NEUT"     >> ${macro}
echo "/generator/kinematics/rooTracker/order consecutive"  >> ${macro}
# echo "/generator/kinematics/rooTracker/first ${first}"     >> ${macro}
echo "/db/set/trajectoryPointCriterion -1 -1 -1 MeV"       >> ${macro}
echo "/run/beamOn ${count}"                                >> ${macro}



# Output directory
output=${SCRIPT_DIR}/../prod/geant4_output/${i}_${j}
execute=ND280GEANT4SIM.exe


${execute} -o ${output} ${macro}

rm -rf ${macro}



#Time 
end_time=`date "+%Y-%m-%d %H:%M:%S"`
run_time=$SECONDS
echo ""
echo ""
echo "========= rooTracker.sh "${i}" "${j}" ========="
echo "======================================="
echo "Summary : "
echo " - Start time : ${start_time}"
echo " - End   time : ${end_time}"
echo " - The Total  : $run_time sec"
echo "======================================="
echo ""
#!/bin/bash 


# Script to start the production from a list of macro files placed in 'geant4_macros' folder, concatenated to end.g4mac end macro.
# It calls start_to_eventana.sh for each macro file
# Usage: bash pgun_prodall.sh [file #] [# of events to generate]

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

i=${1}
N=${2}

cd ${SCRIPT_DIR}/geant4_macros/
# echo $(pwd)
for file in *.g4mac
do
    particle=${file%%.g4mac}
    echo ""
    echo "<><><><><><><><><><><><>"
    echo "Starting ${particle}..."
    echo "<><><><><><><><><><><><>"
    echo ""
    cat $file ../end.g4mac > "tmp_${i}_${file}"
    echo "/run/beamOn ${N}" >> "tmp_${i}_${file}"
    bash ${SCRIPT_DIR}/start_to_eventana.sh ${i} ${particle} "tmp_${i}_${file}"
    rm "tmp_${i}_${file}"
done



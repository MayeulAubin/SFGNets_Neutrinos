#!/bin/bash 


# Script to run the Geant4 Simulation in the ND280 Software for dataset production.
# Usage: bash GEANT4_execute.sh [file #] [div #] [Geant4 macro file]


# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Get input parameters
i=${1}
j=${2}
macro=${3}

# Removes preexisting target file
rm -rf ${SCRIPT_DIR}/../prod/geant4_output/${i}_${j}.root

# Executes Geant 4 with macro named ""
ND280GEANT4SIM.exe -s -o ${SCRIPT_DIR}/../prod/geant4_output/${i}_${j} ${macro}
#!/bin/bash 
# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Calls the neut_to_eventana.sh script inside the Singularity container, to be used in a condor submission scheme
# Usage: bash prod_start_singularity.sh [args of start_to_eventana.sh]

singularity exec --bind /mnt/raid/users/maubin/ND280mnt/:/mnt ~/ND280_custom_installation/container bash /mnt/neut_files/macro/neut_to_eventana.sh ${1} ${2}
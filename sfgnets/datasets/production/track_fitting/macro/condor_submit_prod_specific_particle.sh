#!/bin/bash 
# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Submit the jobs to condor, use the temporary geant4 macros
# Usage: bash condor_submit_prod.sh [imin to start from] [imax] [particle]

# ${1} is the starting i (subrun)
# ${2} is the ending i
# ${3} is the particle (e+, e-, gamma, ...)

# Number of events per file:
N=100

# Number of macro files:
K=0

cd ${SCRIPT_DIR}/geant4_macros/
for file in *.g4mac
do
    # Remove any stray temporary file in the geant4_macros folder
    if [ "${file::3}" != "tmp" ]
    then
        # cat $file ../end.g4mac > "../condor/tmp_${file}"
        # echo "/run/beamOn ${N}" >> "../condor/tmp_${file}"
        file_array[$K]=$file
        (( K ++ ))
    fi
done
cd ${SCRIPT_DIR}

for i in $(seq $1 $2)
do   
    if [ "${file_array[$((i % K))]}" == "${3}.g4mac" ]
    then
        file=${3}".g4mac"
        particle=${3}
        echo "Submitting i: ${i} particle: ${particle}" 
        rm -f ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Executable = /bin/bash" > ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "getenv = True" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Arguments = ${SCRIPT_DIR}/prod_start_singularity.sh ${i} ${particle} /mnt/pgun_files/macro/condor/tmp_${file}" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Output = ${SCRIPT_DIR}/condor/${i}_${particle}.out" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Error = ${SCRIPT_DIR}/condor/${i}_${particle}.err" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Log = ${SCRIPT_DIR}/condor/${i}_${particle}.log" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        echo "Queue" >> ${SCRIPT_DIR}/condor/${i}_${particle}.submit   
        #submit the job
        condor_submit -batch-name Pgun_prod_${particle} ${SCRIPT_DIR}/condor/${i}_${particle}.submit
        # bash ${SCRIPT_DIR}/prod_start_singularity.sh ${i} ${particle} /mnt/pgun_files/macro/condor/tmp_${file}
    fi
done
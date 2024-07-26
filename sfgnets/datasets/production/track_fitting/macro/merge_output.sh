#!/bin/bash 

## Merge root files of the production into a single root file
## Applied here for particle gun production with various particle types
## Usage: bash merge_output.sh [-c] -a        OR      bash merge_output.sh [-c] imin imax

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

merge_all=false # by default merge between imin and imax
clean_copy=false # by default no clean copy (pruning SFG tree)

# particles=("e+" "e-" "gamma" "mu+" "mu-" "n" "p" "pi+" "pi-")
particles=("n")
K=${#particles[@]}

while getopts "ac" flag;do
    case "$flag" in
        a) 
        # all tag, merge all available files
        merge_all=true
        ;;
        c) 
        # clean copy tag, prune the SFG tree (but has errors in copying Hit data)
        clean_copy=true
        ;;
    esac
done

if [ "$merge_all" = false ] ; then

    ##### OPTION 1: MERGE FILES FROM INDEX $1 to $2

    imin=${@:$OPTIND:1} # is the starting i (subrun), imin
    imax=${@:$OPTIND+1:1} # is the ending i, imax
    # j=${@:$OPTIND+2:1} # is the j tag

    FILES_TO_MERGE=("${particles[@]/*/''}")
    output_temp=("${particles[@]/*/''}")
    output=("${particles[@]/*/''}")

    for i in $(seq $imin $imax)
    do
        FILES_TO_MERGE[$((i % K))]="${FILES_TO_MERGE[$((i % K))]} ${SCRIPT_DIR}/../prod/eventana_output/${i}_${particles[$((i % K))]}.root"
    done

    for l in ${!particles[@]}
    do
        echo ""
        echo "Merging ${particles[$l]}..."
        echo ""
        # output file
        output_temp[$l]=${SCRIPT_DIR}/../output/pgun_output_${particles[$l]}_test_temp.root
        output[$l]=${SCRIPT_DIR}/../output/pgun_output_${particles[$l]}_test.root

        # Merge the root files
        hadd -f -n 50 -k ${output_temp[$l]}${FILES_TO_MERGE[$l]}
    done

else
    ##### OPTION 2: MERGE ALL AVAILABLE FILES

    # j=${@:$OPTIND:1}

    FILES_TO_MERGE=("${particles[@]/*/''}")
    output_temp=("${particles[@]/*/''}")
    output=("${particles[@]/*/''}")

    cd ${SCRIPT_DIR}/../prod/eventana_output

    for l in ${!particles[@]}
    do
        echo ""
        echo "Merging ${particles[$l]}..."
        echo ""

        for file in $(ls | grep _${particles[$l]}.root)
        do
            FILES_TO_MERGE[$l]="${FILES_TO_MERGE[$l]} ${SCRIPT_DIR}/../prod/eventana_output/${file}"
        done
        # output file
        output_temp[$l]=${SCRIPT_DIR}/../output/pgun_output_${particles[$l]}_temp.root
        output[$l]=${SCRIPT_DIR}/../output/pgun_output_${particles[$l]}.root

        hadd -f -n 50 -k ${output_temp[$l]}${FILES_TO_MERGE[$l]}
    done
    
fi


if [ "$clean_copy" = true ] ; then
    for l in ${!particles[@]}
    do
        echo ""
        echo "Clean copy ${particles[$l]}..."
        echo ""
        # Copy the file to update the size of the file and remove the temporary file
        rm -rf ${output[$l]}
        echo "Extract the AlgoResults.Hits,AlgoResults.Particles,Nodes.Position,Nodes.Direction,Particles.Hits,Particles.Nodes,Hits*,NHits leaves of the ReconDir/SFG and copy them to a new file..."
        rootmkdir -p ${output[$l]}:ReconDir
        rootslimtree -e "*" -i "AlgoResults.Hits,AlgoResults.Particles,Nodes.Position,Nodes.Direction,Particles.Hits,Particles.Nodes,Hits*,NHits" ${output_temp[$l]}:ReconDir/SFG ${output[$l]}:ReconDir
        echo "Extract the TruthDir to the new file..."
        rootcp -r ${output_temp[$l]}:TruthDir ${output[$l]} # Creates the output file and copy only the TruthDir
    done

else
    for l in ${!particles[@]}
    do
        echo ""
        echo "Pruning ${particles[$l]}..."
        echo ""
        # Clean in the output file the unecessary directories and trees
        echo "Deleting the unecessary directories and trees"
        for dir in Config HeaderDir LowLevelDir
        do
            echo -e "\t Deleting ${dir}"
            rootrm -r ${output_temp[$l]}:${dir}
        done

        for subdir in TREx FGDOnly Global P0D P0DECal SMRD Tracker TrackerECal
        do
            echo -e "\t Deleting ReconDir/${subdir}"
            rootrm -r ${output_temp[$l]}:ReconDir/${subdir}
        done

        echo "Rebuilding the file..."
        rootcp -r --recreate --replace ${output_temp[$l]} ${output[$l]}
    done
fi

echo "Delete temporary file..."
for l in ${!particles[@]}
do
    rm -rf ${output_temp[$l]}
done

echo "Done!"
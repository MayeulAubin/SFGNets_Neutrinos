#!/bin/bash 

# local path:
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


## Merge root files of the production into a single root file, and keep only the SFG information
## Usage: bash merge_output.sh [-c] -a        OR      bash merge_output.sh [-c] imin imax

merge_all=false # by default merge between imin and imax
clean_copy=false # by default no clean copy (pruning SFG tree)

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
    j=${@:$OPTIND+2:1} # is the j tag

    FILES_TO_MERGE=""

    for i in $(seq $imin $imax)
    do
        FILES_TO_MERGE="${FILES_TO_MERGE} ${SCRIPT_DIR}/../prod/eventana_output/${i}_${j}.root"
    done

    # output file
    output_temp=${SCRIPT_DIR}/../output/neutSFG_output_${j}_test_temp.root
    output=${SCRIPT_DIR}/../output/neutSFG_output_${j}_test.root

    # Merge the root files
    hadd -f -n 50 -k ${output_temp}${FILES_TO_MERGE}

else
    ##### OPTION 2: MERGE ALL AVAILABLE FILES

    j=${@:$OPTIND:1}

    FILES_TO_MERGE=""

    cd ${SCRIPT_DIR}/../prod/eventana_output

    for file in $(ls | grep _${j}.root)
    do
        FILES_TO_MERGE="${FILES_TO_MERGE} ${SCRIPT_DIR}/../prod/eventana_output/${file}"
    done

    # output file
    output_temp=${SCRIPT_DIR}/../output/neutSFG_output_${j}_temp.root
    output=${SCRIPT_DIR}/../output/neutSFG_output_${j}.root

    hadd -f -n 50 -k ${output_temp}${FILES_TO_MERGE}
fi


if [ "$clean_copy" = true ] ; then

    # Copy the file to update the size of the file and remove the temporary file
    rm -rf ${output}
    echo "Extract the AlgoResults.Hits,AlgoResults.Vertices,Hits*,NHits,Vertices.Position leaves of the ReconDir/SFG and copy them to a new file..."
    rootmkdir -p ${output}:ReconDir
    rootslimtree -e "*" -i "AlgoResults.Hits,AlgoResults.Vertices,Hits*,NHits,Vertices.Position" ${output_temp}:ReconDir/SFG ${output}:ReconDir
    echo "Extract the TruthDir to the new file..."
    rootcp -r ${output_temp}:TruthDir ${output} # Creates the output file and copy only the TruthDir

else

    # Clean in the output file the unecessary directories and trees
    echo "Deleting the unecessary directories and trees"
    for dir in Config HeaderDir LowLevelDir
    do
        echo -e "\t Deleting ${dir}"
        rootrm -r ${output_temp}:${dir}
    done

    for subdir in TREx FGDOnly Global P0D P0DECal SMRD Tracker TrackerECal
    do
        echo -e "\t Deleting ReconDir/${subdir}"
        rootrm -r ${output_temp}:ReconDir/${subdir}
    done

    echo "Rebuilding the file..."
    rootcp -r --recreate --replace ${output_temp} ${output}
fi

echo "Delete temporary file..."
rm -rf ${output_temp}

echo "Done!"
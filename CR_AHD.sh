#!/bin/bash

# send an email upon start and end of the job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=steffen.elting@univie.ac.at
#SBATCH --partition=apollo_nonreserved
#SBATCH --job-name=cr_ahd
# --output=steffen_test.out

# USAGE() { echo -e "Usage: bash $0 [-di <depot radius>] [-nc <num_carriers>] [] [] \n" 1>&2; exit 1; }

# if (($# == 0))
# then
#     USAGE
# fi

while getopts d:c:n:v:o:r:t:f: opt
do
        case "${opt}" in
                d) distance+=("$OPTARG")
                ;;
                c) num_carriers+=("$OPTARG")
                ;;
                n) num_requests+=("$OPTARG")
                ;;
                v) carrier_max_num_tours+=("$OPTARG")
                ;;
                o) service_area_overlap+=("$OPTARG")
                ;;
                r) run+=("$OPTARG")
                ;;
                t) threads=${OPTARG}
                ;;
                f) fail=${OPTARG}
                ;;
                \? ) echo "Invalid option: -$OPTARG exiting" >&2
                exit
                ;;
        esac
done

# can pass a command line argument on to the python script
options=(  )
if [ ! -z "$distance" ];
then
        options+=( --distance "${distance[@]}");
fi

if [ ! -z "$num_carriers" ];
then
        options+=( --num_carriers "${num_carriers[@]}");
fi

if [ ! -z "$num_requests" ];
then
        options+=( --num_requests "${num_requests[@]}");
fi

if [ ! -z "$carrier_max_num_tours" ];
then
        options+=( --carrier_max_num_tours "${carrier_max_num_tours[@]}");
fi

if [ ! -z "$service_area_overlap" ];
then
        options+=( --service_area_overlap "${service_area_overlap[@]}");
fi

if [ ! -z "$run" ];
then
        options+=( --run "${run[@]}");
fi

if [ ! -z "$threads" ];
then
        options+=( --threads $threads);
fi

if [ ! -z "$fail" ];
then
        options+=( --fail $fail);
fi



# conda --version
# conda env list
source ~/miniconda3/etc/profile.d/conda.sh
conda activate CR_AHD
# conda env list
# can pass a command line argument on to the python script
python Python/src/cr_ahd/main.py "${options[@]}";

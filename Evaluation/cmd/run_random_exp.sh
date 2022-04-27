MODEL=$1
VERSION=$2
DIR_SPLITS=$3
N_JOBS=$4
OPTION=$5

cmd="python scripts/random_exp.py --task ld2018 --model ${MODEL} --version ${VERSION} --index data/splits.all/ld2018_inbox_r0.1_t20.pkl -v -j ${N_JOBS} ${OPTION}"
echo $cmd
eval $cmd

for task in coloc cotim
do
    cmd="python scripts/random_exp.py --task ${task} --model ${MODEL} --version ${VERSION} --index data/splits.all/${task}_r0.2_t20.pkl -v -j ${N_JOBS} ${OPTION}"
    echo $cmd
    eval $cmd
done

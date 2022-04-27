MODEL=$1
VERSION=$2
TASK=$3
OPTION=$4

AGG="cls"
if [ ${MODEL} == "roberta-base" ]; then
    AGG="mean"
fi

for TASK in ld2018 coloc cotim
do
    cmd="python scripts/run_permtest.py evaluation/simple_enc.${MODEL}.${VERSION}/${TASK}_lr_clf.csv --base evaluation/baselines.${MODEL}_${AGG}/${TASK}_lr_clf.csv -v | tee evaluation/permtest/5k_${TASK}_${MODEL}_lite-vs-vanilla.txt"
    echo $cmd
    eval $cmd

    cmd="python scripts/run_permtest.py evaluation/simple_enc.${MODEL}.${VERSION}/${TASK}_lr_clf.cs
 --base evaluation/baselines.pt.v2.${MODEL}.v1_${AGG}/${TASK}_lr_clf.csv -v | tee evaluation/permtest/5k_${TASK}_${MODEL}_lite-vs-da.txt"
    echo $cmd
    eval $cmd

    cmd="python scripts/run_permtest.py evaluation/baselines.${MODEL}_${AGG}/${TASK}_lr_clf.csv --base evaluation/baselines.${MODEL}_${AGG}/${TASK}_lr_clf.csv -v | tee evaluation/permtest/5k_${TASK}_${MODEL}_da-vs-vanilla.txt"
    echo $cmd
    eval $cmd
done

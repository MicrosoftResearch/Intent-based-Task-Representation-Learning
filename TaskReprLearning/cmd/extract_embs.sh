# Usage (under ../TaskReprLearning): bash cmd/extract_embs.sh <path to model directory> <GPU ID (-1 to use CPU)>

DIR_MODEL=$1
CUDA=$2
echo "MODEL: ${DIR_MODEL}"

echo "Toy example"
python src/extract_intent_embs.py data/buy_groceries_vs_call_persons/v1.tsv --data-type tasks -m ${DIR_MODEL} -o ${DIR_MODEL}/buy_groceries_vs_call_persons-v1_embs.txt --cuda ${CUDA}
paste <(cut -f 3 data/buy_groceries_vs_call_persons/v1.tsv) <(sed -e '1d' ${DIR_MODEL}/buy_groceries_vs_call_persons-v1_embs.txt) > ${DIR_MODEL}/buy_groceries_vs_call_persons-v1_embs.tsv

echo "LD2018"

python src/extract_intent_embs.py data/landes_dieugenio_2018/todo-dataset.json --data-type LD2018 -m ${DIR_MODEL} -o ${DIR_MODEL}/ld2018_train_embs.txt --cuda ${CUDA}

echo "AT"

python src/extract_intent_embs.py data/actionable_tasks/Train.tsv --data-type AT -m ${DIR_MODEL} -o ${DIR_MODEL}/at_train_embs.txt --cuda ${CUDA}
python src/extract_intent_embs.py data/actionable_tasks/Test.tsv --data-type AT -m ${DIR_MODEL} -o ${DIR_MODEL}/at_test_embs.txt --cuda ${CUDA}

paste <(sed -e '1d' data/actionable_tasks/Train.tsv) <(sed -e '1d' ${DIR_MODEL}/at_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/at_train_embs.tsv
paste <(sed -e '1d' data/actionable_tasks/Test.tsv) <(sed -e '1d' ${DIR_MODEL}/at_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/at_test_embs.tsv


echo "UIT"

python src/extract_intent_embs.py data/UIT/training.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_train_embs.txt --cuda ${CUDA}
python src/extract_intent_embs.py data/UIT/Test2Columns.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_test_embs.txt --cuda ${CUDA}
python src/extract_intent_embs.py data/UIT/P0.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p0_embs.txt --cuda ${CUDA}
python src/extract_intent_embs.py data/UIT/P1.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p1_embs.txt --cuda ${CUDA}
python src/extract_intent_embs.py data/UIT/P2.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p2_embs.txt --cuda ${CUDA}

paste <(cat data/UIT/training.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_train_embs.tsv
paste <(cat data/UIT/Test2Columns.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_test_embs.tsv
paste <(cat data/UIT/P0.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p0_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p0_embs.tsv
paste <(cat data/UIT/P1.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p1_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p1_embs.tsv
paste <(cat data/UIT/P2.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p2_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p2_embs.tsv

echo "UIT (no list)"

python src/extract_intent_embs.py data/UIT/training.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_nolist_train_embs.txt --cuda ${CUDA} --no-list
python src/extract_intent_embs.py data/UIT/Test2Columns.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_nolist_test_embs.txt --cuda ${CUDA} --no-list
# python src/extract_intent_embs.py data/UIT/P0.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p0_embs.txt --cuda ${CUDA}
# python src/extract_intent_embs.py data/UIT/P1.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p1_embs.txt --cuda ${CUDA}
# python src/extract_intent_embs.py data/UIT/P2.tsv --data-type UIT -m ${DIR_MODEL} -o ${DIR_MODEL}/uit_p2_embs.txt --cuda ${CUDA}

paste <(cat data/UIT/training.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_nolist_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_nolist_train_embs.tsv
paste <(cat data/UIT/Test2Columns.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_nolist_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_nolist_test_embs.tsv
# paste <(cat data/UIT/P0.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p0_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p0_embs.tsv
# paste <(cat data/UIT/P1.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p1_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p1_embs.tsv
# paste <(cat data/UIT/P2.tsv) <(sed -e '1d' ${DIR_MODEL}/uit_p2_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/uit_p2_embs.tsv

echo "CoTL"

python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocTrain.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_train_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocDev.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_dev_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocTest.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_test_embs.txt --cuda ${CUDA}

paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTrain.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_train_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocDev.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_dev_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_dev_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTest.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_test_embs.tsv

python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocTrain.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_train_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocDev.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_dev_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/LocTest.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/coloc_test_embs.txt --cuda ${CUDA}

paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTrain.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_train_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocDev.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_dev_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_dev_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTest.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/coloc_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/coloc_test_embs.tsv

python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/TimTrain.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/cotim_train_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/TimDev.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/cotim_dev_embs.txt --cuda ${CUDA}
python src/extract_intent_embs_pair.py data/CoTL/StratTaskRand/TimTest.txt.tsv --data-type cotl -m ${DIR_MODEL} -o ${DIR_MODEL}/cotim_test_embs.txt --cuda ${CUDA}

paste <(sed -e '1d' data/CoTL/StratTaskRand/TimTrain.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/cotim_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/cotim_train_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/TimDev.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/cotim_dev_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/cotim_dev_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/TimTest.txt.tsv) <(sed -e '1d' ${DIR_MODEL}/cotim_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_MODEL}/cotim_test_embs.tsv

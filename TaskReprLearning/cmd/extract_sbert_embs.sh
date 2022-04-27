MODEL_TYPE=$1
CUDA_VISIBLE_DEVICES=$2
DIR_OUT=sbert_${MODEL_TYPE}
if [ `dirname ${DIR_OUT}` != "save/baselines.pt" ]; then
    DIR_OUT=save/baselines/${DIR_OUT}
fi

mkdir -p ${DIR_OUT}

echo "MODEL: ${MODEL_TYPE}"
echo "OUT: ${DIR_OUT}"

echo "Toy example"

python baseline_encoders/sbert_models.py data/buy_groceries_vs_call_persons/v1.tsv --data-type tasks --model-type ${MODEL_TYPE} -o ${DIR_OUT}/buy_groceries_vs_call_persons-v1_embs.txt -v
paste <(cut -f 3 data/buy_groceries_vs_call_persons/v1.tsv) <(sed -e '1d' ${DIR_OUT}/buy_groceries_vs_call_persons-v1_embs.txt) > ${DIR_OUT}/buy_groceries_vs_call_persons-v1_embs.tsv

echo "LD2018"

python baseline_encoders/sbert_models.py data/landes_dieugenio_2018/todo-dataset.json --data-type LD2018 --model-type ${MODEL_TYPE} -o ${DIR_OUT}/ld2018_train_embs.txt -v


echo "AT"

python baseline_encoders/sbert_models.py data/actionable_tasks/Train.tsv --data-type actionabletasks --model-type ${MODEL_TYPE} -o ${DIR_OUT}/at_train_embs.txt -v
python baseline_encoders/sbert_models.py data/actionable_tasks/Test.tsv --data-type actionabletasks --model-type ${MODEL_TYPE} -o ${DIR_OUT}/at_test_embs.txt -v

paste <(sed -e '1d' data/actionable_tasks/Train.tsv) <(sed -e '1d' ${DIR_OUT}/at_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/at_train_embs.tsv
paste <(sed -e '1d' data/actionable_tasks/Test.tsv) <(sed -e '1d' ${DIR_OUT}/at_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/at_test_embs.tsv

echo "UIT"

python baseline_encoders/sbert_models.py data/UIT/training.tsv --data-type UIT --model-type ${MODEL_TYPE} -o ${DIR_OUT}/uit_train_embs.txt -v
python baseline_encoders/sbert_models.py data/UIT/Test2Columns.tsv --data-type UIT --model-type ${MODEL_TYPE} -o ${DIR_OUT}/uit_test_embs.txt -v

paste <(cat data/UIT/training.tsv) <(sed -e '1d' ${DIR_OUT}/uit_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/uit_train_embs.tsv
paste <(cat data/UIT/Test2Columns.tsv) <(sed -e '1d' ${DIR_OUT}/uit_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/uit_test_embs.tsv


echo "CoLoc"

python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/LocTrain.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/coloc_train_embs.txt --batchsize 1000 -v
python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/LocDev.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/coloc_dev_embs.txt --batchsize 1000 -v
python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/LocTest.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/coloc_test_embs.txt --batchsize 1000 -v

paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTrain.txt.tsv) <(sed -e '1d' ${DIR_OUT}/coloc_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/coloc_train_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocDev.txt.tsv) <(sed -e '1d' ${DIR_OUT}/coloc_dev_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/coloc_dev_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/LocTest.txt.tsv) <(sed -e '1d' ${DIR_OUT}/coloc_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/coloc_test_embs.tsv

echo "CoTim"

python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/TimTrain.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/cotim_train_embs.txt --batchsize 1000 -v
python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/TimDev.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/cotim_dev_embs.txt --batchsize 1000 -v
python baseline_encoders/sbert_models_pair.py data/CoTL/StratTaskRand/TimTest.txt.tsv --data-type CoTL --model-type ${MODEL_TYPE} -o ${DIR_OUT}/cotim_test_embs.txt --batchsize 1000 -v

paste <(sed -e '1d' data/CoTL/StratTaskRand/TimTrain.txt.tsv) <(sed -e '1d' ${DIR_OUT}/cotim_train_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/cotim_train_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/TimDev.txt.tsv) <(sed -e '1d' ${DIR_OUT}/cotim_dev_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/cotim_dev_embs.tsv
paste <(sed -e '1d' data/CoTL/StratTaskRand/TimTest.txt.tsv) <(sed -e '1d' ${DIR_OUT}/cotim_test_embs.txt | cut -d ' ' -f 2- | sed -e 's/ /\t/g') > ${DIR_OUT}/cotim_test_embs.tsv

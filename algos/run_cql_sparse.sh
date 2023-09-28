
mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

CQLWeight=5.0
Lagrange=5.0

SEED=4

GPU=0
for TASKS in 'pen-human-v1 door-human-v1 relocate-human-v1 hammer-human-v1' 'pen-expert-v1 door-expert-v1 relocate-expert-v1 hammer-expert-v1'
do
    for task in ${TASKS}
    do
        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/cql_sparse_trainer.py --seed ${SEED} --task ${task} --use-tqdm 0 --cql-weight ${CQLWeight} --lagrange-threshold ${Lagrange} --with-lagrange 1 > algos/logs_td3bc/cql_sparse_${task}_seed${SEED}_CQLQeight_${CQLWeight}_Lagrange_${Lagrange}.log &
    done
    GPU=$((GPU+1))
done
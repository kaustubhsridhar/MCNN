mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

AlgoType=bc
SEED=4

GPU=0
for TASKS in 'pen-human-v1 hammer-human-v1' 'door-human-v1 relocate-human-v1' 'pen-expert-v1 hammer-expert-v1' 'door-expert-v1 relocate-expert-v1'
do
    for task in ${TASKS}
    do
        for oversampling in 1
        do
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --oversampling ${oversampling} --algo-name ${AlgoType} --task ${task} --use-tqdm 0 > algos/logs_td3bc/${AlgoType}_${task}_oversampling${oversampling}_seed${SEED}.log &
        done  
    done 
    GPU=$((GPU+1))
done

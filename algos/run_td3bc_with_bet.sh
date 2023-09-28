mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

percent=1.0
AlgoType=bet 
SEED=2

for TASKS in 'carla-lane-v0' # 'pen-cloned-v1 door-cloned-v1 relocate-cloned-v1 hammer-cloned-v1'
do
    GPU=1
    for task in ${TASKS}
    do
        # CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer_with_bet.py --seed ${SEED} --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} --use-tqdm 0 > algos/logs_td3bc/${AlgoType}_${task}_seed${SEED}.log &

        for F in 0.1 #0.05 0.025
        do
            for Lipz in 1.0
            do 
                for lamda in 1.0
                do 
                    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer_with_bet.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 > algos/logs_td3bc/mem_${AlgoType}_${task}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}.log &
                done
            done
        done
        # GPU=$((GPU+1))
    done 
done
mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

percent=1.0
SEED=0

for AlgoType in 1nn vinn
do
    for task in hammer-human-v1 pen-human-v1 relocate-human-v1 door-human-v1 hammer-expert-v1 pen-expert-v1 relocate-expert-v1 door-expert-v1 hammer-cloned-v1 pen-cloned-v1 relocate-cloned-v1 door-cloned-v1
    do
        GPU=0
        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/nearest_neighbours.py --seed ${SEED} --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} > algos/logs_td3bc/baseline_${AlgoType}_${task}_seed${SEED}.log &
    done 

    for task in carla-town-v0 carla-lane-v0
    do
        GPU=1
        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/nearest_neighbours.py --seed ${SEED} --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} > algos/logs_td3bc/baseline_${AlgoType}_${task}_seed${SEED}.log &
    done 
done 

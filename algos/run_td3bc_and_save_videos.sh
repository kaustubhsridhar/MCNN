mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

AlgoType=bc
SEED=4

for TASKS in 'pen-expert-v1 hammer-expert-v1' 'relocate-expert-v1 door-expert-v1'
do
    GPU=0
    for task in ${TASKS}
    do
        for percent in 1.0
        do
            # CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} --use-tqdm 0 > algos/logs_td3bc/${AlgoType}_${task}_percent${percent}_seed${SEED}.log &

            for F in 0.1 #0.05 0.025
            do
                for Lipz in 1.0
                do 
                    for lamda in 1.0
                    do 
                        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 --save_videos > algos/logs_td3bc/mem_${AlgoType}_${task}_percent${percent}_frac${F}_Lipz${Lipz}_lamda${lamda}_seed${SEED}_with_saved_videos.log &
                    done
                done
            done
        done 
        GPU=$((GPU+1))
    done 
done

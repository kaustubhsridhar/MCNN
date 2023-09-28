mkdir algos/logs_td3bc
mkdir algos/exp_td3bc

AlgoType=bc
SEED=4

for TASKS in 'pen-human-v1'
do
    GPU=0
    for task in ${TASKS}
    do
        for percent in 1.0
        do
            CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --epoch 1 --chosen-percentage ${percent} --algo-name ${AlgoType} --task ${task} --use-tqdm 0 >> algos/logs_td3bc/inference_${task}.log

            for F in 0.025 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
            do
                for Lipz in 1.0
                do 
                    for lamda in 1.0
                    do 
                        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u algos/td3bc_trainer.py --seed ${SEED} --epoch 1 --chosen-percentage ${percent} --algo-name mem_${AlgoType} --task ${task} --num_memories_frac ${F} --Lipz ${Lipz} --lamda ${lamda} --use-tqdm 0 >> algos/logs_td3bc/inference_${task}.log
                    done
                done
            done
        done 
        GPU=$((GPU+1))
    done 
done

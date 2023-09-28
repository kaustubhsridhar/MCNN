
mkdir logs

GPU=0
SEED=0

for TASKS in 'pen-expert-v1 hammer-expert-v1' 'relocate-expert-v1 door-expert-v1' #'pen-human-v1 hammer-human-v1' 'relocate-human-v1 door-human-v1'
do
    for task in ${TASKS}
    do
        CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --seed ${SEED} --algo bc --env_name ${task} --device 0 --ms online --lr_decay > logs/bc_${task}.log &
        
        for F in 0.1 0.025 0.05
        do
            for lamda in 1.0 #0.1 10.0
            do
                CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --seed ${SEED} --algo mcnn_bc --env_name ${task} --device 0 --ms online --lr_decay --num_memories_frac ${F} --Lipz 1.0 --lamda ${lamda} > logs/mcnn_bc_${task}_F${F}_lamda${lamda}.log &
            done
        done
    done
    GPU=$((GPU+1))
done 

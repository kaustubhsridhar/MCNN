mkdir mems_obs/logs 

for ENV in hammer pen relocate door
do
    for TYPE in expert cloned human
    do
        mkdir mems_obs/logs/${ENV}-${TYPE}-v1
        for F in 0.025 0.05 0.1
        do
            CUDA_VISIBLE_DEVICES=1 nohup python -u mems_obs/update_data_random_mems.py --name ${ENV}-${TYPE}-v1 --num_memories_frac ${F} > mems_obs/logs/${ENV}-${TYPE}-v1/random_update_data_${F}_frac_percentbc.log &
        done 
    done
done


# for TASK in carla-lane-v0 carla-town-v0 
# do
#     mkdir mems_obs/logs/${TASK}
#     for F in 0.025 0.05 0.1
#     do
#         CUDA_VISIBLE_DEVICES=0 nohup python -u mems_obs/update_data_random_mems.py --name ${TASK} --num_memories_frac ${F} > mems_obs/logs/${TASK}/update_data_${F}_frac_percentbc.log &
#     done 
# done
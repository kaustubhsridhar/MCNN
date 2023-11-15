


mkdir ../algos/logs_td3bc

GPU=1
for S in 4
do
    CUDA_VISIBLE_DEVICES=${GPU} nohup python -u train.py --config-dir=. --config-name=kitchen_mcnn_diffusion_cnn.yaml training.seed=${S} training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > ../algos/logs_td3bc/kitchen_seed${S}.log &
done 
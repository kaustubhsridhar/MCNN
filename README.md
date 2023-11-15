# Memory-Consistent Neural Networks

[arXiv](https://arxiv.org/abs/2310.06171) | [Website](https://sites.google.com/view/mcnn-imitation) 

<p align="center">
  <img src="assets/mcnn.gif" width="320" height="240" alt="Videos of MCNN agents performing dexterous manipulation tasks">
</p>

## Setup
Create env, install pytorch, install requirements.
```bash
conda create -n MCNN_env python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

Setup mujoco210 by following the instructions from https://github.com/openai/mujoco-py#install-mujoco.
In case you run across a gcc error, please follow the trouble shooting instructions [here if you have sudo access](https://github.com/openai/mujoco-py#ubuntu-installtion-troubleshooting) or [here otherwise](https://github.com/openai/mujoco-py/issues/627#issuecomment-1383054926).

Install this package
```bash
pip install -e .
```

## Additional setup (only for CARLA)
Instructions to install CARLA can be found [here](https://github.com/Farama-Foundation/d4rl/wiki/CARLA-Setup).
Also note that you have to run the following for any CARLA experiments:

Open a new terminal session, and run the CARLA simulator:
```bash
CUDA_VISIBLE_DEVICES=0 bash CarlaUE4.sh -fps 20
```
In a second terminal window, run
```bash
./PythonAPI/util/config.py --map Town04 --delta-seconds 0.05
```
Use the Town03 map for `carla-town-v0`, and Town04 for `carla-lane-v0`.

## Quickstart 
Download the updated datasets for Adroit:
```bash
python mems_obs/download_updated_datasets.py
```

Train and evaluate MCNN + MLP:
```bash
python algos/td3bc_trainer.py --algo-name mem_bc --task pen-human-v1 --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```

Train and evaluate MCNN + Diffusion Policy:
```bash
cd diffusion_BC
python main.py --algo mcnn_bc --env_name pen-human-v1 --device 0 --ms online --lr_decay --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```
Replace `pen-human-v1` with any of the other tasks such as (hammer-human-v1, pen-human-v1, relocate-human-v1, door-human-v1, hammer-expert-v1, pen-expert-v1, relocate-expert-v1, door-expert-v1, carla-lane-v0).

## Detailed instructions for all methods
### Train / Evaluate with MLP
For MCNN + MLP with neural gas memories:
```bash
python algos/td3bc_trainer.py --algo-name mem_bc --task pen-human-v1 --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```

For MCNN + MLP with random memories:
```bash
python algos/td3bc_trainer.py --algo-name mem_bc --task pen-human-v1 --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0 --use-random-memories 1
```

For MLP-BC:
```bash
python algos/td3bc_trainer.py --algo-name bc --task pen-human-v1
```

For 1NN and VINN:
```bash
python algos/nearest_neighbours.py --algo-name 1nn --task pen-human-v1
python algos/nearest_neighbours.py --algo-name vinn --task pen-human-v1
```

For CQL with sparse reward:
```bash
python algos/cql_sparse_trainer.py --task pen-human-v1
```

### Train / Evaluate with Diffusion BC [Wang et al., ICLR 2023] in Adroit Environments
Move to the folder:
```bash
cd diffusion_BC
```

For MCNN + Diffusion:
```bash
python main.py --algo mcnn_bc --env_name pen-human-v1 --device 0 --ms online --lr_decay --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```

For Diffusion-BC:
```bash
python main.py --algo bc --env_name pen-human-v1 --device 0 --ms online --lr_decay
```

### Train / Evaluate with Diffusion Policy [Chi et al., RSS 2023] in FrankaKitchen Environments
Move to the folder:
```bash
cd diffusion_policy
```

For MCNN + Diffusion:
```bash

```

For Diffusion-BC:
```bash

```

### Train / Evaluate with Behavior Transformer (BeT)
Extra installs:
```
cd miniBET
pip install -e .
```

For MCNN + BeT:
```bash
python algos/td3bc_trainer_with_bet.py --algo-name mem_bet --task pen-human-v1 --num_memories_frac 0.1 --Lipz 1.0 --lamda 1.0
```

For BeT-BC:
```bash
python algos/td3bc_trainer_with_bet.py --algo-name bet --task pen-human-v1
```

## Detailed instructions for creating datasets
### Collect data
Download d4rl datasets, resnet models for CARLA embeddings, and franka kitchen dataset:
```bash
python data/download_d4rl_datasets.py
python data/download_nocrash_models.py
cd diffusion_policy && bash download_kitchen_data.sh && cd ../
```

Gnerate CARLA embeddings
```bash
python data/generate_carla_models.py
```

### Create Memories with Neural Gas
Create memories:
```bash
python mems_obs/create_gng_incrementally.py --name pen-human-v1 --num_memories_frac 0.1
```
Replace name with any of the other tasks and num_memories_frac with any value less than 1. In the paper, we use 0.025, 0.05, and 0.1 for num_memories_frac. (Note: simply use `--name kitchen` for the franka kitchen task)

Update (downloaded) datasets by adding memory and memory_target to every transition:
```bash
python mems_obs/update_data.py --name pen-human-v1 --num_memories_frac 0.1
```
Similar to above, replace name with any of the other tasks and num_memories_frac with any value less than 1.

### Create Random Memories
Create random subset of all observations as memories and update (downloaded) datasets by adding memory and memory_target to every transition:
```bash
python mems_obs/update_data_random_mems.py --name pen-human-v1 --num_memories_frac 0.1
```
Similar to above, replace name with any of the other tasks (use `--name kitchen` for the franka kitchen task) and num_memories_frac with any value less than 1. 

## BibTeX
If you find this codebase or our paper helpful, please consider citing us:
```
@misc{sridhar2023memoryconsistent,
      title={Memory-Consistent Neural Networks for Imitation Learning}, 
      author={Kaustubh Sridhar and Souradeep Dutta and Dinesh Jayaraman and James Weimer and Insup Lee},
      year={2023},
      eprint={2310.06171},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
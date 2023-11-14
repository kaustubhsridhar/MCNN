## Diffusion Policies for Offline RL &mdash; Official PyTorch Implementation

**Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**<br>
Zhendong Wang, Jonathan J Hunt and Mingyuan Zhou <br>
https://arxiv.org/abs/2208.06193 <br>

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.

### Running
Running experiments based our code could be quite easy, so below we use `walker2d-medium-expert-v2` dataset as an example. 

For reproducing the optimal results, we recommend running with 'online model selection' as follows. 
The best_score will be stored in the `best_score_online.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay
```

For conducting 'offline model selection', run the code below. The best_score will be stored in the `best_score_offline.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms offline --lr_decay --early_stop
```

Hyperparameters for Diffusion-QL have been hard coded in `main.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 

## Citation

If you find this open source release useful, please cite in your paper:
```
@article{wang2022diffusion,
  title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
  author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2208.06193},
  year={2022}
}
```


## RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability

####  [[Website]](https://zchuning.github.io/repo-website/) [[Paper]](https://arxiv.org/abs/2309.00082) [[Talk]](https://youtu.be/DQGVD6KaVf8)

[Chuning Zhu<sup>1</sup>](https://homes.cs.washington.edu/~zchuning/), [Max Simchowitz<sup>2</sup>](https://msimchowitz.github.io/), [Siri Gadipudi<sup>1</sup>](https://www.linkedin.com/in/siri-gadipudi-136395221/), [Abhishek Gupta<sup>1</sup>](https://homes.cs.washington.edu/~abhgupta/)<br/>

<sup>1</sup>University of Washington <sup>2</sup>MIT </br>

This is a PyTorch implementation for the RePo algorithm. RePo is a visual model-based reinforcement learning method that learns a minimally task-relevant representation, making it resilient to uncontrollable distractors in the environment. We also provide implementations of Dreamer, TIA, DBC, and DeepMDP.

## Instructions

#### Setting up repo
```
git clone https://github.com/zchuning/repo.git
```

#### Dependencies
- Install [Mujoco](https://www.roboti.us/index.html) 2.1.0
- Install dependencies
```
pip install -r requirements.txt
```


## Distracted DMC experiments
To train on DMC with natural video distractors, download the driving_car videos from Kinetics 400 dataset following these [instructions](https://github.com/Showmax/kinetics-downloader). Then, use one of the following commands to train an agent on distracted Walker Walk. To train on other distracted DMC environments,
replace `walker-walk` with `{domain}-{task}`:

```
# RePo
python experiments/train_repo.py --algo repo --env_id dmc_distracted-walker-walk --expr_name benchmark --seed 0

# Dreamer
python experiments/train_repo.py --algo dreamer --env_id dmc_distracted-walker-walk --expr_name benchmark --seed 0

# TIA
python experiments/train_repo.py --algo tia --env_id dmc_distracted-walker-walk --expr_name benchmark --seed 0

# DBC
python experiments/train_bisim.py --algo bisim --env_id dmc_distracted-walker-walk --expr_name benchmark --seed 0

# DeepMDP
python experiments/train_bisim.py --algo deepmdp --env_id dmc_distracted-walker-walk --expr_name benchmark --seed 0
```

## Maniskill experiments
First, download the background assets from this [link](https://drive.google.com/file/d/1SLh1WOmYn5qzoDUygtlQ89SS8aBSenP0/view?usp=sharing) and place the `data` folder in the root directory of the repository.

Then, use the following command to train an agent on a Maniskill environment, where `{task}` is one of `{PushCubeMatterport, LiftCubeMatterport, TurnFaucetMatterport}`:

```
python experiments/train_repo.py --algo repo --env_id maniskill-{task} --expr_name benchmark --seed 0
```


## Adaptation experiments
To run adaptation experiments, first train an agent on the source domain and save the replay buffer:
```
python experiments/train_repo.py --algo repo --env_id dmc-walker-walk --expr_name benchmark --seed 0 --save_buffer True
```
Then run the adaptation experiment on the target domain using one of the following commands:
```
# Support constraint + calibration
python experiments/adapt_repo.py --algo repo_calibrate --env_id dmc_distracted-walker-walk --expr_name adaptaion --source_dir logdir/repo/dmc-walker-walk/benchmark/0 --seed 0

# Distribution matching + calibration
python experiments/adapt_repo.py --algo repo_calibrate --env_id dmc_distracted-walker-walk --expr_name adaptaion --source_dir logdir/repo/dmc-walker-walk/benchmark/0 --seed 0 --alignment_mode distribution
```

## Bibtex
If you find this code useful, please cite:

```
@inproceedings{
zhu2023repo,
title={RePo: Resilient Model-Based Reinforcement Learning by Regularizing Posterior Predictability},
author={Chuning Zhu and Max Simchowitz and Siri Gadipudi and Abhishek Gupta},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=OIJ3VXDy6s}
}
```

# TacEx: Tactile Curiosity Drives Robot Interaction

[Project website](https://kiten.me/projects/tacex/) · [Paper](https://arxiv.org/abs/2609.40134) · [PDF](https://arxiv.org/pdf/2609.40134) · [Video](https://youtu.be/IkMvuaCo5qw)

**Klemens Iten\*, Alexander Proshkin\*, Bhavya Sukhija\*, Stelian Coros, Andreas Krause, Pieter Abbeel, and Carmelo Sferrazza** (\* denotes equal contribution)

Code accompanying our paper "Tactile Curiosity Drives Robot Interaction", which was accepted to the 10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA. Our algorithm, TacEx, uses tactile predictive uncertainty to guide exploration in robotic manipulation, from reward-free data collection to steering frozen vision-language-action policies. In our simulations, this encourages contact-rich interaction and yields experience that can support downstream manipulation learning.

This repository contains the JAX implementation for two experimental settings:

- **Exploration from scratch:** collect experience without task rewards, then use the fixed dataset for offline downstream pick-and-place learning.
- **Diffusion steering:** learn a noise policy that steers a frozen pretrained π₀ on eight LIBERO-90 tasks, using task rewards together with tactile uncertainty.

## Installation

### Clone the repository and submodules

```bash
git clone https://github.com/klemensiten/tacex.git
cd tacex

git submodule init
git config submodule.LIBERO.url https://github.com/nakamotoo/LIBERO.git
git config submodule.openpi.url https://github.com/nakamotoo/openpi.git
git config submodule.tactile_envs.url https://github.com/carlosferrazza/tactile_envs.git
git submodule update --init --recursive
```

### Create an environment

```bash
conda create -n tacex python=3.11.11
conda activate tacex
python -m pip install --upgrade pip
```

### Install the shared code and tactile simulation packages

Run these commands from the repository root:

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  -e ".[exploration,test]" \
  -e tactile_envs \
  -e tactile_envs/tactile_envs/envs/robosuite \
  -e tactile_envs/tactile_envs/envs/Gymnasium-Robotics

python -m pip check
```

### Add LIBERO/π₀ for diffusion steering

```bash
python -m pip install \
  "jax[cuda12]==0.5.0" \
  -e openpi/packages/openpi-client \
  -e openpi \
  -e LIBERO

python -m pip check
```

## Rendering and logging

For GPU rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

For software rendering, install the system OSMesa library and use `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`. For CPU JAX execution, also set `JAX_PLATFORMS=cpu`. `MUJOCO_GL=disable` is suitable for state-only checks that do not render.

## Exploration examples

Run modules from the repository root.

### State-based MaxInfo-SAC

This example needs no tactile assets or π₀ checkpoint:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=disable \
  python -m examples.exploration.examples.maxinfosac.experiment \
  --entity_name local --alg_name maxinfosac --env_name Pendulum-v1 \
  --num_neurons 32 --num_hidden_layers 2 \
  --max_steps 20 --training_start 5 --batch_size 4 \
  --replay_buffer_size 20 --eval_interval 20 --eval_episodes 1 \
  --use_tqdm 0 --wandb_log 0 --save_video 0
```

### Tactile curiosity on insertion

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa JAX_PLATFORMS=cpu \
  python -m examples.exploration.examples.tactile.insertion.exploration.experiment \
  --entity_name local --env_name insertion --alg_name tactile_drq_explorer \
  --hidden_dims 32 --feature_dim 16 --frame_stack 1 --action_repeat 1 \
  --tactile_size 32 --expl_until 1 \
  --latent_disg_weight 0 --visual_disg_weight 0 --tactile_disg_weight 1 \
  --max_steps 20 --training_start 5 --batch_size 4 \
  --replay_buffer_size 20 --eval_interval 20 --eval_episodes 1 \
  --use_tqdm 0 --wandb_log 0 --save_video 0
```

## LIBERO diffusion steering

The following example adds tactile observations and tactile-only disagreement to the noise-steering learner while keeping π₀ frozen:

```bash
export EXP="$PWD/logs/tacex"
mkdir -p "$EXP"

python -m examples.diffusion_steering.launch_train_sim \
  --algorithm pixel_maxinfosac \
  --env libero --libero_suite libero_90 --libero_task_id 58 \
  --libero_robot Panda --dsrl_action_mode noise \
  --prefix tacex_libero --wandb_project TacEx --seed 0 \
  --batch_size 256 --discount 0.999 --max_steps 500000 \
  --checkpoint_interval 20000 --eval_interval 20000 --log_interval 1000 \
  --eval_episodes 10 --multi_grad_step 20 --start_online_updates 500 \
  --resize_image 64 --query_freq 20 --action_magnitude 1.0 \
  --hidden_dims 128 --add_tactile 1 --use_touch 1 \
  --touch_gripper_type Robotiq85TactileGripper \
  --tactile_shape 32 64 3 --gripper_state_indices 0 3 \
  --ensemble_disagreement_modalities tactile --mask_expl_critic 1
```

For replay-based steering learners, `max_steps` counts gradient updates. `multi_grad_step` controls updates per stored steering transition, and `query_freq` sets the maximum action-chunk length. Logs separately record environment steps. Checkpoints are saved below `EXP` when `checkpoint_interval` is positive.

For the DSRL-SAC baseline, use `--algorithm pixel_sac --add_tactile 0 --use_touch 0` and omit tactile gripper, gripper-state-index, and disagreement options. The scripts in `examples/diffusion_steering/scripts/` also provide tactile-input ablations, PPO/DPPO comparisons, and summed-taxel controls.

## Citation

```bibtex
@article{iten2026tactile,
  title   = {Tactile Curiosity Drives Robot Interaction},
  author  = {Iten, Klemens and Proshkin, Alexander and Sukhija, Bhavya
             and Coros, Stelian and Krause, Andreas and Abbeel, Pieter
             and Sferrazza, Carmelo},
  journal = {arXiv preprint arXiv:2609.40134},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.40134}
}
```

## Acknowledgments

This implementation builds on [DSRL](https://diffusion-steering.github.io/) ([paper](https://arxiv.org/abs/2506.15799)), [MaxInfoRL](https://sukhijab.github.io/projects/maxinforl/), [openpi](https://github.com/Physical-Intelligence/openpi), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [tactile_envs](https://github.com/carlosferrazza/tactile_envs), and [jaxrl2](https://github.com/ikostrikov/jaxrl2). Please also cite the relevant upstream work when using these components.

This research received support from NCCR Automation, a National Centre of Competence in Research funded by the Swiss National Science Foundation (grant 51NF40_225155), from the ONR Multidisciplinary University Research Initiative (MURI) (award N00014-22-1-2773), and from ELSA (European Lighthouse on Secure and Safe AI), funded by the European Union under grant agreement 101070617.

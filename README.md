# TacEx: Tactile Curiosity Drives Robot Interaction

[Project website](https://kiten.me/projects/tacex/) · [Paper](https://arxiv.org/abs/2609.40134) · [PDF](https://arxiv.org/pdf/2609.40134) · [Video](https://youtu.be/IkMvuaCo5qw)

**Klemens Iten\*, Alexander Proshkin\*, Bhavya Sukhija\*, Stelian Coros, Andreas Krause, Pieter Abbeel, and Carmelo Sferrazza**

Code accompanying our paper "Tactile Curiosity Drives Robot Interaction", which was accepted to the 10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA. Our algorithm, TacEx, uses tactile predictive uncertainty to guide exploration in robotic manipulation, from reward-free data collection to steering frozen vision-language-action policies. In our simulations, this encourages contact-rich interaction and yields experience that can support downstream manipulation learning.

This repository contains the JAX implementation for the paper's **diffusion-steering experiments on LIBERO**. Building on DSRL, a reinforcement learning policy selects the diffusion noise supplied to a frozen pretrained π₀ policy. Tactile observations inform the steering policy and critics while the VLA backbone remains fixed. The project website describes the broader TacEx study, including reward-free exploration and offline learning.


## Installation

### 1. Clone the repository and submodules

```bash
git clone https://github.com/klemensiten/dsrl_pi0.git
cd dsrl_pi0

git submodule init
# Submodules
git config submodule.LIBERO.url https://github.com/nakamotoo/LIBERO.git
git config submodule.openpi.url https://github.com/nakamotoo/openpi.git
git config submodule.tactile_envs.url https://github.com/carlosferrazza/tactile_envs.git
git submodule update --init --recursive
```

The submodules provide the DSRL-compatible openpi fork, LIBERO, and the tactile simulation assets and modified robosuite implementation. Keep the recorded revisions when reproducing experiments.

### 2. Create an environment

```bash
conda create -n tacex python=3.11.11
conda activate tacex
python -m pip install --upgrade pip
```

### 3. Install dependencies

Run these commands from the repository root. PyTorch is used by the LIBERO/openpi stack; the reinforcement learning agents and π₀ use JAX.

```bash
python -m pip install torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  -r requirements.txt \
  "jax[cuda12]==0.5.0" \
  -e . \
  -e openpi/packages/openpi-client \
  -e openpi \
  -e LIBERO \
  -e tactile_envs \
  -e tactile_envs/tactile_envs/envs/robosuite

python -m pip check
```

### 4. Configure rendering and logging

For GPU rendering:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export EXP="$PWD/logs/tacex"
mkdir -p "$EXP"

wandb login
```

The first training run downloads the pretrained `pi0_libero` checkpoint from openpi's public asset storage. Allow disk space for the checkpoint, replay data, logs, and videos. `OPENPI_DATA_HOME` can be set to choose the checkpoint cache directory.

## Training on LIBERO

After the submodule release prerequisites above are resolved, the following command configures one run with tactile disagreement, a Panda arm, and the Robotiq tactile gripper:

```bash
python -m examples.launch_train_sim \
  --algorithm pixel_maxinfosac \
  --env libero \
  --libero_suite libero_90 \
  --libero_task_id 58 \
  --libero_robot Panda \
  --dsrl_action_mode noise \
  --prefix tacex_libero \
  --wandb_project TacEx \
  --seed 0 \
  --batch_size 256 \
  --discount 0.999 \
  --max_steps 500000 \
  --checkpoint_interval 20000 \
  --eval_interval 20000 \
  --log_interval 1000 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --query_freq 20 \
  --action_magnitude 1.0 \
  --hidden_dims 128 \
  --add_tactile 1 \
  --use_touch 1 \
  --touch_gripper_type Robotiq85TactileGripper \
  --tactile_shape 32 64 3 \
  --gripper_state_indices 0 3 \
  --ensemble_disagreement_modalities tactile \
  --mask_expl_critic 1
```

For replay-based learners, `max_steps` counts gradient updates; `multi_grad_step` controls updates per collected steering transition, and `query_freq` sets the action-chunk length. Logs separately record environment steps. Checkpoints are saved below `EXP` when `checkpoint_interval` is positive.

For the DSRL-SAC baseline, use `--algorithm pixel_sac --add_tactile 0 --use_touch 0` and omit the tactile gripper, gripper-state indices, and disagreement options. The remaining training settings can stay the same.

The scripts in `examples/scripts/` contain experimental sweeps and baselines, including PPO, DPPO, and summed-taxel-reward variants.

To check tactile sensor registration and contact response without loading π₀, run the setup diagnostic after the submodule integration is available:

```bash
python -m examples.test_libero_touch_setup \
  --mujoco_gl egl \
  --output_dir ./logs/touch_setup
```

The diagnostic saves a video, sensor measurements, and a summary. Its success flag checks tactile response; object lifting is reported separately.

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

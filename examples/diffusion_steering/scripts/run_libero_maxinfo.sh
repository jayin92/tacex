#!/bin/bash
cd "$(dirname "$0")/../../.."
proj_name=DSRL_pi0_Libero
device_id=1
libero_robot="${libero_robot:-Panda}"
if [[ -z "${touch_gripper_type:-}" ]]; then
  if [[ "${libero_robot}" == "Piper" ]]; then
    touch_gripper_type="PiperGripper"
  else
    touch_gripper_type="Robotiq85TactileGripper"
  fi
fi
if [[ -z "${tactile_shape:-}" ]]; then
  tactile_shape="32 64 3"
fi
if [[ -z "${gripper_state_indices:-}" ]]; then
  if [[ "${touch_gripper_type}" == "Robotiq85TactileGripper" ]]; then
    gripper_state_indices="0 3"
  else
    gripper_state_indices="0 1"
  fi
fi

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${device_id}"

export OPENPI_DATA_HOME=./openpi
export EXP="./logs/${proj_name}"
export CUDA_VISIBLE_DEVICES="${device_id}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${EXP}"


python -m examples.diffusion_steering.launch_train_sim \
  --algorithm pixel_maxinfosac \
  --env libero \
  --libero_robot "${libero_robot}" \
  --prefix dsrl_pi0_libero_maxinfosac \
  --wandb_project "${proj_name}" \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --add_tactile 1 \
  --use_touch 1 \
  --touch_gripper_type "${touch_gripper_type}" \
  --tactile_shape ${tactile_shape} \
  --gripper_state_indices ${gripper_state_indices} \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128 \
  --dyn_ent_lr 0.0003 \
  --init_dyn_ent_temperature 1.0 \
  --model_lr 0.0003 \
  --model_wd 0.0 \
  --model_hidden_dims 256 256 \
  --num_model_heads 5 \
  --model_noise_var 1.0 \
  --predict_reward 1 \
  --predict_diff 1 \
  --backup_entropy 1 \
  --model_obs_key state \
  --obs_dim 64 \
  --ensemble_disagreement_modalities "" \
  --mask_expl_critic 0 \
  --tactile_hidden_dims 256 256 \
  --mask_touch 0

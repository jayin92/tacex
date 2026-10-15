# Support both `python -m ...` and direct script execution from the checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

import numpy as np
import os
import sys
import argparse
from typing import Optional
from examples.exploration.examples.utils import Logger, hash_dict


def experiment(
        project_name: str,
        entity_name: str,
        alg_name: str,
        env_name: str,
        action_cost: float = 0.0,
        action_repeat: int = 2,
        lr: float = 1e-4,
        ens_lr: float = 1e-4,
        dyn_ent_lr: float = 1e-4,
        ens_wd: float = 0.0,
        dyn_wd: float = 0.0,
        tau: float = 0.005,
        seed: int = 0,
        wandb_log: bool = True,
        logs_dir: str = './logs',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        updates_per_step: int = 1,
        batch_size: int = 256,
        hidden_dims: int = 256,
        feature_dim: int = 50,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 5,
        exp_hash: str = '',
        n_steps_returns: int = 3,
        init_temperature: float = 1.0,
        sig: float = -1.0,
        pixels_only: bool = False,
        no_rotation: bool = False,
        no_gripping: bool = False,
        start_grasped: bool = False,
        num_init_grasp_steps: int = 150_000,
        frame_stack: int = 3,
        mask_touch: bool = False,
        expl_until: float = 1.0,
        switch_frequency: int = -1,
        tactile_size: Optional[int] = None,
        multi_obj: bool = False,
        expl_agent_update_period: int = 1,
        ensemble_update_period: int = 1,
        n_objects_start: int = 1,
        n_objects_end: int = 4,
        n_objects_steps_to_end: int = 6_000_000,
        agent_update_period: int = 1,
        latent_disg_weight: float = 1.0,
        visual_disg_weight: float = 1.0,
        tactile_disg_weight: float = 1.0,
        tactile_activation: str = 'symlog',
        tactile_clipping: float = -1,
        multi_camera: bool = True,
        rc_loss_coefficient: float = -1,
        update_freq: int = 1,
        grip_range: float = 1.0,
):
    # import jax
    # jax.config.update("jax_debug_nans", True)
    from jaxrl2.utils.tactile.train_utils import train
    if tactile_size is None:
        tactile_size = 32 if env_name == 'insertion' else 16
    if env_name == 'insertion':
        # The bundled insertion environment renders one camera at a time.
        multi_camera = False
    if alg_name in ['tactile_maxinfodrq', 'tactile_drq', 'tactile_drq_explorer']:
        n_steps_returns = -1

    alg_kwargs = {
        'actor_lr': lr,
        'critic_lr': lr,
        'temp_lr': lr,
        'hidden_dims': (hidden_dims, hidden_dims),
        'cnn_features': (32, 32, 32, 32),
        'cnn_strides': (2, 1, 1, 1),
        'cnn_padding': 'VALID',
        'latent_dim': feature_dim,
        'discount': 0.99,
        'tau': tau,
        'target_update_period': 1,
        'target_entropy': None,
        'init_temperature': init_temperature,
        'mask_touch': mask_touch,
    }
    if alg_name == 'tactile_drqv2' or alg_name == 'tactile_maxinfodrqv2':
        del alg_kwargs['init_temperature'], alg_kwargs['temp_lr'], alg_kwargs['target_entropy']
        # alg_kwargs['init_sig'] = 1.0
        # alg_kwargs['final_sig'] = 0.1
        if sig < 0:
            alg_kwargs['init_sig'] = 1.0
            alg_kwargs['final_sig'] = 0.1
        else:
            alg_kwargs['init_sig'] = sig
            alg_kwargs['final_sig'] = sig
        alg_kwargs['steps_to_final_sig'] = 500_000

    replay_buffer_size = min(replay_buffer_size, max_steps)
    if alg_name == 'tactile_maxinfodrq' or alg_name == 'tactile_maxinfodrqv2' or alg_name == 'tactile_drq_explorer':
        alg_kwargs['dyn_ent_lr'] = dyn_ent_lr
        alg_kwargs['dyn_wd'] = dyn_wd
        alg_kwargs['ens_lr'] = ens_lr
        alg_kwargs['ens_wd'] = ens_wd
        alg_kwargs['model_hidden_dims'] = (hidden_dims, hidden_dims)
        # if pixels_only:
        #    alg_kwargs['obs_dim'] = 64
        # else:
        alg_kwargs['obs_dim'] = 32

    if alg_name == 'tactile_drq_explorer':
        del alg_kwargs['dyn_ent_lr'], alg_kwargs['dyn_wd']
        if expl_until >= 1.0:
            exploration_steps = max(max_steps, 10_000_000)
        else:
            exploration_steps = int(expl_until * max_steps)
        alg_kwargs['explore_until'] = exploration_steps
        alg_kwargs['expl_agent_update_period'] = expl_agent_update_period
        alg_kwargs['agent_update_period'] = agent_update_period
        alg_kwargs['ensemble_update_period'] = ensemble_update_period
        alg_kwargs['rc_loss_coefficient'] = rc_loss_coefficient
        alg_kwargs.update({
            'latent_disg_weight': latent_disg_weight,
            'visual_disg_weight': visual_disg_weight,
            'tactile_disg_weight': tactile_disg_weight,
        })
        if switch_frequency == -1:
            alg_kwargs['switch_frequency'] = exploration_steps
        else:
            alg_kwargs['switch_frequency'] = switch_frequency
    if seed == 1:
        camera_idx = [1, 0]
    else:
        camera_idx = [0, 1]
    if not multi_camera:
        camera_idx = [1]
    env_kwargs = {
        'action_cost': action_cost,
        'action_repeat': action_repeat,
        'from_pixels': True,
        'pixels_only': pixels_only,
        'frame_stack': frame_stack,
        'image_size': 84,
        'tactile_size': tactile_size,
        'max_episode_steps': 1_000,
        'return_grasp_flag_as_reward': False,
        'n_objects_start': n_objects_start,
        'n_objects_end': n_objects_end,
        'n_objects_steps_to_end': n_objects_steps_to_end,
        'env_kwargs': {
            'skip_frame': 0,
            'no_rotation': no_rotation,
            'no_gripping': no_gripping,
            'start_grasped': start_grasped,
            'num_init_grasp_steps': num_init_grasp_steps,
            'multi_obj': multi_obj,
            'tactile_activation': tactile_activation,
            'camera_idx': camera_idx,
            'tactile_clipping': tactile_clipping,
            'grip_range': grip_range,
        }
    }

    if env_name == 'insertion':
        if tactile_activation not in ('normal', 'symlog'):
            raise ValueError('Insertion supports normal or symlog tactile activation')
        env_kwargs['env_kwargs'] = {
            'skip_frame': 1,
            'no_rotation': no_rotation,
            'no_gripping': no_gripping,
            'camera_idx': camera_idx[0],
            'symlog_tactile': tactile_activation == 'symlog',
        }

    log_config = {
        'alg_name': alg_name,
        'action_cost': action_cost,
        'lr': lr,
        'ens_lr': ens_lr,
        'ens_wd': ens_wd,
        'dyn_ent_lr': dyn_ent_lr,
        'dyn_wd': dyn_wd,
        'action_repeat': action_repeat,
        'env_name': env_name,
        'hidden_dims': hidden_dims,
        'latent_dim': feature_dim,
        'batch_size': batch_size,
        'n_steps_returns': n_steps_returns,
        'tau': tau,
        'init_temperature': init_temperature,
        'sig': sig,
        'pixels_only': pixels_only,
        'no_rotation': no_rotation,
        'no_gripping': no_gripping,
        'start_grasped': start_grasped,
        'num_init_grasp_steps': num_init_grasp_steps,
        'return_grasp_flag_as_reward': False,
        'frame_stack': frame_stack,
        'tactile_size': tactile_size,
        'mask_touch': mask_touch,
        'switch_frequency': switch_frequency,
        'expl_until': expl_until,
        'n_objects_start': n_objects_start,
        'n_objects_end': n_objects_end,
        'n_objects_steps_to_end': n_objects_steps_to_end,
        'multi_obj': multi_obj,
        'agent_update_period': agent_update_period,
        'expl_agent_update_period': expl_agent_update_period,
        'latent_disg_weight': latent_disg_weight,
        'visual_disg_weight': visual_disg_weight,
        'tactile_disg_weight': tactile_disg_weight,
        'tactile_activation': tactile_activation,
        'tactile_clipping': tactile_clipping,
        'multi_camera': multi_camera,
        'rc_loss_coefficient': rc_loss_coefficient,
        'update_freq': update_freq,
        'updates_per_step': updates_per_step,
        'ensemble_update_period': ensemble_update_period,
        'training_start': training_start,
        'grip_range': grip_range,
    }

    save_video = save_video and (seed == 0 or seed == 1)
    save_train_video = save_video and (seed == 0 or seed == 1)

    train(
        project_name=project_name,
        entity_name=entity_name,
        alg_name=alg_name,
        env_name=env_name,
        alg_kwargs=alg_kwargs,
        env_kwargs=env_kwargs,
        seed=seed,
        wandb_log=wandb_log,
        log_config=log_config,
        logs_dir=logs_dir,
        save_video=save_video,
        save_train_video=save_train_video,
        save_freq=100,
        replay_buffer_size=replay_buffer_size,
        max_steps=max_steps,
        use_tqdm=use_tqdm,
        training_start=training_start,
        updates_per_step=updates_per_step,
        batch_size=batch_size,
        log_interval=log_interval,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        exp_hash=exp_hash,
        n_steps_returns=n_steps_returns,
        update_freq=update_freq,
    )


def main(args):
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    print('exp hash: ', exp_hash)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        env_name=args.env_name,
        action_cost=args.action_cost,
        action_repeat=args.action_repeat,
        lr=args.lr,
        ens_lr=args.ens_lr,
        dyn_ent_lr=args.dyn_ent_lr,
        ens_wd=args.ens_wd,
        dyn_wd=args.dyn_wd,
        tau=args.tau,
        seed=args.seed,
        wandb_log=bool(args.wandb_log),
        logs_dir=args.logs_dir + f'/{args.alg_name}/{exp_hash}',
        save_video=bool(args.save_video),
        replay_buffer_size=args.replay_buffer_size,
        max_steps=args.max_steps,
        use_tqdm=bool(args.use_tqdm),
        training_start=args.training_start,
        updates_per_step=args.updates_per_step,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        feature_dim=args.feature_dim,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        n_steps_returns=args.n_steps_returns,
        init_temperature=args.init_temperature,
        sig=args.sig,
        pixels_only=bool(args.pixels_only),
        no_rotation=bool(args.no_rotation),
        no_gripping=bool(args.no_gripping),
        start_grasped=bool(args.start_grasped),
        num_init_grasp_steps=args.num_init_grasp_steps,
        frame_stack=args.frame_stack,
        tactile_size=args.tactile_size,
        mask_touch=bool(args.mask_touch),
        expl_until=args.expl_until,
        switch_frequency=args.switch_frequency,
        multi_obj=bool(args.multi_obj),
        agent_update_period=args.agent_update_period,
        n_objects_start=args.n_objects_start,
        n_objects_end=args.n_objects_end,
        n_objects_steps_to_end=args.n_objects_steps_to_end,
        expl_agent_update_period=args.expl_agent_update_period,
        ensemble_update_period=args.ensemble_update_period,
        latent_disg_weight=args.latent_disg_weight,
        visual_disg_weight=args.visual_disg_weight,
        tactile_disg_weight=args.tactile_disg_weight,
        tactile_activation=args.tactile_activation,
        tactile_clipping=args.tactile_clipping,
        multi_camera=bool(args.multi_camera),
        rc_loss_coefficient=args.rc_loss_coefficient,
        update_freq=args.update_freq,
        grip_range=args.grip_range,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--entity_name', type=str, default='sukhijab')
    parser.add_argument('--alg_name', type=str, default='tactile_drq_explorer')
    parser.add_argument('--env_name', type=str, default='arm_exploration')
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ens_lr', type=float, default=1e-4)
    parser.add_argument('--dyn_ent_lr', type=float, default=1e-4)
    parser.add_argument('--ens_wd', type=float, default=0.0)
    parser.add_argument('--dyn_wd', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000)
    parser.add_argument('--max_steps', type=int, default=250_000)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--training_start', type=int, default=0)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dims', type=int, default=256)
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=1_000)
    parser.add_argument('--eval_interval', type=int, default=5_000)
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--n_steps_returns', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--init_temperature', type=float, default=1)
    parser.add_argument('--sig', type=float, default=-1.0)
    parser.add_argument('--pixels_only', type=int, default=0)
    parser.add_argument('--no_rotation', type=int, default=1)
    parser.add_argument('--no_gripping', type=int, default=1)
    parser.add_argument('--start_grasped', type=int, default=1)
    parser.add_argument('--num_init_grasp_steps', type=int, default=150_000)
    parser.add_argument('--frame_stack', type=int, default=3)
    parser.add_argument('--tactile_size', type=int, default=None)
    parser.add_argument('--mask_touch', type=int, default=0)
    parser.add_argument('--expl_until', type=float, default=1.0)
    parser.add_argument('--switch_frequency', type=int, default=-1)
    parser.add_argument('--multi_obj', type=int, default=1)
    parser.add_argument('--agent_update_period', type=int, default=1)
    parser.add_argument('--expl_agent_update_period', type=int, default=1)
    parser.add_argument('--ensemble_update_period', type=int, default=1)
    parser.add_argument('--n_objects_start', type=int, default=1)
    parser.add_argument('--n_objects_end', type=int, default=6)
    parser.add_argument('--n_objects_steps_to_end', type=int, default=6_000_000)
    parser.add_argument('--latent_disg_weight', type=float, default=1.0)
    parser.add_argument('--visual_disg_weight', type=float, default=1.0)
    parser.add_argument('--tactile_disg_weight', type=float, default=1.0)
    parser.add_argument('--tactile_activation', type=str, default="symlog")
    parser.add_argument('--tactile_clipping', type=float, default=-1)
    parser.add_argument('--multi_camera', type=int, default=1)
    parser.add_argument('--rc_loss_coefficient', type=float, default=-1)
    parser.add_argument('--update_freq', type=int, default=1)
    parser.add_argument('--grip_range', type=float, default=1.0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)

# Support both `python -m ...` and direct script execution from the checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import numpy as np
import os
import sys
import argparse


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
        sig: float = 0.0,
):
    # import jax
    # jax.config.update("jax_debug_nans", True)
    from jaxrl2.utils.train_utils import train
    if alg_name in ['maxinfodrq', 'drq']:
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
    }
    if alg_name == 'drqv2' or alg_name == 'maxinfodrqv2':
        del alg_kwargs['init_temperature'], alg_kwargs['temp_lr'], alg_kwargs['target_entropy']
        # alg_kwargs['init_sig'] = 1.0
        # alg_kwargs['final_sig'] = 0.1
        if alg_name == 'drqv2':
            alg_kwargs['init_sig'] = 1.0
            alg_kwargs['final_sig'] = 0.1
        else:
            alg_kwargs['init_sig'] = sig
            alg_kwargs['final_sig'] = sig
        alg_kwargs['steps_to_final_sig'] = 500_000

    replay_buffer_size = min(replay_buffer_size, max_steps)
    if alg_name == 'maxinfodrq' or alg_name == 'maxinfodrqv2':
        alg_kwargs['dyn_ent_lr'] = dyn_ent_lr
        alg_kwargs['dyn_wd'] = dyn_wd
        alg_kwargs['ens_lr'] = ens_lr
        alg_kwargs['ens_wd'] = ens_wd
        alg_kwargs['model_hidden_dims'] = (hidden_dims, hidden_dims)
        alg_kwargs['obs_dim'] = 32

    env_kwargs = {'action_cost': action_cost,
                  'action_repeat': action_repeat,
                  'from_pixels': True,
                  'frame_stack': 3,
                  'image_size': 84,
                  'gray_scale': False,
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
    }

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
    )


def main(args):
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
        logs_dir=args.logs_dir,
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
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--entity_name', type=str, required=True)
    parser.add_argument('--alg_name', type=str, default='maxinfodrq')
    parser.add_argument('--env_name', type=str, default='cartpole-swingup_sparse')
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=1)
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
    parser.add_argument('--use_tqdm', type=int, default=0)
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
    parser.add_argument('--sig', type=float, default=0.0)

    args = parser.parse_args()
    main(args)

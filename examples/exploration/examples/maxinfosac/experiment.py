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
        action_repeat: int = 1,
        ens_lr: float = 3e-4,
        dyn_ent_lr: float = 3e-4,
        num_neurons: int = 256,
        num_hidden_layers: int = 2,
        lr: float = 3e-4,
        ens_wd: float = 0.0,
        dyn_wd: float = 0.0,
        seed: int = 0,
        wandb_log: bool = True,
        logs_dir: str = './logs',
        save_video: bool = False,
        replay_buffer_size: int = 1_000_000,
        max_steps: int = 1_000_000,
        use_tqdm: bool = True,
        training_start: int = 0,
        batch_size: int = 256,
        log_interval: int = 1_000,
        eval_interval: int = 5_000,
        eval_episodes: int = 10,
        exp_hash: str = '',
):
    from jaxrl2.utils.train_utils import train

    alg_kwargs = {
        'actor_lr': lr,
        'critic_lr': lr,
        'temp_lr': lr,
        'hidden_dims': (num_neurons,) * num_hidden_layers,
        'discount': 0.99,
        'tau': 0.005,
        'target_update_period': 1,
        'target_entropy': None,
        'backup_entropy': True,
    }
    updates_per_step = 1
    replay_buffer_size = min(replay_buffer_size, max_steps)
    if alg_name  == 'maxinfosac':
        alg_kwargs['dyn_ent_lr'] = dyn_ent_lr
        alg_kwargs['dyn_wd'] = dyn_wd
        alg_kwargs['ens_lr'] = ens_lr
        alg_kwargs['ens_wd'] = ens_wd
        alg_kwargs['model_hidden_dims'] = (num_neurons, ) * num_hidden_layers

    if alg_name  == 'redq':
        alg_kwargs['n'] = 10
        alg_kwargs['m'] = 2
        updates_per_step = 20
    model_update_delay = 1


    env_kwargs = {'action_cost': action_cost,
                  'action_repeat': action_repeat,
                  }

    log_config = {
        'alg_name': alg_name,
        'action_cost': action_cost,
        'ens_lr': ens_lr,
        'ens_wd': ens_wd,
        'lr': lr,
        'dyn_ent_lr': dyn_ent_lr,
        'dyn_wd': dyn_wd,
        'num_neurons': num_neurons,
        'num_hidden_layers': num_hidden_layers,
        'action_repeat': action_repeat,
        'env_name': env_name,
        'model_update_delay': model_update_delay,
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
    )


def main(args):
    """"""
    from pprint import pprint
    print(args)
    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        env_name=args.env_name,
        action_cost=args.action_cost,
        action_repeat=args.action_repeat,
        ens_lr=args.ens_lr,
        dyn_ent_lr=args.dyn_ent_lr,
        num_neurons=args.num_neurons,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        ens_wd=args.ens_wd,
        dyn_wd=args.dyn_wd,
        seed=args.seed,
        wandb_log=bool(args.wandb_log),
        logs_dir=args.logs_dir,
        save_video=bool(args.save_video),
        replay_buffer_size=args.replay_buffer_size,
        max_steps=args.max_steps,
        use_tqdm=bool(args.use_tqdm),
        training_start=args.training_start,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./logs/')
    parser.add_argument('--project_name', type=str, default='MCTest')
    parser.add_argument('--entity_name', type=str, required=True)
    parser.add_argument('--alg_name', type=str, default='maxinfosac')
    parser.add_argument('--env_name', type=str, default='humanoid-walk')
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--ens_lr', type=float, default=3e-4)
    parser.add_argument('--dyn_ent_lr', type=float, default=3e-4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ens_wd', type=float, default=0.0)
    parser.add_argument('--dyn_wd', type=float, default=0.0)
    parser.add_argument('--num_neurons', type=int, default=256)
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--wandb_log', type=int, default=1)
    parser.add_argument('--save_video', type=int, default=0)
    parser.add_argument('--replay_buffer_size', type=int, default=1_000_000)
    parser.add_argument('--max_steps', type=int, default=1_500_000)
    parser.add_argument('--use_tqdm', type=int, default=1)
    parser.add_argument('--training_start', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--log_interval', type=int, default=1_000)
    parser.add_argument('--eval_interval', type=int, default=5_000)
    parser.add_argument('--eval_episodes', type=int, default=5)

    parser.add_argument('--seed', type=int, default=0)


    args = parser.parse_args()
    main(args)

from typing import Tuple

import jax.numpy as jnp
import jax.random

from jaxrl2.data import NstepBatch
from jaxrl.networks.common import InfoDict, Model, Params
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble


def clip(x, min: float = -1.0, max: float = 1.0):
    # use custom clipping to allow gradient flow through the clip function
    clipped_x = jax.lax.stop_gradient(jnp.clip(x, min=min, max=max))
    x = x - jax.lax.stop_gradient(x) + clipped_x
    return x


def update(actor: Model,
           target_actor: Model,
           critic: Model,
           dyn_ent_temp: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           state: jnp.ndarray,
           batch: NstepBatch,
           noise_std: float,
           noise_clip: float,
           key: jax.random.PRNGKey,
           ) -> Tuple[Model, EnsembleState, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[EnsembleState, InfoDict]]:
        actor_key, target_actor_key = jax.random.split(key, 2)
        actions = actor.apply_fn({'params': actor_params}, batch.observations)
        noise = jax.random.normal(key=actor_key, shape=actions.shape) * noise_std
        noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
        actions = clip(actions + noise, min=-1, max=1)
        q, q_expl, _, _ = critic(batch.observations, actions)
        q = jnp.min(q, axis=0)
        q_expl = jnp.min(q_expl, axis=0)
        dyn_ent_coef, _ = dyn_ent_temp()
        total_q = q + dyn_ent_coef * q_expl
        actor_loss = -total_q.mean()

        # getting info gain objective
        target_actions = target_actor(batch.observations)
        noise = jax.random.normal(key=target_actor_key, shape=target_actions.shape) * noise_std
        noise = jnp.clip(noise, min=-noise_clip, max=noise_clip)
        target_actions = clip(target_actions + noise, min=-1, max=1)
        target_inp = jnp.concatenate([state, target_actions], axis=-1)
        inp = jnp.concatenate([state, actions], axis=-1)
        total_inp = jnp.concatenate([inp, target_inp], axis=0)
        info_gain, new_ens_state = ens.get_info_gain(input=total_inp,
                                                     state=ens_state,
                                                     update_normalizer=True)
        info_gain, target_info_gain = info_gain[:actions.shape[0]], info_gain[actions.shape[0]:]
        return actor_loss, (new_ens_state, {'actor_loss': actor_loss,
                                            'info_gain': info_gain.mean(),
                                            'target_info_gain': target_info_gain.mean(),
                                            })

    new_actor, (new_ens_state, info) = actor.apply_gradient(actor_loss_fn)

    return new_actor, new_ens_state, info

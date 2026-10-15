import jax.random
from jaxrl2.networks.ensemble_model import EnsembleState, DeterministicEnsemble
import jax.numpy as jnp
import jax.random as jr
from distrax import Normal

EPS = 1e-8


def get_unsupervised_discovery_reward(
        observation: jnp.ndarray,
        skill: jnp.ndarray,
        ensemble: DeterministicEnsemble,
        ens_state: EnsembleState,
        rng: jax.Array,
        skill_sampling_dist: str = 'Gaussian',
        num_skill_samples: int = 5,
        sampling_std: float = 1.0,
        squash_skill: bool = True,
):
    """
    Given (s, z) return reward r(s, z) for the skill conditioned policy pi(a|s, z).
    r(s, z) = E_{z' \sim p(z)}[KL(p(s'|s, z) || p(s'|s, z'))]
    """
    assert len(observation.shape) == len(skill.shape) == 1, "the function is only written for a single batch, " \
                                                            "vmap it to get reward across batches"
    # sample skills to approximate the expectation
    if skill_sampling_dist == 'Gaussian':
        sampled_skills = jr.normal(shape=(num_skill_samples,) + skill.shape, key=rng) * sampling_std
        if squash_skill:
            sampled_skills = jnp.tanh(sampled_skills)
    elif skill_sampling_dist == 'Uniform':
        sampled_skills = jr.uniform(key=rng, shape=(num_skill_samples,) + skill.shape, maxval=1.0, minval=-1.0)
    else:
        raise NotImplementedError

    # get next state prediction for all different skills: p(s'|s, z)
    next_state_fn = lambda z: ensemble(input=jnp.concatenate([observation, z], axis=-1), state=ens_state,
                                       denormalize_output=False)

    all_skills = jnp.concatenate([sampled_skills, skill[jnp.newaxis]], axis=0)

    # get next state predictions
    next_state, next_state_std = jax.vmap(next_state_fn)(all_skills)
    next_state, next_state_skill = next_state[:-1], next_state[-1]
    next_state_std, next_state_skill_std = next_state_std[:-1], next_state_std[-1]
    # do moment matching such that p(s'|s, z) = N(\mu(s,z), \sigma^2(s, z)), where sigma comes from the epistemic std
    ns_skill_mean, ns_skill_eps_std = jnp.mean(next_state_skill, axis=0), jnp.std(next_state_skill, axis=0)
    # total variance = Epistemic var + aleatoric var
    total_var = jnp.square(ns_skill_eps_std) + jnp.square(jnp.mean(next_state_skill_std, axis=0))
    dist_skill = Normal(loc=ns_skill_mean, scale=jnp.sqrt(total_var))

    def kl_dist(ns, ns_std):
        # do moment matching such that p(s'|s, z) = N(\mu(s,z), \sigma^2(s, z)),
        # where sigma comes from the epistemic std
        ns_mean, ns_eps_std = jnp.mean(ns, axis=0), jnp.std(ns, axis=0)
        # total variance = Epistemic var + aleatoric var
        total_var = jnp.square(ns_eps_std) + jnp.square(jnp.mean(ns_std, axis=0))
        dist_1 = Normal(loc=ns_mean, scale=jnp.sqrt(total_var))
        kl = dist_skill.kl_divergence(dist_1).mean(axis=-1)
        return kl

    # take the average kl to the skill dist across samples
    average_kl = jax.vmap(kl_dist)(next_state, next_state_std)
    average_kl = average_kl.mean(axis=0)
    return average_kl

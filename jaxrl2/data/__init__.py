from jaxrl2.data.replay_buffer import ReplayBuffer


def __getattr__(name):
    if name in {'NstepReplayBuffer', 'NstepBatch'}:
        from jaxrl2.data import nstep_buffer

        value = getattr(nstep_buffer, name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

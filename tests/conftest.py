import pytest

from fsbench import FSConfig, generate_env


@pytest.fixture
def make_env(tmp_path):
    counter = iter(range(10_000))

    def _make(task="reconcile", cfg=None, seed=7, **kw):
        out = tmp_path / f"env{next(counter)}"
        manifest = generate_env(task, cfg or FSConfig(), out, world_seed=seed, task_seed=seed,
                                layout_seed=seed, **kw)
        return out, manifest

    return _make

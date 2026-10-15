from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]

setup(
    name="jaxrl2",
    version="0.0.0",
    description="TacEx reinforcement learning agents and experiment entry points",
    python_requires=">=3.11",
    packages=find_packages(include=("jaxrl2", "jaxrl2.*", "examples", "examples.*"),
                           exclude=("examples.diffusion_steering.tests", "examples.diffusion_steering.tests.*")),
    install_requires=requirements,
    extras_require={
        "exploration": [
            # Gymnasium-compatible fork; PyPI jaxrl uses a different API.
            "jaxrl @ git+https://github.com/sukhijab/jaxrl.git@268bbdc039033184078d121a237d4f27b9178f19",
        ],
        "metaworld": [
            "jaxrl @ git+https://github.com/sukhijab/jaxrl.git@268bbdc039033184078d121a237d4f27b9178f19",
            "metaworld @ git+https://github.com/sukhijab/Metaworld.git@2ae2cf61413f09ede5ee582737e2c9722878c475",
        ],
        "humanoid-bench": [
            "jaxrl @ git+https://github.com/sukhijab/jaxrl.git@268bbdc039033184078d121a237d4f27b9178f19",
            "humanoid-bench @ git+https://github.com/sukhijab/humanoid-bench.git@edc0438ad9f280b73f5e00f7a061c4d650294ee5",
            "mujoco-mjx==3.8.1",
            "brax==0.12.1",
            "gymnax==0.0.9",
        ],
        "test": ["pytest>=8.3,<10"],
    },
)

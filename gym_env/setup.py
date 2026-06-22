from setuptools import setup, find_packages

setup(
    name="panda_vla_env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "mujoco>=3.0.0",
        "numpy",
    ],
    entry_points={
        "gymnasium.envs": [
            "PandaVLA-v0 = gym_env.panda_vla_env:PandaVLAEnv",
        ],
    },
)

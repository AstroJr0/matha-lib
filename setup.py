import numpy as np
from setuptools import Extension, setup

MODULES = ["_numtheory", "_vector", "_linalg", "_stats", "_trig", "_geometry", "_special", "_ml"]

setup(
    ext_modules=[
        Extension(
            f"matha.{m}",
            [f"matha/{m}.c"],
            include_dirs=[np.get_include(), "matha"],
            depends=["matha/_npc.h"],
        )
        for m in MODULES
    ],
)

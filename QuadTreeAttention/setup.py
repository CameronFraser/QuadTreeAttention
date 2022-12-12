#!/usr/bin/env python

import os

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_file = os.path.dirname(__file__)

setup(
    name="qta",
    ext_modules=[
        CUDAExtension(
            "score_computation_cuda",
            [
                "qta/src/score_computation.cpp",
                "qta/src/score_computation_kernal.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
        CUDAExtension(
            "value_aggregation_cuda",
            [
                "qta/src/value_aggregation.cpp",
                "qta/src/value_aggregation_kernel.cu"
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
    ],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)

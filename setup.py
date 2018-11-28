import os
import setuptools

# NOTE: If dijkstra.cpp does not exist:
# cython -3 --fast-fail -v --cplus dijkstra.pyx

import numpy as np

setuptools.setup(
  setup_requires=['pbr', 'numpy'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=[
    setuptools.Extension(
      'dijkstra3d',
      sources=[ 'dijkstra3d.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=[
        '-std=c++11', '-O3', '-ffast-math'
      ]
    )
  ],
  pbr=True)






import os
import setuptools
import sys

# NOTE: If dijkstra.cpp does not exist:
# cython -3 --fast-fail -v --cplus dijkstra.pyx

import numpy as np

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

extra_compile_args = [
  '-std=c++11', '-O3', '-ffast-math', 
]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  name="dijkstra3d",
  version="1.9.0",
  python_requires="~=3.6", # >= 3.6 < 4.0
  setup_requires=['numpy'],
  ext_modules=[
    setuptools.Extension(
      'dijkstra3d',
      sources=[ 'dijkstra3d.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=extra_compile_args,
    )
  ],
  url="https://github.com/seung-lab/dijkstra3d/",
  author="William Silversmith",
  author_email="ws9@princeton.edu",
  packages=setuptools.find_packages(),
  package_data={
    'dijkstra3d': [
      'LICENSE',
    ],
  },
  description="Implementation of Dijkstra's Shortest Path algorithm on 3D images.",
  long_description=read('README.md'),
  long_description_content_type="text/markdown",
  license = "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ]
)






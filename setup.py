import os
import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

def read(fname):
  with open(os.path.join(os.path.dirname(__file__), fname), 'rt') as f:
    return f.read()

extra_compile_args = [
  '-std=c++11', '-O3',
]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  name="dijkstra3d",
  version="1.15.2",
  python_requires=">=3.8,<4.0",
  setup_requires=['numpy','cython'],
  ext_modules=[
    setuptools.Extension(
      'dijkstra3d',
      sources=[ 'dijkstra3d.pyx' ],
      language='c++',
      include_dirs=[ str(NumpyImport()) ],
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
  license = "GPL-3.0-or-later",
  classifiers=[
    "Intended Audience :: Developers",
    "Development Status :: 5 - Production/Stable",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows :: Windows 10",
  ]
)






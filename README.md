[![Build Status](https://travis-ci.org/seung-lab/dijkstra3d.svg?branch=master)](https://travis-ci.org/seung-lab/dijkstra3d)

# dijkstra3d
Dijkstra's Shortest Path for 3D Volumes. 

Perform dijkstra's shortest path algorithm on a 3D image grid. Vertices are voxels and edges are the 26 nearest neighbors (except for the edges of the image where the number of edges is reduced). For given input voxels A and B, the edge weight from A to B is B and from B to A is A. All weights must be non-negative (incl. negative zero).  

## C++ Use 

```cpp
#include <vector>
#include "dijkstra3d.hpp"

// 3d array represented as 1d array
float* labels = new float[512*512*512](); 

// x + sx * y + sx * sy * z
int source = 0 + 512 * 5 + 512 * 512 * 3; // coordinate <0, 5, 3>
int target = 128 + 512 * 128 + 512 * 512 * 128; // coordinate <128, 128, 128>

vector<unsigned int> path = dijkstra::dijkstra3d<float>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512,
  source, target
);
```

## Python Installation

*Requires a C++ compiler.*

```bash
pip install -r requirements.txt
python setup.py develop
```

## Python Use

```python
from dijkstra import dijkstra
import numpy as np

x = np.ones((512, 512, 512), dtype=np.int32)
y = dijkstra(x, (0,0), (511, 511, 511))
print(y.shape)
```
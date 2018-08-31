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

float* field = dijkstra::distance_field3d<float>(labels, /*sx=*/512, /*sy=*/512, /*sz=*/512, source);
```

## Python Installation

*Requires a C++ compiler.*

```bash
pip install -r requirements.txt
python setup.py develop
```

## Python Use

```python
import dijkstra
import numpy as np

x = np.ones((512, 512, 512), dtype=np.int32)
y = dijkstra.dijkstra(x, (0,0,0), (511, 511, 511))
print(y.shape)

y = dijkstra.distance_field(x, (0,0,0), (511, 511, 511))
```

## Performance

On a field of ones from the bottom left corner to the top right corner of a 512x512x512 float32 image, it takes about 41 seconds, for a performance rating of about 3 MVx/sec on a 3.7 GHz Intel i7-4920K CPU.  

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/dijkstra3d/master/dijkstra3d.png" alt="A memory benchmark of a 512x512x512 field of ones run.." /><br>
Fig. 1: A benchmark of dijkstra.dijkstra run on a 512<sup>3</sup> voxel field of ones from bottom left source to top right target.
</p>


### What is that pairing_heap.hpp?

Early on, I anticipated using decrease key in my heap and implemented a pairing heap, which is supposed to be an improvement on the Fibbonacci heap. However, I ended up not using decrease key, and the STL priority queue ended up being faster. If you need a pairing heap outside of boost, check it out.

## References

1. E. W. Dijkstra. "A Note on Two Problems in Connexion with Graphs" Numerische Mathematik 1. pp. 269-271. (1959)  
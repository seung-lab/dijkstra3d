[![Build Status](https://travis-ci.org/seung-lab/dijkstra3d.svg?branch=master)](https://travis-ci.org/seung-lab/dijkstra3d) [![PyPI version](https://badge.fury.io/py/dijkstra3d.svg)](https://badge.fury.io/py/dijkstra3d)  

# dijkstra3d
Dijkstra's Shortest Path variants for 26-connected 3D Image Volumes or 8-connected 2D images. 

Perform dijkstra's shortest path algorithm on a 3D image grid. Vertices are voxels and edges are the 26 nearest neighbors (except for the edges of the image where the number of edges is reduced). For given input voxels A and B, the edge weight from A to B is B and from B to A is A. All weights must be non-negative (incl. negative zero).  

## What Problem does this Package Solve?

This package was developed in the course of exploring TEASAR skeletonization of 3D image volumes (now available in [Kimimaro](https://github.com/seung-lab/kimimaro)). Other commonly available packages implementing Dijkstra used matricies or object graphs as their underlying implementation. In either case, these generic graph packages necessitate explicitly creating the graph's edges and vertices, which turned out to be a significant computational cost compared with the search time. Additionally, some implementations required memory quadratic in the number of vertices (e.g. an NxN matrix for N nodes) which becomes prohibitive for large arrays. In some cases, a compressed sparse matrix representation was used to remain within memory limits.  

Neither of graph construction nor quadratic memory pressure are necessary for an image analysis application. The edges between voxels (3D pixels) are regular and implicit in the rectangular structure of the image. Additionally, the cost of each edge can be stored a single time instead of 26 times in contiguous uncompressed memory regions for faster performance.  

## Available Dijkstra Variants

The following variants are available in 2D and 3D:

- **dijkstra** - Shortest path between source and target. Early termination on finding the target. Bidirectional version available.
- **parental_field / query_shortest_path** - Compute shortest path between source and all targets. Use query_shortest_path to make repeated queries against the result set.  
- **euclidean_distance_field** - Given a boolean label field and a source vertex, compute the anisotropic euclidean distance from the source to all labeled vertices.
- **distance_field** - Given a numerical field, for each directed edge from adjacent voxels A and B, use B as the edge weight. In this fashion, compute the distance from a source point for all finite voxels.


## Python Use

```python
import dijkstra3d
import numpy as np

field = np.ones((512, 512, 512), dtype=np.int32)
source = (0,0,0)
target = (511, 511, 511)

path = dijkstra3d.dijkstra(field, source, target) # terminates early
path = dijkstra3d.dijkstra(field, source, target, bidirectional=True) # 2x memory usage, faster algorithm
print(path.shape)

parents = dijkstra3d.parental_field(field, source=(0,0,0))
path = dijkstra3d.path_from_parents(parents, target=(511, 511, 511))
print(path.shape)

dist_field = dijkstra3d.euclidean_distance_field(field, source=(0,0,0), anisotropy=(4,4,40))
dist_field = dijkstra3d.distance_field(field, source=(0,0,0))
```

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

vector<unsigned int> path = dijkstra::bidirectional_dijkstra3d<float>(
  labels, /*sx=*/512, /*sy=*/512, /*sz=*/512,
  source, target
);

uint32_t* parents = dijkstra::parental_field3d<float>(labels, /*sx=*/512, /*sy=*/512, /*sz=*/512, source);
vector<unsigned int> path = dijkstra::query_shortest_path(parents, target);


float* field = dijkstra::euclidean_distance_field3d<float>(
  labels, 
  /*sx=*/512, /*sy=*/512, /*sz=*/512, 
  /*wx=*/4, /*wy=*/4, /*wz=*/40, 
  source);

float* field = dijkstra::distance_field3d<float>(labels, /*sx=*/512, /*sy=*/512, /*sz=*/512, source);
```

## Python `pip` Binary Installation

```bash
pip install dijkstra3d
```

## Python `pip` Source Installation

*Requires a C++ compiler.*

```bash
pip install numpy
pip install dijkstra3d
```

## Python Direct Installation

*Requires a C++ compiler.*

```bash
git clone https://github.com/seung-lab/dijkstra3d.git
cd dijkstra3d
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop
```

## Performance

I ran the algorithm on a field of ones from the bottom left corner to the top right corner of a 512x512x512 int8 image using a 3.7 GHz Intel i7-4920K CPU. Unidirectional search takes about 39.5 seconds (3.4 MVx/sec) with a maximum memory usage of about 1300 MB. In the unidirectional case, this test forces the algorithm to process nearly all of the volume (dijkstra aborts early when the target is found). In the bidirectional case, the volume is processed in about 11.5 seconds (11.7 MVx/sec) with a peak memory usage of about 2300 MB.

Theoretical unidirectional memory allocation breakdown: 128 MB source image, 512 MB distance field, 512 MB parents field (1152 MB). Theoretical bidirectional memory allocation breakdown: 128 MB source image, 2x 512 distance field, 2x 512 MB parental field (2176 MB).

<p style="font-style: italics;" align="center">
<img height=384 src="https://raw.githubusercontent.com/seung-lab/dijkstra3d/master/dijkstra3d.png" alt="Fig. 1: A benchmark of dijkstra.dijkstra run on a 512^3 voxel field of ones from bottom left source to top right target. (black) bidirectional search (blue) unidirectional search." /><br>
Fig. 1: A benchmark of dijkstra.dijkstra run on a 512<sup>3</sup> voxel field of ones from bottom left source to top right target. (black) bidirectional search (blue) unidirectional search.
</p>

```python 
import numpy as np
import time
import dijkstra3d

field = np.ones((512,512,512), order='F', dtype=np.int8)

s = time.time()
path = dijkstra3d.dijkstra(x, source=(0,0,0), target=(511, 511, 511), bidirectional=True) # or False 
print(time.time() - s)
```


### What is that pairing_heap.hpp?

Early on, I anticipated using decrease key in my heap and implemented a pairing heap, which is supposed to be an improvement on the Fibbonacci heap. However, I ended up not using decrease key, and the STL priority queue ended up being faster. If you need a pairing heap outside of boost, check it out.

## References

1. E. W. Dijkstra. "A Note on Two Problems in Connexion with Graphs" Numerische Mathematik 1. pp. 269-271. (1959)  
2. E. W. Dijkstra. "Go To Statement Considered Harmful". Communications of the ACM. Vol. 11, No. 3, pp. 147-148. (1968)
3. Pohl, Ira. "Bi-directional Search", in Meltzer, Bernard; Michie, Donald (eds.), Machine Intelligence, 6, Edinburgh University Press, pp. 127–140. (1971)

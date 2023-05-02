# cython: language_level=3
"""
Cython binding for C++ dijkstra's shortest path algorithm
applied to 3D images. 

Contains:
  dijkstra - Find the shortest 26-connected path from source
    to target using the values of each voxel as edge weights.\

  distance_field - Compute the distance field from a source
    voxel in an image using image voxel values as edge weights.

  euclidean_distance_field - Compute the euclidean distance
    to each voxel in a binary image from a source point.

  parental_field / path_from_parents - Same as dijkstra,
    but if you're computing dijkstra multiple times on
    the same image, this can be much much faster.


Author: William Silversmith
Affiliation: Seung Lab, Princeton Neuroscience Institute
Date: August 2018-February 2020
"""

from libc.stdlib cimport calloc, free
from libc.stdint cimport (
   int8_t,  int16_t,  int32_t,  int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t
)
from cpython cimport array 
import array
import sys

from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

__VERSION__ = '1.12.0'

ctypedef fused UINT:
  uint8_t
  uint16_t
  uint32_t
  uint64_t

class DimensionError(Exception):
  pass

cdef extern from "dijkstra3d.hpp" namespace "dijkstra":
  cdef vector[OUT] dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity, uint32_t* voxel_graph
  )
  cdef vector[OUT] bidirectional_dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity, uint32_t* voxel_graph
  )
  cdef vector[OUT] value_target_dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, T target,
    int connectivity, 
    uint32_t* voxel_connectivity_graph
  )
  cdef vector[OUT] compass_guided_dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity, float normalizer,
    uint32_t* voxel_graph
  )
  cdef float* distance_field3d[T](
    T* field,
    size_t sx, size_t sy, size_t sz,
    vector[size_t] source, size_t connectivity,
    uint32_t* voxel_graph, size_t &max_loc
  )
  cdef OUT* parental_field3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, OUT* parents,
    int connectivity, uint32_t* voxel_graph
  )
  cdef float* euclidean_distance_field3d(
    uint8_t* field,
    size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    vector[size_t] source,  
    float free_space_radius,
    float* dist, 
    uint32_t* voxel_graph,
    size_t &max_loc
  )
  cdef vector[T] query_shortest_path[T](
    T* parents, T target
  ) 
  
def format_voxel_graph(voxel_graph):
  while voxel_graph.ndim < 3:
    voxel_graph = voxel_graph[..., np.newaxis]

  if not np.issubdtype(voxel_graph.dtype, np.uint32):
    voxel_graph = voxel_graph.astype(np.uint32, order="F")
  
  return np.asfortranarray(voxel_graph)

def dijkstra(
  data, source, target, 
  bidirectional=False, connectivity=26, 
  compass=False, compass_norm=-1,
  voxel_graph=None
):
  """
  Perform dijkstra's shortest path algorithm
  on a 3D image grid. Vertices are voxels and
  edges are the 26 nearest neighbors (except
  for the edges of the image where the number
  of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   Data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate of starting voxel
   target: (x,y,z) coordinate of target voxel
   bidirectional: If True, use more memory but conduct
    a bidirectional search, which has the potential to be 
    much faster.
   connectivity: 6 (faces), 18 (faces + edges), and 
    26 (faces + edges + corners) voxel graph connectivities 
    are supported. For 2D images, 4 gets translated to 6,
    8 gets translated to 18.
   compass: If True, A* search using the chessboard 
    distance to target as the heuristic. This has the 
    effect of guiding the search like using a compass.
    This option has no effect when bidirectional search
    is enabled as it is not supported yet.
   compass_norm: Allows you to manipulate the relative
    greed of the A* search. By default set to -1, which
    means the norm will be the field minimum, but you 
    can choose whatever you want if you know what you're
    doing.
   voxel_graph: a bitwise representation of the premitted
    directions of travel between voxels. Generated from
    cc3d.voxel_connectivity_graph. 
    (See https://github.com/seung-lab/connected-components-3d)
  
  Returns: 1D numpy array containing indices of the path from
    source to target including source and target.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6, 18, 26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32, order='F')

  _validate_coord(data, source)
  _validate_coord(data, target)

  if dims == 2:
    data = data[:, :, np.newaxis]
    source = list(source) + [ 0 ]
    target = list(target) + [ 0 ]

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)

  cdef size_t cols = data.shape[0]
  cdef size_t rows = data.shape[1]
  cdef size_t depth = data.shape[2]

  path = _execute_dijkstra(
    data, source, target, connectivity, 
    bidirectional, compass, compass_norm,
    voxel_graph
  )

  return _path_to_point_cloud(path, dims, rows, cols)

def railroad(
  data, source, 
  connectivity=26, voxel_graph=None
):
  """
  A "rail road" (a term defined by us) is the shortest
  last-mile path (the "road") from a target point to a 
  "rail network" that includes the source point. A "rail"
  is a zero weighted path that acts as a strong attractor
  for the search algorithm. In the context of the 
  skeletonization problem, such networks are assembled
  as a matter of course. It becomes more and more efficient
  to search for the rail network than for the target point.

  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   Data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate of starting voxel
   connectivity: 6 (faces), 18 (faces + edges), and 
    26 (faces + edges + corners) voxel graph connectivities 
    are supported. For 2D images, 4 gets translated to 6,
    8 gets translated to 18.
   voxel_graph: a bitwise representation of the premitted
    directions of travel between voxels. Generated from
    cc3d.voxel_connectivity_graph. 
    (See https://github.com/seung-lab/connected-components-3d)
  
  Returns: 1D numpy array containing indices of the path from
    source to target including source and destination.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6, 18, 26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32, order='F')

  _validate_coord(data, source)

  if dims == 2:
    data = data[:, :, np.newaxis]
    source = list(source) + [ 0 ]

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)

  cdef size_t cols = data.shape[0]
  cdef size_t rows = data.shape[1]

  path = _execute_value_target_dijkstra(
    data, source,
    connectivity, voxel_graph
  )

  return _path_to_point_cloud(path, dims, rows, cols)

def distance_field(
  data, source, connectivity=26, 
  voxel_graph=None, return_max_location=False
):
  """
  Use dijkstra's shortest path algorithm
  on a 3D image grid to generate a weighted 
  distance field from one or more source voxels. Vertices are 
  voxels and edges are the 26 nearest neighbors 
  (except for the edges of the image where 
  the number of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate or list of coordinates 
    of starting voxels.
   connectivity: 26, 18, or 6 connected.
   return_max_location: returns the coordinates of one
     of the possibly multiple maxima.
  
  Returns: 
    let field = 2D or 3D numpy array with each index
      containing its distance from the source voxel.

    if return_max_location:
      return (field, (x,y,z) of max distance)
    else:
      return field
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6, 18, 26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )
  
  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)

  source = np.array(source, dtype=np.uint64)
  if source.ndim == 1:
    source = source[np.newaxis, :]

  for src in source:
    _validate_coord(data, src)

  if source.shape[1] < 3:
    tmp = np.zeros((source.shape[0], 3), dtype=np.uint64)
    tmp[:, :source.shape[1]] = source[:,:]
    source = tmp

  while data.ndim < 3:
    data = data[..., np.newaxis]

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)

  field, max_loc = _execute_distance_field(data, source, connectivity, voxel_graph)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  if return_max_location:
    return field, np.unravel_index(max_loc, data.shape, order="F")[:dims]
  else:
    return field

# parents is either uint32 or uint64
def _path_from_parents_helper(cnp.ndarray[UINT, ndim=3] parents, target):
  cdef UINT[:,:,:] arr_memview
  cdef vector[UINT] path
  cdef UINT* path_ptr
  cdef UINT[:] vec_view

  cdef size_t sx = parents.shape[0]
  cdef size_t sy = parents.shape[1]
  cdef size_t sz = parents.shape[2]

  cdef UINT targ = target[0] + sx * (target[1] + sy * target[2])

  arr_memview = parents
  path = query_shortest_path(&arr_memview[0,0,0], targ)
  path_ptr = <UINT*>&path[0]
  vec_view = <UINT[:path.size()]>path_ptr
  buf = bytearray(vec_view[:])
  return np.frombuffer(buf, dtype=parents.dtype)[::-1]

def path_from_parents(parents, target):
  ndim = parents.ndim
  while parents.ndim < 3:
    parents = parents[..., np.newaxis]

  while len(target) < 3:
    target += (0,)

  cdef size_t sx = parents.shape[0]
  cdef size_t sy = parents.shape[1]

  numpy_path = _path_from_parents_helper(parents, target)
  ptlist = _path_to_point_cloud(numpy_path, 3, sy, sx)
  return ptlist[:, :ndim]

def parental_field(data, source, connectivity=26, voxel_graph=None):
  """
  Use dijkstra's shortest path algorithm
  on a 3D image grid to generate field of
  voxels that point to their parent voxel. 

  This is used to execute dijkstra's algorithm
  once, and then query different targets for
  the path they represent. 

  Parameters:
    data: Input weights in a 2D or 3D numpy array. 
    source: (x,y,z) coordinate of starting voxel
    connectivity: 6 (faces), 18 (faces + edges), and 
      26 (faces + edges + corners) voxel graph connectivities 
      are supported. For 2D images, 4 gets translated to 6,
      8 gets translated to 18.

  Returns: 2D or 3D numpy array with each index
    containing the index of its parent. The root
    of a path has index 0.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  if dims == 2:
    if connectivity == 4:
      connectivity = 6
    elif connectivity == 8:
      connectivity = 18 # or 26 but 18 might be faster

  if connectivity not in (6,18,26):
    raise ValueError(
      "Only 6, 18, and 26 connectivities are supported. Got: " + str(connectivity)
    )

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)

  if dims == 1:
    data = data[:, np.newaxis, np.newaxis]
    source = ( source[0], 0, 0 )
  if dims == 2:
    data = data[:, :, np.newaxis]
    source = ( source[0], source[1], 0 )

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  _validate_coord(data, source)

  data = np.asfortranarray(data)

  field = _execute_parental_field(data, source, connectivity, voxel_graph)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field

def euclidean_distance_field(
  data, source, anisotropy=(1,1,1), 
  free_space_radius=0, voxel_graph=None,
  return_max_location=False
):
  """
  Use dijkstra's shortest path algorithm
  on a 3D image grid to generate a weighted 
  euclidean distance field from a source voxel 
  from a binary image. Vertices are 
  voxels and edges are the 26 nearest neighbors 
  (except for the edges of the image where 
  the number of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate of starting voxel
   anisotropy: (wx,wy,wz) weights for each axial direction.
   free_space_radius: (float, optional) if you know that the 
    region surrounding the source is free space, we can use
    a much faster algorithm to fill in that volume. Value
    is physical radius (can get this from the EDT). 
   return_max_location: returns the coordinates of one
     of the possibly multiple maxima.
  
  Returns: 
    let field = 2D or 3D numpy array with each index
      containing its distance from the source voxel.

    if return_max_location:
      return (field, (x,y,z) of max distance)
    else:
      return field
  """
  dims = len(data.shape)
  if dims > 3:
    raise DimensionError(f"Only 2D and 3D image sources are supported. Got: {dims}")

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)
  
  source = np.array(source, dtype=np.uint64)
  if source.ndim == 1:
    source = source[np.newaxis, :]

  if source.shape[1] < 3:
    tmp = np.zeros((source.shape[0], 3), dtype=np.uint64)
    tmp[:, :source.shape[1]] = source[:,:]
    source = tmp

  while data.ndim < 3:
    data = data[..., np.newaxis]

  for src in source:
    _validate_coord(data, src)

  if voxel_graph is not None:
    voxel_graph = format_voxel_graph(voxel_graph)

  data = np.asfortranarray(data)

  field, max_loc = _execute_euclidean_distance_field(
    data, source, anisotropy, 
    free_space_radius, voxel_graph
  )
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  if return_max_location:
    return field, np.unravel_index(max_loc, data.shape, order="F")[:dims]
  else:
    return field

def _validate_coord(data, coord):
  dims = len(data.shape)

  if len(coord) != dims:
    raise IndexError(
      "Coordinates must have the same dimension as the data. coord: {}, data shape: {}"
        .format(coord, data.shape)
    )

  for i, size in enumerate(data.shape):
    if coord[i] < 0 or coord[i] >= size:
      raise IndexError("Selected voxel {} was not located inside the array.".format(coord))

def _path_to_point_cloud(path, dims, rows, cols):
  ptlist = np.zeros((path.shape[0], dims), dtype=np.uint32)

  cdef size_t sxy = rows * cols
  cdef size_t i = 0

  if dims == 3:
    for i, pt in enumerate(path):
      ptlist[ i, 0 ] = pt % cols
      ptlist[ i, 1 ] = (pt % sxy) / cols
      ptlist[ i, 2 ] = pt / sxy
  else:
    for i, pt in enumerate(path):
      ptlist[ i, 0 ] = pt % cols
      ptlist[ i, 1 ] = (pt % sxy) / cols

  return ptlist

def _execute_value_target_dijkstra(
  data, source, 
  int connectivity, voxel_graph=None
):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])

  cdef vector[uint32_t] output32
  cdef vector[uint64_t] output64

  sixtyfourbit = data.size > np.iinfo(np.uint32).max
  
  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[float, uint64_t](
        &arr_memviewfloat[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[float, uint32_t](
        &arr_memviewfloat[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[double, uint64_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[double, uint32_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[uint64_t, uint64_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[uint64_t, uint32_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[uint32_t, uint64_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[uint32_t, uint32_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[uint16_t, uint64_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[uint16_t, uint32_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    if sixtyfourbit:
      output64 = value_target_dijkstra3d[uint8_t, uint64_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )
    else:
      output32 = value_target_dijkstra3d[uint8_t, uint32_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, 0, connectivity,
        voxel_graph_ptr
      )

  cdef uint32_t* output_ptr32
  cdef uint64_t* output_ptr64

  cdef uint32_t[:] vec_view32
  cdef uint64_t[:] vec_view64

  if sixtyfourbit:
    output_ptr64 = <uint64_t*>&output64[0]
    if output64.size() == 0:
      return np.zeros((0,), dtype=np.uint64)
    vec_view64 = <uint64_t[:output64.size()]>output_ptr64
    buf = bytearray(vec_view64[:])
    output = np.frombuffer(buf, dtype=np.uint64)
  else:
    output_ptr32 = <uint32_t*>&output32[0]
    if output32.size() == 0:
      return np.zeros((0,), dtype=np.uint32)
    vec_view32 = <uint32_t[:output32.size()]>output_ptr32
    buf = bytearray(vec_view32[:])
    output = np.frombuffer(buf, dtype=np.uint32)

  return output[::-1]

def _execute_dijkstra(
  data, source, target, int connectivity, 
  bidirectional, compass, float compass_norm=-1,
  voxel_graph=None
):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])
  cdef size_t sink = target[0] + sx * (target[1] + sy * target[2])

  cdef vector[uint32_t] output32
  cdef vector[uint64_t] output64

  sixtyfourbit = data.size > np.iinfo(np.uint32).max
  
  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[float, uint64_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[float, uint64_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[float, uint64_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = bidirectional_dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
      else:
        output32 = compass_guided_dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm,
          voxel_graph_ptr
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )
      else:
        output32 = dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity,
          voxel_graph_ptr
        )

  cdef uint32_t* output_ptr32
  cdef uint64_t* output_ptr64

  cdef uint32_t[:] vec_view32
  cdef uint64_t[:] vec_view64

  if sixtyfourbit:
    output_ptr64 = <uint64_t*>&output64[0]
    if output64.size() == 0:
      return np.zeros((0,), dtype=np.uint64)
    vec_view64 = <uint64_t[:output64.size()]>output_ptr64
    buf = bytearray(vec_view64[:])
    output = np.frombuffer(buf, dtype=np.uint64)
  else:
    output_ptr32 = <uint32_t*>&output32[0]
    if output32.size() == 0:
      return np.zeros((0,), dtype=np.uint32)
    vec_view32 = <uint32_t[:output32.size()]>output_ptr32
    buf = bytearray(vec_view32[:])
    output = np.frombuffer(buf, dtype=np.uint32)

  if bidirectional:
    return output
  else:
    return output[::-1]

def _execute_distance_field(data, sources, connectivity, voxel_graph):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef vector[size_t] src
  for source in sources:
    src.push_back(source[0] + sx * (source[1] + sy * source[2]))

  cdef float* dist
  cdef size_t max_loc = data.size + 1

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    dist = distance_field3d[float](
      &arr_memviewfloat[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  elif dtype == np.float64:
    arr_memviewdouble = data
    dist = distance_field3d[double](
      &arr_memviewdouble[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    dist = distance_field3d[uint64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    dist = distance_field3d[uint32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    dist = distance_field3d[uint16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    dist = distance_field3d[uint8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz,
      src, connectivity,
      voxel_graph_ptr, max_loc
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef size_t voxels = sx * sy * sz
  cdef float[:] dist_view = <float[:voxels]>dist

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(dist_view[:])
  free(dist)
  buf = np.frombuffer(buf, dtype=np.float32).reshape(data.shape, order='F')
  return buf, max_loc

def _execute_parental_field(data, source, connectivity, voxel_graph):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])

  sixtyfourbit = data.size > np.iinfo(np.uint32).max

  cdef cnp.ndarray[uint32_t, ndim=3] parents32
  cdef cnp.ndarray[uint64_t, ndim=3] parents64

  if sixtyfourbit:
    parents64 = np.zeros( (sx,sy,sz), dtype=np.uint64, order='F' )
  else:
    parents32 = np.zeros( (sx,sy,sz), dtype=np.uint32, order='F' )

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    if sixtyfourbit:
      parental_field3d[float,uint64_t](
        &arr_memviewfloat[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[float,uint32_t](
        &arr_memviewfloat[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if sixtyfourbit:
      parental_field3d[double,uint64_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[double,uint32_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if sixtyfourbit:
      parental_field3d[uint64_t,uint64_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[uint64_t,uint32_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    if sixtyfourbit:
      parental_field3d[uint32_t,uint64_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[uint32_t,uint32_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if sixtyfourbit:
      parental_field3d[uint16_t,uint64_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[uint16_t,uint32_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  elif dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    if sixtyfourbit:
      parental_field3d[uint8_t,uint64_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
    else:
      parental_field3d[uint8_t,uint32_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity,
        voxel_graph_ptr
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  if sixtyfourbit:
    return parents64
  else:
    return parents32

def _execute_euclidean_distance_field(
  data, sources, anisotropy, float free_space_radius=0,
  voxel_graph=None
):
  cdef uint8_t[:,:,:] arr_memview8

  cdef uint32_t[:,:,:] voxel_graph_memview
  cdef uint32_t* voxel_graph_ptr = NULL
  if voxel_graph is not None:
    voxel_graph_memview = voxel_graph
    voxel_graph_ptr = <uint32_t*>&voxel_graph_memview[0,0,0]

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef float wx = anisotropy[0]
  cdef float wy = anisotropy[1]
  cdef float wz = anisotropy[2]

  cdef vector[size_t] src
  for source in sources:
    src.push_back(source[0] + sx * (source[1] + sy * source[2]))

  cdef cnp.ndarray[float, ndim=3] dist = np.zeros( (sx,sy,sz), dtype=np.float32, order='F' )

  dtype = data.dtype
  cdef size_t max_loc = data.size + 1

  if dtype in (np.int8, np.uint8, bool):
    arr_memview8 = data.astype(np.uint8)
    euclidean_distance_field3d(
      &arr_memview8[0,0,0],
      sx, sy, sz,
      wx, wy, wz,
      src, free_space_radius,
      &dist[0,0,0],
      voxel_graph_ptr,
      max_loc
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  if max_loc == data.size + 1:
    raise ValueError(f"Something went wrong during processing. max_loc: {max_loc}")

  return dist, max_loc

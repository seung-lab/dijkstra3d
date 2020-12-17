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

__VERSION__ = '1.7.0'

class DimensionError(Exception):
  pass

cdef extern from "dijkstra3d.hpp" namespace "dijkstra":
  cdef vector[OUT] dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity
  )
  cdef vector[OUT] bidirectional_dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity
  )
  cdef vector[OUT] compass_guided_dijkstra3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, size_t target,
    int connectivity, float normalizer
  )
  cdef float* distance_field3d[T](
    T* field,
    size_t sx, size_t sy, size_t sz,
    size_t source
  )
  cdef OUT* parental_field3d[T,OUT](
    T* field, 
    size_t sx, size_t sy, size_t sz, 
    size_t source, OUT* parents,
    int connectivity
  )
  cdef float* euclidean_distance_field3d(
    uint8_t* field,
    size_t sx, size_t sy, size_t sz,
    float wx, float wy, float wz,
    size_t source,  
    float free_space_radius,
    float* dist
  )
  cdef vector[T] query_shortest_path[T](
    T* parents, T target
  ) 
  
def dijkstra(
  data, source, target, 
  bidirectional=False, connectivity=26, 
  compass=False, compass_norm=-1
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

  data = np.asfortranarray(data)

  cdef size_t cols = data.shape[0]
  cdef size_t rows = data.shape[1]
  cdef size_t depth = data.shape[2]

  path = _execute_dijkstra(
    data, source, target, connectivity, 
    bidirectional, compass, compass_norm
  )

  return _path_to_point_cloud(path, dims, rows, cols)

def distance_field(data, source):
  """
  Use dijkstra's shortest path algorithm
  on a 3D image grid to generate a weighted 
  distance field from a source voxel. Vertices are 
  voxels and edges are the 26 nearest neighbors 
  (except for the edges of the image where 
  the number of edges is reduced).
  
  For given input voxels A and B, the edge
  weight from A to B is B and from B to A is
  A. All weights must be non-negative (incl. 
  negative zero).
  
  Parameters:
   Data: Input weights in a 2D or 3D numpy array. 
   source: (x,y,z) coordinate of starting voxel
  
  Returns: 2D or 3D numpy array with each index
    containing its distance from the source voxel.
  """
  dims = len(data.shape)
  if dims not in (2,3):
    raise DimensionError("Only 2D and 3D image sources are supported. Got: " + str(dims))

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)

  if dims == 1:
    data = data[:, np.newaxis, np.newaxis]
    source = ( source[0], 0, 0 )
  if dims == 2:
    data = data[:, :, np.newaxis]
    source = ( source[0], source[1], 0 )

  _validate_coord(data, source)

  data = np.asfortranarray(data)

  field = _execute_distance_field(data, source)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field

def path_from_parents(parents, target):
  cdef size_t sx = parents.shape[0]
  cdef size_t sy = parents.shape[1]
  cdef size_t sz = parents.shape[2]

  cdef size_t targ = target[0] + sx * (target[1] + sy * target[2])

  cdef uint32_t[:,:,:] arr_memview32
  cdef vector[uint32_t] path32
  cdef uint32_t* path_ptr32
  cdef uint32_t[:] vec_view32

  cdef uint64_t[:,:,:] arr_memview64
  cdef vector[uint64_t] path64
  cdef uint64_t* path_ptr64
  cdef uint64_t[:] vec_view64

  if parents.dtype == np.uint64:
    arr_memview64 = parents
    path64 = query_shortest_path[uint64_t](&arr_memview64[0,0,0], targ)
    path_ptr64 = <uint64_t*>&path64[0]
    vec_view64 = <uint64_t[:path64.size()]>path_ptr64
    buf = bytearray(vec_view64[:])
    numpy_path = np.frombuffer(buf, dtype=np.uint64)[::-1]
  else:
    arr_memview32 = parents
    path32 = query_shortest_path[uint32_t](&arr_memview32[0,0,0], targ)
    path_ptr32 = <uint32_t*>&path32[0]
    vec_view32 = <uint32_t[:path32.size()]>path_ptr32
    buf = bytearray(vec_view32[:])
    numpy_path = np.frombuffer(buf, dtype=np.uint32)[::-1]

  return _path_to_point_cloud(numpy_path, 3, sy, sx)

def parental_field(data, source, connectivity=26):
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

  _validate_coord(data, source)

  data = np.asfortranarray(data)

  field = _execute_parental_field(data, source, connectivity)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field

def euclidean_distance_field(data, source, anisotropy=(1,1,1), free_space_radius=0):
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
  
  Returns: 2D or 3D numpy array with each index
    containing its distance from the source voxel.
  """
  dims = len(data.shape)
  assert dims <= 3

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.float32)

  if dims == 1:
    data = data[:, np.newaxis, np.newaxis]
    source = ( source[0], 0, 0 )
  if dims == 2:
    data = data[:, :, np.newaxis]
    source = ( source[0], source[1], 0 )

  _validate_coord(data, source)

  data = np.asfortranarray(data)

  field = _execute_euclidean_distance_field(data, source, anisotropy, free_space_radius)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

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

def _execute_dijkstra(
  data, source, target, int connectivity, 
  bidirectional, compass, float compass_norm=-1
):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

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
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[float, uint64_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[float, uint64_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[float, uint32_t](
          &arr_memviewfloat[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[double, uint64_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[double, uint32_t](
          &arr_memviewdouble[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint64_t, uint64_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[uint64_t, uint32_t](
          &arr_memview64[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint32_t, uint64_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[uint32_t, uint32_t](
          &arr_memview32[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint16_t, uint64_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[uint16_t, uint32_t](
          &arr_memview16[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    if bidirectional:
      if sixtyfourbit:
        output64 = bidirectional_dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = bidirectional_dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
    elif compass:
      if sixtyfourbit:
        output64 = compass_guided_dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
      else:
        output32 = compass_guided_dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, 
          connectivity, compass_norm
        )
    else:
      if sixtyfourbit:
        output64 = dijkstra3d[uint8_t, uint64_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )
      else:
        output32 = dijkstra3d[uint8_t, uint32_t](
          &arr_memview8[0,0,0],
          sx, sy, sz,
          src, sink, connectivity
        )

  cdef uint32_t* output_ptr32
  cdef uint64_t* output_ptr64

  cdef uint32_t[:] vec_view32
  cdef uint64_t[:] vec_view64

  if sixtyfourbit:
    output_ptr64 = <uint64_t*>&output64[0]
    vec_view64 = <uint64_t[:output64.size()]>output_ptr64
    buf = bytearray(vec_view64[:])
    output = np.frombuffer(buf, dtype=np.uint64)
  else:
    output_ptr32 = <uint32_t*>&output32[0]
    vec_view32 = <uint32_t[:output32.size()]>output_ptr32
    buf = bytearray(vec_view32[:])
    output = np.frombuffer(buf, dtype=np.uint32)

  if bidirectional:
    return output
  else:
    return output[::-1]

def _execute_distance_field(data, source):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])

  cdef float* dist

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    dist = distance_field3d[float](
      &arr_memviewfloat[0,0,0],
      sx, sy, sz,
      src
    )
  elif dtype == np.float64:
    arr_memviewdouble = data
    dist = distance_field3d[double](
      &arr_memviewdouble[0,0,0],
      sx, sy, sz,
      src
    )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    dist = distance_field3d[uint64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz,
      src
    )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    dist = distance_field3d[uint32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz,
      src
    )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    dist = distance_field3d[uint16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz,
      src
    )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    dist = distance_field3d[uint8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz,
      src
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef size_t voxels = sx * sy * sz
  cdef float[:] dist_view = <float[:voxels]>dist

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(dist_view[:])
  free(dist)
  # I don't actually understand why order F works, but it does.
  return np.frombuffer(buf, dtype=np.float32).reshape(data.shape, order='F')

def _execute_parental_field(data, source, connectivity):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

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
        connectivity
      )
    else:
      parental_field3d[float,uint32_t](
        &arr_memviewfloat[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  elif dtype == np.float64:
    arr_memviewdouble = data
    if sixtyfourbit:
      parental_field3d[double,uint64_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity
      )
    else:
      parental_field3d[double,uint32_t](
        &arr_memviewdouble[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    if sixtyfourbit:
      parental_field3d[uint64_t,uint64_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity
      )
    else:
      parental_field3d[uint64_t,uint32_t](
        &arr_memview64[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    if sixtyfourbit:
      parental_field3d[uint32_t,uint64_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity
      )
    else:
      parental_field3d[uint32_t,uint32_t](
        &arr_memview32[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    if sixtyfourbit:
      parental_field3d[uint16_t,uint64_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity
      )
    else:
      parental_field3d[uint16_t,uint32_t](
        &arr_memview16[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    if sixtyfourbit:
      parental_field3d[uint8_t,uint64_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, &parents64[0,0,0],
        connectivity
      )
    else:
      parental_field3d[uint8_t,uint32_t](
        &arr_memview8[0,0,0],
        sx, sy, sz,
        src, &parents32[0,0,0],
        connectivity
      )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  if sixtyfourbit:
    return parents64
  else:
    return parents32

def _execute_euclidean_distance_field(data, source, anisotropy, float free_space_radius = 0):
  cdef uint8_t[:,:,:] arr_memview8

  cdef size_t sx = data.shape[0]
  cdef size_t sy = data.shape[1]
  cdef size_t sz = data.shape[2]

  cdef float wx = anisotropy[0]
  cdef float wy = anisotropy[1]
  cdef float wz = anisotropy[2]

  cdef size_t src = source[0] + sx * (source[1] + sy * source[2])

  cdef cnp.ndarray[float, ndim=3] dist = np.zeros( (sx,sy,sz), dtype=np.float32, order='F' )

  dtype = data.dtype

  if dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    euclidean_distance_field3d(
      &arr_memview8[0,0,0],
      sx, sy, sz,
      wx, wy, wz,
      src, free_space_radius,
      &dist[0,0,0]
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  return dist

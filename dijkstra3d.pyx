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

__VERSION__ = '1.2.0'

class MemoryOrderError(Exception):
  pass

cdef extern from "dijkstra3d.hpp" namespace "dijkstra":
  cdef vector[uint32_t] dijkstra3d[T](
    T* field, 
    int sx, int sy, int sz, 
    int source, int target
  )
  cdef vector[uint32_t] bidirectional_dijkstra3d[T](
    T* field, 
    int sx, int sy, int sz, 
    int source, int target
  )
  cdef float* distance_field3d[T](
    T* field,
    int sx, int sy, int sz,
    int source
  )
  cdef uint32_t* parental_field3d[T](
    T* field, 
    int sx, int sy, int sz, 
    int source, uint32_t* parents
  )
  cdef float* euclidean_distance_field3d(
    uint8_t* field,
    int sx, int sy, int sz,
    float wx, float wy, float wz,
    int source, float* dist
  )
  cdef vector[uint32_t] query_shortest_path(
    uint32_t* parents, uint32_t target
  ) 

def dijkstra(data, source, target, bidirectional=False):
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
  
  Returns: 1D numpy array containing indices of the path from
    source to target including source and target.
  """
  dims = len(data.shape)
  assert dims in (2, 3)

  if data.size == 0:
    return np.zeros(shape=(0,), dtype=np.uint32, order='F')

  _validate_coord(data, source)
  _validate_coord(data, target)

  if dims == 2:
    data = data[:, :, np.newaxis]
    source = list(source) + [ 0 ]
    target = list(target) + [ 0 ]

  data = np.asfortranarray(data)

  cdef int cols = data.shape[0]
  cdef int rows = data.shape[1]
  cdef int depth = data.shape[2]

  if bidirectional:
    path = _execute_bidirectional_dijkstra(data, source, target)
  else:
    path = _execute_dijkstra(data, source, target)
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

  field = _execute_distance_field(data, source)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field

def path_from_parents(parents, target):
  cdef int sx = parents.shape[0]
  cdef int sy = parents.shape[1]
  cdef int sz = parents.shape[2]

  cdef int targ = target[0] + sx * (target[1] + sy * target[2])

  cdef uint32_t[:,:,:] arr_memview32 = parents

  cdef vector[uint32_t] path = query_shortest_path(&arr_memview32[0,0,0], targ)
  cdef uint32_t* path_ptr = <uint32_t*>&path[0]
  cdef uint32_t[:] vec_view = <uint32_t[:path.size()]>path_ptr

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(vec_view[:])
  numpy_path = np.frombuffer(buf, dtype=np.uint32)[::-1]
  return _path_to_point_cloud(numpy_path, 3, sy, sx)

def parental_field(data, source):
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
  
  Returns: 2D or 3D numpy array with each index
    containing the index of its parent. The root
    of a path has index 0.
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

  if not np.isfortran(data):
    raise MemoryOrderError("Input array must be Fortran ordered.")

  field = _execute_parental_field(data, source)
  if dims < 3:
    field = np.squeeze(field, axis=2)
  if dims < 2:
    field = np.squeeze(field, axis=1)

  return field

def euclidean_distance_field(data, source, anisotropy=(1,1,1)):
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

  field = _execute_euclidean_distance_field(data, source, anisotropy)
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

  cdef int sxy = rows * cols

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

def _execute_dijkstra(data, source, target):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef int sx = data.shape[0]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[2]

  cdef int src = source[0] + sx * (source[1] + sy * source[2])
  cdef int sink = target[0] + sx * (target[1] + sy * target[2])

  cdef vector[uint32_t] output

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    output = dijkstra3d[float](
      &arr_memviewfloat[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype == np.float64:
    arr_memviewdouble = data
    output = dijkstra3d[double](
      &arr_memviewdouble[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    output = dijkstra3d[uint64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    output = dijkstra3d[uint32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    output = dijkstra3d[uint16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    output = dijkstra3d[uint8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz,
      src, sink
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef uint32_t* output_ptr = <uint32_t*>&output[0]
  cdef uint32_t[:] vec_view = <uint32_t[:output.size()]>output_ptr

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(vec_view[:])
  return np.frombuffer(buf, dtype=np.uint32)[::-1]

def _execute_bidirectional_dijkstra(data, source, target):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef int sx = data.shape[0]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[2]

  cdef int src = source[0] + sx * (source[1] + sy * source[2])
  cdef int sink = target[0] + sx * (target[1] + sy * target[2])

  cdef vector[uint32_t] output

  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    output = bidirectional_dijkstra3d[float](
      &arr_memviewfloat[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype == np.float64:
    arr_memviewdouble = data
    output = bidirectional_dijkstra3d[double](
      &arr_memviewdouble[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    output = bidirectional_dijkstra3d[uint64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int32, np.uint32):
    arr_memview32 = data.astype(np.uint32)
    output = bidirectional_dijkstra3d[uint32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    output = bidirectional_dijkstra3d[uint16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz,
      src, sink
    )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    output = bidirectional_dijkstra3d[uint8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz,
      src, sink
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  cdef uint32_t* output_ptr = <uint32_t*>&output[0]
  cdef uint32_t[:] vec_view = <uint32_t[:output.size()]>output_ptr

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(vec_view[:])
  return np.frombuffer(buf, dtype=np.uint32)

def _execute_distance_field(data, source):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef int sx = data.shape[0]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[2]

  cdef int src = source[0] + sx * (source[1] + sy * source[2])

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

  cdef int voxels = sx * sy * sz
  cdef float[:] dist_view = <float[:voxels]>dist

  # This construct is required by python 2.
  # Python 3 can just do np.frombuffer(vec_view, ...)
  buf = bytearray(dist_view[:])
  free(dist)
  # I don't actually understand why order F works, but it does.
  return np.frombuffer(buf, dtype=np.float32).reshape(data.shape, order='F')


def _execute_parental_field(data, source):
  cdef uint8_t[:,:,:] arr_memview8
  cdef uint16_t[:,:,:] arr_memview16
  cdef uint32_t[:,:,:] arr_memview32
  cdef uint64_t[:,:,:] arr_memview64
  cdef float[:,:,:] arr_memviewfloat
  cdef double[:,:,:] arr_memviewdouble

  cdef int sx = data.shape[0]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[2]

  cdef int src = source[0] + sx * (source[1] + sy * source[2])

  cdef cnp.ndarray[uint32_t, ndim=3] parents = np.zeros( (sx,sy,sz), dtype=np.uint32, order='F' )
  dtype = data.dtype

  if dtype == np.float32:
    arr_memviewfloat = data
    parental_field3d[float](
      &arr_memviewfloat[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  elif dtype == np.float64:
    arr_memviewdouble = data
    parental_field3d[double](
      &arr_memviewdouble[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  elif dtype in (np.int64, np.uint64):
    arr_memview64 = data.astype(np.uint64)
    parental_field3d[uint64_t](
      &arr_memview64[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  elif dtype in (np.uint32, np.int32):
    arr_memview32 = data.astype(np.uint32)
    parental_field3d[uint32_t](
      &arr_memview32[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  elif dtype in (np.int16, np.uint16):
    arr_memview16 = data.astype(np.uint16)
    parental_field3d[uint16_t](
      &arr_memview16[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  elif dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    parental_field3d[uint8_t](
      &arr_memview8[0,0,0],
      sx, sy, sz,
      src, &parents[0,0,0]
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  return parents

def _execute_euclidean_distance_field(data, source, anisotropy):
  cdef uint8_t[:,:,:] arr_memview8

  cdef int sx = data.shape[0]
  cdef int sy = data.shape[1]
  cdef int sz = data.shape[2]

  cdef float wx = anisotropy[0]
  cdef float wy = anisotropy[1]
  cdef float wz = anisotropy[2]

  cdef int src = source[0] + sx * (source[1] + sy * source[2])

  cdef cnp.ndarray[float, ndim=3] dist = np.zeros( (sx,sy,sz), dtype=np.float32, order='F' )

  dtype = data.dtype

  if dtype in (np.int8, np.uint8, np.bool):
    arr_memview8 = data.astype(np.uint8)
    euclidean_distance_field3d(
      &arr_memview8[0,0,0],
      sx, sy, sz,
      wx, wy, wz,
      src, &dist[0,0,0]
    )
  else:
    raise TypeError("Type {} not currently supported.".format(dtype))

  return dist

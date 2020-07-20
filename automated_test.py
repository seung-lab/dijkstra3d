import pytest 

import dijkstra3d
import numpy as np
from math import sqrt

TEST_TYPES = (
  np.float32, np.float64,
  np.uint64, np.uint32, np.uint16, np.uint8,
  np.int64, np.int32, np.int16, np.int8,
  np.bool
)

@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("connectivity", [ 18, 26 ])
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra2d_10x10_26(dtype, bidirectional, connectivity, compass):
  values = np.ones((10,10,1), dtype=dtype)

  path = dijkstra3d.dijkstra(
    values, (1,1,0), (1,1,0), 
    bidirectional=bidirectional, connectivity=connectivity,
    compass=compass
  )
  assert len(path) == 1
  assert np.all(path == np.array([ [1,1,0] ]))
  
  path = dijkstra3d.dijkstra(
    values, (0,0,0), (3,0,0), 
    bidirectional=bidirectional, connectivity=connectivity,
    compass=compass
  )

  assert len(path) == 4
  assert np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [2,1,0],
    [3,0,0],
  ])) or np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [2,0,0],
    [3,0,0],    
  ])) or np.all(path == np.array([
    [0,0,0],
    [1,0,0],
    [2,1,0],
    [3,0,0],    
  ])) or np.all(path == np.array([
    [0,0,0],
    [1,0,0],
    [2,0,0],
    [3,0,0],    
  ])) 

  path = dijkstra3d.dijkstra(
    values, (0,0,0), (5,5,0), 
    bidirectional=bidirectional, connectivity=connectivity,
    compass=compass
  )

  assert len(path) == 6
  assert np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [2,2,0],
    [3,3,0],
    [4,4,0],
    [5,5,0],
  ]))

  path = dijkstra3d.dijkstra(
    values, (0,0,0), (9,9,0), 
    bidirectional=bidirectional, connectivity=connectivity,
    compass=compass
  )
  
  assert len(path) == 10
  assert np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [2,2,0],
    [3,3,0],
    [4,4,0],
    [5,5,0],
    [6,6,0],
    [7,7,0],
    [8,8,0],
    [9,9,0]
  ]))

  path = dijkstra3d.dijkstra(values, (2,1,0), (3,0,0), compass=compass)

  assert len(path) == 2
  assert np.all(path == np.array([
    [2,1,0],
    [3,0,0],
  ]))

  path = dijkstra3d.dijkstra(values, (9,9,0), (5,5,0), compass=compass)

  assert len(path) == 5
  assert np.all(path == np.array([
    [9,9,0],
    [8,8,0],
    [7,7,0],
    [6,6,0],
    [5,5,0],
  ]))

# There are many more equal distance paths 
# for 6 connected... so we have to be less specific.
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra2d_10x10_6(dtype, bidirectional, compass):
  values = np.ones((10,10,1), dtype=dtype)

  path = dijkstra3d.dijkstra(
    values, (1,1,0), (1,1,0), 
    bidirectional=bidirectional, connectivity=6,
    compass=compass
  )
  assert len(path) == 1
  assert np.all(path == np.array([ [1,1,0] ]))
  
  path = dijkstra3d.dijkstra(
    values, (0,0,0), (3,0,0), 
    bidirectional=bidirectional, connectivity=6,
    compass=compass
  )
  
  assert len(path) == 4
  assert np.all(path == np.array([
    [0,0,0],
    [1,0,0],
    [2,0,0],
    [3,0,0],
  ]))

@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra2d_10x10_off_origin(bidirectional, dtype, compass):
  values = np.ones((10,10,1), dtype=dtype)
  
  path = dijkstra3d.dijkstra(
    values, (2,0,0), (3,0,0), 
    bidirectional=bidirectional, compass=compass
  )

  assert len(path) == 2
  assert np.all(path == np.array([
    [2,0,0],
    [3,0,0],
  ]))

  path = dijkstra3d.dijkstra(
    values, (2,1,0), (3,0,0), 
    bidirectional=bidirectional, compass=compass
  )

  assert len(path) == 2
  assert np.all(path == np.array([
    [2,1,0],
    [3,0,0],
  ]))

  path = dijkstra3d.dijkstra(
    values, (9,9,0), (5,5,0), 
    bidirectional=bidirectional, compass=compass
  )

  assert len(path) == 5
  assert np.all(path == np.array([
    [9,9,0],
    [8,8,0],
    [7,7,0],
    [6,6,0],
    [5,5,0],
  ]))

@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra3d_3x3x3_26(bidirectional, dtype, compass):
  values = np.ones((3,3,3), dtype=dtype)

  path = dijkstra3d.dijkstra(
    values, (1,1,1), (1,1,1), 
    bidirectional=bidirectional, connectivity=26,
    compass=compass
  )
  assert len(path) == 1
  assert np.all(path == np.array([ [1,1,1] ]))

  path = dijkstra3d.dijkstra(
    values, (0,0,0), (2,2,2), 
    bidirectional=bidirectional, connectivity=26
  )

  assert np.all(path == np.array([
    [0,0,0],
    [1,1,1],
    [2,2,2]
  ]))

  path = dijkstra3d.dijkstra(
    values, (2,2,2), (0,0,0), 
    bidirectional=bidirectional, connectivity=26,
    compass=compass
  )
  assert np.all(path == np.array([
    [2,2,2],
    [1,1,1],
    [0,0,0]
  ]))

@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra3d_3x3x3_18(bidirectional, dtype, compass):
  values = np.ones((3,3,3), dtype=dtype)

  path = dijkstra3d.dijkstra(
    values, (1,1,1), (1,1,1), 
    bidirectional=bidirectional, connectivity=18,
    compass=compass
  )
  assert len(path) == 1
  assert np.all(path == np.array([ [1,1,1] ]))

  path = dijkstra3d.dijkstra(
    values, (0,0,0), (2,2,2), 
    bidirectional=bidirectional, connectivity=18,
    compass=compass
  )
  assert np.all(path == np.array([
    [0,0,0],
    [1,0,1],
    [1,1,2],
    [2,2,2],
  ])) or np.all(path == np.array([
    [0,0,0],
    [0,1,1],
    [1,1,2],
    [2,2,2],
  ])) or np.all(path == np.array([
    [0,0,0],
    [0,1,1],
    [1,2,1],
    [2,2,2],
  ])) or np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [1,2,1],
    [2,2,2],
  ]))

  path = dijkstra3d.dijkstra(
    values, (2,2,2), (0,0,0), 
    bidirectional=bidirectional, connectivity=18,
    compass=compass
  )
  assert np.all(path == np.array([
    [2,2,2],
    [1,2,1],
    [1,1,0],
    [0,0,0]
  ])) or np.all(path == np.array([
    [2,2,2],
    [1,2,1],
    [0,1,1],
    [0,0,0]
  ])) or np.all(path == np.array([
    [2,2,2],
    [2,1,1],
    [1,0,1],
    [0,0,0]
  ])) or np.all(path == np.array([
    [2,2,2],
    [1,1,2],
    [1,0,1],
    [0,0,0]
  ])) or np.all(path == np.array([
    [2,2,2],
    [1,1,2],
    [1,1,0],
    [0,0,0]
  ])) or np.all(path == np.array([
    [2,2,2],
    [2,1,1],
    [1,1,0],
    [0,0,0]
  ]))

@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra3d_3x3x3_6(bidirectional, dtype, compass):
  values = np.ones((3,3,3), dtype=dtype)

  path = dijkstra3d.dijkstra(
    values, (1,1,1), (1,1,1), 
    bidirectional=bidirectional, connectivity=6,
    compass=compass
  )
  assert len(path) == 1
  assert np.all(path == np.array([ [1,1,1] ]))

  path = dijkstra3d.dijkstra(
    values, (0,0,0), (2,2,2), 
    bidirectional=bidirectional, connectivity=6,
    compass=compass
  )
  assert len(path) == 7
  assert tuple(path[0]) == (0,0,0)
  assert tuple(path[-1]) == (2,2,2)

  path = dijkstra3d.dijkstra(
    values, (2,2,2), (0,0,0), 
    bidirectional=bidirectional, connectivity=6,
    compass=compass
  )
  assert len(path) == 7
  assert tuple(path[0]) == (2,2,2)
  assert tuple(path[-1]) == (0,0,0)

def test_bidirectional():
  x = 20000
  values = np.array([
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
    [x, 1, x, x, x, 6, x, x, x, x],
    [x, x, 1, x, 4, x, 7, x, x, x], # two paths: cost 22, length 8
    [x, x, x, 1, 8, x, 1, x, x, x], #            cost 23, length 9
    [x, x, x, x, x, 8, x, 1, x, x],
    [x, x, x, x, x, x, x, x, 1, x],
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
  ])

  path_reg = dijkstra3d.dijkstra(np.asfortranarray(values), (3,1), (7, 8), bidirectional=False)
  path_bi = dijkstra3d.dijkstra(np.asfortranarray(values), (3,1), (7, 8), bidirectional=True)

  print(path_reg)
  print(path_bi)

  assert np.all(path_reg == path_bi)

  assert len(path_bi) == 8
  assert np.all(path_bi == [
    [3,1],
    [4,2],
    [5,3],
    [5,4], # critical
    [6,5], 
    [5,6],
    [6,7],
    [7,8]
  ])

  values = np.array([
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
    [x, x, x, 6, 6, 6, 6, x, x, x],
    [1, 1, 1, x, x, x, x, 6, 6, 6], # 42, 45
    [x, x, 9, x, x, x, 7, x, x, x],
    [x, x, 1, x, x, x, 1, x, x, x],
    [x, x, 1, x, x, x, 1, x, x, x],
    [x, x, 1, 1, 1, 1, 1, x, x, x],
    [x, x, x, x, x, x, x, x, x, x],
  ])

  path_reg = dijkstra3d.dijkstra(np.asfortranarray(values), (4,0), (4, 9), bidirectional=False)
  path_bi = dijkstra3d.dijkstra(np.asfortranarray(values), (4,0), (4, 9), bidirectional=True)

  print(path_reg)
  print(path_bi)

  assert np.all(path_reg == path_bi)

  assert len(path_bi) == 14
  assert np.all(path_bi == [
    [4,0],
    [4,1],
    # [4,2],
    [5,2], 
    [6,2], 
    [7,2],
    # [8,2],
    [8,3],
    [8,4],
    [8,5],
    # [8,6],
    [7,6],
    [6,6],
    [5,6],
    [4,7],
    [4,8],
    [4,9]
  ])



@pytest.mark.parametrize("bidirectional", [ False, True ])
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra_2d_loop(bidirectional, compass):
  x = 20000
  values = np.array([
    [x, x, x, x, x, x, 0, x, x, x],
    [x, x, x, x, x, x, 0, x, x, x],
    [x, x, 1, x, 0, 0, 0, x, x, x],
    [x, x, 2, x, 0, x, 0, x, x, x],
    [x, 0, x, 3, x, x, 0, x, x, x],
    [x, 0, x, 4, 0, 0, 0, x, x, x],
    [x, 0, x, 5, x, x, x, x, x, x],
    [x, 0, x, 6, x, x, x, x, x, x],
    [x, 0, x, 7, x, x, x, x, x, x],
    [x, x, x, 1, 8, 9,10, x, x, x],
    [x, x, x, 4, x, x,11,12, x, x],
    [x, x, x, x, x, x, x, x,13,14],
  ], order='F')

  path = dijkstra3d.dijkstra(values, (2,2), (11, 9), bidirectional=bidirectional, compass=compass)
  correct_path = np.array([
    [2, 2],
    [3, 2],
    [4, 3],
    [5, 4],
    [6, 3],
    [7, 3],
    [8, 3],
    [9, 4],
    [9, 5],
    [9, 6],
    [10, 7],
    [11, 8],
    [11, 9]
  ])

  assert np.all(path == correct_path)

@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_distance_field_2d(dtype):
  values = np.ones((5,5), dtype=dtype)
  
  field = dijkstra3d.distance_field(values, (0,0))

  assert np.all(field == np.array([
    [
      [0, 1, 2, 3, 4],
      [1, 1, 2, 3, 4],
      [2, 2, 2, 3, 4],
      [3, 3, 3, 3, 4],
      [4, 4, 4, 4, 4],
    ]
  ]))

  field = dijkstra3d.distance_field(values, (4,4))

  assert np.all(field == np.array([
    [
      [4, 4, 4, 4, 4],
      [4, 3, 3, 3, 3],
      [4, 3, 2, 2, 2],
      [4, 3, 2, 1, 1],
      [4, 3, 2, 1, 0],
    ]
  ]))

  field = dijkstra3d.distance_field(values, (2,2))

  assert np.all(field == np.array([
    [
      [2, 2, 2, 2, 2],
      [2, 1, 1, 1, 2],
      [2, 1, 0, 1, 2],
      [2, 1, 1, 1, 2],
      [2, 2, 2, 2, 2],
    ]
  ]))


  field = dijkstra3d.distance_field(values * 2, (2,2))

  assert np.all(field == np.array([
    [
      [4, 4, 4, 4, 4],
      [4, 2, 2, 2, 4],
      [4, 2, 0, 2, 4],
      [4, 2, 2, 2, 4],
      [4, 4, 4, 4, 4],
    ]
  ]))

@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_distance_field_2d_asymmetric(dtype):
  values = np.ones((5, 10), dtype=dtype)

  assert np.all(field == np.array([
    [
      [0, 1, 2, 3, 4],
      [1, 1, 2, 3, 4],
      [2, 2, 2, 3, 4],
      [3, 3, 3, 3, 4],
      [4, 4, 4, 4, 4],
    ]
  ]))

  field = dijkstra3d.distance_field(values, (4,4))

  assert np.all(field == np.array([
    [
      [4, 4, 4, 4, 4],
      [4, 3, 3, 3, 3],
      [4, 3, 2, 2, 2],
      [4, 3, 2, 1, 1],
      [4, 3, 2, 1, 0],
    ]
  ]))

  field = dijkstra3d.distance_field(values, (2,2))

  assert np.all(field == np.array([
    [
      [2, 2, 2, 2, 2],
      [2, 1, 1, 1, 2],
      [2, 1, 0, 1, 2],
      [2, 1, 1, 1, 2],
      [2, 2, 2, 2, 2],
    ]
  ]))

  field = dijkstra3d.distance_field(values * 2, (2,2))

  assert np.all(field == np.array([
    [
      [4, 4, 4, 4, 4],
      [4, 2, 2, 2, 4],
      [4, 2, 0, 2, 4],
      [4, 2, 2, 2, 4],
      [4, 4, 4, 4, 4],
    ]
  ]))

@pytest.mark.parametrize("dtype", TEST_TYPES)
def test_distance_field_2d_asymmetric(dtype):
  values = np.ones((5, 10), dtype=dtype)

  answer = np.array([
    [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    [1, 1, 1, 2, 3, 4, 5, 6, 7, 8],
    [2, 2, 2, 2, 3, 4, 5, 6, 7, 8],
    [3, 3, 3, 3, 3, 4, 5, 6, 7, 8],
    [4, 4, 4, 4, 4, 4, 5, 6, 7, 8],
  ], dtype=np.float32)

  field = dijkstra3d.distance_field(values, (0,1))
  assert np.all(field == answer)

@pytest.mark.parametrize('free_space_radius', (0,1,2,3,4,5,10))
def test_euclidean_distance_field_2d(free_space_radius):
  values = np.ones((2, 2), dtype=bool)

  sq2 = sqrt(2)
  sq3 = sqrt(3)

  answer = np.array([
    [0, 1],
    [1, sq2],
  ], dtype=np.float32)

  field = dijkstra3d.euclidean_distance_field(values, (0,0), free_space_radius=free_space_radius)
  assert np.all(np.abs(field - answer) < 0.00001)

  values = np.ones((5, 5), dtype=bool)

  answer = np.array(
    [[0,        1.        , 2.       , 3.        , 4.       ],
     [1,        1.4142135 , 2.4142137, 3.4142137 , 4.4142137],
     [2,        2.4142137 , 2.828427 , 3.828427  , 4.8284273],
     [3,        3.4142137 , 3.828427 , 4.2426405 , 5.2426405],
     [4,        4.4142137 , 4.8284273, 5.2426405 , 5.656854 ]], 
  dtype=np.float32)

  field = dijkstra3d.euclidean_distance_field(values, (0,0,0), (1,1,1), free_space_radius=free_space_radius)
  assert np.all(np.abs(field - answer) < 0.00001) 

  answer = np.array([
    [
      [0, 1],
      [1, sq2],
    ],
    [
      [1,   sq2],
      [sq2, sq3],
    ]
  ], dtype=np.float32)

  values = np.ones((2, 2, 2), dtype=bool)
  field = dijkstra3d.euclidean_distance_field(values, (0,0,0), (1,1,1), free_space_radius=free_space_radius)
  assert np.all(np.abs(field - answer) < 0.00001)   

  values = np.ones((2, 2, 2), dtype=bool)
  field = dijkstra3d.euclidean_distance_field(values, (1,1,1), (1,1,1), free_space_radius=free_space_radius)

  answer = np.array([
    [
      [sq3, sq2],
      [sq2, 1],
    ],
    [
      [sq2,   1],
      [1, 0],
    ]
  ], dtype=np.float32)  

  assert np.all(np.abs(field - answer) < 0.00001)   

@pytest.mark.parametrize('point', (np.random.randint(0,256, size=(3,)),))
def test_euclidean_distance_field_3d_free_space_eqn(point):
  point = tuple(point)
  print(point)
  values = np.ones((256, 256, 256), dtype=bool)
  field_dijk = dijkstra3d.euclidean_distance_field(values, point, free_space_radius=0) # free space off
  field_free = dijkstra3d.euclidean_distance_field(values, point, free_space_radius=10000) # free space 100% on

  assert np.all(np.abs(field_free - field_dijk) < 0.001) # there's some difference below this

def test_compass():
  field = np.array([
   [6, 9, 7, 7, 1, 7, 4, 3, 5, 9],
   [4, 8, 7, 8, 1, 2, 5, 8, 3, 9],
   [5, 9, 4, 5, 7, 9, 2, 1, 5, 1],
   [1, 3, 6, 9, 6, 1, 7, 9, 5, 8],
   [2, 7, 3, 6, 1, 8, 9, 2, 1, 5],
   [7, 3, 7, 2, 9, 9, 8, 8, 9, 6],
   [3, 3, 8, 9, 3, 6, 8, 1, 6, 4],
   [9, 7, 5, 7, 9, 7, 8, 6, 7, 2],
   [6, 3, 7, 1, 1, 5, 2, 1, 3, 9],
   [2, 4, 8, 2, 9, 5, 2, 3, 3, 2],
  ])
  start = (8,1)
  target = (1,5)
  dijkstra_path = dijkstra3d.dijkstra(field, start, target, compass=False)
  compass_path = dijkstra3d.dijkstra(field, start, target, compass=True)

  def path_len(path):
    length = 0
    for p in path:
      length += field[tuple(p)]
    return length

  if not np.all(dijkstra_path == compass_path):
    print(field)
    print(dijkstra_path)
    print("dijkstra cost: %d" % path_len(dijkstra_path))
    print(compass_path)
    print("compass cost: %d" % path_len(compass_path))

  assert np.all(dijkstra_path == compass_path)

@pytest.mark.parametrize("dtype", TEST_TYPES)
@pytest.mark.parametrize("compass", [ False, True ])
def test_dijkstra_parental(dtype, compass):
  values = np.ones((10,10,1), dtype=dtype, order='F')
  
  parents = dijkstra3d.parental_field(values, (0,0,0))
  path = dijkstra3d.path_from_parents(parents, (3,0,0))

  assert len(path) == 4
  assert np.all(path == np.array([
    [0,0,0],
    [1,1,0],
    [2,1,0],
    [3,0,0],
  ]))

  def path_len(path, values):
    length = 0
    for p in path:
      length += values[tuple(p)]
    return length

  # Symmetric Test
  for _ in range(500):
    values = np.random.randint(1,10, size=(10,10,1))
    values = np.asfortranarray(values)

    start = np.random.randint(0,9, size=(3,))
    target = np.random.randint(0,9, size=(3,))
    start[2] = 0
    target[2] = 0

    parents = dijkstra3d.parental_field(values, start)
    path = dijkstra3d.path_from_parents(parents, target)

    path_orig = dijkstra3d.dijkstra(values, start, target, compass=compass)

    if path_len(path, values) != path_len(path_orig, values):
      print(start, target)
      print(path)
      print(path_orig)
      print(values[:,:,0])
      print('parents_path')
      for p in path:
        print(values[tuple(p)])
      print('compass_path')
      for p in path_orig:
        print(values[tuple(p)])

    assert path_len(path, values) == path_len(path_orig, values)

    if compass == False:
      assert np.all(path == path_orig)

  # Asymmetric Test
  for _ in range(500):
    values = np.random.randint(1,255, size=(11,10,10))
    values = np.asfortranarray(values)

    start = np.random.randint(0,9, size=(3,))
    target = np.random.randint(0,9, size=(3,))
    start[0] = np.random.randint(0,10)
    target[0] = np.random.randint(0,10)

    parents = dijkstra3d.parental_field(values, start)
    path = dijkstra3d.path_from_parents(parents, target)

    path_orig = dijkstra3d.dijkstra(values, start, target, compass=compass)

    if path_len(path, values) != path_len(path_orig, values):
      print(start, target)
      print(path)
      print(path_orig)

    assert path_len(path, values) == path_len(path_orig, values)

    if compass == False:
      assert np.all(path == path_orig)

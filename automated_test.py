import dijkstra3d
import numpy as np
from math import sqrt

TEST_TYPES = (
  np.float32, np.float64,
  np.uint64, np.uint32, np.uint16, np.uint8,
  np.int64, np.int32, np.int16, np.int8,
  np.bool
)

def test_dijkstra2d_10x10():
  for dtype in TEST_TYPES:
    values = np.ones((10,10,1), dtype=dtype)

    path = dijkstra3d.dijkstra(values, (1,1,0), (1,1,0))
    assert len(path) == 1
    assert np.all(path == np.array([ [1,1,0] ]))
    
    path = dijkstra3d.dijkstra(values, (0,0,0), (3,0,0))

    assert len(path) == 4
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,0],
      [2,1,0],
      [3,0,0],
    ]))

    path = dijkstra3d.dijkstra(values, (0,0,0), (5,5,0))

    assert len(path) == 6
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,0],
      [2,2,0],
      [3,3,0],
      [4,4,0],
      [5,5,0],
    ]))

    path = dijkstra3d.dijkstra(values, (0,0,0), (9,9,0))
    
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

def test_dijkstra2d_10x10_off_origin():
  for dtype in TEST_TYPES:
    values = np.ones((10,10,1), dtype=dtype)
    
    path = dijkstra3d.dijkstra(values, (2,0,0), (3,0,0))

    assert len(path) == 2
    assert np.all(path == np.array([
      [2,0,0],
      [3,0,0],
    ]))

    path = dijkstra3d.dijkstra(values, (2,1,0), (3,0,0))

    assert len(path) == 2
    assert np.all(path == np.array([
      [2,1,0],
      [3,0,0],
    ]))

    path = dijkstra3d.dijkstra(values, (9,9,0), (5,5,0))

    assert len(path) == 5
    assert np.all(path == np.array([
      [9,9,0],
      [8,8,0],
      [7,7,0],
      [6,6,0],
      [5,5,0],
    ]))

def test_dijkstra3d_3x3x3():
  for dtype in TEST_TYPES:
    values = np.ones((3,3,3), dtype=dtype)

    path = dijkstra3d.dijkstra(values, (1,1,1), (1,1,1))
    assert len(path) == 1
    assert np.all(path == np.array([ [1,1,1] ]))

    path = dijkstra3d.dijkstra(values, (0,0,0), (2,2,2))
    assert np.all(path == np.array([
      [0,0,0],
      [1,1,1],
      [2,2,2]
    ]))

    path = dijkstra3d.dijkstra(values, (2,2,2), (0,0,0))
    assert np.all(path == np.array([
      [2,2,2],
      [1,1,1],
      [0,0,0]
    ]))

def test_dijkstra_2d_loop():
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
  ])

  path = dijkstra3d.dijkstra(np.asfortranarray(values), (2,2), (11, 9))
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


def test_distance_field_2d():
  for dtype in TEST_TYPES:
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

def test_distance_field_2d_asymmetric():
  for dtype in TEST_TYPES:
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


def test_euclidean_distance_field_2d():
  values = np.ones((2, 2), dtype=bool)

  sq2 = sqrt(2)
  sq3 = sqrt(3)

  answer = np.array([
    [0, 1],
    [1, sq2],
  ], dtype=np.float32)

  field = dijkstra3d.euclidean_distance_field(values, (0,0))
  assert np.all(np.abs(field - answer) < 0.00001)

  values = np.ones((5, 5), dtype=bool)

  answer = np.array(
    [[0,        1.        , 2.       , 3.        , 4.       ],
     [1,        1.4142135 , 2.4142137, 3.4142137 , 4.4142137],
     [2,        2.4142137 , 2.828427 , 3.828427  , 4.8284273],
     [3,        3.4142137 , 3.828427 , 4.2426405 , 5.2426405],
     [4,        4.4142137 , 4.8284273, 5.2426405 , 5.656854 ]], 
  dtype=np.float32)

  field = dijkstra3d.euclidean_distance_field(values, (0,0,0), (1,1,1))
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
  field = dijkstra3d.euclidean_distance_field(values, (0,0,0), (1,1,1))
  assert np.all(np.abs(field - answer) < 0.00001)   

  values = np.ones((2, 2, 2), dtype=bool)
  field = dijkstra3d.euclidean_distance_field(values, (1,1,1), (1,1,1))

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


def test_dijkstra_parental():
  for dtype in TEST_TYPES:
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

    # Symmetric Test
    for _ in range(50):
      values = np.random.randint(1,255, size=(10,10,10))
      values = np.asfortranarray(values)

      start = np.random.randint(0,9, size=(3,))
      target = np.random.randint(0,9, size=(3,))

      parents = dijkstra3d.parental_field(values, start)
      path = dijkstra3d.path_from_parents(parents, target)

      path_orig = dijkstra3d.dijkstra(values, start, target)

      assert np.all(path == path_orig)

    # Asymmetric Test
    for _ in range(50):
      values = np.random.randint(1,255, size=(11,10,10))
      values = np.asfortranarray(values)

      start = np.random.randint(0,9, size=(3,))
      target = np.random.randint(0,9, size=(3,))

      parents = dijkstra3d.parental_field(values, start)
      path = dijkstra3d.path_from_parents(parents, target)

      path_orig = dijkstra3d.dijkstra(values, start, target)

      print(start, target)
      print(path)
      print(path_orig)

      assert np.all(path == path_orig)

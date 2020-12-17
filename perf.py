import dijkstra3d
import numpy as np
import time

def edf():
  print("Running edf.")
  N = 1
  sx, sy, sz = 256, 256, 256
  values = np.ones((sx,sy,sz), dtype=np.bool)
  for i in range(5):
    s = time.time()
    dijkstra3d.euclidean_distance_field(values, (100,100,100))
    e = time.time()
    accum = e-s
    mvx = N * sx * sy * sz / accum / 1000000
    print(f"{mvx:.3f} MVx/sec ({accum:.3f} sec)")  

def diagonal_ones():
  print("Running diagonal_ones.")
  N = 1
  sx, sy, sz = 512, 512, 512
  values = np.ones((sx,sy,sz), dtype=np.uint32)
  s = time.time()
  dijkstra3d.dijkstra(values, (0,0,0), (sx-1,sy-1,sz-1), compass=False)
  e = time.time()
  accum = e-s
  mvx = N * sx * sy * sz / accum / 1000000
  print(f"{mvx:.3f} MVx/sec ({accum:.3f} sec)")

def random_paths():
  print("Running random_paths.")
  values = np.random.randint(1,255, size=(7,7,7))
  values = np.asfortranarray(values)

  start = np.random.randint(0,7, size=(3,))
  target = np.random.randint(0,7, size=(3,))

  N = 1
  sx, sy, sz = 500, 500, 500
  for n in range(1, 100, 1):
    accum = 0
    for i in range(N):
      values = np.random.randint(1,n+1, size=(sx,sy,sz))
      values = np.asfortranarray(values)
      # values = np.ones((sx,sy,sz)) / 1000
      start = np.random.randint(0,min(sx,sy,sz), size=(3,))
      target = np.random.randint(0,min(sx,sy,sz), size=(3,))  

      s = time.time()
      path_orig = dijkstra3d.dijkstra(values, start, target, compass=False)
      accum += (time.time() - s)

    mvx = N * sx * sy * sz / accum / 1000000
    print(f"{n} {mvx:.3f} MVx/sec ({accum:.3f} sec)")

edf()
# diagonal_ones()
# random_paths()

#include "pairing_heap.hpp"


int main () {
  MinPairingHeap *heap = new MinPairingHeap(11.0, 0);

  for (float i = 10.0; i >= 0; i --) {
    heap->insert(i, 0);
    printf("inserted %.1f %lu\n", i, (unsigned long)(heap->root));
  }

  printf("OKAY! Now deleting mins %lu.\n", (unsigned long)(heap->root));

  for (int i = 0; i < 10; i++) {
    if (!heap->root) { 
      printf("NULL ROOT\n");
      break; 
    }
    printf("root key %.1f\n", heap->root->key);
    heap->delete_min();
  }

  delete heap;

  return 0;
}
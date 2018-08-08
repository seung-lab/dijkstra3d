#include "pairing_heap.hpp"


int main () {
  MinPairingHeap *heap = new MinPairingHeap(0.0, 0);

  for (float i = 10.0; i >= 1.0; i--) {
    heap->insert(i, 0);
    printf("inserted %.1f %p\n", i, heap->root);
  }

  // heap->print_keys();
  // heap->delete_min();
  // heap->print_keys();
// /  printf("OKAY! Now deleting mins %lu.\n", (unsigned long)(heap->root));

  for (int i = 0; i < 10; i++) {
    if (!heap->root) { 
      printf("NULL ROOT\n");
      break; 
    }
    heap->delete_min();
  }

  heap->print_keys();

  // delete heap;

  return 0;
}
/*
 * An implementation of a pairing heap.
 *
 * Michael L. FredmanRobert SedgewickDaniel D. SleatorRobert E. Tarjan
 * "The pairing heap: A new form of self-adjusting heap."
 * Algorithmica. Nov. 1986, Vol. 1, Iss. 1-4, pp. 111-129
 * doi: 10.1007/BF01840439
 *
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018
 */

#include <cmath>
#include <cstdint>
#include <stdio.h>

#ifndef PAIRING_HEAP_HPP
#define PAIRING_HEAP_HPP

class PHNode {
public:
  PHNode *left;
  PHNode *right;
  PHNode *parent;
  uint32_t index; 

  PHNode(uint32_t idx) {
    left = NULL;
    right = NULL;
    parent = NULL;
    index = idx;
  }

  PHNode(PHNode *lt, PHNode *rt, PHNode *p, uint32_t idx) {
    left = lt;
    right = rt;
    parent = p;
    index = idx;
  }

  ~PHNode () {}
}

class PairingHeap {
  PHNode *root;

  ~PairingHeap() {
    recursive_delete(root);
  }

  int find_min () {
    if (root == NULL) {
      return -1;
    }

    return root.index;
  }

  void recursive_delete (const PHNode *n) {
    if (n == NULL) {
      return;
    }
    else if (n.left == NULL && n.right == NULL) {
      delete n;
    }

    if (n.left != NULL) {
      recursive_delete(n.left);
    }

    if (n.right != NULL) {
      recursive_delete(n.right);
    }
  }

  PHNode* insert(uint32_t x) {
    PHNode *I = new PHNode(x);
    PHNode *tmp = NULL;

    if (x > root.index) {
      if (root.left == NULL) {
        root.left = I;
      }
      else {
        tmp = root.left;
      }
    }
    else {
      I.left = root;
      root = I;
    }

    return root;
  }

  PHNode* meld(PHNode* a, PHNode* b) {
    
  }

}



#endif
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
#include <vector>

#ifndef PAIRING_HEAP_HPP
#define PAIRING_HEAP_HPP

class PHNode {
public:
  PHNode *left;
  PHNode *right;
  float key; 
  uint32_t value;

  // pp. 114: "In order to make "decrease key" and "delete"
  // more efficient, we must store with each node a third pointer, 
  // to its parent in the binary tree."

  PHNode *parent; 

  PHNode() {
    left = NULL;
    right = NULL;
    parent = NULL;
    key = 0;
    value = 0;
  }

  PHNode(float k, uint32_t val) {
    left = NULL;
    right = NULL;
    parent = NULL;
    key = k;
    value = val;
  }

  PHNode(PHNode *lt, PHNode *rt, PHNode *p, float k, uint32_t val) {
    left = lt;
    right = rt;
    parent = p;
    key = k;
    value = val;
  }

  PHNode (const PHNode &p) {
    left = p.left;
    right = p.right;
    parent = p.parent;
    key = p.key;
    value = p.value;
  }

  ~PHNode () {}
};

// O(1)
PHNode* meld(PHNode* h1, PHNode *h2) {
  if (h1->key <= h2->key) {
    h2->right = h1->left;
    h1->left = h2;
    h2->parent = h1;
    return h1;
  }
  
  h2->left = h1;
  h2->parent = NULL;
  h1->right = h1->left;
  h1->left = NULL;
  h1->parent = h2;

  return h1;
}

// O(log n) amortized?
PHNode* delmin (PHNode* root) {
  PHNode *subtree = root->left;
  delete root;

  if (!subtree) {
    return NULL;
  }

  std::vector<PHNode*> forest;
  while (subtree->right) {
    forest.push_back(subtree->right);
    // subtree->parent = NULL; // probably not necessary
    subtree = subtree->right;
  }

  // forward pass
  size_t last = (forest.size() >> 1) << 1; // if odd, size - 1
  for (size_t i = 0; i < last; i += 2) {
    forest[i >> 1] = meld(forest[i], forest[i + 1]); 
  }
  last >>= 1;

  if (forest.size() & 0x1) {
    last++;
    forest[last] = forest[forest.size() - 1];
  }

  // backward pass
  for (size_t i = last; i > 0; i--) {
    forest[i-1] = meld(forest[i], forest[i - 1]);
  }

  return forest[0];
}


class MinPairingHeap {
public:
  PHNode *root;

  MinPairingHeap() {}

  MinPairingHeap (float key, const uint32_t val) {
    root = new PHNode(key, val);
  }

  // O(n)
  ~MinPairingHeap() {
    recursive_delete(root);
  }

  bool empty () {
    return root == NULL;
  }

  // O(1)
  PHNode* find_min () {
    return root;
  }

  // O(1)
  PHNode* insert(float key, const uint32_t val) {
    PHNode *I = new PHNode(key, val);
    return insert(I);
  }

  // O(1)
  PHNode* insert(PHNode* I) {
    if (root->key <= I->key) {
      I->right = root->left;
      root->left = I;
      I->parent = root;
    }
    else {
      I->left = root;
      I->parent = NULL;
      root->right = root->left;
      root->left = NULL;
      root->parent = I;
      root = I;
    }

    return I;
  }

  // O(1)
  void decrease_key (float delta, PHNode* x) {
    x->key -= delta;

    if (x == root) {
      return;
    }
    
    // Assuming I do this right,
    // x not being root should be sufficent
    // to mean it has a parent.
    // if (x->parent) {

    if (x->parent->left == x) {
      x->parent->left = NULL;
    }
    else {
      x->parent->right = NULL;
    }

    insert(x);
  }

  // O(log n) amortized?
  void delete_min () {
    if (!root) {
      return;
    }

    root = delmin(root);
  }

  void delete_node (PHNode *x) {
    if (x == root) {
      root = delmin(root);
      return;
    } 

    if (x->parent->left == x) {
      x->parent->left = NULL;
    }
    else {
      x->parent->right = NULL;
    }

    // probably unnecessary line
    x->parent = NULL;

    insert(delmin(x));
  }

private:
  void recursive_delete (const PHNode *n) {
    if (n == NULL) {
      return;
    }
    else if (n->left == NULL && n->right == NULL) {
      delete n;
    }

    if (n->left != NULL) {
      recursive_delete(n->left);
    }

    if (n->right != NULL) {
      recursive_delete(n->right);
    }
  }
};


#endif
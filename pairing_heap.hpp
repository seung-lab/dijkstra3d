/*
 * An implementation of a minimum pairing heap for use with
 * 3D djikstra. Can be easily generalized.
 *
 * Michael L. Fredman, Robert Sedgewick, Daniel D. Sleator, 
 * and Robert E. Tarjan
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
#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <unistd.h>

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

  void print () {
    printf("PHNode[%p](%.1f, %d, %p, %p, %p)\n", this, key, value, left, right, parent);
  }
};

void really_print_keys(PHNode *n, const int depth) {
  printf("(%d) %1.f \n", depth, n->key);

  if (depth > 20) {
    return;
  }

  if (n->left != NULL) {
    printf("L");
    really_print_keys(n->left, depth+1);
  }

  if (n->right != NULL) {
    printf("R");
    really_print_keys(n->right, depth+1);
  }    
}

// O(1)
PHNode* meld(PHNode* h1, PHNode *h2) {
  if (h1->key <= h2->key) {
    h2->right = h1->left;
    h1->left = h2;
    h2->parent = h1;
    return h1;
  }
  
  h1->right = h2->left;
  h2->left = h1;
  h1->parent = h2;
  
  return h2;
}

// O(log n) amortized?
PHNode* delmin (PHNode* root) {
  PHNode *subtree = root->left;
  
  if (!subtree) {
    delete root;
    return NULL;
  }

  PHNode* forest[512];
  int j = 0;
  while (subtree) {
    forest[j] = subtree;
    subtree = subtree->right;
    j++;
  }

  const size_t forest_size = j;

  for (int i = 0; i < forest_size; i++) {
    forest[i]->parent = NULL;
    forest[i]->right = NULL;
  }

  if (forest_size == 1) {
    delete root;
    return forest[0];
  }

  // need to deal with lone subtrees?

  // forward pass
  size_t last = forest_size & 0xfffffffe; // if odd, size - 1
  for (size_t i = 0; i < last; i += 2) {
    forest[i >> 1] = meld(forest[i], forest[i + 1]); 
  }
  last >>= 1;

  if (forest_size & 0x1) { // if odd
    forest[last] = forest[forest_size - 1];
  }
  else {
    last--;
  }

  // backward pass
  for (size_t i = last; i > 0; i--) {
    forest[i-1] = meld(forest[i], forest[i - 1]);
  }

  delete root;
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
    root = NULL;
  }

  bool empty () {
    return root == NULL;
  }

  float min_key () {
    if (root) {
      return root->key;
    }

    throw "No min key.";
  }

  uint32_t min_value () {
    if (root) {
      return root->value;
    }

    throw "No min value.";
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
    if (!root) {
      root = I;
      return I;
    }

    if (root->key <= I->key) {
      I->right = root->left;
      root->left = I;
      I->parent = root;
    }
    else {
      root->right = I->left;
      I->left = root;
      root->parent = I;
      root = I;
    }

    return I;
  }

  void decrease_key (float delta) {
    if (root) {
      root->key -= delta;
    }
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

  void print_keys () {
    if (root) {
      really_print_keys(root, 0);
    }
  }

private:
  void recursive_delete (PHNode *n) {
    if (n == NULL) {
      return;
    }

    if (n->left != NULL) {
      recursive_delete(n->left);
    }

    if (n->right != NULL) {
      recursive_delete(n->right);
    }

    if (n->parent) {
      if (n->parent->left == n) {
        n->parent->left = NULL;
      }
      else if (n->parent->right == n) {
        n->parent->right = NULL;
      }
    }

    delete n;
  }
};


#endif
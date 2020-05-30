"""
Directed Acyclic Graph generator with adjacency matrix and feature matrix.

@author: Tyler
"""

import numpy as np

# A - adjacency matrix (node and edges)
A = np.array([[1,0,0],
              [1,0,1],
              [0,1,0]])

# F - feature list (node type)
F = np.array([[0,1,0],
              [0,0,1],
             [1,0,0]])

print(f'Adjacency Matrix:\n{A}')
print()
print(f'\nFeature List:\n{F}')

# print(A[0,:])
# print()
# print(A[:,0])
# print()
# print(A[:,:2])

import
"""Device primitives shared by solvers: sorting, scanning, hashing, block-sparse matrices and their solvers.

Each primitive works on its own storage and on the arrays it is given, so that any solver can use it. Resources are
'@qd.data_oriented' classes holding their own storage, and phases are '@qd.func(requires_top_level=True)' functions that
a graph kernel or a kernel of the owning solver calls.
"""

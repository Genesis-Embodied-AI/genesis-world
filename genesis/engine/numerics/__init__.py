"""Device primitives shared by solvers: sorting, scanning, hashing, block-sparse matrices and their solvers.

Nothing here depends on a solver, an entity or a scene. Resources are plain '@qd.data_oriented' classes holding their
own storage, and phases are '@qd.func(requires_top_level=True)' functions that a graph kernel or a kernel of the
owning solver calls.
"""

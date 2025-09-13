# Algorithms

!!! warning "Work-in-progress"

    This page of the docs is still a work-in-progress. Check back later!

```@docs
GCPAlgorithms
GCPAlgorithms.AbstractAlgorithm
GCPAlgorithms.objective
GCPAlgorithms.grad_U!
```

```@autodocs
Modules = [GCPAlgorithms]
Filter = t -> t in subtypes(GCPAlgorithms.AbstractAlgorithm) || (t isa Function && t != GCPAlgorithms._gcp)
```

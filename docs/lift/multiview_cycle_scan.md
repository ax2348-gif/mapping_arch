# Cycle scans for views at the same time step (design note, DRAFT)

Status: the user's idea, 2026-10-16, written down for later. Not built, not frozen.

## Setting

Several views share one time step, for example several cameras or agents, each summarised by a register token. Within-frame patch operations are cheap spatial processing and are set aside here. The problem is relating the registers. A linear-attention or delta-rule scan imposes an ORDER on a set that has none, because the scan is asymmetric.

## The cycle scan: 3 views, a nice special case

- Scan the 3 registers as a cycle (a looped scan, with wrap-around), in both directions.
- The symmetry group of a bidirectional cycle on 3 elements is the dihedral group D3, and **D3 ≅ S3**: for three elements the cycle C3 is the complete graph K3, so every pair of views is adjacent.
- So 2 scans (one per direction) give exact permutation equivariance over the 3 views.
- This is special to n = 3. For n ≥ 4, C_n is a proper subgraph of K_n and D_n a proper subgroup of S_n, so a single cycle is not enough.

## Lifting: one cycle scan per triangular face of the simplex

- Regard n views as the vertices of an (n-1)-simplex, and apply the 3-view trick to every triangular face.
- **4 views:** the tetrahedron has 4 triangular faces, so 4 × 2 = 8 scans.
- **n views:** C(n, 3) faces, so 2·C(n, 3) scans. That is 2, 8, 20, 40 for n = 3, 4, 5, 6: cubic growth.
- The set of all triangular faces is invariant under S_n. Aggregating the face scans symmetrically (e.g. averaging per register) is therefore EXACTLY permutation-equivariant. Each pair of views shares n - 2 faces.
- Beyond small n this cost is the limit, which is why more views "need separate consideration".

## Geometry: the loop is natural

- The transitions between views are their relative poses, and the relative poses around any cycle compose to the identity (cycle consistency). A memory transported around a face's loop by exact rotations returns to itself.
- So the cycle scan respects a property the rotation mechanism already has, beyond restoring symmetry.
- Caution: exact rotations are orthogonal transitions. Looping to a fixed point does not converge, because the geometric series needs spectral radius < 1. Go once around.

## Open (for the user)

- How face-scan outputs are aggregated per register: mean, or a learned symmetric pooling.
- Whether the writes use the order-free simultaneous delta update (frame-aligned chunk; `(I + βKKᵀ)⁻¹` in place of the triangular solve), so only the transitions carry order.
- Where this enters: the multi-agent arm (rendezvous, shared terrain) and the camera rung with several cameras.

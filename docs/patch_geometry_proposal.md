# Suggested path for a shared patch geometry interface

Proposed direction, 2026-10-17: make a small calibrated pinhole camera measurement the common observation and rendering primitive, initially sampled at 3x3 pixels. Each time step has its own implicit SE3 local frame. Input and output tokens specify a viewing frustum and image-plane sampling grid relative to that frame; the model learns the relationships between frames and the spatial memory needed to answer queries. The existing SE2 task is an exact planar special case.

This replaces fixed learned patch identities with explicit local geometry on both sides of the model. It offers a natural interface for variable image dimensions and zoom while preserving the central task of learned temporal spatial integration. A first implementation candidate and its validation limits are recorded in [hybrid6](models/hybrid6.md). This proposal defines the local-calibration allowance for that variant; it does not retroactively change earlier task contracts or results.

Model name: **hybrid6**. Keep hybrid5 associated with the prospective PaTH hybrid in [the lift plan](lift/plan.md) and [the LVSM memory review](lvsm_memory_review.md), avoiding a rename of those historical references. The calibrated patch interface is the preferred next design candidate; the numbering does not require implementing hybrid5 first, and PaTH is not included in this proposal. The initial 100k-step Torch setting has been approved; the job gates training on GPU preflight checks.

## The boundary of the learned problem

The proposed boundary is:

> Supply the geometry of measurements within a local frame. Do not compute cross-frame alignment, accumulated poses, or query-to-source correspondences.

An observation image is a collection of patches whose positions, orientations, and sampling footprints are known relative to that image's frame. A requested image likewise specifies a collection of patches relative to its chosen anchor frame. Expanding either image into patch measurements uses camera geometry, without solving where its anchor lies relative to other observations.

| Supplied by the interface | Learned by the model |
| --- | --- |
| Patch positions within an image | Relationships between observation frames |
| Camera coordinate conventions, intrinsics, and pixel grid | Composition and use of local odometry across time |
| Requested patch poses relative to an arbitrary anchor frame | Relationship between that anchor and relevant stored observations |
| Sampling footprint and output grid | Retention, association, retrieval, and appearance reconstruction |

An operational check for query preprocessing is whether it can be computed from the rendering request and camera specification alone, without reading source images, trajectory history, or scene state. For observations, the corresponding allowance is each image's own local sampling geometry. Local odometry remains a supplied measurement between successive frames; accumulating it or using it analytically to align content remains outside this allowance.

This does not claim that the representation cannot improve learning. It removes the burden of learning a fixed camera convention and may substantially help optimization. The defensible claim is that it supplies no solution to the relationships across the observation history. End-to-end learning here concerns temporal spatial integration and memory, with a specified sensor and rendering interface.

## Common camera and sampling specification

### Frames sensors and transitions

Each time t names a local reference frame F_t. Tokens have three distinct roles: an observation supplies a measurement from a posed sensor, a memory render requests a measurement from that same kind of sensor, and an odometry token specifies the next frame relative to the current one. These share local-frame semantics, but are not all just poses.

For notation only, let G_t map coordinates from F_t to an arbitrary world frame. Define:

```text
D_t = inverse(G_t) * G_(t+1)    # next frame expressed in F_t
P   = sensor-to-F_t transform  # local sensor pose
C   = sensor calibration and sampling specification
```

The model receives local D_t, P, and C, not G_t. D_t maps coordinates from F_(t+1) into F_t; applying it to coordinates expressed in F_t requires its inverse. This direction must be explicit in the implementation. Current continuous odometry already gives the translation of the next origin in the current body frame and the relative rotation; discrete SE2 odometry encodes the corresponding restricted transform as a symbol. Do not reinterpret translation as expressed in the destination frame.

| Role | Local specification | Supplied value |
| --- | --- | --- |
| Observation | Sensor pose P and calibration C in F_t | Measured pixels |
| Memory render | The same sensor pose and calibration language in F_t | No measured pixels; predict them |
| Transition | Next-frame pose D_t in F_t | Relative motion measurement, without sensor intrinsics |

Observation and rendering are dual measurement operations. They should share the sensor specification and its embedding as an initial design, while retaining separate role indicators and content input/output paths. An unobserved query must not be confused with an observed all-zero patch. Shared measurement semantics do not imply that the content encoder and decoder must be inverses or share weights.

A possible factorization of the token embedding is:

```text
observation = observation_role + E_pose(P) + E_sensor(C) + E_content(pixels)
render      = render_role      + E_pose(P) + E_sensor(C)
transition  = transition_role  + E_pose(D_t)
```

Sharing E_pose is a hypothesis enabled by a common rigid-transform convention, not a requirement to conflate sensor intrinsics with motion. A joint pose/calibration encoder is also compatible with the semantics. Sensor geometry remains meaningful for both observation and rendering; transitions have no fictitious camera or pixel resolution.

The initial fixed-3x3 version has exactly three learned role vectors: observation, render, and transition. There are no learned patch-index vectors. Parameter-dependent pixel, pose, and sensor encoders are additional learned functions. A separate constant 3x3 sampling embedding is unnecessary; it can be absorbed into the observation/render role vectors. Introduce sampling-resolution embeddings only when supporting multiple grids.

Every patch of one image shares the same extrinsic pose P. Patch-specific crop intrinsics K_i select its portion of that image; sampling metadata specifies its pixel grid. For a single observation camera defining F_t, P is the identity transform: zero translation and identity rotation. For a render, P is the requested image camera pose relative to the selected anchor. Do not move that image-level translation into patch intrinsics, including in SE2. Adding these embeddings combines token attributes; it does not compose geometric transforms.

For multiview observations at one time, define F_t as a rig or reference-camera frame. Each camera has its own calibrated sensor-to-F_t pose, shared across its patches. Odometry must then describe transitions between those same reference frames. This uses the existing pose field without changing token semantics. Training only on identity observation extrinsics does not establish generalization to nonidentity ones; varied camera offsets and orientations require training or fine-tuning and explicit evaluation, even with a pose encoder shared with renders.

Odometry describes a measured transition. A commanded action is not generally a measured pose change: unknown dynamics, noise, and contact may intervene. If actions are introduced, give them a separate role and representation unless their deterministic conversion to D_t is part of the declared task.

### Camera calibration and sampling

Associate each observation or render token with a time index t and a local camera specification: center o, orientation Q, and calibrated image-plane extent. Together these define a viewing frustum. Separately specify the sampling grid, initially 3x3, and pixel footprints within that extent. Near and far clipping planes are unnecessary unless the task uses them. The time index names the associated frame and controls temporal placement; it does not expose an absolute pose.

Use intrinsics K in pixel coordinates together with explicit grid dimensions and a sampling convention. This is C in the token equations above. K and the grid determine the frustum; do not supply a redundant independent frustum parameter. Changing resolution at fixed frustum requires updating K. A normalized image-plane parameterization is an equivalent possible encoding, not another required input.

The sensor specification contains focal lengths, principal point, and any other declared intrinsic calibration, but no scene-plane distance h or world-units-per-pixel s. Those latter quantities appear below only to explain the planar renderer. In SE3, physical pixel footprints depend on unknown scene depth and are not camera calibration inputs.

Share the meaning of geometry between observations and queries, with a shared geometry encoder as the initial hypothesis. The physical measurement is a ray bundle, but the learned embedding does not automatically know the ray interpretation. Supplying camera parameters need not introduce ray tracing, scene intersection, or geometric retrieval into the network.

For homogeneous image coordinate u, the specified ray is:

```text
ray(u, lambda) = o + lambda * Q * inverse(K) * u,  lambda > 0
```

Use consistent units for o. A crop of one pinhole image retains that image's camera center: cropping changes intrinsics and the image-plane domain, not the center of projection. Orientation, angular extent, and sampling structure are all required; the center ray alone is insufficient.

All queries retain their selected anchor time, which can be any observation time rather than only the final frame. Local camera calibration and requested local camera poses are supplied; absolute world poses, cumulative odometry, query-to-source transforms, and analytically aligned world writes are not. Keep causal or bidirectional access and inert query behavior fixed in comparisons.

## Exact planar special case

Embed SE2 frames in SE3 by restricting frame motion to xy translation and rotation about z. Work in the observation camera's local frame: its extrinsic is identity and the texture plane is at z=1. Equivalently, in world coordinates one can place the texture at z=0 and the camera at (x_t, y_t, -1). The nonidentity world pose is not the local sensor extrinsic and is not supplied to the model.

The initially matched task uses focal lengths f_x=f_y=1 in pixel units and a centered 3x3 sampling grid. Let d be the requested image camera's xy displacement relative to the anchor, phi its relative heading, a_i the center of patch i in centered image pixel coordinates, and u a centered pixel coordinate within that patch. All image patches share one camera extrinsic; only their crop intrinsics differ:

```text
x_i(u) = d + R(phi) * (a_i + u)

c_i = -a_i
K_i = [[1, 0, c_i_x],
       [0, 1, c_i_y],
       [0, 0,     1]]
Q = block_diag(R(phi), 1)
o = (d_x, d_y, 0)
ray_i(u, lambda) = o + lambda * Q * inverse(K_i) * u_homogeneous
```

Here inverse(K_i) * u_homogeneous = (u_x+a_ix, u_y+a_iy, 1). Intersecting the rays with z=1, at lambda=1, therefore reproduces the existing sample positions exactly. Image translation and orientation reside in the shared extrinsic (o, Q); patch offsets reside in the principal point c_i. The principal point is distinct from the optical center o.

For later zoom experiments, replace the diagonal ones in K_i by focal length f. On this fixed plane the samples become x_i(u)=d+(1/f)*R(phi)*(a_i+u). More generally, for a frontoparallel plane at distance h they have spacing s=h/f. Here h describes the scene and s is a derived sampling footprint; neither is an additional sensor parameter. The initial h=1 and f=1 happen to match numerically, but zoom changes f while the plane remains at z=1.

For observations whose camera defines F_t, d=0 and phi=0, so the extrinsic is identity for every patch. For renders, use the supplied relative image pose. These calculations use only local camera calibration and the requested pose, never an integrated heading. The equations use conventional (x, y) coordinates; implementation must explicitly convert to the repository's (dy, dx), clockwise-angle convention. The formula c_i=-a_i assumes a centered image principal point and crops without resizing; general calibration must transform the original intrinsics under the actual crop and resize.

This construction reproduces the existing planar samples while preserving the same extrinsic/intrinsic separation needed for SE3. Known-plane intersection belongs to the SE2 renderer, not to model preprocessing for unknown SE3 geometry. No alternative that folds image-camera translation into intrinsics is proposed.

## Requirements for an SE3 extension

The interface must retain full three-dimensional translation, rotation, ray origins, and calibration. Do not bake in zero camera offsets, roll-only rotations, a shared plane, or a fixed depth in the architecture just because those suffice for SE2. Choose a rotation encoding with explicit conventions and handle its ambiguities; simply adding independent Euler-angle embeddings does not enforce SE3 composition. Additive token embeddings encode transforms, not an algebra in which adding embeddings composes them.

Keep these distinctions explicit:

- **Camera pose versus intrinsics.** Translate an actual camera through its extrinsics; represent cropping and zoom through calibration. Neither a principal-point shift nor a focal-length change substitutes for translation in a scene with depth. Preserve the requested viewing direction, since appearance may depend on it.
- **Ray geometry versus scene geometry.** Origins, directions, and angular footprints are known from the sensor. Depths, surface intersections, visibility, and occlusion are not. Do not supply those to the learned model through query preprocessing or a hidden world-coordinate lookup.
- **Metric motion versus angular sampling.** Translation units must agree between odometry and sensor offsets. Intrinsics determine angular sampling, not physical footprint without depth. Fix the planar h=1 convention only for the planar renderer; do not use unknown depth to normalize SE3 translations.
- **Anchor frame versus requested scene time.** For the initial static-scene task, t selects a reference frame and a memory access point. If dynamic scenes are introduced, requested scene time must be specified separately; the choice of coordinate anchor must not silently determine when an object is rendered.
- **Visible support versus occupied volume.** A query ray passing through a previously viewed frustum does not establish that its visible surface was observed. SE3 evaluation must distinguish supported reconstruction from disocclusion or unseen-content prediction. Visibility-based labels may be computed by the evaluation renderer, but must not leak scene geometry into model inputs.

The model may compose local transitions internally and learn sensor-to-memory alignment. The interface must not precompute the products of D_t, transform every observation into a common frame, or provide query-to-source transforms. A conceptual source-to-query relation depends on such products; providing local sensor calibration is not permission to calculate that relation externally.

Planar success alone will not validate these requirements. Before claiming SE3 readiness, use small scenes at multiple depths with lateral camera translation, forward translation, and rotations about multiple axes. Include an occluder and compare translation, principal-point shifts, and focal-length changes: effects that coincide on the plane must become distinguishable when geometry requires it. Check that a common rigid change of world coordinates leaves local inputs and targets unchanged, and that identical physical sensor requests anchored in different frames give consistent predictions where memory access permits. These are proposed future checks, not experiments run for this note.

## Additive geometry and temporal position

Keep additive camera-geometry embeddings and the existing time-only RoPE initially. Spatial coordinates at different times belong to different frames; their raw difference is not a physical displacement. Ordinary axis-wise spatial RoPE across all tokens would introduce coordinate-difference structure before those frames are related. The issue is unknown cross-frame alignment, not a claim that every multidimensional RoPE imposes a single Euclidean distance mixing time and space.

Within-frame spatial attention could use meaningful local geometry later, but it is not required for this interface. Treat the sequence as ordered local frames with attached measurements; serialization order within a frame does not define its spatial geometry.

## Image size and scale

Start with 3x3 samples per token. Image size determines how many patches are supplied or requested; it does not require a new positional identity table or decoder at this fixed patch resolution. An image becomes a collection of calibrated small camera measurements sharing a frame. Rectangular grids and irregular sets fit the same interface.

Distinguish three experiments:

| Variation | Fixed quantities | Consequence |
| --- | --- | --- |
| Image size | Pixel spacing and patch size | More patches and a larger field of view |
| Zoom | Image dimensions and patch size | Different world footprint and detail per pixel |
| Resolution at fixed field of view | World extent | Dimensions increase as pixel spacing decreases |

In the planar task, smaller s corresponds to larger f and zooming in. Keep the image camera extrinsic fixed for optical zoom and update calibration for the requested grid. At fixed crop pixel offsets a_i, zoom changes both the patch footprints and their centers on the plane around the shared image center. In SE3, calibration specifies angular scale; physical footprint also depends on depth and surface orientation. Keep odometry units independent of image dimensions and zoom.

An early extension can use distinct learned sampling embeddings for a small set of resolutions, such as 3x3 and 5x5, with compatible content encoders and output heads. At fixed frustum these are different samplings of the same viewing region. This changes samples per token, distinct from increasing image size by adding tokens. An embedding alone cannot change the number of input or output pixels. Learned resolution IDs support trained resolutions; generalization to unseen grids would call for numeric sampling features or a shared decoder queried at individual sample locations.

Image size ceases to be a restriction of the token interface, but generalization is not automatic. More patches still change compute, recurrent write counts, and forgetting. Likewise, a scale descriptor does not ensure that the content encoder learns scale-dependent appearance or that missing fine detail becomes recoverable.

## Relationship to current code and contracts

The current [SE2d hybrid4](../mapping/models/se2d/hybrid4.py) uses learned observation and query position tables for nine patches. Query tokens add a shared image-center pose embedding to patch-specific learned embeddings. The [SE2c hybrid4c](../mapping/models/se2c/hybrid4c.py) parameterizes the configured grid, but still uses fixed position tables and a common configured patch count for observations and queries.

The proposed interface replaces both position tables and expands a whole-image query into patch camera specifications. Runtime observation and query counts must be handled separately, including token ordering, query layout, and output assembly. Geometry encoding can build on the conventions in [se2c encoding](../mapping/se2c/encoding.py) and [geometry](../mapping/se2c/geometry.py), without importing trajectory integration into the model path. The exact parameterization and feature map for camera calibration remain implementation choices to compare; the measurement specification and cross-frame boundary are the design commitments.

The [continuous task contract](../CONTRACT_SE2C.md) currently states that poses must not be composed outside learned layers. Before implementing this variant, document the explicit local camera-calibration interface: preserve the supplied image pose and derive crop intrinsics from the image grid. The proposed token construction does not require composing image poses into patch-center poses. Preserve the prohibition on composition across the observation history, and do not relabel earlier results as using the new interface.

## Suggested evaluation path

1. Compare the existing interface, explicit query geometry alone, and explicit geometry on both inputs and queries. Keep images, scale, trajectories, temporal access, patch counts, and training budget matched initially. Account for parameter-count changes.
2. Vary observation size and query size independently at fixed pixel spacing. Include held-out grid sizes and shapes. Report both fixed-frame-count and fixed-total-patch-budget comparisons.
3. Check whether the same patch prediction changes when requested alone, with neighboring patches, or in a reordered query set. Inertness does not itself guarantee independence within a query. Inspect seams and overlapping predictions when a region is partitioned into different patch grids.
4. Verify that the camera construction reproduces the existing planar sample coordinates, including shifted patches, quarter turns, continuous rotations, and scale changes. Keep the renderer's sampling rule matched for the initial comparison.
5. Add calibrated zoom, first shared within an episode, then varying independently between observations and requests. Evaluate matched-field-of-view images using consistent filtering. Separately introduce multiple sampling resolutions per token, initially with learned resolution embeddings and corresponding encoders and heads.
6. Combine size and scale variation with longer sequences after establishing their individual effects. Extend to genuine SE3 scenes as a separate task with depth, parallax, and occlusion, without assuming planar equivalences still hold.

Preserve the intended trajectory difficulty. Larger fields of view can create overlap or loop closures absent in the original no-overlap data. Recheck footprints explicitly; scaling trajectory steps to compensate also changes world distances and must be reported.

Zoom experiments require scale-dependent antialiasing. Define pixel footprints through the calibrated grid rather than treating center rays as a complete filtering specification. Geometric coverage alone is insufficient: a coarse observation may cover a target while lacking its fine detail. Separate supported reconstruction from superresolution, and filter the final texture field consistently before sampling. Comparisons should use matched worlds and trajectories where possible.

Finally, inspect memory behavior under changing patch counts. Per-token decay can produce different retention across a frame when there are more observation tokens, and row-major serialization changes adjacency at row boundaries. These are possible limits of the memory architecture, distinct from the proposed geometric interface.

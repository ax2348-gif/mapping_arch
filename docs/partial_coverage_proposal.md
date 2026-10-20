# Future direction: motion with gaps and explicit unknown regions

Proposed by the user, 2026-10-18. This is a task-design proposal for SE2C and Ocean, not an implemented generator change or authorization for additional training runs.

## Motivation

The working concern is that these models readily exploit shortcuts, and observation overlap beyond SE2D can severely confound whether a model has learned spatial memory and motion composition. Nearby redundant views may support local matching or interpolation without requiring the long-range computation we intend to study. Overlap is not intrinsically invalid, and the current experiments do not establish that every successful continuous model relies on a shortcut. In particular, the recent constrained-path screen also changes displacement distributions and, at 15px steps, exceeds the current training motion range.

The proposal combines two changes:

1. **Allow gaps in the observed map.** Broaden motion independently of the requirement that observations form a densely covered region. Include larger translations, sparse observation spacing, diverse turns, and trajectories with controllable overlap and revisitation.
2. **Allow partially or entirely unsupported queries.** Render the supported parts from memory and pad the unobserved parts with black or an explicit unknown value. Stop requiring every query pixel to have been observed.

Together these changes remove the need to constrain motion and query sampling merely to guarantee complete coverage. They offer more freedom to design SE2C and Ocean tasks, while potentially requiring the model to represent both what it has observed and what remains unknown. Changing only motion while rejecting all partially covered queries would retain a strong coverage-selection bias; changing only query padding would leave overlap-based shortcuts available.

## Proposed target semantics

For a query at time t, derive a per-pixel support target from observations available through t only. Let m indicate support and c the target appearance where supported. A padded image target is `m*c + (1-m)*u`, where u is the declared unknown fill. Future observations must not contribute to either support or appearance supervision for that query.

Black padding is a simple initial variant. However, real black texture and unknown space then have the same appearance, so a black output alone cannot demonstrate knowledge of coverage. An explicit support channel is the preferred comparison: predict appearance plus known/unknown status, display unknown pixels in a distinct style, and score appearance on supported pixels. A dedicated sentinel is another option if it is distinguishable from valid intensities and is not erased by output clipping or display normalization. The exact output representation remains undecided.

Ground-truth support is a supervision/evaluation product of the generator, not an input to the model. Do not provide integrated world poses, coverage maps, target visibility, or scene geometry that would solve the intended memory problem externally. The model still receives observations, local motion, and query specifications through its normal interface.

“Unknown” here means unsupported by the available observation history under a declared sampling rule. It does not mean empty space, black material, or a calibrated estimate of model uncertainty. Predicting support correctly would be evidence of learned observation bookkeeping; stronger claims about uncertainty require separate tests.

## SE2C and Ocean details

For SE2C, compute support from the union of past observation footprints in the world used by the generator. Define treatment of pixel footprints, resampling, and boundary mixtures explicitly. Geometric inclusion alone need not imply enough sampling resolution to reconstruct a fine target pixel: avoid silently turning this into a superresolution task.

For Ocean's height-field setting, use visibility of the relevant scene surface, not just overlap in projected image coordinates or horizontal footprints. A surface hidden behind an occluder can remain unknown despite a nearby observed projection. Newly visible surfaces should stay unknown until observed. Distinguish never-observed surface, known background/no-hit, and outside-domain cases in the target contract; specify any view-dependent appearance assumptions. This visibility computation belongs to target generation and must not leak into model inputs.

## Evaluation and controls

Compare the existing dense/full-coverage task with sparse motion plus partial-query supervision, and include intermediate variants where feasible to separate the two interventions. Control texture, horizon, query age and displacement, rotation, model size, and training budget. Measure overlap explicitly rather than relying only on a trajectory label.

Sample fully known, mixed, and fully unknown queries with reported proportions. Include queries whose support is split across distant observations, and revisits after long gaps. Prevent a trivial all-unknown/black predictor from looking successful when missing area dominates. Report separately:

- Reconstruction error on supported pixels, including distant sources and overlap strata.
- Support precision/recall or IoU, stratified by known fraction; boundary accuracy separately.
- False-known predictions in unseen regions and false-unknown predictions in observed regions.
- Performance on entirely known and entirely unknown queries, and all-unknown/black baseline scores.
- Shuffled-odometry and removed-observation controls, with visibility targets recomputed for any changed available history when appropriate.

A particularly useful mechanistic test is to add or remove one observation while keeping the query fixed: predicted support should change at the corresponding spatial region, and retained known content should stay aligned. Disconnected regions should remain correctly related by supplied motion, even without overlapping visual evidence between them. Overlap can then be varied as an experimental factor rather than enforced as a prerequisite for valid queries.

## Relation to the curriculum proposal

This complements the proposed 15px discrete-to-continuous curriculum. Hybrid6's shared geometry interface could support dense discrete pretraining followed by sparse continuous/Ocean tasks or mixed training. Keep task redesign and architecture/curriculum changes distinguishable in comparisons. Neither the curriculum nor unknown padding by itself guarantees elimination of shortcuts; the intended benefit is greater control over what information is available and what computation successful retrieval requires.

Next design work: choose support/output semantics, define SE2C sampling and Ocean visibility rules, build generator-level oracle checks, then preregister motion/overlap/query-coverage strata before training. No current task contract or active run is changed by this note.

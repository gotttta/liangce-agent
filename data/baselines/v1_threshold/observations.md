# v1 Threshold Baseline Observations

This baseline is a deterministic comparison point, not an accepted defect detector. The red lines in each `result_contour.png` are contours derived from the raw threshold mask.

| Sample | Baseline result | Known failure |
|---|---:|---|
| `in_film_particle_left_pattern.jpg` | 29 measured components, 837 px | Large numbers of normal periodic bright-line fragments are treated as defects. |
| `in_film_particle_middle_defect.jpg` | 1 measured component, 10,935 px | Bright image borders connect into one dominant component; the detected area does not represent the particle boundary. |
| `in_film_particle_middle_defect_tight.jpg` | 3 measured components, 147 px | Only scattered bright fragments are retained; most of the particle body and boundary are missed. |
| `in_film_particle_right_defect_tight.jpg` | 11 measured components, 2,542 px | The particle is split into bright fragments and its darker interior is missed. |
| `in_film_particle_right_zoom.jpg` | 19 measured components, 9,530 px | Normal horizontal lines are detected together with partial particle edges. |

## Gate 2 Improvement Target

The fixed v2 pipeline must suppress normal periodic structures, produce a single coherent particle candidate where one particle is visible, and retain both bright and darker particle regions sufficiently to form a closed contour. Quantitative acceptance remains governed by the Ground Truth metrics in `docs/v2-agent-project-plan.md`.

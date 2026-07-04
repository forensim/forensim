# Sample Scene: Crime Scene 01

A synthetic indoor crime scene dataset for testing ForenSim's reconstruction and inference pipeline.

## Contents
- 8 synthetic photographs (640×480, JPEG)
- Simulated indoor room with 3 evidence markers (EV-1, EV-2, EV-3)
- Camera positions approximate a real photogrammetry session

## Evidence Tags
| Marker | Tag | Description |
|---|---|---|
| EV-1 | blood_spatter | Simulated blood spatter pattern near east wall |
| EV-2 | shell_casing | Expended brass shell casing on floor |
| EV-3 | impact_mark | Projectile impact mark on north wall |

## Usage
Load the `images/` directory in ForenSim's Evidence tab,
set the workspace to `workspace/`, then run reconstruction.

This repository contains a set of models...

Physical Description
---


Data
---

South Pole Lake

Siple Coast


Models
---

1) Analytical Solutions

Robin, Rez, Meyer, Perol
All four solutions are provided as functions in:
- ./ice_temperature/lib/analytical_solutions.py

2) Numerical Solution

Based on Weertman (1968)
is the predominant model explored by Hills et al. (2022).
- ./ice_temperature/lib/numerical_model.py


Testing
---

Unit testing is done for all scripts.
- ./cylindricalstefan/tests/

Dependencies
---

Python 3 (other versions may work, but they are not tested). Also, numpy and scipy.


References
---
- Hills, B. H., Winebrenner, D. P., Elam, W. T., Kintner P. (2020). Avoiding slush for hot-point drilling of glacier boreholes. Annals of Glaciology.
Robin
Rez
Meyer
Perol
Weertman
Cuffey and Paterson

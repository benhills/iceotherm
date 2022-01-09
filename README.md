This repository contains a set of models for glacier and ice-sheet temperature. Each is based on some previously published model from the glaciology literature. More will be added if/when the literature expands.

Data
---

Constants are defined in a class at:
- ./icetemperature/lib/constants.py
with most being from Cuffey and Paterson (2010)

South Pole Lake is included as a case study. The relevant data (Kahle et al., 2021; Price et al., 2002) are included at:
- ./data/

Siple Coast ice stream temperatures are included as a case study. The relevant data are embedded within the jupyter notebook.

Models
---

1) Analytical Solutions

Robin (1955), Rez, Meyer and Minchew (2018), Perol and Rice (2015)
All four solutions are provided as functions in:
- ./icetemperature/lib/analytical_solutions.py

2) Numerical Solution

Based on Weertman (1968)
This is the predominant model explored by Hills et al. (2022).
- ./icetemperature/lib/numerical_model.py

Testing
---

Unit testing is done for all scripts.
- ./icetemperature/tests/

Dependencies
---

Python 3 (other versions may work, but they are not tested). Also, numpy and scipy.

References
---

- Cuffey, K., & Paterson, W. S. B. (2010). The Physics of Glaciers (Fourth). Butterworth-Heinemann.
- Hills, B. H., Christianson, K., Hoffman, A. O., Fudge, T. J., Holschuh, N., Kahle, E. C., Conway, H., Christian, J. E., Horlings, A. N., O’Connor, G. K., Steig, E. J.
(2022). Geophysics and thermodynamics at South Pole Lake indicate stability and a regionally thawed bed. Geophysical Research Letters.
- Kahle, E. C., Steig, E. J., Jones, T. R., Fudge, T. J., Koutnik, M. R., Morris, V. A., Vaughn, B. H., Schauer, A. J., Stevens, C. M., Conway, H., Waddington, E. D., Buizert, C., Epifanio, J., & White, J. W. C. (2021). Reconstruction of Temperature, Accumulation Rate, and Layer Thinning From an Ice Core at South Pole, Using a Statistical Inverse Method. Journal of Geophysical Research: Atmospheres, 126(13), 1–20. https://doi.org/10.1029/2020jd033300
- Meyer, C. R., & Minchew, B. M. (2018). Temperate ice in the shear margins of the Antarctic Ice Sheet: Controlling processes and preliminary locations. Earth and Planetary Science Letters, 498, 17–26. https://doi.org/10.1016/j.epsl.2018.06.028
- Perol, T., & Rice, J. R. (2015). Shear heating and weakening of the margins of Western Antarctic ice streams. Geophysical Research Letters, 42, 3406–3413. https://doi.org/10.1002/2015GL063638.Received
- Price, P. B., Nagornov, O. V, Bay, R., Chirkin, D., He, Y., Miocinovic, P., Richards, A., Woschnagg, K., Koci, B., & Zagorodnov, V. (2002). Temperature profile for glacial ice at the South Pole : Implications for life in a nearby subglacial lake. Proceedings of the National Academy of Sciences, 99(12), 7844–7847. https://doi.org/10.1073/pnas.082238999
- Rezvanbehbahani, S., van der Veen, C. J., & Stearns, L. A. (2019). An Improved Analytical Solution for the Temperature Profile of Ice Sheets. Journal of Geophysical Research: Earth Surface, 124(2), 271–286. https://doi.org/10.1029/2018JF004774
- Robin, G. de Q. (1955). Ice movement and temperature distribution in glaciers and ice sheets. Journal of Glaciology, 2(18), 523–532.
Weertman, J. (1968). Comparison between Measured and Theoretical Temperature Profiles of the Camp Century, Greenland, Borehole. Journal of Geophysical Research, 73(8), 2691–2700. https://doi.org/10.1029/JB073i008p02691

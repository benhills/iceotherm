# iceotherm

This repository contains a set of 1.n-dimensional thermal models for glacier and ice-sheet temperature. Each is based on some previously published model from the glaciology literature. More will be added if/when the literature expands.

**1) Analytical Solutions**

Robin (1955), Rezvanbehbahani et al. (2019), Meyer and Minchew (2018), Perol and Rice (2015)
All four solutions are provided as functions in:

./icetemperature/lib/analytical_solutions.py

A quasi-analytical ODE solution from Perol and Rice (2015) is a melting-point bounded version of their analytical solution for temperature in ice-stream shear margins. We reproduce that solution here:

./icetemperature/lib/shearmargin_ode.py

**2) Numerical Solution**

Weertman (1968)
This is the predominant model explored by Hills et al. (2022a, 2022b).

./icetemperature/lib/numerical_model.py

Tests of the numerical model against analytical solutions are included at ./notebooks/analytical_solutions.ipynb

Case studies
---

**Ice temperature and flow history at South Pole Lake.** 

The relevant data include surface boundary conditions from the South Pole Ice Core (Kahle et al., 2021) and in-situ temperature measurements from the IceCube/AMANDA array (Price et al., 2002). Relevant data are saved at ./data/ and example jupyter notebooks which recreate the modeling experiments from Hills et al. (2022a) are in ./notebooks/publication_figures/south_pole_lake_example.ipynb.

**Siple Coast ice-stream temperature and shear-margin thermodynamics.** 

Hills et al. (2022b) argue that ice-stream shear margins are colder than previously thought due to longitudinal ice advection, so the ice stream acts as a conveyor of cold ice from upstream. All the relevant data for this example are embedded within the jupyter notebook at ./notebooks/publication_figures/siple_coast_example.ipynb.


**Slush formation in thermally-drilled glacier boreholes.** 

This work was done to aid in design of a hot-point drill, the Ice Diver, at the Applied Physics Lab, University of Washington (Hills et al., 2021). Mechanically-drilled glacier boreholes are easily stabilized with an antifreeze solution and can be held open for years. However, attempts to stabilize thermally-drilled holes have resulted in a plug of slush freezing in the hole which effectively freezes the hole shut. In essense, molecular diffusion (movement of the solute particles in the solution) is at least an order of magnitude slower than thermal diffusion in the solution, which creates an area of constitutional supercooling inside the hole (Worster, 2000). Therefore, refreezing happens in small particles within the solution rather than accretion on the borehole wall.

Dependencies
---

Python 3 (other versions may work, but they are not tested). Also, numpy and scipy.

Optional:
[FEniCS]. I recommend either the Anaconda or the Docker install.

References
---

- Carslaw, H. S., & Jaeger, J. C. (1959). Conduction of Heat in Solids (Second). London: Oxford University Press.
- Crepeau, J., & Siahpush, A. (2008). Analytical solutions to the Stefan problem with internal heat generation. Heat and Mass Transfer, 44, 787–794.
- Cuffey, K., & Paterson, W. S. B. (2010). The Physics of Glaciers (Fourth). Butterworth-Heinemann.
- Flick, E. W. (1998). Industrial Solvents Handbook (5th ed.). Westwood, NJ: Noyes Data Corporation.
- Hills, B. H., Winebrenner, D. P., Elam, W. T., Kintner P. (2020). Avoiding slush for hot-point drilling of glacier boreholes. Annals of Glaciology.
- Hills, B. H., Christianson, K., Hoffman, A. O., Fudge, T. J., Holschuh, N., Kahle, E. C., Conway, H., Christian, J. E., Horlings, A. N., O’Connor, G. K., Steig, E. J. (2022). Geophysics and thermodynamics at South Pole Lake indicate stability and a regionally thawed bed. Geophysical Research Letters.
- Humphrey, N., & Echelmeyer, K. (1990). Hot-water drilling and bore-hole closure in cold ice. Journal of Glaciology, 36(124), 287–298.
- Meyer, C. R., & Minchew, B. M. (2018). Temperate ice in the shear margins of the Antarctic Ice Sheet: Controlling processes and preliminary locations. Earth and Planetary Science Letters, 498, 17–26.
- Robin, G. de Q. (1955). Ice movement and temperature distribution in glaciers and ice sheets. Journal of Glaciology, 2(18), 523–532.
- Rezvanbehbahani, S., van der Veen, C. J., & Stearns, L. A. (2019). An Improved Analytical Solution for the Temperature Profile of Ice Sheets. Journal of Geophysical Research: Earth Surface, 124(2), 271–286.
- Perol, T., & Rice, J. R. (2015). Shear heating and weakening of the margins of Western Antarctic ice streams. Geophysical Research Letters, 42, 3406–3413.
- Weertman, J. (1968). Comparison between Measured and Theoretical Temperature Profiles of the Camp Century, Greenland, Borehole. Journal of Geophysical Research, 73(8), 2691–2700.
- Worster, M. G. (2000). Solidification of Fluids. In G. K. Batchelor, H. K. Moffat, & M. G. Worster (Eds.), Perspectives in Fluid Dynamics. Cambridge University Press.

[FEniCS]: https://fenicsproject.org/

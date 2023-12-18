This folder contains the code for the Master’s Thesis ”Optimal Labor Market Policy in a 
HANK & SAM Model” by Jens Brøndum Petersen. It includes:

1. 4 notebooks respectively used for: 1) calibration of the model, 2) investigations of the 
steady state of the model, 3) investigations of the transition path of the model and 4) a 
sensitivity analysis.

3. The files ”HANKSAMModel”, “household_problem”, “steady_state” and “blocks”
used for solving the model. The file “HANKSAMModel” is used for initializing the 
problem, while “household_problem” contains the code used for solving the problem 
of the households. “steady_state” contains the algorithm for solving the steady state 
while “blocks” contains the block structure used for solving the transition path.

In order to run the code, it is necessary to install the GEModelTools package, developed by
Jeppe Druedahl, Emil Holst Partsch and others. For the present thesis the versions of 
February 2023 have been used. An installation guide can be found at: 
https://github.com/NumEconCopenhagen/GEModelTools.

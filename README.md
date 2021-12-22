MATH 607E (2021WT1) Term Project: The Modified Preconditioned Conjugate Gradient (MPCG) Method for Cloth Simulation
==============================

Contains source code and the project report.

# Summary
Investigated the Preconditioned Conjugate Gradient (PCG) method following [Shewchuk's introductory text](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf). Implemented an MPCG solver and integrated it to my 4x4 square-shaped cloth simulator.

# File Structure
* `sim.ipynb`: MPCG-integrated cloth simulator
* `solvers/`: Contains various solvers - SD (Steepest Descent), CG (Conjugate Gradient), MPCG
* `visualizations/`: code to visualize stuff for the final report
* `docs/`: contains the final report.
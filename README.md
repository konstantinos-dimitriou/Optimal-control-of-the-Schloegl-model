# Numerical Experiments: Optimal Control of the Schlögl Model

> **Important Version Note:** > The code in this repository was implemented using Python 3 with the legacy versions of **FEniCS** and **FEniCS-Adjoint**. It is not directly compatible with the modern FEniCSx environment. 
> 
> A new, more compact version of this code utilizing **FEniCSx** is currently in development. That upcoming release will also feature implementations for **time-delayed nonlocal feedback control**, which are omitted in this legacy version.

## Overview

This repository contains the Python code used for the numerical experiments in my Master's thesis. It focuses on the numerical solution and optimal control of reaction-diffusion systems, specifically the **1-dimensional and 2-dimensional Schlögl models**. 

The primary objective of these scripts is to manipulate traveling wave fronts—specifically stopping, generating, and turning them. The optimization routines allow for the comparison of different control strategies, including standard **L2** control costs and sparse **L2 + L1** penalty functionals.

## Repository Structure

The codebase consists of 7 primary scripts. While there is some repetition between the optimal control files, they are separated by dimension and specific control objectives for clarity.

### PDE Solvers (Uncontrolled State)
* `Schloegl_1D.py`: Solves the baseline 1D Schlögl partial differential equation.
* `Schloegl_2D.py`: Solves the baseline 2D Schlögl partial differential equation.

### Optimal Control Scripts 1D
* `Optimal_control_Schloegl_1D.py`: Solves the optimal control problem in the 1-dimensional setting, allowing for the penalization of the control function to induce sparsity.

### Optimal Control Scripts 2D (Wave Front Manipulation)
These three scripts solve the optimal control problem in a 2-dimensional spatial domain, each targeting a specific wave front behavior:
* `Extinguish_wave_fonts_2D.py`: Applies control to stop/extinguish an expanding wave front.
* `Generate_wave_fonts_2D.py`: Applies control to an initial zero-state to generate a specific wave front pattern by the terminal time.
* `Turning_wave_fonts_2D.py`: Applies control to continuously turn an existing wave front in a counterclockwise direction while restraining its expansion.
* `purely_time_dependent_controls.py`: Solves the optimal control problem utilizing **purely time-dependent controls**. *(Note: The theoretical foundation for this approach is studied in Section 2.7 of the thesis, though these specific numerical experiments were added subsequently).*

## Dependencies

To run these scripts, you will need the legacy FEniCS stack:
* Python 3
* FEniCS (version 2019.2.0 or compatible legacy version)
* FEniCS-Adjoint (version 2019.1.0)
* SciPy (for the L-BFGS-B optimization algorithm)

## License and Citation

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details. If you find this code or the associated numerical experiments useful for your work, please cite the Master's thesis:

> **Konstantinos Dimitriou.** *On different control strategies for optimal control problems governed by prototypical reaction-diffusion equations.* Master's thesis, 2023.
> 
> > [**Read the full thesis on ResearchGate**](https://www.researchgate.net/publication/381887437_On_different_control_strategies_for_optimal_control_problems_governed_by_prototypical_reaction_diffusion_equations)


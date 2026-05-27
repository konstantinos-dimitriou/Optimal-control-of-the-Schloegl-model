# Author: Konstantinos Dimitriou
# Department: Mathematics
# Institution: University Bonn
# Date: 2/4/2023

import numpy as np
from fenics import *
from fenics_adjoint import *


def solve_schloegl_model_2D():
    """
    This is our main solution routine
    """

    # Define time span and number of time steps
    start_time = 0          # Start time
    end_time = 40.0         # Final time
    number_of_time_steps = 400         # number of time steps
    dt = (end_time - start_time) / number_of_time_steps        # Time step size

    # Create mesh and define function space
    nx = ny = 141
    mesh = RectangleMesh(Point(0, 0), Point(70, 70), nx, ny)
    Solution_function_space = FunctionSpace(mesh, 'Lagrange', 1)

    # Define constants
    y_1 = 0
    y_2 = 0.25
    y_3 = 1

    # Define reaction term
    def R(y):
        return (y-y_1)*(y-y_2)*(y-y_3)
    
    # Define boundary conditions
    boundary_conditions = []  # Neumann boundary conditions = 0, are natural in FEniCS

    # Define initial condtion
    y_initial = Expression('(x[0]<=140/3) * (x[0]>=70/3)', degree=1) # start

    # Project onto solution space (needed for equation)
    y_prev = project(y_initial, Solution_function_space)

    # Define variational problem
    y = Function(Solution_function_space)
    dy = TrialFunction(Solution_function_space)
    v = TestFunction(Solution_function_space)

    # Define weak form of Schloegl equation
    schloegl_weak_form_residuum = (
        y * v * dx 
        + dt * dot(grad(y), grad(v)) * dx
        + dt * R(y) * v * dx 
        - y_prev * v * dx
    )
    
    # Precompute Jacobian
    Jacobian = derivative(schloegl_weak_form_residuum, y, dy)

    # Create VTK file to save solution (if requested)
    vtkfile_of_state = File('DATA/Schloegl_2D/solution.pvd')

    # Time-stepping
    t = start_time
    while t <= end_time:
        # Solve PDE
        solve(schloegl_weak_form_residuum == 0, y, boundary_conditions, J=Jacobian,\
             solver_parameters={"newton_solver":{"maximum_iterations":1000}})

        # Save to file
        vtkfile_of_state << (y, t)

        # Update previous solution
        y_prev.assign(y)

        # Update current time
        t += dt


if __name__=='__main__':
    solve_schloegl_model_2D()


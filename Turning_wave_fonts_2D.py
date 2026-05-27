# Author: Konstantinos Dimitriou
# Department: Mathematics
# Institution: University Bonn
# Date: 2/4/2023

import numpy as np
from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import time


def solve_schloegl_model(simulation_settings, reaction_term, desired_state, controls, save_to_file, file_location, file_name_suffix):
    """
    simulation_settings = [start_time, end_time, number_of_time_steps, Solution_function_space, boundary_conditions]
    """
    # Copy settings from settings array
    start_time = simulation_settings[0]
    end_time = simulation_settings[1]
    number_of_time_steps = simulation_settings[2]
    Solution_function_space = simulation_settings[3]
    boundary_conditions = simulation_settings[4]

    R = reaction_term

    # Compute fixed time step size
    dt = (end_time - start_time)/number_of_time_steps

    # Define initial condtion
    y_initial = Expression('(x[0]<=x[1]+10) * (x[0]>=x[1]-10)', degree=1) # start

    # Project onto solution space (needed for equation)
    y_prev = project(y_initial, Solution_function_space)

    # Define variational problem
    y = Function(Solution_function_space)
    dy = TrialFunction(Solution_function_space)
    v = TestFunction(Solution_function_space)

    # Define desired state
    d = Function(Solution_function_space)
    u = Function(Solution_function_space)

    # Define weak form of Schloegl equation
    schloegl_weak_form_residuum = (
        y * v * dx 
        + dt * dot(grad(y), grad(v)) * dx
        + dt * R(y) * v * dx 
        - y_prev * v * dx
        - dt * u * v * dx
    )
    
    # Precompute Jacobian
    Jacobian = derivative(schloegl_weak_form_residuum, y, dy)

    # Assemble objective functional
    # L2 penalty
    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + 0.5*float(dt)*assemble((u)**2 * dx)
    # L2 + L1  penalty
    mu = 1.0  
    kappa = 0.1
    j = 0.5*float(dt)*assemble((y - d)**2 * dx) + kappa*0.5*float(dt)*assemble((u)**2 * dx) + float(mu) * float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5) *dx)

    # Create VTK file to save solution (if requested)
    if save_to_file == True:
        vtkfile_of_state = File(file_location + '/schloegl_2d_state_' + file_name_suffix + '.pvd')
        vtkfile_of_control = File(file_location + '/schloegl_2d_control_' + file_name_suffix + '.pvd')

    # Time-stepping
    t = start_time
    while t <= end_time:
        # Source function from controls
        u.assign(controls[t])

        # Solve PDE
        solve(schloegl_weak_form_residuum == 0, y, boundary_conditions, J=Jacobian,\
             solver_parameters={"newton_solver":{"maximum_iterations":1000}})

        # Compute objective functional
        # Implement a trapezoidal rule
        if t > end_time - float(dt):
            weight = 0.5
        else:
            weight = 1

        # Update desired state (because it depends on the time t)
        desired_state.t = t
        d.assign(interpolate(desired_state, Solution_function_space))

        # L2 penalty
        #j += weight*float(dt)*assemble((y - d)**2*dx) + weight*float(dt)*assemble((u)**2*dx)
        # L2 + L1 penalty     
        j += weight*float(dt)*assemble((y - d)**2*dx) + weight*float(dt)*assemble((u)**2*dx) + float(mu) * weight*float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5)*dx)

        # Save to file and plot solution (if requested)
        if save_to_file == True:
            vtkfile_of_state << (y, t)
            vtkfile_of_control << (u, t)

        # Update previous solution
        y_prev.assign(y)

        # Update current time
        t += dt

    return y, d, j


def main_opt_control():
    '''This is our main optimal control routine'''
    # The following line supresses fenics prints when solving each PDE
    set_log_level(50)

    # Start timer
    stopwatch_start = time.time()

    # Define time span and number of time steps
    start_time = 0          # Start time
    end_time = 20.0         # Final time
    number_of_time_steps = 200         # number of time steps
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

    # Define boundary condition
    bcs = []

    # Define desired state
    # Turning continuously
    desired_state = Expression('(x[0]<=(((1)*(T-t)+(-1)*t)/T)*x[1]+(10*(T-t)+80*t)/T) * (x[0]>=(((1)*(T-t)+(-1)*t)/T)*x[1]+(-10*(T-t)+60*t)/T)', degree=1, t=start_time, T=end_time)

    # Create simulation settings array
    simulation_settings = [start_time, end_time, number_of_time_steps, Solution_function_space, bcs]

    # Create control functions
    controls = OrderedDict()
    t = 0
    while t <= end_time:
        # This is also our initial guess
        controls[t] = Function(Solution_function_space)
        #controls[t] = Constant(0.0)  # Constant controls
        t += float(dt)

    # Data settings
    save_to_file = True
    file_location = 'DATA/Data_Turning_wave_fonts'
    file_name_suffix = 'Initial'

    # Initila solve of equation
    y, d, j = solve_schloegl_model(simulation_settings, R, desired_state, controls, save_to_file, file_location, file_name_suffix)
    J = j

    # Define control for minimization problem
    m = []
    t = 0
    lower_bounds = []
    upper_bounds = []
    while t < end_time:
        m.append(Control(controls[t]))
        
        #Define lower and upper bounds for controls
        lower_bounds.append(-100.0)
        upper_bounds.append(100.0)
        t += float(dt)

    # Define bound array
    bounds = [lower_bounds, upper_bounds]

    # Define reduced functional
    reduced_functional = ReducedFunctional(J, m)

    # Solve minimization problem
    #opt_ctrls = minimize(reduced_functional, bounds=bounds, method='L-BFGS-B', tol=1e-6, options={"gtol":1e-3, "disp":True, "maxiter": 60})
    opt_ctrls = minimize(reduced_functional, bounds=bounds, method='L-BFGS-B', tol=1e-6, options={"gtol":1e-3, "disp":True, "maxiter": 5})

    # Turn optimal control again into a directory with float indices so that we can use it in our main solve function
    opt_ctrls_as_directory = OrderedDict()
    t = 0
    for control_value in opt_ctrls:
        opt_ctrls_as_directory[t] = control_value
        t += float(dt)

    # Solve for optimal state and plot
    # Set new data settings
    save_to_file = True
    file_location = 'DATA/Data_Turning_wave_fonts'
    file_name_suffix = 'Optimal_Terminal'

    solve_schloegl_model(simulation_settings, R, desired_state, opt_ctrls_as_directory, save_to_file, file_location, file_name_suffix)

    # Stop timer
    stopwatch_stop = time.time()
    print('Total computation time: ',stopwatch_stop - stopwatch_start)


if __name__=='__main__':
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("Start of simulation: ", current_time)

    main_opt_control()

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("End of simulation: ", current_time)


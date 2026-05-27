# Author: Konstantinos Dimitriou
# Department: Mathematics
# Institution: University Bonn
# Date: 2/4/2023


import numpy as np
from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import time
import matplotlib.pyplot as plt



def solve_schloegl_model(simulation_settings, reaction_term, desired_state, controls, do_plot):
    """
    simulation_settings = [start_time, end_time, number_of_time_steps, Solution_function_space, boundary_conditions, mesh]
    """
    # Copy settings from settings array
    start_time = simulation_settings[0]
    end_time = simulation_settings[1]
    number_of_time_steps = simulation_settings[2]
    Solution_function_space = simulation_settings[3]
    boundary_conditions = simulation_settings[4]
    mesh = simulation_settings[5]

    R = reaction_term

    # Compute fixed time step size
    dt = (end_time - start_time)/number_of_time_steps

    # Define initial value

    # Extinguish wave font

    # y_initial = Expression(
    #     '0.5*(1 - tanh(x[0]/(2*sq2)))'
    #     , degree=1
    #     , sq2 = np.sqrt(2)
    #     , pi = np.pi
    # )
    # y_initial = Expression(
    #     '(0 <= x[0])*(x[0] <= 14)*(-0.5*(1-cos((x[0]-7 +7)*(pi/7) ))) + (9 <= x[0])*(x[0] <= 23)*(0.125*(1-cos( (x[0]-16+7)*(pi/7) ))) + (17<= x[0])*(x[0] <= 31)*(-0.5*0.111*(1-cos( (x[0]-24+7)*(pi/7) ))) + (26<= x[0])*(x[0] <= 40) *(0.5*0.0625*(1-cos( (x[0]-33+7)*(pi/7) ))) '
    #     , degree=1
    #     , pi = np.pi
    # )

    # Generate wave font

    y_initial = Expression('0*x[0]', degree=1)

    # Project initial condition onto solution function space (needed for problem setup)
    y_prev = interpolate(y_initial, Solution_function_space)

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
    mu = 0.5
    kappa = 0.5
    #kappa = 1.0
    # L2 penalty
    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + float(kappa)*0.5*float(dt)*assemble((u)**2 * dx)

    # L2 + L1 penalty
    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + float(kappa)*0.5*float(dt)*assemble((u)**2 * dx) + 0.5*float(mu) * float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5) *dx)

    # Final state
    # L2 controls
    #j = 0.1*0.5*float(dt)*assemble((u)**2 * dx)
    #j = float(kappa)*0.5*float(dt)*assemble((u)**2 * dx)
    # L2 + L1 controls
    j = float(kappa)*0.5*float(dt)*assemble((u)**2 * dx) + float(mu) * float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5) *dx)

    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + 0.001*0.5*float(dt)*assemble((u)**2 * dx)
    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + 0.5*float(dt)*assemble((u)**2 * dx) + 0.5*float(dt)*assemble(u * dx)

    # Reaching a final desired state
    #j = 0.5*float(dt)*assemble((u)**2 * dx)

    # L2 + L1 penalty
    #j = 0.5*float(dt)*assemble((y - d)**2 * dx) + 0.5*float(dt)*assemble(((u)**2)**(0.5) * dx) + 0.5*float(dt)*assemble((u)**2 * dx)

    # Time-stepping
    t = start_time

    # Create result arrays
    solution_history = []
    time_point_array = []

    while t <= end_time:
        # Update source term from control array       
        u.assign(controls[t])

        # Solve PDE
        solve(schloegl_weak_form_residuum == 0, y, boundary_conditions, J=Jacobian,\
             solver_parameters={"newton_solver":{"maximum_iterations":1000}})

        # Save state and time
        current_state = []
        for mesh_point in mesh.coordinates():
            current_state.append(y(mesh_point))
        solution_history.append(current_state)
        time_point_array.append(t)

        # Compute objective functional
        # Implement a trapezoidal rule
        if t > end_time - float(dt):
            weight = 0.5
        else:
            weight = 1

        # Update data function (because it depends on the time t)
        desired_state.t = t
        d.assign(interpolate(desired_state, Solution_function_space))

        # Reaching a desired state in whole time interval
        # L2 penalty
        #j += weight*float(dt)*assemble((y - d)**2*dx) + weight*float(dt)*assemble((u)**2*dx)

        # L2 + L1 penalty      
        #j += weight*float(dt)*assemble((y - d)**2*dx) + float(kappa)*weight*float(dt)*assemble((u)**2*dx) + float(mu) * weight*float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5) *dx)

        # Reaching a desired Final state
        # L2 controls
        #j += 0.1*weight*float(dt)*assemble((u)**2*dx)
        #j += float(kappa)*weight*float(dt)*assemble((u)**2*dx)
        # L2 + L1 controls
        j += float(kappa)*weight*float(dt)*assemble((u)**2*dx) + float(mu) * weight*float(dt)*assemble((u**2+DOLFIN_EPS)**(0.5) *dx)
        if t>end_time-10*dt:
            j += weight*float(dt)*assemble((y - d)**2*dx) 

        # Reaching desired state after halfway
        # if t > end_time/2:
        #     j += weight*float(dt)*assemble((y - d)**2*dx) + weight*float(dt)*assemble((u)**2*dx)
        # else:
        #     j += weight*float(dt)*assemble((u)**2*dx)

        # Update previous solution
        y_prev.assign(y)

        # Update current time
        t += dt
    
    solution_history = np.array(solution_history)   # This is needed so that we can later reshape the data

    if do_plot == True:
        # Plot solution
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X, Y = np.meshgrid(mesh.coordinates(), time_point_array)
        Z = solution_history.reshape(X.shape)

        # Plot solution
        ax.plot_surface(X, Y, Z, cmap="plasma")
        ax.set_xlabel('Spatial interval')
        ax.set_ylabel('Time')
        ax.set_zlabel('State function')
        fig.savefig('Schloegl_1d_optimal_state_L2_L1.png',format='png')
        fig.savefig('Schloegl_1d_optimal_state_L2_L1.svg',format='svg')
        plt.show()

    return y, d, j, solution_history


def main_opt_control():
    '''This is our main optimal control routine'''
    # The following line supresses fenics prints when solving each PDE
    set_log_level(50)

    # Start timer
    stopwatch_start = time.time()

    # Define time span and number of time steps
    start_time = 0          # Start time
    #end_time = 40.0         # Final time
    #end_time = 20.0         # Final time
    end_time = 5.0         # Final time
    #end_time = 7.5         # Final time
    #end_time = 10.0         # Final time
    #number_of_time_steps = 400         # number of time steps
    number_of_time_steps = 100         # number of time steps
    #number_of_time_steps = 200         # number of time steps
    dt = (end_time - start_time) / number_of_time_steps        # Time step size

    # Create mesh and define function space
    number_of_cells = 201
    #start_point = -25.0
    #end_point = 25.0
    start_point = 0.0
    end_point = 40.0
    #start_point = -10.0
    #end_point = 30.0
    mesh = IntervalMesh(number_of_cells , start_point , end_point)

    #Solution_function_space = fe.FunctionSpace(mesh, 'P', 1)
    Solution_function_space = FunctionSpace(mesh, 'Lagrange', 1)
    #Solution_function_space = FunctionSpace(mesh, 'CG', 1)

    #Control_function_space = FunctionSpace(mesh, 'Lagrange', 1)

    # Define constants
    # Since we work with the standard form y_1=0, 0<y_2<1 , y_3=1
    y_1 = 0
    #y_2 = 0.25   # Threshold parameter - This we might change later
    y_2 = 0.125
    y_3 = 1

    # Define reaction term
    def R(y):
        #return (y-y_1)*(y-y_2)*(y-y_3)
        return 1/3*y**3 - y

    # Define boundary condition
    bcs = []

    # Define desired state

    # Extinquish wave
    #desired_state = Expression("0", degree=0)

    # Generate wave
    desired_state = Expression(
        '(0 <= x[0])*(x[0] <= 14)*(-0.5*(1-cos((x[0]-7 +7)*(pi/7) ))) + (9 <= x[0])*(x[0] <= 23)*(0.125*(1-cos( (x[0]-16+7)*(pi/7) ))) + (17<= x[0])*(x[0] <= 31)*(-0.5*0.111*(1-cos( (x[0]-24+7)*(pi/7) ))) + (26<= x[0])*(x[0] <= 40) *(0.5*0.0625*(1-cos( (x[0]-33+7)*(pi/7) ))) '
        , degree=1
        , pi = np.pi
    )


    # Create simulation settings array
    simulation_settings = [start_time, end_time, number_of_time_steps, Solution_function_space, bcs, mesh]

    # Create control functions
    controls = OrderedDict()
    t = 0
    while t <= end_time:
        # This is also our initial guess
        controls[t] = Function(Solution_function_space)
        #controls[t] = Constant(0.0)  # Constant controls
        t += float(dt)

    do_plot = True

    # Initila solve of equation
    y, d, j, solution_history = solve_schloegl_model(simulation_settings, R, desired_state, controls, do_plot)

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
    #opt_ctrls = minimize(reduced_functional, bounds=bounds, method='L-BFGS-B', tol=1e-6, options={"gtol":1e-3, "disp":True, "maxiter": 500})
    opt_ctrls = minimize(reduced_functional, bounds=bounds, method='L-BFGS-B', tol=1e-6, options={"gtol":1e-3, "disp":True, "maxiter": 500})

    # Turn optimal control again into a directory with float indices so that we can use it in our main solve function
    opt_ctrls_as_directory = OrderedDict()
    t = 0
    for control_value in opt_ctrls:
        opt_ctrls_as_directory[t] = control_value
        t += float(dt)

    # Solve for optimal state and plot
    do_plot = True
    solve_schloegl_model(simulation_settings, R, desired_state, opt_ctrls_as_directory, do_plot)

    # Plot optimal control
    # Save state and time
    control_history = []
    time_point_array = []
    t = 0
    while t < end_time:
        current_state = []
        for mesh_point in mesh.coordinates():
            current_state.append(opt_ctrls_as_directory[t](mesh_point))
        control_history.append(current_state)
        time_point_array.append(t)
        t += float(dt)

    control_history = np.array(control_history)  # needed for reshape()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(mesh.coordinates(), time_point_array)
    Z = control_history.reshape(X.shape)

    # Plot control
    ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.set_xlabel('Spatial interval')
    ax.set_ylabel('Time')
    ax.set_zlabel('Control function')
    fig.savefig('Schloegl_1d_optimal_control_L2_L1.png',format='png')
    fig.savefig('Schloegl_1d_optimal_control_L2_L1.svg',format='svg')
    plt.show()

    # Stop timer
    stopwatch_stop = time.time()
    print('Total computation time: ',stopwatch_stop - stopwatch_start)


def main_single_solve():
    # Start timer
    stopwatch_start = time.time()

    # Define time span and number of time steps
    start_time = 0.0        # Start time
    end_time = 5.0          # Final time
    end_time = 40          # Final time
    num_steps = 300         # Number of time steps
    dt = (end_time - start_time) / num_steps        # Time step size

    # Create mesh and define function space
    number_of_cells = 100
    number_of_cells = 401
    start_point = -20.0
    end_point = 20.0
    mesh = IntervalMesh(number_of_cells , start_point , end_point)
    #solution_function_space = fe.FunctionSpace(mesh, 'P', 1)
    solution_function_space = FunctionSpace(mesh, 'Lagrange', 1)

    # Define constants
    # Since we work with the standard form y_1=0, 0<y_2<1 , y_3=1
    y_1 = 0
    y_2 = 0.25   # Threshold parameter - This we might change later
    y_3 = 1
    c = wave_velocity = (1/np.sqrt(2))*(1-2*y_2)
    D = 1       # diffusion coefficient

    # Define reaction term
    def R(y):
        return (y-y_1)*(y-y_2)*(y-y_3)
        #return 1/3*y**3 - y
        #return 1/3*y**3 - y

    # Define initial value and project onto function space
    y_0 = Expression(
        '0.5*(1 - tanh(x[0]/(2*sq2)))'
        , degree=1
        , sq2 = np.sqrt(2)
        , pi = np.pi
    )

   
    #y_prev = fe.interpolate(y_0, solution_function_space)
    y_prev = project(y_0, solution_function_space)

    # Define variational problem
    y = Function(solution_function_space)
    dy = TrialFunction(solution_function_space)
    v = TestFunction(solution_function_space)

    schloegl_weak_form_residuum = (
        y * v * dx 
        + D * dt * dot(grad(y), grad(v)) * dx
        + dt * R(y) * v * dx 
        - y_prev * v * dx
    )

    # Compute Jacobian
    Jacobian = derivative(schloegl_weak_form_residuum, y, dy)

    # Time-stepping
    t = 0

    # Create result arrays
    solution_history = []
    time_point_array = []

    # Start simulation loop
    #for n in range(num_steps):
    for n in range(num_steps):
        # Compute solution
        solve(schloegl_weak_form_residuum == 0, y, J=Jacobian,\
             solver_parameters={"newton_solver":{"maximum_iterations":500}})

        # Save function state
        current_state = []
        for mesh_point in mesh.coordinates():
            current_state.append(y(mesh_point))
        solution_history.append(current_state)
        time_point_array.append(t)

        # Save to file and plot solution
        #vtkfile << (y, t)

        # Update previous solution
        y_prev.assign(y)
        # Update current time
        t += dt

    solution_history = np.array(solution_history)   # This is needed so that we can later reshape the data

    stopwatch_end = time.time()
    
    print('Total computation time: ', stopwatch_end - stopwatch_start)


    # Plot solution
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(mesh.coordinates(), time_point_array)
    Z = solution_history.reshape(X.shape)


    # Plot a basic wireframe.
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    #ax.plot_surface(X, Y, Z, cmap='coolwarm')
    #ax.plot_surface(X, Y, Z, cmap="autumn")
    ax.plot_surface(X, Y, Z, cmap="plasma")
    #ax.plot_surface(X, Y, Z, cmap="cividis")


    ax.set_xlabel('Spatial interval')
    ax.set_ylabel('Time')
    ax.set_zlabel('Function value')

    plt.show()


if __name__=='__main__':
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("Start of simulation: ", current_time)
    
    #main_single_solve()
    main_opt_control()

    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print("End of simulation: ", current_time)


# Author: Konstantinos Dimitriou
# Department: Mathematics
# Institution: University Bonn
# Date: 2/4/2023


import numpy as np
from ufl import *
from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import time
import matplotlib.pyplot as plt



#def solve_schloegl_model(simulation_settings, reaction_term, desired_state, controls, do_plot):
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

    # u_1 = Function(Solution_function_space)
    # u_2 = Function(Solution_function_space)
    # u_3 = Function(Solution_function_space)
    # u_4 = Function(Solution_function_space)

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
    mu = 2.0
    kappa = 0.0
    # All sumations are done in the while loop
    j = 0

    # Time-stepping
    t = start_time

    # Create result arrays
    solution_history = []
    time_point_array = []


    
    x = SpatialCoordinate(mesh)
    while t <= end_time:
        # Update source term from control array     
        u_1 = controls[t][0]
        u_2 = controls[t][1]
        u_3 = controls[t][2]
        u_4 = controls[t][3]
        u_5 = controls[t][4]
        u_6 = controls[t][5]

        #source_function = (x[0]<10)*u_1 + (x[0]>10)*(x[0]<20)*u_2
        # source_function = (tanh(100*(x[0]-0)) - tanh(100*(x[0]-10)))*u_1 + \
        #     (tanh(100*(x[0]-10)) - tanh(100*(x[0]-20)))*u_2
        # source_function = (tanh(100*(x[0]-0)) - tanh(100*(x[0]-10)))*u_1 + \
        #     (tanh(100*(x[0]-10)) - tanh(100*(x[0]-20)))*u_2 + \
        #     (tanh(100*(x[0]-20)) - tanh(100*(x[0]-30)))*u_3 + \
        #     (tanh(100*(x[0]-30)) - tanh(100*(x[0]-40)))*u_4

        # source_function = (tanh(100*(x[0]-0)) - tanh(100*(x[0]-7)))*u_1 + \
        #     (tanh(100*(x[0]-7)) - tanh(100*(x[0]-14)))*u_2 + \
        #     (tanh(100*(x[0]-14)) - tanh(100*(x[0]-21)))*u_3 + \
        #     (tanh(100*(x[0]-21)) - tanh(100*(x[0]-28)))*u_4 + \
        #     (tanh(100*(x[0]-28)) - tanh(100*(x[0]-35)))*u_5 + \
        #     (tanh(100*(x[0]-35)) - tanh(100*(x[0]-40)))*u_6
        source_function = (tanh(100*(x[0]-0)) - tanh(100*(x[0]-5)))*u_1 + \
            (tanh(100*(x[0]-7)) - tanh(100*(x[0]-12)))*u_2 + \
            (tanh(100*(x[0]-14)) - tanh(100*(x[0]-19)))*u_3 + \
            (tanh(100*(x[0]-21)) - tanh(100*(x[0]-26)))*u_4 + \
            (tanh(100*(x[0]-28)) - tanh(100*(x[0]-33)))*u_5 + \
            (tanh(100*(x[0]-35)) - tanh(100*(x[0]-40)))*u_6

        projection_source = project(source_function, Solution_function_space)

        u.assign(projection_source)

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
        if t > end_time - float(dt) or t==start_time:
            weight = 0.5
        else:
            weight = 1

        # Update data function (because it depends on the time t)
        desired_state.t = t
        d.assign(interpolate(desired_state, Solution_function_space))

        
        u_1 = project(u_1, Solution_function_space)
        u_2 = project(u_2, Solution_function_space)
        u_3 = project(u_3, Solution_function_space)
        u_4 = project(u_4, Solution_function_space)
        u_5 = project(u_5, Solution_function_space)
        u_6 = project(u_6, Solution_function_space)

        # Reaching a desired state in whole time interval
        # L2 penalty
        j += weight*float(kappa)*0.5*float(dt)*0.5 * assemble(((u_1)**2 + (u_2)**2 + (u_3)**2 + (u_4)**2 + (u_5)**2 + (u_6)**2)*dx)*(1/40)
        # L1 penalty
        j += weight*float(mu)*0.5*float(dt)*0.5 * assemble(((u_1**2+DOLFIN_EPS)**(0.5) + (u_2**2+DOLFIN_EPS)**(0.5) +\
                                                             (u_3**2+DOLFIN_EPS)**(0.5) + (u_4**2+DOLFIN_EPS)**(0.5) +\
                                                                (u_5**2+DOLFIN_EPS)**(0.5) + (u_6**2+DOLFIN_EPS)**(0.5))*dx)*(1/40)

        # State penalty
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
    end_time = 5.0         # Final time
    number_of_time_steps = 100         # number of time steps
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

    Solution_function_space = FunctionSpace(mesh, 'Lagrange', 1)

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
        #controls[t] = Function(Solution_function_space)
        #controls[t] = Constant(0.0)  # Constant controls

        #controls[t] = [Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0)]
        controls[t] = [Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0), Constant(0.0)]

        #controls[t] = [Constant(0.0), Constant(1.0)]
        t += float(dt)

    do_plot = True

    # Initila solve of equation
    #y, d, j, solution_history = solve_schloegl_model(simulation_settings, R, desired_state, controls, do_plot)
    #y, d, j, solution_history = solve_schloegl_model(simulation_settings, R, desired_state, all_control_functions, do_plot)
    y, d, j, solution_history = solve_schloegl_model(simulation_settings, R, desired_state, controls, do_plot)

    J = j

    # Define control for minimization problem
    m = []
    t = 0
    lower_bounds = []
    upper_bounds = []
    while t < end_time:
        m.append(Control(controls[t][0]))
        m.append(Control(controls[t][1]))
        m.append(Control(controls[t][2]))
        m.append(Control(controls[t][3]))
        m.append(Control(controls[t][4]))
        m.append(Control(controls[t][5]))
        
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
    opt_ctrls = minimize(reduced_functional, method='L-BFGS-B', tol=1e-6, options={"gtol":1e-3, "disp":True, "maxiter": 500})

    # Turn optimal control again into a directory with float indices so that we can use it in our main solve function
    opt_ctrls_as_directory = OrderedDict()
    t = 0
    #for control_value in opt_ctrls:
    #for i in np.arange(0,len(opt_ctrls),2):
    #for i in np.arange(0,len(opt_ctrls),4):
    for i in np.arange(0,len(opt_ctrls),6):
        #opt_ctrls_as_directory[t] = [opt_ctrls[i], opt_ctrls[i+1]]
        opt_ctrls_as_directory[t] = [opt_ctrls[i], opt_ctrls[i+1], opt_ctrls[i+2], opt_ctrls[i+3], opt_ctrls[i+4], opt_ctrls[i+5]]
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
        # current_control_function = Expression("(tanh(100*(x[0]-0)) - tanh(100*(x[0]-10)))*u_1 + \
        #     (tanh(100*(x[0]-10)) - tanh(100*(x[0]-20)))*u_2 + \
        #     (tanh(100*(x[0]-20)) - tanh(100*(x[0]-30)))*u_3 + \
        #     (tanh(100*(x[0]-30)) - tanh(100*(x[0]-40)))*u_4",
        #     u_1 = opt_ctrls_as_directory[t][0],
        #     u_2 = opt_ctrls_as_directory[t][1],
        #     u_3 = opt_ctrls_as_directory[t][2],
        #     u_4 = opt_ctrls_as_directory[t][3], degree=1)
        # current_control_function = Expression("(tanh(100*(x[0]-0)) - tanh(100*(x[0]-15)))*u_1 + \
        #     (tanh(100*(x[0]-10)) - tanh(100*(x[0]-25)))*u_2 + \
        #     (tanh(100*(x[0]-20)) - tanh(100*(x[0]-35)))*u_3 + \
        #     (tanh(100*(x[0]-30)) - tanh(100*(x[0]-40)))*u_4",
        #     u_1 = opt_ctrls_as_directory[t][0],
        #     u_2 = opt_ctrls_as_directory[t][1],
        #     u_3 = opt_ctrls_as_directory[t][2],
        #     u_4 = opt_ctrls_as_directory[t][3], degree=1)
        # current_control_function = Expression("(tanh(100*(x[0]-0)) - tanh(100*(x[0]-7)))*u_1 + \
        #     (tanh(100*(x[0]-7)) - tanh(100*(x[0]-14)))*u_2 + \
        #     (tanh(100*(x[0]-14)) - tanh(100*(x[0]-21)))*u_3 + \
        #     (tanh(100*(x[0]-21)) - tanh(100*(x[0]-28)))*u_4 + \
        #     (tanh(100*(x[0]-28)) - tanh(100*(x[0]-35)))*u_5 + \
        #     (tanh(100*(x[0]-35)) - tanh(100*(x[0]-40)))*u_6",
        #     u_1 = opt_ctrls_as_directory[t][0],
        #     u_2 = opt_ctrls_as_directory[t][1],
        #     u_3 = opt_ctrls_as_directory[t][2],
        #     u_4 = opt_ctrls_as_directory[t][3],
        #     u_5 = opt_ctrls_as_directory[t][4],
        #     u_6 = opt_ctrls_as_directory[t][5],
        #     degree=1)
        current_control_function = Expression("(tanh(100*(x[0]-0)) - tanh(100*(x[0]-5)))*u_1 + \
            (tanh(100*(x[0]-7)) - tanh(100*(x[0]-12)))*u_2 + \
            (tanh(100*(x[0]-14)) - tanh(100*(x[0]-19)))*u_3 + \
            (tanh(100*(x[0]-21)) - tanh(100*(x[0]-26)))*u_4 + \
            (tanh(100*(x[0]-28)) - tanh(100*(x[0]-33)))*u_5 + \
            (tanh(100*(x[0]-35)) - tanh(100*(x[0]-40)))*u_6",
            u_1 = opt_ctrls_as_directory[t][0],
            u_2 = opt_ctrls_as_directory[t][1],
            u_3 = opt_ctrls_as_directory[t][2],
            u_4 = opt_ctrls_as_directory[t][3],
            u_5 = opt_ctrls_as_directory[t][4],
            u_6 = opt_ctrls_as_directory[t][5],
            degree=1)
        for mesh_point in mesh.coordinates():
            current_state.append(current_control_function(mesh_point))
        control_history.append(current_state)
        time_point_array.append(t)
        t += float(dt)

    control_history = np.array(control_history)  # needed for reshape()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y = np.meshgrid(mesh.coordinates(), time_point_array)
    Z = control_history.reshape(X.shape)

    # Plot control
    #ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.set_zlim(-1.51, 1.51)
    ax.plot_surface(X, Y, Z, cmap="CMRmap")
    #ax.plot_surface(X, Y, Z, cmap="twilight")
    ax.set_xlabel('Spatial interval')
    ax.set_ylabel('Time')
    ax.set_zlabel('Control function')
    plt.show()

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


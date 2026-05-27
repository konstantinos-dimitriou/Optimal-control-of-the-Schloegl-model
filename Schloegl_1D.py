# Author: Konstantinos Dimitriou
# Department: Mathematics
# Institution: University Bonn
# Date: 2/4/2023


import numpy as np
from fenics import *
from fenics_adjoint import *
import time
import matplotlib.pyplot as plt


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

    # Define reaction term
    def R(y):
        return (y-y_1)*(y-y_2)*(y-y_3)
        #return 1/3*y**3 - y

    # Define initial value and project onto function space
    y_0 = Expression(
        '0.5*(1 - tanh(x[0]/(2*sq2)))'
        , degree=1
        , sq2 = np.sqrt(2)
        , pi = np.pi
    )

    # y_0 = Expression(
    #     '(0 <= x[0])*(x[0] <= 14)*(-0.5*(1-cos((x[0]-7 +7)*(pi/7) ))) + (9 <= x[0])*(x[0] <= 23)*(0.125*(1-cos( (x[0]-16+7)*(pi/7) ))) + (17<= x[0])*(x[0] <= 31)*(-0.5*0.111*(1-cos( (x[0]-24+7)*(pi/7) ))) + (26<= x[0])*(x[0] <= 40) *(0.5*0.0625*(1-cos( (x[0]-33+7)*(pi/7) ))) '
    #     , degree=1
    #     , pi = np.pi
    # )
   
    #y_prev = fe.interpolate(y_0, solution_function_space)
    y_prev = project(y_0, solution_function_space)

    # Define variational problem
    y = Function(solution_function_space)
    dy = TrialFunction(solution_function_space)
    v = TestFunction(solution_function_space)

    schloegl_weak_form_residuum = (
        y * v * dx 
        + dt * dot(grad(y), grad(v)) * dx
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
    for n in range(num_steps):
        # Compute solution
        solve(schloegl_weak_form_residuum == 0, y, J=Jacobian,\
             solver_parameters={"newton_solver":{"maximum_iterations":100}})

        # Save function state
        current_state = []
        for mesh_point in mesh.coordinates():
            current_state.append(y(mesh_point))
        solution_history.append(current_state)
        time_point_array.append(t)

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


    # Plot solution
    ax.plot_surface(X, Y, Z, cmap="plasma")
    ax.set_xlabel('Spatial interval')
    ax.set_ylabel('Time')
    ax.set_zlabel('Function value')
    fig.savefig('final_state.png',format='png')
    #plt.show()


if __name__=='__main__':   
    main_single_solve()


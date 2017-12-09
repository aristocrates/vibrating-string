import numpy as np
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physicalstring import *
import integrator
import os.path
import random
import sys

def zero_pad(n, length):
    ans = str(n)
    ans = "0"*(length - len(ans)) + ans
    return ans

def triangle_midpoint(length, mid, np_array = True):
    """
    Returns a triangle waveform ranging from 0 to 1, with peak at mid
    """
    triangle_filter = [i / mid for i in range(mid)] \
                      + [1 - i / (length - mid - 1) for i in range(length - mid)]
    if np_array:
        return np.array(triangle_filter)
    else:
        return triangle_filter

def write_plot(x, y, n, fileformat, directory="plots", xlim = [0, 1],
               ylim = [-1, 1], scatter = True, plot = False):
    if scatter:
        plt.scatter(x, y)
    if plot:
        plt.plot(x, y)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(os.path.join(directory, "figure"
                             + zero_pad(n, 4) + "." + fileformat))
    plt.close()

def sinusoid_drive_left(num_elements, num_snapshots, fileformat,
                        stepsize = 1e-3, steps_per_snapshot=10,
                        verbose = False, directory = "plots_sinusoid"):
    if verbose:
        print("Directory is %s" % directory)
    boundary = [BoundaryCondition(lambda t: 0.25 * np.sin(10 * t), 0),
                BoundaryCondition(lambda t: 0, 0)]
    x = np.arange(0, 1, 1. / num_elements)
    y = np.array([0] * num_elements)

    # intially motionless
    initial_data = [k for k in y] + [0 for k in x]

    st = String(boundary, num_elements, initial_data)

    finite_int = FiniteElementIntegrator(st, integrator.RungeKutta,
                                         {-1:1, 0:-2, 1:1}, 1e-2)
    # finite_int = FiniteElementIntegrator(st, integrator.DormandPrince,
    #                                     {-1:1, 0:-2, 1:1}, 1e-2,
    #                                     adaptive = False)
    # finite_int = FiniteElementIntegrator(st, integrator.ForwardEuler,
    #                                      {-1:1, 0:-2, 1:1}, 1e-2)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory)
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory)

def fixed_ends_ergodic(num_elements, num_snapshots, fileformat,
                       pluck_loc = 0.65, stepsize = 1e-3,
                       steps_per_snapshot = 10, verbose = False,
                       directory = "plots_fixed_ergo"):
    boundary = [BoundaryCondition(lambda t: 0, 0), BoundaryCondition(lambda t: 0, 0)]
    num_elements = 100
    x = np.arange(0, 1, 1. / num_elements)
    y = 20 * triangle_midpoint(num_elements, int(0.65 * num_elements))
    initial_data = [k for k in y] + [0 for k in x]
    st = String(boundary, num_elements, initial_data)

    lambda_var = 10.
    rho = 1.
    k = 1.
    extended_stencil_dict = [{'stencil':{-1:1, 0:-2, 1:1}, 'coefficient':(k/rho), 'power':1}]
    phi_time_deriv_dict = [{'deriv':0, 'coefficient': (-1 * lambda_var / rho), 'power':3}]

    # for testing purposes, should produce equivalent results
    extended_stencil_dict_alt = [{'stencil':{-1:1, 0:-2, 1:1}, 'coefficient':(k/rho), 'power':1},
                                 {'stencil':{0:1}, 'coefficient': (-1 * lambda_var / rho), 'power':3}]
    phi_time_deriv_dict_alt = []

    #finite_int = FiniteElementIntegratorGeneralized(st, integrator.DormandPrince, extended_stencil_dict_alt, phi_time_deriv_dict_alt, 1e-2, adaptive=False)
    finite_int = FiniteElementIntegratorGeneralized(st, integrator.RungeKutta, extended_stencil_dict, phi_time_deriv_dict, 1e-2)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory, ylim = [-20, 20])
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory, ylim = [-20, 20])

def fixed_ends_nonergodic(num_elements, num_snapshots, fileformat,
                       pluck_loc = 0.65, stepsize = 5e-4,
                       steps_per_snapshot = 10, verbose = False,
                       directory = "plots_fixed_nonergo"):
    boundary = [BoundaryCondition(lambda t: 0, 0), BoundaryCondition(lambda t: 0, 0)]
    num_elements = 100
    x = np.linspace(0, 2 * np.pi / 10, num_elements)
    y = np.array([0] + [k for k in 10 * np.sin(5 * x[1:-1])] + [0]) #20 * triangle_midpoint(num_elements, int(0.65 * num_elements))
    initial_data = [k for k in y] + [0 for k in x]
    st = String(boundary, num_elements, initial_data)

    #lambda_var = 10.
    alpha = 0
    rho = 1.
    k = 1.
    #phi_time_deriv_dict = [{'deriv':0, 'coefficient': (-1 * lambda_var / rho), 'power':1}]
    finite_int = FiniteElementIntegratorFermiPastaUlamTsingou(st, integrator.RungeKutta, 1.0, 2 * np.pi / 10 / num_elements)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory, xlim = [0, 2 * np.pi / 10],
                   ylim = [-20, 20])
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory, xlim = [0, 2 * np.pi / 10],
               ylim = [-20, 20])

def fixed_ends_anderson(num_elements, num_snapshots, fileformat,
                       pluck_loc = 0.65, stepsize = 1e-3,
                       steps_per_snapshot = 10, verbose = False,
                       directory = "plots_fixed_anderson"):
    boundary = [BoundaryCondition(lambda t: 0, 0), BoundaryCondition(lambda t: 0, 0)]
    #boundary = [BoundaryCondition(lambda t: 0.25 * np.sin(10 * t), 0), BoundaryCondition(lambda t: 0, 0)]
    num_elements = 100
    x = np.arange(0, num_elements)
    # y = [0]
    # for i in range(num_elements - 2):
    #     y.append(y[i] + random.uniform(-1, 1))
    # y.append(0)
    # y = np.array(y)
    #y = 10 * triangle_midpoint(num_elements, int(pluck_loc * num_elements))
    y = 7 * np.array([0] * int( (num_elements - 20) / 2) + triangle_midpoint(20, 10, np_array = False) + [0] * int(num_elements - 20 - (num_elements - 20) / 2)) #np.array([0] + [random.uniform(-1, 1) for k in range(num_elements - 2)] + [0])#np.array([0 for k in x])#np.sin(30 * np.pi * x)#np.array([0] + [5 * random.uniform(-1, 1) for k in range(num_elements - 2)] + [0])#20 * triangle_midpoint(num_elements, int(0.65 * num_elements))
    initial_data = [k for k in y] + [0 for k in x]
    st = String(boundary, num_elements, initial_data)

    # attempt to be somewhat smooth in setting mu
    # mu_derivs =
    #mu_var = [0]
    #for i in range(num_elements - 2):
    #     mu_var.append(mu_var[i] + random.uniform(-1, 1))
    # mu_var.append(0)
    # mu_var = np.array(mu_var)
    mu_var = np.array([random.uniform(0, 1) for k in range(num_elements)])
    rho = 1.
    k = 1.
    extended_stencil_dict = [{'stencil':{-1:1, 0:-2, 1:1}, 'coefficient':[(k/rho)] * num_elements, 'power':1}]
    phi_time_deriv_dict = [{'deriv':0, 'coefficient': (-1 * mu_var / rho), 'power':1}]

    # for testing purposes, should produce equivalent results
    extended_stencil_dict_alt = [{'stencil':{-1:1, 0:-2, 1:1}, 'coefficient':([k/rho] * num_elements), 'power':1},
                                 {'stencil':{0:1}, 'coefficient': (-1 * mu_var / rho), 'power':1}]
    phi_time_deriv_dict_alt = []

    #finite_int = FiniteElementIntegratorGeneralized(st, integrator.DormandPrince, extended_stencil_dict_alt, phi_time_deriv_dict_alt, 1e-2, adaptive=False)
    finite_int = FiniteElementIntegratorGeneralized(st, integrator.RungeKutta, extended_stencil_dict, phi_time_deriv_dict, 1, spatial_param_dependence = True)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory, xlim = [0, 100], ylim = [-20, 20],
                   plot = True)
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory, xlim = [0, 100], ylim = [-20, 20],
               plot = True)

def fixed_ends(num_elements, num_snapshots, fileformat, pluck_loc = 0.65,
               stepsize = 1e-3, steps_per_snapshot = 10,
               verbose = False, directory = "plots_fixed"):
    """
    Simulates a string with ends fixed at zero, plucked at some point

    pluck_loc: relative location of the string pluck (value from 0 to 1)
    """
    if verbose:
        print("Directory is %s" % directory)
    
    boundary = [BoundaryCondition(lambda x: 0, 0),
                BoundaryCondition(lambda x: 0, 0)]
    x = np.arange(0, num_elements, 1) / num_elements
    assert(0 <= pluck_loc <= 1)
    y = triangle_midpoint(num_elements,
                                     int(pluck_loc * num_elements))
    initial_data = [k for k in y] + [0 for k in x]
    
    st = String(boundary, num_elements, initial_data)
    
    finite_int = FiniteElementIntegrator(st, integrator.DormandPrince,
                                         {-1:1, 0:-2, 1:1}, 1e-2,
                                         adaptive = False)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory)
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory)

def fixed_ends_damped(num_elements, num_snapshots, fileformat,
                      pluck_loc = 0.65, stepsize = 1e-3,
                      steps_per_snapshot = 10, verbose = False,
                      directory = "plots_fixed_damped"):
    """
    Simulates a string with ends fixed at zero, plucked at some point
    Includes a damping term

    pluck_loc: relative location of the string pluck (value from 0 to 1)
    """
    if verbose:
        print("Directory is %s" % directory)
    
    boundary = [BoundaryCondition(lambda x: 0, 0),
                BoundaryCondition(lambda x: 0, 0)]
    x = np.arange(0, num_elements, 1) / num_elements
    assert(0 <= pluck_loc <= 1)
    y = triangle_midpoint(num_elements,
                                     int(pluck_loc * num_elements))
    initial_data = [k for k in y] + [0 for k in x]
    
    st = String(boundary, num_elements, initial_data)

    k = 1.
    rho = 1.
    gamma = -1.
    extended_stencil_dict = [{'stencil':{-1:1, 0:-2, 1:1}, 'coefficient':(k/rho), 'power':1}]
    phi_time_deriv_dict = [{'deriv':1, 'coefficient': (gamma / rho), 'power':1.}]
    
    finite_int = FiniteElementIntegratorGeneralized(st, integrator.RungeKutta, extended_stencil_dict, phi_time_deriv_dict, 1e-2, spatial_param_dependence = False)

    def current_values_position():
        return finite_int.integrator.current_data()[:num_elements]

    for i in range(num_snapshots):
        write_plot(x, current_values_position(), i, fileformat,
                   directory = directory)
        for j in range(steps_per_snapshot):
            finite_int.step(stepsize)
        if verbose:
            print(i)
    write_plot(x, current_values_position(), i, fileformat,
               directory = directory)

def generate_plots(fileformat = "png"):
    fixed_ends(100, 500, fileformat, pluck_loc = 0.65, stepsize = 1e-3,
               verbose = True, directory='plots_fixed')
    sinusoid_drive_left(100, 500, fileformat, verbose = True,
                        directory='plots_sinusoid')
    fixed_ends_ergodic(100, 500, fileformat, pluck_loc=0.65,
                       stepsize=0.001, steps_per_snapshot=10,
                       verbose=True, directory='plots_fixed_ergo')
    fixed_ends_nonergodic(100, 1000, fileformat, pluck_loc=0.65,
                          stepsize=5e-4, steps_per_snapshot=10,
                          verbose=True, directory='plots_fixed_nonergo')
    fixed_ends_anderson(100, 1500, fileformat, pluck_loc = 0.65,
                        stepsize = 0.005, steps_per_snapshot=10,
                        verbose = True, directory='plots_fixed_anderson')
    fixed_ends_damped(100, 500, fileformat, pluck_loc = 0.5,
                       stepsize=1e-3, steps_per_snapshot=10,
                       verbose=True, directory='plots_fixed_damped')

if __name__ == "__main__":
    fileformat = "png"
    if len(sys.argv) > 1:
        print("Output will be pdf files")
        fileformat = "pdf"
    generate_plots(fileformat)

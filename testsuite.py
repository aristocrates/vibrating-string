"""
Test suite for the integrators

Nicholas Meyer
"""
import numpy as np
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import problem, integrator

def all_tests():
    test_harmonic("test_plots/convergence.pdf")
    test_boundary_conditions()

def test_harmonic(filename):
    """
    Produces convergence plots for 
    """
    harmonic_convergence_plot(filename)

def test_boundary_conditions():
    pass

def harmonic_convergence_plot(filename, largest_step = 5e-1,
                              smallest_step = 5e-5, **kwargs):
    step_sizes_prime = np.logspace(np.log10(largest_step),
                                   np.log10(smallest_step),
                                   num = 1 + abs(int(np.log10(smallest_step
                                                              / largest_step))
                                   ))
    #print(step_sizes)
    # add intermediate step sizes
    step_sizes = []
    for i, v in enumerate(step_sizes_prime):
        step_sizes.append(v)
        if i < len(step_sizes_prime) - 1:
            step_sizes.append(5 * step_sizes_prime[i + 1])
    #print(step_sizes)
    positions_euler = []
    velocities_euler = []
    positions_rk = []
    velocities_rk = []
    positions_dp = []
    velocities_dp = []
    for step in step_sizes:
        # set up a fresh problem each time
        simple_mass_spring = problem.Harmonic(k = 1, m = 1)
        euler = integrator.ForwardEuler(simple_mass_spring)
        rk = integrator.RungeKutta(simple_mass_spring)
        dp = integrator.DormandPrince(simple_mass_spring, step, 1,
                                      adaptive = False)
        
        num_steps = int(np.round(10 * largest_step / step))
        for i in range(num_steps):
            euler.step(delta = step)
            rk.step(delta = step)
            dp.step(delta = step)
        final_euler = euler.current_data()
        final_rk = rk.current_data()
        final_dp = dp.current_data()
        # the analytic solution is x(t) = cos(t), v(t) = -sin(t)
        # TODO: make sure this isn't off by one step size
        # if correct, will look like exponential decay
        # if wrong, might "flatline", indicating systematic offset error
        # final_analytic = [np.cos(10 * largest_step),
        #                   -np.sin(10 * largest_step),
        #                   10 * largest_step]
        positions_euler.append(final_euler[0])
        velocities_euler.append(final_euler[1])
        positions_rk.append(final_rk[0])
        velocities_rk.append(final_rk[1])
        positions_dp.append(final_dp[0])
        velocities_dp.append(final_dp[1])
    error_position_euler = np.array([np.abs(x - np.cos(10 * largest_step))
                                     for x in positions_euler])
    error_velocity_euler = np.array([np.abs(v + np.sin(10 * largest_step))
                                     for v in velocities_euler])
    error_position_rk = np.array([np.abs(x - np.cos(10 * largest_step))
                                  for x in positions_rk])
    error_velocity_rk = np.array([np.abs(v + np.sin(10 * largest_step))
                                  for v in velocities_rk])
    error_position_dp = np.array([np.abs(x - np.cos(10 * largest_step))
                                  for x in positions_dp])
    error_velocity_dp = np.array([np.abs(v + np.sin(10 * largest_step))
                                  for v in velocities_dp])
    plt.title("Euler, RK4, Dormand Prince Convergence")
    plt.xlabel("Step size")
    plt.ylabel("Absolute Error")
    plt.loglog(step_sizes, error_position_euler, label="Euler x")
    plt.loglog(step_sizes, error_velocity_euler, label="Euler v")
    plt.loglog(step_sizes, error_position_rk, label="RK4 x")
    plt.loglog(step_sizes, error_velocity_rk, label="RK4 v")
    plt.loglog(step_sizes, error_position_dp, label="DP x")
    plt.loglog(step_sizes, error_velocity_dp, label="DP v")
    plt.legend()
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    return [step_sizes, error_position_euler, error_velocity_euler,
            error_position_rk, error_velocity_rk, error_position_dp,
            error_velocity_dp]

if __name__ == "__main__":
    all_tests()

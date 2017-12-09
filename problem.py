"""
Encapsulation of problems that can be expressed by a system of coupled
first order (in time) ordinary differential equations.

To create a class which extends Problem, and which encapsulates the
details of the equations and variables, note that there should be an
equation for each of the n_vars variables, and the initial data should
have length (n_vars + 1) (the last index is time).

Equations with explicit dependence on time have not been tested, but
should work with all provided integration methods.
"""
import numpy as np

class Problem:
    def __init__(self, n_vars, equations, initial_data,
                 equation_deriv = True):
        """
        * n_vars is a number representing the number of parameters
        * equations is either a single equation or a list of callables
        * representing equations governing the system
        * initial_data gives the initial values of the parameters
        * equation_deriv is either a single value or a tuple of n_vars values.
         If equation_deriv is a single value v, this is equivalent to the
         tuple [v] * n_vars.
          A value of False in index k means that equation k is the explicit
         formula for variable k, and a value of True (default) means that
         equation k is the formula for the time derivative of variable k.
          If a value of False is given for equation k, then equation k should
         have no dependence on variable k (otherwise unspecified behavior).
          If objects that are not booleans or 1D lists of booleans are
         provided here, the behavior is unspecified.

        equations must represent f_i where the equations governing the system
        are in the following form: 
        \dot{x}_i (or x_i) = f_1(x_1, x_2, ..., x_n, t)
        
        For a one dimensional problem \dot{x} = x with initial conditions
        x(t = 0) = 1 (where one variable depends on time only),
        can set n_vars = 1, equations = [lambda x: x[0]], initial_data = [1, 0]

        For [] with boundary conditions []:
        n_vars = 2, equations = [], initial_data = [],
        equation_deriv = [False, True, True]

        Every function passed in equations should accept n_vars variables
        in a consistent order. Note that the last index in the variable
        accepted by each equation must be for time, even if there is no
        explicit time dependence; thus, the number of variables accepted by
        each equation should be (n_vars + 1).
        """
        self.n_vars = n_vars
        self.equations = equations
        # some sanity checks
        assert(len(equations) == n_vars)
        self.initial_data = initial_data

        # set whether the equations are governing the value or derivative
        try:
            assert(len(equation_deriv) == n_vars)
            self.equation_deriv = equation_deriv
        except TypeError as e:
            self.equation_deriv = [equation_deriv] * n_vars
        self.equation_deriv = self.equation_deriv

        assert(len(self.equation_deriv) == n_vars)

        self.deriv_mask = np.array(self.equation_deriv + [False]) * 1
        assert(len(self.deriv_mask) == n_vars + 1)
        assert(self.deriv_mask[-1] == 0)
        self.inv_deriv_mask = self.deriv_mask ^ 1
        # time index still zero
        self.inv_deriv_mask[-1] = 0
        assert(len(self.inv_deriv_mask) == n_vars + 1)
        assert(self.inv_deriv_mask[-1] == 0)
        # sanity check
        for i in range(n_vars):
            assert(self.deriv_mask[i] ^ self.inv_deriv_mask[i] == 1)

    def value_n(self, n, data):
        """
        n is which variable to calculate the current first derivative (or
        current value if equation_deriv[n] is False)
        data is the "current" values of all parameters

        0 indexing is used

        data is a list of values for all variables
        """
        return self.equations[n](data)

class Harmonic(Problem):
    """
    Single dimensional harmonic oscillator
    """
    def __init__(self, k = 1.0, m = 1.0, velocity_drag = lambda v: 0,
                 initial_position = 1.0, initial_velocity = 0.0,
                 initial_time = 0.0):
        """
        k: scalar, spring constant
        m: scalar, mass
        velocity_drag: callable which accepts velocity and returns
                       f(v) such that the system is governed by equation
                       m\ddot{x} + f(\dot{x})\dot{x} + kx = 0
        
        For a linear velocity dependent drag, use velocity_drag = lambda v: c
        where c > 0.

        Note that no attempt is made to convert integer arguments into floats
        """
        # x0 = x, x1 = \dot{x}, in equations variable x = (x0, x1)

        # \dot{x0} = x1
        eq1 = lambda x: x[1]
        # \dot{x1} = -f(x1)x1 / m - k/m x0
        eq2 = lambda x: -velocity_drag(x[1]) * x[1] / m - k / m * x[0]
        super().__init__(2, [eq1, eq2],
                         [initial_position, initial_velocity, initial_time])

class Lorenz(Problem):
    """
    Lorenz attractor

    Implements three variable problem of the form
    \dot{x} = \sigma(y - x)
    \dot{y} = x(\rho - z) - y
    \dot{z} = xy - \beta z
    """
    def __init__(self, sigma = 10, rho = 28, beta = 8/3.,
                 initial_data = [1, -0.5, 2, 0]):
        eq1 = lambda x: sigma * (x[1] - x[0])
        eq2 = lambda x: x[0] * (rho - x[2]) - x[1]
        eq3 = lambda x: x[0] * x[1] - beta * x[2]
        super().__init__(3, [eq1, eq2, eq3], initial_data)

class DoublePendulum(Problem):
    """
    A double pendulum consisting of two linked massive rods of equal
    mass and length

    Equations of motion from https://en.wikipedia.org/wiki/Double_pendulum
    """
    def __init__(self, m = 1, l = 1, g = 1,
                 # [theta1, theta2, p1, p2]
                 initial_data = [np.pi / 6., np.pi / 3., 0., 0., 0.]):
        self.params = {"m":m, "l":l, "g":g}
        eq1 = lambda x: (6. / (m * l**2)
                         * (2. * x[2] - 3. * np.cos(x[0] - x[1]) * x[3])
                         / (16. - 9. * np.cos(x[0] - x[1])**2))
        eq2 = lambda x: (6. / (m * l**2)
                         * (8. * x[3] - 3. * np.cos(x[0] - x[1]) * x[2])
                         / (16. - 9. * np.cos(x[0] - x[1])**2))
        eq3 = lambda x: (-1. / 2. * m * l**2 *
                         (eq1(x) * eq2(x) * np.sin(x[0] - x[1])
                          + 3. * g / l * np.sin(x[0])))
        eq4 = lambda x: (-1. / 2. * m * l**2 *
                         (-eq1(x) * eq2(x) * np.sin(x[0] - x[1])
                          + 1. * g / l * np.sin(x[1])))
        super().__init__(4, [eq1, eq2, eq3, eq4], initial_data)

    def locations(self, current_data):
        """
        Returns [rod1x, rod1y, rod2x, rod2y]
        end positions based on the input data (the fixed axis of the
        first rod is the origin)

        current_data: [theta1, theta2, p1, p2, t]
        """
        assert(len(current_data) == 5)
        l = self.params["l"]
        pos1x = l * np.sin(current_data[0])
        pos1y = -l * np.cos(current_data[0])
        pos2x = pos1x + l * np.sin(current_data[1])
        pos2y = pos1y - l * np.cos(current_data[1])
        return [pos1x, pos1y, pos2x, pos2y]

    def poincare(self, current_data, threshold = 1e-2):
        """
        Returns the (theta, p) of the second pendulum if theta_1 is
        sufficiently close to 0 and \dot{theta}_1 is negative.
        Otherwise returns None.
        """
        if (abs(current_data[0]) < threshold
            and current_data[2] < 0):
            return [current_data[1], current_data[3]]
        else:
            #print(abs(current_data[0]))
            return None

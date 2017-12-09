"""
Provides consistent interface for ODE integrators in several variables
"""
import numpy as np

class Integrator:
    """
    Encapsulation of an integrator that evolves a system forward in time
    """
    def __init__(self, problem):
        """
        """
        self.problem = problem
        # make a copy of the list so the original problem is never modified
        self._current_data = np.array(list(self.problem.initial_data))
        self._current_time = self._current_data[-1]
        # copy the derivative masks
        self._deriv_mask     = self.problem.deriv_mask
        self._inv_deriv_mask = self.problem.inv_deriv_mask

    def current_data(self):
        """
        Returns a copy of the current state
        """
        return self._current_data[:]
        
    def step(self, delta = None):
        """
        If delta is None, then the default current step size is used
        """
        pass

class ForwardEuler(Integrator):
    """
    The forward Euler method freezes the current values of all variables and
    then updates every variable using the derivatives based on the current
    variable values
    """
    def __init__(self, problem):
        super().__init__(problem)

    def step(self, delta = None):
        """
        If delta is None, does nothing
        """
        if delta is None:
            pass
        else:
            function_vals = []
            for i in range(self.problem.n_vars):
                function_vals.append(self.problem.value_n(i,self._current_data))
            # append a time value of zero
            function_vals.append(0)

            function_vals = np.array(function_vals)
            assert(len(function_vals) == self.problem.n_vars + 1)

            deriv_vals    = function_vals * self._deriv_mask
            explicit_vals = function_vals * self._inv_deriv_mask

            self._current_data = self._current_data * self._deriv_mask
            self._current_data = self._current_data + explicit_vals
            self._current_data = self._current_data + delta * deriv_vals
            
            self._current_time += delta
            # the previous block sets the time index to zero
            self._current_data[-1] = self._current_time

class RungeKutta(Integrator):
    """
    Implements RK4
    """
    def __init__(self, problem):
        super().__init__(problem)

    def step(self, delta = None):
        if delta is None:
            pass
        else:
            function_vals = []
            for i in range(self.problem.n_vars):
                function_vals.append(self.problem.value_n(i,self._current_data))

            # append a time value of zero
            function_vals.append(0)

            function_vals = np.array(function_vals)
            assert(len(function_vals) == self.problem.n_vars + 1)

            # these values are fixed for all k_i and the final update
            explicit_vals = function_vals * self._inv_deriv_mask

            # for each of the k arrays, need to apply the derivative mask
            k1 = []
            for i in range(self.problem.n_vars):
                k1.append(self.problem.value_n(i, self._current_data))
            # dummy time index
            k1.append(1)
            k1 = np.array(k1)
            k1 = k1 * self._deriv_mask
            k1[-1] = 1

            k2 = []
            k1_arr = self._current_data + k1 * delta / 2. * self._deriv_mask
            # k1_arr = [self._current_data[i] + k * delta / 2
            #           for i, k in enumerate(k1)] \
            #               + [self._current_time + delta / 2]
            for i in range(self.problem.n_vars):
                # add in the time separately, note that these need to be
                # normal (not numpy) arrays to work properly
                k2.append(self.problem.value_n(i, k1_arr))
            k2.append(1)
            k2 = np.array(k2)
            k2 = k2 * self._deriv_mask
            k2[-1] = 1
            
            k3 = []
            k2_arr = self._current_data + k2 * delta / 2. * self._deriv_mask
            # k2_arr = [self._current_data[i] + k * delta / 2
            #           for i, k in enumerate(k2)] \
            #               + [self._current_time + delta / 2]
            for i in range(self.problem.n_vars):
                k3.append(self.problem.value_n(i, k2_arr))
            k3.append(1)
            k3 = np.array(k3)
            k3 = k3 * self._deriv_mask
            k3[-1] = 1

            k4 = []
            k3_arr = self._current_data + k3 * delta * self._deriv_mask
            # k3_arr = [self._current_data[i] + k * delta
            #           for i, k in enumerate(k3)] \
            #               + [self._current_time + delta]
            for i in range(self.problem.n_vars):
                k4.append(self.problem.value_n(i, k3_arr))
            k4.append(1)
            k4 = np.array(k4)
            k4 *= self._deriv_mask

            # change = []
            # # everything except time
            # for i in range(self.problem.n_vars):
            #     change.append(delta / 6. * (k1[i] + 2 * k2[i]
            #                                 + 2 * k3[i] + k4[i]))
            # # time
            # change.append(delta)

            # # add the changed value to the current data
            # for i, val in enumerate(change):
            #     self._current_data[i] += val
            # self._current_time += delta

            # avoiding +=, *= because of past experience with numpy bugs
            self._current_data = self._current_data * self._deriv_mask
            self._current_data = self._current_data + explicit_vals
            self._current_data = self._current_data + \
                                 delta / 6. * (k1 + 2 * k2 + 2 * k3
                                               + k4) * self._deriv_mask

            self._current_time += delta
            # the previous block sets the time index to zero
            self._current_data[-1] = self._current_time

class DormandPrince(Integrator):
    """
    Implements the Dormand Prince adaptive Runge Kutta routine
    """
    def __init__(self, problem, step_size = 1e-3, tolerance = 1e-12,
                 adaptive = True, growth_factor = 1.5):
        super().__init__(problem)
        self.step_size = step_size
        self.tolerance = tolerance
        self.adaptive = adaptive
        self.growth_factor = growth_factor

    def step(self, delta = None):
        h = self.step_size
        # if not adaptive, use the provided value for delta
        if self.adaptive == False:
            if delta is not None:
                h = delta

        current_data_np = np.array(self._current_data)
        explicit_vals   = current_data_np * self._inv_deriv_mask
        def value_all(data):
            """
            Always returns 0 for the time part
            """
            return np.array([self.problem.value_n(i, data)
                             for i in range(self.problem.n_vars)] + [0])
        unit_time = np.array([0] * self.problem.n_vars + [1])
        k1_temp = value_all(current_data_np)
        k1 = k1_temp * self._inv_deriv_mask + h * k1_temp * self._deriv_mask
        
        k2_arg = (current_data_np + 1/5. * k1 * self._deriv_mask
                  + 1/5. * h * unit_time)
        k2_temp = value_all(k2_arg)
        k2 = k2_temp * self._inv_deriv_mask + h * k2_temp * self._deriv_mask
        
        k3_arg = (current_data_np + (3/40. * k1
                                     + 9/40. * k2) * self._deriv_mask
                  + 3/10. * h * unit_time)
        k3_temp = value_all(k3_arg)
        k3 = k3_temp * self._inv_deriv_mask + h * k3_temp * self._deriv_mask
        
        k4_arg = (current_data_np + (44/45. * k1
                                     - 56/15. * k2
                                     + 32/9. * k3) * self._deriv_mask
                  + 4/5. * h * unit_time)
        k4_temp = value_all(k4_arg)
        k4 = k4_temp * self._inv_deriv_mask + h * k4_temp * self._deriv_mask
        
        k5_arg = (current_data_np + (19372/6561. * k1
                                     - 25360/2187. * k2
                                     + 64448/6561. * k3
                                     - 212/729. * k4) * self._deriv_mask
                  + 8/9. * h * unit_time)
        k5_temp = value_all(k5_arg)
        k5 = k5_temp * self._inv_deriv_mask + h * k5_temp * self._deriv_mask
        
        k6_arg = (current_data_np + (9017/3168. * k1
                                     - 355/33. * k2
                                     + 46732/5247. * k3
                                     + 49/176. * k4
                                     - 5103 / 18656. * k5) * self._deriv_mask
                  + h * unit_time)
        k6_temp = value_all(k6_arg)
        k6 = k6_temp * self._inv_deriv_mask + h * k6_temp * self._deriv_mask
        
        k7_arg = (current_data_np + (35/384. * k1
                                     + 500/1113. * k3
                                     + 125 / 192. * k4
                                     - 2187/6784. * k5
                                     + 11/84. * k6) * self._deriv_mask
                  + h * unit_time)
        k7_temp = value_all(k7_arg)
        k7 = k7_temp * self._inv_deriv_mask + h * k7_temp * self._deriv_mask
        z = (current_data_np + (5179 / 57600. * k1
                                + 7571 / 16695. * k3
                                + 393/640. * k4
                                - 92097 / 339200. * k5
                                + 187 / 2100. * k6
                                + 1/40. * k7) * self._deriv_mask
             + h * unit_time)

        # need to manually add in the explicit part
        self._current_data = k7_arg * self._deriv_mask \
                             + k6_temp * self._inv_deriv_mask
        self._current_time += h
        self._current_data[-1] = self._current_time

        if self.adaptive:
            # update the step size
            err = (sum([k**2 for k in [z[i] - k7_arg[i]
                                       for i, v in enumerate(z)]]))**0.5
            # prevents the step size from growing too much
            # restricts each successive step to
            # [(growth factor)^-1, (growth factor)] * (current step size)
            self.step_size = min(max((self.tolerance * h
                                      / (2 * err))**(1.0 / 5.0),
                                     self.growth_factor**-1),
                                 self.growth_factor) * self.step_size

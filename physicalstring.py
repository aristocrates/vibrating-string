"""
Solves for the dynamics of a string in two dimensions with initial and
boundary conditions

Nicholas Meyer
"""
import numpy as np
import problem, integrator

class BoundaryCondition:
    """
    Encapsulates a boundary condition.

    Does not store the surface where the boundary condition applies; this
    should be handles by the external code
    """
    def __init__(self, value = lambda x: 0, nderiv = 0):
        """
        Indicates a boundary condition of value on the specified spacial
        derivative of the field value

        value is a callable that accepts a scalar input parameter (time)
        and outputs a scalar
        """
        self.value = value
        self.nderiv = nderiv

class FieldTheory:
    def __init__(self):
        pass
        
class String(FieldTheory):
    """
    Solves a field theory with the equation of motion

    d^2 \phi / dt^2 = k/rho d^2 \phi / dx^2
    """
    def __init__(self, boundary, n_elements, initial_data, k = 1., rho = 1.):
        """
        boundary: vector of 2 elements
        n_elements: a scalar representing the number of points
        initial: vector of 2 * n elements representing the initial
                 [field values, field derivatives] at each
                 point at time 0
        k, rho: spring constant and density (note that default values make
                the string change very slowly)
        """
        self.boundary = boundary
        self.n_elements = n_elements
        assert(len(initial_data) == 2 * n_elements)
        self.initial_data = initial_data
        self.k = k
        self.rho = rho

class FiniteElementIntegrator:
    def __init__(self, string, ode_integrator, stencil_dict, delta, **kwargs):
        """
        field_theory: problem to solve (instance of FieldTheory)
        ode_integrator: subclass of integrator.Integrator
        stencil_dict: dictionary representing stencil coefficients
                      e.g. first order second derivative d^2/x^2
                      given by (phi(x - delta) - 2 phi(x) + phi(x + delta))
                      would be {-1:1, 0:-2, 1:1}
                      Note that the minimum / maximum keys are used to
                      determine the leftmost / rightmost elements to use the
                      stencil to evolve
        delta: spacing between pixels
        boundary_conditions: array of 2n callables (first n corresponding to
                             the field values, the second n corresponding to
                             the field derivatives), each accepting one
                             argument (time)
                              - boundary conditions supercede the stencil is
                             applied (and at the very start of the simulation)
                              - Passing None indicates no boundary condition on
                             that element
        kwargs: special arguments to pass to construct an instance
                           ode_integrator
        """
        self.string = string
        self.delta = delta
        self.ode_integrator = ode_integrator
        self.setup_integrator(ode_integrator, stencil_dict, delta,
                              string, **kwargs)

    def setup_integrator(self, ode_integrator, stencil, delta, string,
                         **kwargs):
        n = string.n_elements
        boundary_conditions = [None] * 2 * n
        # to set boundary conditions
        eq_deriv = [True] * 2 * n

        if string.boundary[0].nderiv == 0:
            stationary_node = lambda x: string.boundary[0].value(x[2 * n])
            boundary_conditions[0] = stationary_node
            eq_deriv[0] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        if string.boundary[1].nderiv == 0:
            stationary_node = lambda x: string.boundary[1].value(x[2 * n])
            boundary_conditions[n - 1] = stationary_node
            eq_deriv[n - 1] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        # last index is for initial time = 0
        initial_conditions = string.initial_data + [0]

        minimum_access = min(stencil.keys())
        maximum_access = max(stencil.keys())
        # Don't access anything left of the minimum access point
        start_index = max(0, -1 * minimum_access)
        stop_index = -1 * min(0, -1 * maximum_access)
        assert(start_index >= 0)
        assert(stop_index >= 0)

        equations = []
        # n equations 0th derivative
        for i in range(n):
            # each parameter x is has 2n variables, equation i
            # is the equation for the time derivative of the ith variable
            assert( (i + n) < 2 * n )
            # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
            equations.append(lambda x, i = i, n = n: x[i + n])

        # n equations 1st derivative
        for i in range(n):
            # don't look beyond bounds
            if i < start_index:
                assert(i - start_index < 0)
                equations.append(lambda x: 0)
            elif i >= (n - stop_index):
                assert(i + stop_index >= n)
                equations.append(lambda x: 0)
            # otherwise apply stencil
            else:
                assert(i - start_index >= 0)
                assert(i + stop_index < n)
                # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
                def stencil_func(x, i = i):
                    val = 0
                    for k in stencil.keys():
                        val += stencil[k] * x[i + k]
                    return val / delta**2
                equations.append(stencil_func)

        # set the boundary conditions
        for i, b in enumerate(boundary_conditions):
            if b is not None:
                equations[i] = b

        prob = problem.Problem(2 * n, equations, initial_conditions,
                               equation_deriv = eq_deriv)
        self.integrator = ode_integrator(prob, **kwargs)

    def step(self, step_size):
        self.integrator.step(step_size)

class FiniteElementIntegratorGeneralized:
    def __init__(self, string, ode_integrator, extended_stencil_dict,
                 phi_time_deriv_dict, delta,
                 spatial_param_dependence = False, **kwargs):
        """
        The same as FiniteElementIntegrator, but allows for greater
        flexibility in spatial derivative dependence and
        dependence on the first time derivative for Phi in the equation of
        motion for d^2 Phi/dt

        Argument differences from FiniteElementIntegrator:

        extended_stencil_dict: list of dictionaries [k] where k has the form
                               {'stencil':{-1:1, 0:-2, 1:1},
                                'coefficient':2,
                                'power':3}
                               This can be used to construct arbitrary
                               (uncoupled, "polynomial") dependence on
                               spatial derivatives, i.e. Phi^3, 2(dPhi/dt)^2
                               by choosing the stencil appropriately and
                               by choosing the desired coefficients and
                               powers
        phi_time_deriv_dict: list of dictionaries [d] where d has the form
                             {'deriv': 1, 'coefficient':-1, 'power':1}
        """
        self.string = string
        self.delta = delta
        self.ode_integrator = ode_integrator
        self.setup_integrator(ode_integrator, extended_stencil_dict,
                              phi_time_deriv_dict, delta, string,
                              spatial_param_dependence, **kwargs)

    def setup_integrator(self, ode_integrator, extended_stencil,
                         time_derivs, delta, string,
                         spatial_param_dependence, **kwargs):
        n = string.n_elements
        boundary_conditions = [None] * 2 * n
        # to set boundary conditions
        eq_deriv = [True] * 2 * n

        def deriv_spec(index_i, data):
            """
            Evaluates the sum of all derivative terms for index_i
            index_i should be between n and 2n - 1, corresponding to
            the equation for the derivative of the first derivative
            of the field at position index_i

            data is an array of length (2n + 1), the current data
            """
            assert(n <= index_i < 2 * n)
            ans = 0
            for d in time_derivs:
                if d['deriv'] < 0:
                    msg = "Time derivative (%s) cannot be negative" \
                          % str(d['deriv'])
                    raise ValueError(msg)
                if d['deriv'] > 1:
                    msg = "Time derivatives beyond 1 (%s) not yet supported" \
                          % str(d['deriv'])
                    raise ValueError(msg)
                # the first derivative is index_i, the second is index_i - n
                if d['deriv'] == 1:
                    if spatial_param_dependence:
                        ans += d['coefficient'][index_i - n] * (data[index_i])**(d['power'])
                        #print(len(d['coefficient']))
                    else:
                        ans += d['coefficient'] * (data[index_i])**(d['power'])
                else:
                    assert(d['deriv'] == 0)
                    if spatial_param_dependence:
                        ans += d['coefficient'][index_i - n] * (data[index_i - n])**(d['power'])
                        #print(len(d['coefficient']))
                    else:
                        ans += d['coefficient'] * (data[index_i - n])**(d['power'])
            return ans
        # end deriv_spec

        def stencil_spec(index_i, data):
            """
            Evaluates the sum of all stencil terms for index_i
            index_i should be between n and 2n - 1, corresponding to
            the equation for the derivative of the first derivative
            of the field at position index_i

            data: same as deriv_spec
            """
            assert(n <= index_i < 2 * n)
            def stencil_helper(dat, index, stencil, delta):
                val = 0
                for k in stencil.keys():
                    val += stencil[k] * dat[index + k]
                #print(val)
                return val / delta**2
            ans = 0
            for stenc in extended_stencil:
                stencil_val = stencil_helper(data, index_i - n,
                                             stenc['stencil'], delta)
                assert(not np.isnan(stencil_val))
                if spatial_param_dependence:
                    ans += stenc['coefficient'][index_i - n] * (stencil_val)**(stenc['power'])
                else:
                    ans += stenc['coefficient'] * (stencil_val)**(stenc['power'])
                #print(stencil_val)
            return ans
        # end stencil_spec

        if string.boundary[0].nderiv == 0:
            stationary_node = lambda x: string.boundary[0].value(x[2 * n])
            boundary_conditions[0] = stationary_node
            eq_deriv[0] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        if string.boundary[1].nderiv == 0:
            stationary_node = lambda x: string.boundary[1].value(x[2 * n])
            boundary_conditions[n - 1] = stationary_node
            eq_deriv[n - 1] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        # last index is for initial time = 0
        initial_conditions = string.initial_data + [0]

        # modify this to get the minimum from all stencils
        minimum_access = min([min(stencil.keys())
                              for s in extended_stencil
                              for stencil in [s['stencil']]])
        maximum_access = max([max(stencil.keys())
                              for s in extended_stencil
                              for stencil in [s['stencil']]])
        # Don't access anything left of the minimum access point
        start_index = max(0, -1 * minimum_access)
        stop_index = -1 * min(0, -1 * maximum_access)
        assert(start_index >= 0)
        assert(stop_index >= 0)

        equations = []
        # n equations 0th derivative
        for i in range(n):
            # each parameter x is has 2n variables, equation i
            # is the equation for the time derivative of the ith variable
            assert( (i + n) < 2 * n )
            # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
            equations.append(lambda x, i = i, n = n: x[i + n])

        # n equations 1st derivative
        for i in range(n):
            # don't look beyond bounds
            if i < start_index:
                assert(i - start_index < 0)
                equations.append(lambda x: 0)
            elif i >= (n - stop_index):
                assert(i + stop_index >= n)
                equations.append(lambda x: 0)
            # otherwise apply stencil
            else:
                assert(i - start_index >= 0)
                assert(i + stop_index < n)
                # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
                def stencil_func(x, i = i, n = n):
                    # print("deriv spec: " + str(deriv_spec(i + n, x)))
                    # print("stencil spec: " + str(stencil_spec(i + n, x)))
                    return (deriv_spec(i + n, x) + stencil_spec(i + n, x))
                    # val = 0
                    # for k in stencil.keys():
                    #     val += stencil[k] * x[i + k]
                    # return val / delta**2
                equations.append(stencil_func)

        # set the boundary conditions
        for i, b in enumerate(boundary_conditions):
            if b is not None:
                equations[i] = b

        prob = problem.Problem(2 * n, equations, initial_conditions,
                               equation_deriv = eq_deriv)
        self.integrator = ode_integrator(prob, **kwargs)

    def step(self, step_size):
        self.integrator.step(step_size)
        
class Central(FiniteElementIntegrator):
    """
    Implements a second derivative, second order finite difference scheme
    """
    def __init__(self, string, ode_integrator, delta, **kwargs):
        """
        delta is the (horizontal) separation distance of each point
        (this should be moved into the string class)
        """
        super().__init__(string, ode_integrator, delta, **kwargs)
        # construct a single integrator

        # the 4 variables are (in order) \phi_{-1}, \phi, \phi_{+1},
        # and d\phi/dt
        # the time derivative of d\phi/dt is given by the stencil applied
        # to \phi_{-1}, \phi, \phi_{+1}

        # note that the equations of motion will only touch \phi and d\phi/dt
        # since \phi_{-1} and \phi_{+1} will be modified directly by calling
        # Integrator.set_current_data
        def stencil(xminus1, x, xplus1, delta):
            return (1. * xminus1 - 2. * x + 1. * xplus1) / delta**2
        equations = [lambda x: 0., lambda x: x[3], lambda x: 0.,
                     lambda x: string.k / string.rho *
                     stencil(x[0], x[1], x[2], delta)]
        # the initial data is irrelevant here because the current data will
        # be set every time before calling step()
        # [0] * 5 because the last index is time
        finite_element_prob = problem.Problem(n_vars = 4, equations = equations, initial_data = [0.] * 5)
        self.integrator = ode_integrator(finite_element_prob, **kwargs)

        self.string = string
        # initially, the time derivative everywhere on the string is 0
        # perhaps this array should be moved inside the string class
        self.time_derivatives = [0] * self.string.n_elements
        self.time = 0.

    def step(self, step_size):
        # create buffer for new values of the string
        data_buffer = []
        deriv_buffer = []
        # haven't implemented derivative boundary conditions yet
        if (self.string.boundary[0].nderiv != 0
            or self.string.boundary[1].nderiv != 0):
            msg = "Derivative boundary conditions not yet supported"
            raise(NotImplementedError(msg))
        # enforce left boundary condition
        data_buffer.append(self.string.boundary[0].value)
        deriv_buffer.append(0)

        for i in range(self.string.n_elements - 2):
            data_segment = ([k for k in self.string.current_data[i:i+3]]
                            + self.time_derivatives[i:i+1] + [self.time])
            #if i == 10:
            #    print(data_segment)
            # stencil using 3 elements
            assert(len(data_segment) == 5)
            # TODO: check this
            self.integrator.set_current_data(data_segment)
            self.integrator.step(delta = step_size)
            # just pick out the center value and derivative
            data_buffer.append(self.integrator.current_data()[1])
            deriv_buffer.append(self.integrator.current_data()[3])
        # enforce right boundary condition
        data_buffer.append(self.string.boundary[1].value)
        deriv_buffer.append(0)

        self.time += step_size
        self.string.current_data = data_buffer[:]
        self.time_derivatives = deriv_buffer[:]
        #print(max(self.time_derivatives))
        #print(min(self.time_derivatives))


class FiniteElementIntegratorFermiPastaUlamTsingou:
    def __init__(self, string, ode_integrator, alpha, delta, **kwargs):
        """
        field_theory: problem to solve (instance of FieldTheory)
        ode_integrator: subclass of integrator.Integrator
        stencil_dict: dictionary representing stencil coefficients
                      e.g. first order second derivative d^2/x^2
                      given by (phi(x - delta) - 2 phi(x) + phi(x + delta))
                      would be {-1:1, 0:-2, 1:1}
                      Note that the minimum / maximum keys are used to
                      determine the leftmost / rightmost elements to use the
                      stencil to evolve
        delta: spacing between pixels
        boundary_conditions: array of 2n callables (first n corresponding to
                             the field values, the second n corresponding to
                             the field derivatives), each accepting one
                             argument (time)
                              - boundary conditions supercede the stencil is
                             applied (and at the very start of the simulation)
                              - Passing None indicates no boundary condition on
                             that element
        kwargs: special arguments to pass to construct an instance
                           ode_integrator
        """
        self.string = string
        self.delta = delta
        self.ode_integrator = ode_integrator
        self.setup_integrator(ode_integrator, alpha, delta,
                              string, **kwargs)

    def setup_integrator(self, ode_integrator, alpha, delta, string,
                         **kwargs):
        n = string.n_elements
        boundary_conditions = [None] * 2 * n
        # to set boundary conditions
        eq_deriv = [True] * 2 * n

        if string.boundary[0].nderiv == 0:
            stationary_node = lambda x: string.boundary[0].value(x[2 * n])
            boundary_conditions[0] = stationary_node
            eq_deriv[0] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        if string.boundary[1].nderiv == 0:
            stationary_node = lambda x: string.boundary[1].value(x[2 * n])
            boundary_conditions[n - 1] = stationary_node
            eq_deriv[n - 1] = False
        else:
            err = "Boundary conditions on the first derivative not supported"
            raise NotImplementedError(err)
        # last index is for initial time = 0
        initial_conditions = string.initial_data + [0]

        minimum_access = -1
        maximum_access = 1
        # Don't access anything left of the minimum access point
        start_index = max(0, -1 * minimum_access)
        stop_index = -1 * min(0, -1 * maximum_access)
        assert(start_index >= 0)
        assert(stop_index >= 0)

        equations = []
        # n equations 0th derivative
        for i in range(n):
            # each parameter x is has 2n variables, equation i
            # is the equation for the time derivative of the ith variable
            assert( (i + n) < 2 * n )
            # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
            equations.append(lambda x, i = i, n = n: x[i + n])

        # n equations 1st derivative
        for i in range(n):
            # don't look beyond bounds
            if i < start_index:
                assert(i - start_index < 0)
                equations.append(lambda x: 0)
            elif i >= (n - stop_index):
                assert(i + stop_index >= n)
                equations.append(lambda x: 0)
            # otherwise apply stencil
            else:
                assert(i - start_index >= 0)
                assert(i + stop_index < n)
                # https://stackoverflow.com/questions/21053988/lambda-function-acessing-outside-variable scoping workaround
                def stencil_func(x, i = i):
                    np.seterr(all="raise")
                    try:
                        ans = ((x[i - 1] - 2 * x[i] + x[i + 1])
                               * (1.0 - alpha * (x[i + 1] - x[i - 1])) / delta**2)
                    except FloatingPointError:
                        import pdb; pdb.set_trace()
                    return ans
                #+ alpha * (x[i + 1] - x[i - 1]))) / delta**2
                    # val = 0
                    # for k in stencil.keys():
                    #     val += stencil[k] * x[i + k]
                    # return val / delta**2
                equations.append(stencil_func)

        # set the boundary conditions
        for i, b in enumerate(boundary_conditions):
            if b is not None:
                equations[i] = b

        prob = problem.Problem(2 * n, equations, initial_conditions,
                               equation_deriv = eq_deriv)
        self.integrator = ode_integrator(prob, **kwargs)

    def step(self, step_size):
        self.integrator.step(step_size)

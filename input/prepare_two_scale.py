#!/usr/bin/env python3
import sympy
from sympy.parsing.sympy_parser import parse_expr
import re

def laplace(f, vars):
    return sum([sympy.diff(f, i, i) for i in vars])


def grad(f, vars):
    return [sympy.diff(f, i) for i in vars]


def n_deriv(f, vars, normal):
    return sum([grad(f, vars)[j] * normal[j] for j in range(len(vars))])


def time_deriv(f, t):
    return sympy.diff(f, t)


def boundary_flux(f, vars, axis):
    """
    Flux term (\\nabla f \\cdot n) for a single axis (horizontal or vertical boundary).
    Abusing the fact that n = xe_0 / n=ye_1 / ... recovers horizontal / vertical / ... normal in case of [-1,1]^d domain.

    :param f: function to take flux from
    :param vars: variables to obtain the gradient (and normal) of
    :param axis: 0 (horizontal) or 1 (vertical) or (2 (azymuthal), if your system can handle it)
    :return: boundary flux expression
    """
    return grad(f, vars)[axis] * vars[axis]


def boundary_integral(f, vars):
    if len(vars) == 2:
        x0, x1 = vars
        normals = ((1, 0), (0, 1), (-1, 0), (0, -1))  # Can for sure be improved
        vals = [sympy.integrate(n_deriv(f, vars, normals[0]).subs(x0, 1), (x1, -1, 1)),
                sympy.integrate(n_deriv(f, vars, normals[1]).subs(x1, 1), (x0, -1, 1)),
                sympy.integrate(n_deriv(f, vars, normals[2]).subs(x0, -1), (x1, -1, 1)),
                sympy.integrate(n_deriv(f, vars, normals[3]).subs(x1, -1), (x0, -1, 1))]
        return sum(vals)
    else:
        raise NotImplemented("Not working for %dD" % len(vars))


def bulk_integral(f, vars):
    for var in vars:
        f = sympy.integrate(f, (var, -1, 1))
    return f


def macro_functional(pi, rho, yvars, consts, nonlinear):
    if nonlinear:
        abs_pi = sympy.Abs(pi)
        abs_rho = sympy.Abs(bulk_integral(rho, yvars))
        return consts['theta'] * sympy.Min(abs_pi, sympy.sqrt(abs_pi)) * sympy.Min(1, abs_rho)
    else:
        return consts['theta'] * pi * bulk_integral(rho, yvars)


def compute_solution_set(pi, rho, xvars, yvars, t, consts, nonlinear):
    del_pi = laplace(pi, xvars)
    del_rho = laplace(rho, yvars)
    macro_rhs = - macro_functional(pi, rho, yvars, consts, nonlinear) - consts['A'] * del_pi
    micro_rhs = time_deriv(rho, t) - consts['D'] * del_rho
    HORIZONTAL, VERTICAL = 0, 1
    neumann_rhs = consts['D'] * boundary_flux(rho, yvars, HORIZONTAL)
    robin_rhs = consts['D'] * boundary_flux(rho, yvars, VERTICAL) - consts['kappa'] * (
            pi + consts['p_F'] - consts['R'] * rho)
    init_rho = rho.subs(t, 0)
    funcs = {"macro_rhs": macro_rhs, "micro_rhs": micro_rhs,
             "macro_bc": pi, "micro_bc_neumann": neumann_rhs,
             "micro_bc_robin": robin_rhs, "init_rho": init_rho,
             "macro_solution": pi, "micro_solution": rho, "nonlinear": nonlinear}
    return funcs


def write_param_file(filename, funcs, parameters):
    data = funcs.copy()
    data.update(parameters)
    data['micro_geometry'] = "[-1,1]x[-1,1]"
    data['macro_geometry'] = "[-1,1]x[-1,1]"

    with open('%s.prm' % filename, 'w') as param_file:
        param_file.write("# This parameter file has been automatically generated and is deal.II compliant\n")
        for key, val in data.items():
            formatted_val = str(val)
            formatted_val = re.sub(r'\bAbs\b', 'abs', formatted_val)
            formatted_val = re.sub(r'\bTrue\b', 'true', formatted_val)
            formatted_val = re.sub(r'\bMin\b', 'min', formatted_val)
            formatted_val = re.sub(r'\bre\b', '', formatted_val)
            formatted_val = formatted_val.replace('**', '^')
            param_file.write("set %s = %s\n" % (key, formatted_val))


def create_new_case(name, pi_def, rho_def, nonlinear):
    xvars = sympy.symbols('x0 x1')
    yvars = sympy.symbols('y0 y1')
    t = sympy.symbols('t')
    const_vals = {'A': 3.0, 'D': 1, 'theta': 0.5, 'kappa': 1, 'p_F': 4, 'R': 2}
    const_symbols = sympy.symbols(list(const_vals))
    consts = {char: symb for char, symb in zip(const_vals.keys(), const_symbols)}
    pi = parse_expr(pi_def)
    rho = parse_expr(rho_def)
    funcs = compute_solution_set(pi, rho, xvars, yvars, t, consts, nonlinear)
    write_param_file(name, funcs, const_vals)


if __name__ == '__main__':
    print("Manufactured function creator.\n"
          "We solve an elliptic-parabolic system.\n"
          "Dimensions 2, use x0,...,xd for the macro variable, "
          "y0,...,yd for the micro variable and t for time variable.", flush=True)
    pi = input("Supply macro function. pi(x0,...,xd,t) = ")
    rho = input("Supply micro function. rho(x0,...,xd,y0,...,yd,t) = ")
    if not pi or not rho:
        print("Going for default sets", flush=True)
        pi = "(cos(exp(-D*t)*2*sin(1)*sqrt(theta/A)*x0) + cos(exp(-D*t)*2*sin(1)*sqrt(theta/A)*x1))"
        rho = "exp(-2*D*t)*cos(y0)*cos(y1)"

    nonlinear = bool(input("Running nonlinear version? [1/0]: "))
    print("Nonlinearity set to %s" % nonlinear)
    name = input("Supply solution set name: ")
    print("pi(x0,...,xd) = %s\nv(x0,...,xd,y0,...,yd) = %s\nStoring in '%s.prm'" % (pi, rho, name), flush=True)
    input("If happy, press enter, else Ctrl-C: ")
    create_new_case(name, pi, rho, nonlinear)
    print("Successfully written new parameter set")

# u = sin(x0*x1) + cos(x0 + x1)
# v = exp(y0**2 + y1**2) + x0**2 + x1**2
# a,b = symbols('a b', nonzero=True)
# w = exp(a*x0+b*x1)
#
# del_u = laplace(u,[x0,x1])
# del_v = laplace(v,[y0,y1])
#
# print(grad(w,[x0,x1]))
# print(boundary_integral(w,[x0,x1]))

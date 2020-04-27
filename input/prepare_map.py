#!/usr/bin/env python3
from sympy import *
from sympy.parsing.sympy_parser import parse_expr


def laplace(f, vars):
    return sum([diff(f, i, i) for i in vars])

def grad(f, vars):
    return [diff(f, i) for i in vars]


def map_function(u, map, vars):
    return u.subs({vars[i]: map[i] for i in range(len(vars))}, simultaneous=True)

def n_deriv(f, vars, normal):
    return sum([grad(f, vars)[j] * normal[j] for j in range(len(vars))])


def boundary_integral(f, vars):
    if len(vars) == 2:
        x0, x1 = vars
        normals = ((1, 0), (0, 1), (-1, 0), (0, -1))  # Can for sure be improved
        vals = [integrate(n_deriv(f, vars, normals[0]).subs(x0, 1), (x1, -1, 1)),
                integrate(n_deriv(f, vars, normals[1]).subs(x1, 1), (x0, -1, 1)),
                integrate(n_deriv(f, vars, normals[2]).subs(x0, -1), (x1, -1, 1)),
                integrate(n_deriv(f, vars, normals[3]).subs(x1, -1), (x0, -1, 1))]
        return sum(vals)
    else:
        raise NotImplemented("Not working for %dD" % len(vars))


def bulk_integral(f, vars):
    for var in vars:
        f = integrate(f, (var, -1, 1))
    return f


def compute_solution_set(u, v, chi_inv, xvars, yvars, micro_integration='bulk'):
    del_u = laplace(u, xvars)
    map_v = map_function(v, chi_inv, yvars)
    del_map_v = laplace(map_v, yvars)

    if micro_integration == 'flux':
        integral_v = boundary_integral(map_v, yvars)
    elif micro_integration == 'bulk':
        integral_v = bulk_integral(map_v, yvars)
    else:
        raise ValueError("Choose micro_integration to be either 'bulk' or 'flux', not %s" % micro_integration)
    macro_rhs = - integral_v - del_u
    micro_rhs = - u - del_map_v # I think?
    funcs = {"macro_rhs": macro_rhs, "micro_rhs": micro_rhs,
             "macro_bc": u, "micro_bc": v,
             "macro_solution": u, "micro_solution": v}
    return funcs


def write_param_file(filename, funcs):
    data = funcs.copy()
    data['micro_geometry'] = "[-1,1]x[-1,1]"
    data['macro_geometry'] = "[-1,1]x[-1,1]"

    with open('%s.prm' % filename, 'w') as param_file:
        for key, val in data.items():
            formatted_val = str(val).replace('**', '^')
            param_file.write("set %s = %s\n" % (key, formatted_val))


def create_new_case(name, u_def, v_def, chi_def, chi_inv_def):
    xvars = symbols('x0 x1')
    yvars = symbols('y0 y1')
    u = parse_expr(u_def)
    v = parse_expr(v_def)
    chi_inv = parse_expr(chi_inv_def)
    funcs = compute_solution_set(u, v, chi, chi_inv, xvars, yvars)
    write_param_file(name, funcs)


if __name__ == '__main__':
    print("Manufactured function creator.\n"
          "We solve a elliptic-elliptic system.\n"
          "Dimensions 2, use x0,...,xd for the macro variable "
          "and y0,...,yd for the micro variable.", flush=True)
    u = input("Supply macro function. u(x0,...,xd) = ")
    v = input("Supply micro function. v(x0,...,xd,y0,...,yd) = ")
    chi = input("Supply mapping function. chi(x0,...,xd,y0,...,yd) = ")
    chi_inv = input("Supply inverse mapping function. chi^-1(x0,...,xd,y0,...,yd) = ")
    if not (u and v and chi and chi_inv):
        print("Going for default sets", flush=True)
        u = "sin(x0*x1) + cos(x0 + x1)"
        v = "exp(x0**2 + x1**2) + (1-y0**2)*(1-y1**2)"
        chi = "y0,y1"  # fixme
    name = input("Supply solution set name: ")
    print("u(x0,...,xd) = %s\nv(x0,...,xd,y0,...,yd) = %s\nStoring in '%s.prm'" % (u, v, name), flush=True)
    input("If happy, press enter, else Ctrl-C: ")
    create_new_case(name, u, v)
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

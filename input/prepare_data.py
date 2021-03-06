#!/usr/bin/env python3
from sympy import *
from sympy.parsing.sympy_parser import parse_expr
import sys, re
from configparser import ConfigParser


def laplace(f, vars):
    return sum([diff(f, i, i) for i in vars])  # replace with matrices


def grad(f, vars):
    return [diff(f, i) for i in vars]


def map_function(u, map, vars):
    return u.subs({vars[i]: map[i] for i in range(len(vars))}, simultaneous=True)


def n_deriv(f, vars, normal):
    return sum([grad(f, vars)[j] * normal[j] for j in range(len(vars))])


def jacobian(map, vars):
    m_vars = Matrix(vars)
    return map.jacobian(m_vars)


def boundary_integral(f, vars):
    # todo: Implement map for this # check line integral
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


def parse_mapping(chi_def, sep=';', dim=2):
    chis_def = chi_def.split(sep)
    if len(chis_def) != dim:
        raise ValueError("Mapping has %d dimensions instead of %d", (len(chis_def), dim))
    chi = Matrix([parse_expr(cdef) for cdef in chis_def])
    return chi


def bulk_integral(f, vars):
    for var in vars:
        f = integrate(f, (var, -1, 1))
    return f


def matrix_repr(matrix):
    return ';'.join([str(el) for el in matrix])


def compute_elliptic_problem(u, v, chi, xvars, yvars, micro_integration='bulk'):
    del_u = laplace(u, xvars)
    del_v = laplace(v, yvars)
    map_v = map_function(v, chi, yvars)

    jac = jacobian(chi, yvars)
    # Use reference domain to integrate, but we want the integral on the deformed domain
    integral_v = bulk_integral(map_v * jac.det(), yvars)
    macro_rhs = - integral_v - del_u
    micro_rhs = - u - del_v

    funcs = {"macro_rhs": macro_rhs, "micro_rhs": micro_rhs,
             "macro_solution": u, "micro_solution": v,
             "macro_bc": u, "micro_bc": v,
             "mapping": matrix_repr(chi), "jac_mapping": matrix_repr(jac)}
    return funcs


def write_param_file(filename, funcs):
    data = funcs.copy()
    data['micro_geometry'] = "[-1,1]x[-1,1]"
    data['macro_geometry'] = "[-1,1]x[-1,1]"

    with open('%s.prm' % filename, 'w') as param_file:
        for key, val in data.items():
            formatted_val = str(val).replace('**', '^')
            param_file.write("set %s = %s\n" % (key, formatted_val))


def create_new_elliptic_case(name, u_def, v_def, chi_def):
    xvars = symbols('x0 x1')
    yvars = symbols('y0 y1')
    u = parse_expr(u_def.replace('^', '**'))
    v = parse_expr(v_def.replace('^', '**'))
    chi = parse_mapping(chi_def.replace('^', '**'))
    funcs = compute_elliptic_problem(u, v, chi, xvars, yvars)
    write_param_file(name, funcs)


def elliptic_wizard():
    print("We solve an elliptic-elliptic system.\n"
          "Dimensions 2, use x0,...,xd for the macro variable "
          "and y0,...,yd for the micro variable.", flush=True)
    u = input("Supply macro function. u(x0,...,xd) = ")
    v = input("Supply micro function. v(x0,...,xd,y0,...,yd) = ")
    chi = input("Supply mapping function. chi(x0,...,xd,y0,...,yd) = ")
    if not (u and v and chi):
        print("Going for default sets", flush=True)
        u = "sin(x0*x1) + cos(x0 + x1)"
        v = "exp(x0**2 + x1**2) + (1-y0**2)*(1-y1**2)"
        chi = "y0;y1"
    name = input("Supply solution set name: ")
    if not name:
        name = "test"
    print("u(x0,...,xd) = %s\nv(x0,...,xd,y0,...,yd) = %s\nchi(x0,...,xd,y0,...,yd) = %s\nStoring in '%s.prm'" % (
        u, v, chi, name), flush=True)
    input("If happy, press enter, else Ctrl-C: ")
    create_new_elliptic_case(name, u, v, chi)
    print("Successfully written new parameter set")


def read_functions(config_file):
    config = ConfigParser()
    config.read(config_file)
    try:
        name = re.search(r'(.*/)?([^/]+)\.ini', config_file).group(2)
    except AttributeError:
        raise ValueError("Make sure config file ends with '.ini'")
    u = config.get('functions', 'u')
    v = config.get('functions', 'v')
    chi = config.get('functions', 'chi')
    create_new_elliptic_case(name, u, v, chi)
    print("Successfully written new parameter set")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Manufactured system creation wizard")
        elliptic_wizard()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        read_functions(filename)
    else:
        raise ValueError("Too many arguments")

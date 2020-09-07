#!/usr/bin/env python3
from sympy import *
from sympy.core import expr
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


def mapped_micro_measures(boundary, map, vars):
    t = symbols('t')
    move = 1 - 2 * t
    directions = {'up': (move, 1), 'right': (1, -move), 'down': (-move, -1), 'left': (-1, move)}
    if boundary not in directions:
        raise AttributeError("Direction %s not present" % boundary)
    params = directions[boundary]
    mapped_boundary = map.subs({vars[i]: params[i] for i in range(len(vars))})
    path_integrand = mapped_boundary.diff(t).norm()
    path_length = integrate(path_integrand, (t, 0, 1))
    return path_length


def mapped_normal(direction, map, vars):
    directions = {'up': (0, 1), 'right': (1, 0), 'down': (0, -1), 'left': (-1, 0)}
    if direction not in directions:
        raise AttributeError("Direction %s not present" % direction)
    normal = directions[direction]
    inv_jac = jacobian(map, vars).inv()
    m_normal = inv_jac.T * Matrix(normal)
    m_normal = m_normal / m_normal.norm()
    return m_normal


def mapped_boundary_integral(direction, f, map, vars):
    t = symbols('t')
    move = 1 - 2 * t
    directions = {'up': (move, 1), 'right': (1, -move), 'down': (-move, -1), 'left': (-1, move)}
    if direction not in directions:
        raise AttributeError("Direction %s not present" % direction)
    boundary = directions[direction]
    mapped_boundary = map.subs({vars[i]: boundary[i] for i in range(len(vars))})
    integrand = f.subs({vars[i]: mapped_boundary[i] for i in range(len(vars))})
    jac = mapped_boundary.diff('t').norm()
    result = integrate(integrand * jac, (t, 0, 1))
    return result


def mapped_n_deriv(direction, f, map, vars):
    normal = mapped_normal(direction, map, vars)
    grad_f = grad(f, vars)
    return grad_f[0] * normal[0] + grad_f[1] * normal[1]


def get_macro_n_deriv(axis, f, vars):
    normal = [0] * len(vars)
    if axis not in vars:
        raise AttributeError("Axis %s not in %s" % (axis, vars))
    normal[vars.index(axis)] = axis  # Normal vector adjusted for direction
    grad_f = grad(f, vars)
    return sum([grad_f[i] * normal[i] for i in range(len(vars))])


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


def compute_biomath_problem(u, v, w, chi, xvars, yvars):
    const_vals = {'kappa_1': 1.0, 'kappa_2': 1, 'kappa_3': 1, 'kappa_4': -1, 'D_1': 4, 'D_2': 1}
    const_symbols = symbols(list(const_vals))
    k_1, k_2, k_3, k_4, D_1, D_2 = const_symbols
    INFLOW_BOUNDARY = 'left'
    OUTFLOW_BOUNDARY = 'right'
    del_u = laplace(u, xvars)
    del_v = laplace(v, yvars)
    del_w = laplace(w, xvars)
    jac = jacobian(chi, yvars)
    f_v = -D_2 * del_v
    g_1_v = D_2 * mapped_n_deriv(INFLOW_BOUNDARY, v, chi, yvars) - (k_1 * u - k_2 * v)
    g_2_v = D_2 * mapped_n_deriv(OUTFLOW_BOUNDARY, v, chi, yvars) - (k_3 * w - k_4 * v)
    g_3_v = D_2 * mapped_n_deriv('up', v, chi, yvars)
    g_4_v = D_2 * mapped_n_deriv('down', v, chi, yvars)
    inflow_func = mapped_micro_measures(INFLOW_BOUNDARY, chi, yvars)
    outflow_func = mapped_micro_measures(OUTFLOW_BOUNDARY, chi, yvars)
    g_1_u = u
    g_2_u = get_macro_n_deriv(xvars[1], u, xvars)
    g_1_w = D_1 * get_macro_n_deriv(xvars[0], w, xvars)
    g_2_w = D_1 * get_macro_n_deriv(xvars[1], w, xvars)
    mbi = mapped_boundary_integral(INFLOW_BOUNDARY, k_1 * u - k_2 * v + g_1_v, chi, yvars)
    f_u = -del_u + mbi
    f_w = -D_1 * del_w + mapped_boundary_integral(OUTFLOW_BOUNDARY, k_3 * w - k_4 * v + g_2_v, chi, yvars)

    funcs = {"solution_u": u, "bulk_rhs_u": f_u, "bc_u_1": g_1_u, "bc_u_2": g_2_u, "solution_w": w, "bulk_rhs_w": f_w,
             "bc_w_1": g_1_w, "bc_w_2": g_2_w, "inflow_measure": inflow_func, "outflow_measure": outflow_func,
             "solution_v": v, "bulk_rhs_v": f_v, "bc_v_1": g_1_v, "bc_v_2": g_2_v, "bc_v_3": g_3_v, "bc_v_4": g_4_v,
             "mapping": chi, "jac_mapping": jac, 'micro_geometry': "[-1,1]x[-1,1]",
             'macro_geometry': "[-1,1]x[-1,1]", **const_vals}
    return funcs


def compute_single_scale_problem(funcs):
    const_vals = {'kappa_1': 0, 'kappa_2': 1, 'kappa_3': -1, 'kappa_4': 0, 'D_2': 1, 'D_1': 0, 'x0': -1, 'x1': -1}
    funcs_ = funcs.copy()
    x, y = symbols('x y')
    for func, val in funcs.items():
        if isinstance(val, expr.Expr) or isinstance(val, Matrix):
            funcs_[func] = val.subs(const_vals)
            funcs_[func] = funcs_[func].subs({'y0': x, 'y1': y})
    # Turn y0 and y1 in x and y
    ref_solution = map_function(funcs_['solution_v'], funcs_['mapping'], (x, y))
    single_scale_funcs = {'solution': funcs_['solution_v'], 'rhs': funcs_['bulk_rhs_v'], 'mapping': funcs_['mapping'],
                          'jac_mapping': funcs_['jac_mapping'], 'left_robin': funcs_['bc_v_1'],
                          'right_robin': funcs_['bc_v_2'], 'up_neumann': funcs_['bc_v_3'],
                          'down_neumann': funcs_['bc_v_4'], 'ref_solution': ref_solution}
    return single_scale_funcs


def write_param_file(filename, data):
    with open('%s.prm' % filename, 'w') as param_file:
        for key, val in data.items():
            if isinstance(val, Matrix):
                val = matrix_repr(val)
            formatted_val = str(val).replace('**', '^')
            param_file.write("set %s = %s\n" % (key, formatted_val))


def create_new_biomath_case(name, u_def, v_def, w_def, chi_def):
    xvars = symbols('x0 x1')
    yvars = symbols('y0 y1')
    u = parse_expr(u_def.replace('^', '**'))
    v = parse_expr(v_def.replace('^', '**'))
    w = parse_expr(w_def.replace('^', '**'))
    chi = parse_mapping(chi_def.replace('^', '**'))
    funcs = compute_biomath_problem(u, v, w, chi, xvars, yvars)
    write_param_file(name, funcs)
    single_funcs = compute_single_scale_problem(funcs)
    write_param_file('validate.' + name, single_funcs)


def bio_wizard():
    print("We solve an elliptic-elliptic system.\n"
          "Dimensions 2, use x0,...,xd for the macro variable "
          "and y0,...,yd for the micro variable.", flush=True)
    u = input("Supply macro u function. u(x0,...,xd) = ")
    w = input("Supply macro w function. w(x0,...,xd,y0,...,yd) = ")
    v = input("Supply micro v function. v(x0,...,xd,y0,...,yd) = ")
    chi = input("Supply mapping function. chi(x0,...,xd,y0,...,yd) = ")
    if not (u and v and w and chi):
        print("Going for default sets", flush=True)
        u = "sin(x0*x1) + cos(x0 + x1)"
        v = "exp(x0**2 + x1**2) + (1-y0**2)*(1-y1**2)"
        w = "sin(x0*x1) + cos(x0 + x1)"
        chi = "y0;y1"
    name = input("Supply solution set name: ")
    if not name:
        name = "test"
    print(
        "u(x0,...,xd) = %s\nw(x0,...,xd) = %s\nv(x0,...,xd,y0,...,yd) = %s\nchi(x0,...,xd,y0,...,yd) = %s\nStoring in "
        "'%s.prm'" % (u, w, v, chi, name), flush=True)
    input("If happy, press enter, else Ctrl-C: ")
    create_new_biomath_case(name, u, v, w, chi)
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
    w = config.get('functions', 'w')
    chi = config.get('functions', 'chi')
    create_new_biomath_case(name, u, v, w, chi)
    print("Successfully written new parameter set")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Manufactured system creation wizard")
        bio_wizard()
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        read_functions(filename)
    else:
        raise ValueError("Too many arguments")

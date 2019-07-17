from sympy import *
from sympy.parsing.sympy_parser import parse_expr


def laplace(f, vars):
    return sum([diff(f, i, i) for i in vars])


def grad(f, vars):
    return [diff(f, i) for i in vars]


def n_deriv(f, vars, normal):
    return sum([grad(f, vars)[j] * normal[j] for j in range(len(vars))])


def boundary_integral(f, vars):
    if len(vars) == 2:
        x0,x1 = vars
        normals = ((1, 0), (0, 1), (-1, 0), (0, -1))  # Can for sure be improved
        vals = [integrate(n_deriv(f, vars, normals[0]).subs(x0, 1), (x1, -1, 1)),
                integrate(n_deriv(f, vars, normals[1]).subs(x1, 1), (x0, 1, -1)),
                integrate(n_deriv(f, vars, normals[2]).subs(x0, -1), (x1, 1, -1)),
                integrate(n_deriv(f, vars, normals[3]).subs(x1, -1), (x0, -1, 1))]
        return sum(vals)
    else:
        raise NotImplemented("Not working for %dD" % len(vars))


def compute_solution_set(u, v, xvars, yvars):
    del_u = laplace(u, xvars)
    del_v = laplace(v, yvars)
    boundary_v = boundary_integral(v, yvars)
    macro_rhs = - boundary_v - del_u
    micro_rhs = - u - del_v
    funcs = {"macro_rhs": macro_rhs, "micro_rhs": micro_rhs,
             "macro_bc": u, "micro_bc": v}
    return funcs


def write_param_file(filename, funcs):
    data = funcs.copy()
    data['geometry'] = "[-1,1]x[-1,1]"

    with open('%s.prm' % filename, 'w') as param_file:
        for key, val in data.items():
            param_file.write("set %s = %s\n" % (key, val))


def create_new_case(name, u_def, v_def):
    xvars = symbols('x0 x1')
    yvars = symbols('y0 y1')
    u = parse_expr(u_def)
    v = parse_expr(v_def)
    funcs = compute_solution_set(u, v, xvars, yvars)
    write_param_file(name, funcs)

if __name__=='__main__':
    u = "sin(x0*x1) + cos(x0 + x1)"
    v = "exp(x0**2 + x1**2) + y0*y1"
    create_new_case("test",u,v)

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

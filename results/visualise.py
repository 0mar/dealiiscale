import vtk
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, convert_xor
import re


def variables(num_dims):
    result = {}
    for dim in range(num_dims):
        x = f"x{dim}"
        y = f"y{dim}"
        result[x], result[y] = sp.symbols([x, y])
    return result


def matrix_function(filename, var):
    with open(filename, 'r') as config:
        data = config.read()
    if has_line := re.search('^set mapping = (.+)$', data, re.MULTILINE):
        map_eq_line = has_line.group(1)
        map_eq_lines = map_eq_line.split(';')
        map_eq = [parse_expr(map_eq_line, local_dict=var, transformations=[convert_xor]) for map_eq_line in
                  map_eq_lines]
        map_eq_matrix = sp.Matrix(map_eq)
        return map_eq_matrix
    else:
        raise ValueError("Not found any mapping in config file %s" % filename)


def get_transform_function(filename):
    var = variables(2)
    matrix = matrix_function(filename, var)
    macro_part = matrix.subs({var['y0']: 0, var['y1']: 0})
    rc = matrix.jacobian(sp.Matrix([var['y0'], var['y1']]))
    if rc.free_symbols:
        raise ValueError("Map cannot be parsed due to leftover symbols. Probably because map is not linear")

    def transform_function(x):
        x0, x1 = x
        offset = macro_part.subs({var['x0']: x0, var['x1']: x1})
        transformation = np.eye(4)
        transformation[:2, :2] = rc
        transformation[:2, 3] = list(offset)
        return transformation

    return transform_function


param_filename = '../input/biocase.prm'
transform_func = get_transform_function(param_filename)

sol_filename = 'u-solution.vtk'
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(sol_filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
output = reader.GetOutput()
transform = vtk.vtkTransform()
np_trans = transform_func([1,-1])

transform.SetMatrix(np_trans.flatten())
filter_ = vtk.vtkTransformFilter()
filter_.SetTransform(transform)
filter_.SetInputDataObject(output)
filter_.Update()
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName('test_output.vtk')
writer.SetInputData(filter_.GetOutput())
writer.Write()

import vtk
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, convert_xor
import re


def variables(num_dims):
    """
    Get multiscale variables in any dimension:
    (NB: in this script we assume same dimensionality for both scales)
    Macroscopic variables are denoted with x_i
    Microscopic variables are denoted with y_i

    :param num_dims: Number of dimensions
    :return: dictionary with variables
    """
    result = {}
    for dim in range(num_dims):
        x = f"x{dim}"
        y = f"y{dim}"
        result[x], result[y] = sp.symbols([x, y])
    return result


def matrix_function(filename, var):
    """
    Obtain a vector function that provides the microscopic grid transformation for each macroscopic point.

    :param filename: file in which SymPy expression for grid deformation lives
    :param var: dictionary of (multiscale) variables
    :return: vector as a function of multiscale vars
    """
    map_eq_line = "y0;y1"
    map_eq_lines = map_eq_line.split(';')
    map_eq = [parse_expr(map_eq_line, local_dict=var, transformations=[convert_xor]) for map_eq_line in
              map_eq_lines]
    map_eq_matrix = sp.Matrix(map_eq)
    return map_eq_matrix


def transform_dataset(name, transformation_matrix, out_name=None):
    """
    Wrapper method over VTK functions that reads, transforms and writes an unstructured grid

    :param name: file name of unstructured grid file
    :param transformation_matrix: 4D matrix that contains linear transformation of the unstructured grid
    :param out_name: name of output file
    :return:
    """
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(name)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    output = reader.GetOutput()
    transform = vtk.vtkTransform()

    transform.SetMatrix(transformation_matrix.flatten())
    filter_ = vtk.vtkTransformFilter()
    filter_.SetTransform(transform)
    filter_.SetInputDataObject(output)
    filter_.Update()
    writer = vtk.vtkUnstructuredGridWriter()

    if not out_name:
        out_name = name.replace('-solution', '-warped')
    writer.SetFileName(out_name)
    writer.SetInputData(filter_.GetOutput())
    writer.Write()


def transform_macro(name, level):
    """
    Transform the macroscopic scale to fit in a visualisation of the microscopic scale.

    :param name: file name of a microscopic function
    :param level: 1 or 2, representing a shift in z-direction so is visible in ParaView
    :return:
    """
    trans_matrix = np.eye(4)
    trans_matrix[2, 3] = level * 0.25  # Translation: lift it up from axis z=0 that has microscale
    transform_dataset(name, trans_matrix)


def get_transform_function(filename, num_points_x0, num_points_x1):
    """
    Obtain a function that returns a matrix representing the domain deformation.
    This matrix contains:
    1) the mapping function from the PDE
    2) a scaling to make all microdata fit in a macroscopic domain
    3) a translation to display all micro-systems side to side at their macro-locations

    :param filename: file name of the macroscopic solution
    :param num_points_x0: Number of points along the horizontal axis
    :param num_points_x1: Number of points along the vertical axis
    :return: function that takes macroscopic point and returns transformation matrix
    """
    var = variables(2)
    matrix = matrix_function(filename, var)
    macro_part = matrix.subs({var['y0']: 0, var['y1']: 0})
    rc = matrix.jacobian(sp.Matrix([var['y0'], var['y1']]))
    if rc.free_symbols & {var['y0'], var['y1']}:
        raise ValueError("Map cannot be parsed due to leftover microscopic symbols. Probably because map is not linear")

    def transform_function(x):
        x0, x1 = x
        # We ignore micro-offset, it is not important for the visualisation
        macro_offset = x0, x1
        transformation = np.eye(4)
        transformation[:2, :2] = rc.subs({var['x0']: x0, var['x1']: x1})
        transformation /= num_points_x0
        transformation[:2, 3] = macro_offset
        print(transformation)
        return transformation

    return transform_function


# Transform macro-coordinates
u_sol_file = 'u-solution.vtk'
transform_macro(u_sol_file, level=1)

# Transform micro-coordinates
import sys
param_filename = sys.argv[1]
loc_file = 'micro-solutions/grid_locations.txt'
coords = np.loadtxt(loc_file)
num_els_x0 = num_els_x1 = int(np.sqrt(len(coords)))
transform_func = get_transform_function(param_filename, num_els_x0, num_els_x1)
sol_string = 'micro-solutions/v-solution-%d.vtk'
out_string = 'micro-solutions/v-warped-%d.vtk'
for i in range(len(coords)):
    np_trans = transform_func(coords[i])
    transform_dataset(sol_string % i, np_trans, out_name=out_string%i)
    print("Wrote number %d on %s" % (i, coords[i]))

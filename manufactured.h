/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef MANUFACTURED_H
#define MANUFACTURED_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim>
class MicroBoundary : public Function<dim> {
public:
    MicroBoundary() : Function<dim>() {

    }

    /**
     * Compute the value of the microscopic boundary at a given point
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

    /**
     * Compute the analytic gradient of the boundary at point p. Necessary for Robin/Neumann boundary conditions and
     * exact evaluation of the error.
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return gradient of the microscopic boundary condition at p
    */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;

    /**
     * Set a macroscopic solution point for the boundary (corresponding to the computed value on the grid)
     * @param macro_solution Value of the macroscopic solution for the corresponding microscopic system.
     */
    void set_macro_solution(const Vector<double> &macro_solution);

    void set_macro_cell_index(unsigned int index);

private:
    Vector<double> macro_sol;
    unsigned int macro_cell_index = 0;
};

template<int dim>
class MacroBoundary : public Function<dim> {
public:
    MacroBoundary() : Function<dim>() {

    }

    /**
     * Creates a macroscopic boundary (only Dirichlet at this point)
     * @param p The point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary condition at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;


    /**
     * Compute the analytic gradient of the boundary at point p. Necessary for Robin/Neumann boundary conditions and
     * exact evaluation of the error.
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return gradient of the microscopic boundary condition at p
    */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;

private:
    const double lambda = std::sqrt(8. / 3.); // Coming from manufactured problem
};

template<int dim>
class MicroSolver {
public:

    /**
     * Create a Microsolver that resolves the microscopic systems.
     * @param macro_dof_handler The macroscopic degrees of freedom
     * @param macro_solution The macroscopic solution
     */
    MicroSolver(Vector<double> *macro_solution, unsigned int refine_level);

    /**
     * Collection method for setting up all necessary tools for the microsolver
     */
    void setup();

    /**
     * Collection method for solving the micro systems
     */
    void run();

    void refine_grid();

    /**
     * Compute the right hand side of the Macroscopic function;
     * a contribution from the microscopic solution.
     * @param dof_index Degree of freedom corresponding to the microscopic grid.
     * @return double with the value of the integral/other RHS function
     */
    double get_macro_contribution(unsigned int cell_index);

    void get_macro_rhs(Triangulation<dim> &macro_triangulation, Vector<double> &macro_rhs);

    /**
     * Output the results/write them to file/something
     */
    void output_results();

    /**
     * Post-process the solution (write convergence estimates and other stuff)
     */
    void process_solution();

    unsigned int refine_level; // todo: Set to private when we've asserted implementation works.

    unsigned int num_grids() const;
private:

    /**
     * Create a domain and make a triangulation
     */
    void make_grid();

    /**
     * Common data structures are created in this method
     * Distribute the degrees of freedom
     * Create a common sparsity pattern based on the mesh
     */
    void setup_system();

    /**
     * Specific data structures are created in this method.
     * Create specific right hand sides and solution vectors.
     */
    void setup_scatter();

    /**
     * Actual important method: Create the system matrices and create the right hand side vectors
     * by looping over all the cells and computing the discrete weak forms
     */
    void assemble_system();

    /**
     * Solve the system we obtained in `assemble_system`.
     */
    void solve();


    const double laplacian = 4.;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    ConvergenceTable convergence_table;
    unsigned int cycle;


    SparsityPattern sparsity_pattern;
    std::vector<SparseMatrix<double>> system_matrices;
    Vector<double> *macro_solution;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> righthandsides;

public:
    MicroBoundary<dim> boundary;
};

template<int dim>
class MacroSolver {
public:
    MacroSolver(unsigned int refine_level);

    void run();

    void process_solution();

    void compute_exact_value(Vector<double> &exact_values);

private:
    void make_grid(unsigned int refine_level);

    void setup_system();

    void assemble_system();

    /**
     * Interpolate a finite element function defined on the present grid to a single value per cell.
     * @param fe Finite element function corresponding to the finite element space
     * @return Vector (finite element function) with length `#cells`
     */
    void interpolate_function(const Vector<double> &func, Vector<double> &interp_func);

    void solve();

    void output_results();

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> interpolated_solution;
    Vector<double> system_rhs;
    MicroSolver<dim> micro;
    MacroBoundary<dim> boundary;
    ConvergenceTable convergence_table;
    int cycle;

};

int main() {
    deallog.depth_console(0);
    for (int i = 2; i < 10; i++) {
        MacroSolver<2> macro(i);
    }
    return 0;
}

#endif //MANUFACTURED_H

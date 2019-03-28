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
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */


#ifndef MANUFACTURED_H
#define MANUFACTURED_H

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
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
#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>
// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;

template<int dim>
class MicroBoundary : public Function<dim> {
public:
    MicroBoundary() : Function<dim>() {

    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;

    void set_macro_solution(double macro_solution);

private:
    double macro_solution = 0;
};

template<int dim>
class MacroBoundary : public Function<dim> {
public:
    MacroBoundary() : Function<dim>() {

    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;

private:
    const double lambda = std::sqrt(8. / 3.); // Coming from manufactured problem
};

template<int dim>
class MicroSolver {
public:
    MicroSolver(DoFHandler<dim> *macro_dof_handler, Vector<double> *macro_solution);

    void setup();

    void run();

    double get_macro_contribution(unsigned int dof_index);

    void output_results();

private:

    void make_grid();

    void setup_system();

    void setup_scatter();

    void assemble_system();

    void solve();

    void process_solution();


    const double laplacian = 4.;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> *macro_dof_handler;
    ConvergenceTable convergence_table;
    unsigned int cycle;


    SparsityPattern sparsity_pattern;
    std::vector<SparseMatrix<double>> system_matrices;
    Vector<double> *macro_solution;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> righthandsides;
    MicroBoundary<dim> boundary;
};

template<int dim>
class MacroSolver {
public:
    MacroSolver(unsigned int refine_level);

    void run();

    void process_solution();

private:
    void make_grid(unsigned int refine_level);

    void setup_system();

    void assemble_system();

    void solve();

    void output_results();

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
    MicroSolver<dim> micro;
    MacroBoundary<dim> boundary;
    ConvergenceTable convergence_table;
    int cycle;

};

int main(int argc, char *argv[]) {
    deallog.depth_console(0);
//    {
//        MacroSolver<2> laplace_problem_2d;
//        laplace_problem_2d.run();
//    }
//
//    {
//        MacroSolver<3> laplace_problem_3d;
//        laplace_problem_3d.run();
//    }
    for (unsigned int i = 0; i < 5; i++) {
        MacroSolver<2> macro(i);
    }
    return 0;
}

#endif //MANUFACTURED_H
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


// @sect3{Include files}

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

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <string>
#include <memory>

// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;


template<int dim>
class PlaceSolver {
public:
    PlaceSolver();


private:
    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    void build_function();

    void build_matrix();

    void solve();

    Vector<double> function;

    double real_function(const Point<dim> &p);

    void save_function();

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
};

template<int dim>
PlaceSolver<dim>::PlaceSolver(): fe(1), dof_handler(triangulation) {
    GridGenerator::hyper_cube(triangulation, -2, 2);
    triangulation.refine_global(5);
    dof_handler.distribute_dofs(fe);
    this->build_function();
    this->build_matrix();
    this->solve();
    this->save_function();

}

template<int dim>
void PlaceSolver<dim>::build_function() {
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    Vector<double> cell_function(dofs_per_cell);
    function.reinit(dof_handler.n_dofs());
    function = 0;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_function = 0;
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                cell_function(i) +=
                        fe_values.shape_value(i, q_index) * real_function(fe_values.quadrature_point(q_index)) *
                        fe_values.JxW(q_index);

            }
        }
        // Get corresponding global locations of the local ones.
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            function(local_dof_indices[i]) += cell_function(i);
        }
    }
}

template<int dim>
void PlaceSolver<dim>::build_matrix() {
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    Vector<double> cell_function(dofs_per_cell);

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_function = 0;
        cell_matrix = 0;
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    cell_matrix(i, j) += fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) *
                                         fe_values.JxW(q_index);
                }
            }
        }
        // Get corresponding global locations of the local ones.
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
        }
    }
}

template<int dim>
void PlaceSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, function,
                 PreconditionIdentity());

    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}


template<int dim>
void PlaceSolver<dim>::save_function() {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/function.vtk");
    data_out.write_vtk(output);
}


template<>
// In 2D, I like GNU plot
void PlaceSolver<2>::save_function() {
    DataOut<2> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/function.gpl");
    data_out.write_gnuplot(output);
}

template<int dim>
double PlaceSolver<dim>::real_function(const Point<dim> &p) {
    double return_value = 0.0;
    for (unsigned int i = 0; i < dim; i++) {
        return_value += 4.0 * std::pow(p(i), 2.0);
    }
    return return_value;
}

template<>
double PlaceSolver<3>::real_function(const Point<3> &p) {
    double first_term = p(0) * p(0) + 9. / 4 * p(1) * p(1) + p(2) * p(2) - 1;
    double second_term = std::pow(p(0) * p(2), 2) * p(2);
    double third_term = 9. / 80 * std::pow(p(1) * p(2), 2) * p(2);
    return std::pow(first_term, 3) - second_term - third_term;
}

//template<>
//double PlaceSolver<2>::real_function(const Point<2>& p) {
//    double first_term = 2*p(0)*p(0) + p(1)*p(1) - 1;
//    double second_term = std::pow(p(0)*p(1),2)*p(1)/10;
//    return std::pow(first_term,3) - second_term;
//}

template<int dim>
class MicroSolver {
public:
    MicroSolver(DoFHandler<dim> *macro_dof_handler, Vector<double> *macro_solution);

    void setup();

    void run();

    double get_macro_contribution(unsigned int dof_index);

    double initial_condition(const Point<dim> &p);

private:

    void make_grid();

    void setup_system();

    void setup_scatter();

    void assemble_system();

    void solve();

    void output_results() const;

    const double time_delta = 0.01;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;
    DoFHandler<dim> *macro_dof_handler;


    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> *macro_solution;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> righthandsides;
};

template<int dim>
class MacroSolver {
public:
    MacroSolver();

    void run();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void output_results() const;

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
    MicroSolver<dim> micro;
};
//
//template<int dim>
//class RightHandSide : public Function<dim> {
//public:
//    RightHandSide(MacroSolver<dim> &macro);
//
//    MacroSolver<dim> macro;
//
//    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;
//
//};
//
//template<int dim>
//RightHandSide<dim>::RightHandSide(MacroSolver<dim> &macro) : Function<dim>(), macro(macro) {
//
//}
//
//template<int dim>
//double RightHandSide<dim>::value(const Point<dim> &p,
//                                 const unsigned int /*component*/) const {
//    double return_value = 0.0;
//    for (unsigned int i = 0; i < dim; ++i)
//        return_value += 4.0 * std::pow(p(i), 4.0);
//
//    return return_value;
//}

template<int dim>
class BoundaryValues : public Function<dim> {
public:
    BoundaryValues() : Function<dim>() {}

    virtual double value(const Point<dim> &p,
                         const unsigned int component = 0) const;
};

template<int dim>
double BoundaryValues<dim>::value(const Point<dim> &p,
                                  const unsigned int /*component*/) const {
    return p.square();
}

template<int dim>
MacroSolver<dim>::MacroSolver(): fe(1), dof_handler(triangulation), micro(&dof_handler,&solution){
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    make_grid();
    setup_system();
    micro.setup();
    micro.run();
    this->run();
}

template<int dim>
void MacroSolver<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(2);

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
}

template<int dim>
void MacroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

template<int dim>
void MacroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(2);


//    const RightHandSide<dim> right_hand_side;


    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index) *
                                          fe_values.JxW(q_index));

                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                micro.get_macro_contribution(local_dof_indices[i]) *
                                fe_values.JxW(q_index));
            }


        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }


    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
}


template<int dim>
void MacroSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());

    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}


template<int dim>
void MacroSolver<dim>::output_results() const {
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/macro-solution.gpl");
    data_out.write_gnuplot(output);
}


template<int dim>
void MacroSolver<dim>::run() {
    assemble_system();
    solve();
    output_results();
}



template<int dim>
MicroSolver<dim>::MicroSolver(DoFHandler<dim> *macro_dof_handler, Vector<double> *macro_solution): fe(1), dof_handler(
        triangulation) {
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    this->macro_dof_handler = macro_dof_handler;
    this->macro_solution = macro_solution;
}
template <int dim>
void MicroSolver<dim>::setup() {

    make_grid();
    setup_system();
    setup_scatter();
}

template<int dim>
void MicroSolver<dim>::make_grid() {
    std::cout << "Setting up micro grid" << std::endl;
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(4);

    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Total number of cells: "
              << triangulation.n_cells()
              << std::endl;
}

template<int dim>
void MicroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}

template<int dim>
void MicroSolver<dim>::setup_scatter() {
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int i = 0; i < macro_dof_handler->n_dofs(); i++) {
        Vector<double> solution(n_dofs);
        solutions.push_back(solution);

        Vector<double> rhs(n_dofs);
        righthandsides.push_back(rhs);
    }
}

template <int dim>
double MicroSolver<dim>::get_macro_contribution(unsigned int dof_index) {
    // Simple start: f(r) = r(x,x)
    return solutions.at(dof_index)(dof_index);
}

template <int dim>
double MicroSolver<dim>::initial_condition(const Point<dim> &p) {
    double val = 0;
    for (unsigned int i=0;i<dim;i++) {
        val += (1-p(i))*(1-p(i));
    }
    return val;
}

template<int dim>
void MicroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(2);


//    const RightHandSide<dim> right_hand_side;


    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int macro_dofs = macro_dof_handler->n_dofs();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) * fe_values.shape_grad(j, q_index)  +
                                          fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) /
                                          this->time_delta) *
                                          fe_values.JxW(q_index);
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
            }
        }
        for (unsigned int k = 0; k < macro_dofs; k++) {
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
                    cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                    initial_condition(fe_values.quadrature_point(q_index)) )*
                                    fe_values.JxW(q_index);
                }
                righthandsides.at(k)(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }


    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,0,BoundaryValues<dim>(),boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,system_matrix,solutions.at(0),righthandsides.at(0));
}


template<int dim>
void MicroSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    for (unsigned int k=0;k<macro_dof_handler->n_dofs();k++) {
        solver.solve(system_matrix, solutions.at(k), righthandsides.at(k),PreconditionIdentity());
    }


    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}


template<int dim>
void MicroSolver<dim>::output_results() const {
    for (unsigned int k=0;k<macro_dof_handler->n_dofs();k++) {
        DataOut<dim> data_out;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solutions.at(k), "solution");

        data_out.build_patches();

        std::ofstream output("results/solution-" + std::to_string(k) + ".gpl");
        data_out.write_gnuplot(output);
    }

}


template<int dim>
void MicroSolver<dim>::run() {
    assemble_system();
    solve();
    output_results();
}

int main() {
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
    {
        MacroSolver<2> macro;
    }

    return 0;
}

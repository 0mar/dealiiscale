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


#include "manufactured.h"
using namespace dealii;

template<int dim>
double MicroBoundary<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = 0;
    for (unsigned int i = 0; i < dim; i++) {
        val += p(i) * p(i) * macro_solution;
    }
    return val;
}

template<int dim>
Tensor<1, dim> MicroBoundary<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;

    return_val[0] = 2 * p(0) * macro_solution;
    return_val[1] = 2 * p(1) * macro_solution;
    return return_val;
}

template<int dim>
void MicroBoundary<dim>::set_macro_solution(const double macro_solution) {
    this->macro_solution = macro_solution;
}


template<int dim>
double MacroBoundary<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = 0; // Todo: prescribe pi and solve rho, then prescribe rho and solve pi
    val = std::sin(lambda * p(0)) + std::cos(lambda * p(1));
    return val;
}

template<int dim>
Tensor<1, dim> MacroBoundary<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;

    return_val[0] = lambda * std::cos(lambda * p(0));
    return_val[1] = -lambda * std::sin(lambda * p(1));
    return return_val;
}

template<int dim>
MacroSolver<dim>::MacroSolver(unsigned int refine_level): fe(1), dof_handler(triangulation), micro(&dof_handler, &solution), boundary() {
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    make_grid(refine_level);
    setup_system();
    cycle = 0;
    micro.setup();
    for (unsigned int i = 0; i < 1; i++) {
//        micro.run();
        this->run();
        cycle++;
    }
//    micro.output_results();
    this->output_results();
}

template<int dim>
void MacroSolver<dim>::make_grid(unsigned int refine_level) {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);

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
    system_matrix = 0;
    system_rhs = 0;

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                            fe_values.shape_grad(j, q_index) - micro.get_macro_contribution(local_dof_indices[i])
                            * fe_values.shape_value(i,q_index)* fe_values.shape_value(j,q_index))*fe_values.JxW(q_index);


                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                0* // debug
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
                                             boundary,
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
    printf("Convergence after %d CG iterations\n", solver_control.last_step());
    cycle++;
}

template<int dim>
void MacroSolver<dim>::process_solution() {
    const unsigned int n_active = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();
    Vector<float> difference_per_cell(n_active);
    VectorTools::integrate_difference(dof_handler, solution, boundary, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::L2_norm);
    double l2_error = difference_per_cell.l2_norm();
    VectorTools::integrate_difference(dof_handler, solution, boundary, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::H1_seminorm);
    double h1_error = difference_per_cell.l2_norm();
    printf("Cycle: %d\n, Number of active cells: %d\n, Number of DoFs, %d\n", cycle, n_active, n_dofs);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", l2_error);
    convergence_table.add_value("H1", h1_error);

}

template<int dim>
void MacroSolver<dim>::output_results() {

    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.write_text(std::cout);
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
    process_solution();
}

template<int dim>
MicroSolver<dim>::MicroSolver(DoFHandler<dim> *macro_dof_handler, Vector<double> *macro_solution): fe(1), dof_handler(
        triangulation), boundary() {
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    this->macro_dof_handler = macro_dof_handler;
    this->macro_solution = macro_solution;
    cycle = 0;
}

template<int dim>
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
}

template<int dim>
void MicroSolver<dim>::setup_scatter() {
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int i = 0; i < macro_dof_handler->n_dofs(); i++) {
        Vector<double> solution(n_dofs);
        solutions.push_back(solution);

        Vector<double> rhs(n_dofs);
        righthandsides.push_back(rhs);

        SparseMatrix<double> system_matrix;
        system_matrices.push_back(system_matrix);
    }
}

template<int dim>
double MicroSolver<dim>::get_macro_contribution(unsigned int dof_index) {
    // manufactured as: f(x) = \int_Y \rho(x,y)dy
    double integral = 0;
//    QGauss<dim> quadrature_formula(2);
//    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
//    const unsigned int dofs_per_cell = fe.dofs_per_cell;
//    const unsigned int n_q_points = quadrature_formula.size();
//    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
//    for (const auto &cell: dof_handler.active_cell_iterators()) {
//        fe_values.reinit(cell);
//        cell->get_dof_indices(local_dof_indices);
//        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
//            for (unsigned int i = 0; i < dofs_per_cell; i++) {
//                integral += 1 * solutions.at(dof_index)(local_dof_indices.at(i)) * fe_values.JxW(q_index);
//            }
//        }
//    }
    integral = 8./3;//*(*macro_solution)(dof_index); // debug
    return integral;
}

template<int dim>
void MicroSolver<dim>::assemble_system() { // todo: L2 norm, solo implementation and test
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
    for (unsigned int k = 0; k < macro_dofs; k++) {
        righthandsides.at(k) = 0;
        solutions.at(k) = 0;
        system_matrices.at(k).reinit(sparsity_pattern);

    }
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    cell_matrix(i, j) += fe_values.shape_grad(i, q_index)
                                         * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index);
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                for (unsigned int k = 0; k < macro_dofs; k++) {
                    system_matrices.at(k).add(local_dof_indices[i],
                                              local_dof_indices[j],
                                              cell_matrix(i, j));
                }
            }
        }
        for (unsigned int k = 0; k < macro_dofs; k++) {
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
                    cell_rhs(i) += -laplacian * (*macro_solution)(k) * fe_values.JxW(q_index);
                }
                righthandsides.at(k)(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    for (unsigned int k = 0; k < macro_dofs; k++) {
        this->boundary.set_macro_solution((*macro_solution)(k)); // todo: evaluate the point from dof
        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler, 0, boundary, boundary_values);
        MatrixTools::apply_boundary_values(boundary_values, system_matrices.at(k), solutions.at(k),
                                           righthandsides.at(k));
    }
}


template<int dim>
void MicroSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    for (unsigned int k = 0; k < macro_dof_handler->n_dofs(); k++) {
        solver.solve(system_matrices.at(k), solutions.at(k), righthandsides.at(k), PreconditionIdentity());
    }


    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
    cycle++;
}


template<int dim>
void MicroSolver<dim>::process_solution() {
    for (unsigned int k = 0; k < macro_dof_handler->n_dofs(); k++) {
        boundary.set_macro_solution((*macro_solution)(k));
        const unsigned int n_active = triangulation.n_active_cells();
        const unsigned int n_dofs = dof_handler.n_dofs();
        Vector<float> difference_per_cell(n_active);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell, QGauss<dim>(3),
                                          VectorTools::L2_norm);
        double l2_error = difference_per_cell.l2_norm();
        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell, QGauss<dim>(3),
                                          VectorTools::H1_seminorm);
        double h1_error = difference_per_cell.l2_norm();
//        printf("Cycle: %d\n, Number of active cells: %d\n, Number of DoFs, %d\n", cycle, n_active, n_dofs);
        convergence_table.add_value("cycle", cycle);
        convergence_table.add_value("cells", n_active);
        convergence_table.add_value("dofs", n_dofs);
        convergence_table.add_value("L2", l2_error);
        convergence_table.add_value("H1", h1_error);
    }
}

template<int dim>
void MicroSolver<dim>::output_results() {
    for (unsigned int k = 0; k < macro_dof_handler->n_dofs(); k++) {
        std::ofstream micro_file("results/micro_solution" + std::to_string(k) + ".txt", std::ofstream::app);
        convergence_table.write_text(micro_file);
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
    process_solution();
}


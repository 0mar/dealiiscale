/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
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
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe_field.h>
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
#include <deal.II/base/convergence_table.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template<int dim>
class RightHandSide : public Function<dim> {
public:
    RightHandSide() : Function<dim>() {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const {
    double radius_sqrd = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        radius_sqrd += p[i] * p[i];
    }
    return 8 - 4 * radius_sqrd;
}


template<int dim>
class NeumannData : public Function<dim> {
public:
    NeumannData() : Function<dim>() {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double NeumannData<dim>::value(const Point<dim>, const unsigned int) const {
    return -1.875;
}

template<int dim>
class Solution : public Function<dim> {
public:
    Solution() : Function<dim>() {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
    double radius_sqrd = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        radius_sqrd += p[i] * p[i];
    }
    return radius_sqrd * radius_sqrd / 4. - 2 * radius_sqrd;
}

class DiskProblem {
public:
    DiskProblem();

    void run();

    void output_results();

    void refine();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void process_solution();


    Triangulation<2> triangulation;
    FE_Q<2> fe;
    DoFHandler<2> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
    const double kappa = .5;
    const double c = -7.75;
    int cycle;
    ConvergenceTable convergence_table;
    const unsigned int NEUMANN_BOUNDARY = 1;
    const unsigned int ROBIN_BOUNDARY = 2;
};

DiskProblem::DiskProblem() :
        fe(1),
        dof_handler(triangulation),
        cycle(0) {
    make_grid();
}

void DiskProblem::make_grid() {
    const Point<2> center(0, 0);
    const double inner_radius = 0.5;
    const double outer_radius = 1.0;
    GridGenerator::hyper_shell(triangulation, center, inner_radius, outer_radius, 10);
    triangulation.refine_global(4);
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double radius_sqrd = cell->face(face_number)->center()(0) * cell->face(face_number)->center()(0) +
                                           cell->face(face_number)->center()(1) * cell->face(face_number)->center()(1);
                if (radius_sqrd < 0.75 * 0.75) {
                    cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
                } else {
                    cell->face(face_number)->set_boundary_id(ROBIN_BOUNDARY);
                }
            }
        }
    }
}

void DiskProblem::refine() {
    triangulation.refine_global(1);
}

void DiskProblem::setup_system() {
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

void DiskProblem::assemble_system() {
    int integration_order = 2;
    const int dim = 2;
    QGauss<dim> quadrature_formula(integration_order);
    QGauss<dim - 1> face_quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_gradients | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_normal_vectors | update_quadrature_points |
                                     update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_face_points = face_quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    const RightHandSide<dim> rhs;
    const NeumannData<dim> neumann_data;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const DoFHandler<dim>::active_cell_iterator &cell:dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index) *
                                          fe_values.JxW(q_index));
                }
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                rhs.value(fe_values.quadrature_point(q_index)) *
                                fe_values.JxW(q_index));
            }
        }
        for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                if (cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                    fe_face_values.reinit(cell, face_number);
                    for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                                if (cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                                    cell_matrix(i, j) += (fe_face_values.shape_value(i, q_index) *
                                                          fe_face_values.shape_value(j, q_index) *
                                                          kappa * fe_face_values.JxW(q_index));
                                }
                            }
                            if (cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                                cell_rhs(i) += fe_face_values.shape_value(i, q_index) * kappa * c *
                                               fe_face_values.JxW(q_index);
                            } else {
                                cell_rhs(i) += neumann_data.value(fe_face_values.quadrature_point(q_index)) *
                                               fe_face_values.shape_value(i, q_index) *
                                               fe_face_values.JxW(q_index);
                            }
                        }
                    }
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                system_matrix.add(local_dof_indices[i],
                                  local_dof_indices[j],
                                  cell_matrix(i, j));
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

void DiskProblem::solve() {
    SolverControl solver_control(10000, 1e-10);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());
}

void DiskProblem::process_solution() {
    const int dim = 2;
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();
    std::cout << "Cycle " << cycle << ':'
              << std::endl
              << "   Number of active cells:       "
              << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs
              << std::endl;
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    cycle++;
}

void DiskProblem::output_results() {
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/solution.gpl");
        data_out.write_gnuplot(output);
    }
    {

    }

    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L2", true);
//    convergence_table.set_tex_caption("cells", "\\# cells");
//    convergence_table.set_tex_caption("dofs", "\\# dofs");
//    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
//    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
//    convergence_table.set_tex_format("cells", "r");
//    convergence_table.set_tex_format("dofs", "r");
    std::ofstream ofs("results/robin.txt");
    convergence_table.write_text(ofs);
    ofs.close();
}

void DiskProblem::run() {
    setup_system();
    assemble_system();
    solve();
    process_solution();
}

int main() {
    deallog.depth_console(2);
    DiskProblem poisson_problem;
    poisson_problem.refine();
    poisson_problem.run();
    poisson_problem.output_results();
    return 0;
}

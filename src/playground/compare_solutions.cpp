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
#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/solution_transfer.h>

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
    return radius_sqrd * radius_sqrd / 4. - 2. * radius_sqrd;
}

class SolutionComparer {
public:
    SolutionComparer();

    void run();

    void output_results(const std::string &filename);

    void setup(unsigned int _refine_level);

    Vector<double> solution;

    void compare_with_finest(unsigned int _refine_level, const Vector<double> &finest_solution);

    void compare_with_finest(unsigned int _refine_level, const Vector<double> &finest_solution,
                             const DoFHandler<2> &finest_dof_handler);

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();


    Triangulation<2> triangulation;
    FE_Q<2> fe;
public:
    DoFHandler<2> dof_handler;
private:
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> system_rhs;
    const double kappa = .5;
    const double c = -7.75;
    int refine_level;
    ConvergenceTable convergence_table;
};

SolutionComparer::SolutionComparer() :
        fe(1),
        dof_handler(triangulation),
        refine_level(1) {
}

void SolutionComparer::make_grid() {
    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(refine_level);
    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}

void SolutionComparer::setup_system() {
    dof_handler.distribute_dofs(fe);
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

void SolutionComparer::assemble_system() {
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
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                            cell_matrix(i, j) += (fe_face_values.shape_value(i, q_index) *
                                                  fe_face_values.shape_value(j, q_index) *
                                                  kappa * fe_face_values.JxW(q_index));
                        }
                        cell_rhs(i) += fe_face_values.shape_value(i, q_index) * kappa * c * fe_face_values.JxW(q_index);
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

void SolutionComparer::solve() {
    SolverControl solver_control(10000, 1e-10);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());
}

void SolutionComparer::output_results(const std::string &filename) {
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
//    convergence_table.set_tex_caption("cells", "\\# cells");
//    convergence_table.set_tex_caption("dofs", "\\# dofs");
//    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
//    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
//    convergence_table.set_tex_format("cells", "r");
//    convergence_table.set_tex_format("dofs", "r");
    std::ofstream ofs(filename, std::ios_base::app);
    convergence_table.write_text(ofs);
    ofs.close();
}

void SolutionComparer::setup(const unsigned int _refine_level) {
    refine_level = _refine_level;
    make_grid();
    setup_system();
}

void SolutionComparer::compare_with_finest(unsigned int _refine_level, const Vector<double> &finest_solution) {
    int spare_refine = _refine_level - refine_level;
    printf("Refining with %d extra levels\n", spare_refine);
    Assert(spare_refine > 0, ExcNotImplemented("No accounting for coarsening comparisons"))
    SolutionTransfer<2> solution_transfer(dof_handler);
    // Look, solution transfer never directly sees the triangulation.
    // Maybe we can manipulate the dof_handler directly and require no changing of the mesh
    solution_transfer.prepare_for_pure_refinement();
    triangulation.refine_global(spare_refine);
    Vector<double> tmp(solution);
    dof_handler.distribute_dofs(fe);
    Vector<double> extrapolated_solution(dof_handler.n_dofs());
    solution_transfer.refine_interpolate(tmp, extrapolated_solution);
    Vector<double> difference(finest_solution);
    difference -= extrapolated_solution;
    const int dim = 2;
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      difference,
                                      ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    VectorTools::integrate_difference(dof_handler,
                                      difference,
                                      ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::H1_norm);
    const double H1_error = difference_per_cell.l2_norm();
    const unsigned int n_dofs = dof_handler.n_dofs();
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);

}

void SolutionComparer::compare_with_finest(unsigned int _refine_level, const Vector<double> &finest_solution,
                                           const DoFHandler<2> &finest_dof_handler) {
    const int dim = 2;
    int spare_refine = _refine_level - refine_level;
    printf("Refining with %d extra levels\n", spare_refine);
    Vector<double> extrapolated_solution(finest_dof_handler.n_dofs());
    VectorTools::interpolate_to_different_mesh(dof_handler, solution, finest_dof_handler, extrapolated_solution);
    Vector<double> difference(finest_solution);
    difference -= extrapolated_solution;
    Vector<float> difference_per_cell(finest_dof_handler.get_triangulation().n_active_cells());
    VectorTools::integrate_difference(finest_dof_handler,
                                      difference,
                                      ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    VectorTools::integrate_difference(finest_dof_handler,
                                      difference,
                                      ZeroFunction<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::H1_norm);
    const double H1_error = difference_per_cell.l2_norm();
    const unsigned int n_dofs = finest_dof_handler.n_dofs();
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);


}

void SolutionComparer::run() {
    assemble_system();
    solve();
}

int main() {
    const int finest_level = 9;
    SolutionComparer fine_sol_com;
    fine_sol_com.setup(finest_level);
    fine_sol_com.run();
    std::string filename = "results/test_compare1.txt";
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 9; i++) {
        SolutionComparer sol_com;
        sol_com.setup(i);
        sol_com.run();
        sol_com.compare_with_finest(finest_level, fine_sol_com.solution);//, fine_sol_com.dof_handler);
        sol_com.output_results(filename);
    }
    return 0;
}

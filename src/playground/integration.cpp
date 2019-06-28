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

#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Test how we can do bulk and boundary integrals, and how precise they are.
 *
 */


template<int dim>
class RightHandSide : public Function<dim> {
public:
    RightHandSide() : Function<dim>() {}

    virtual double value(const Point <dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double RightHandSide<dim>::value(const Point <dim> &p, const unsigned int) const {
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

    virtual double value(const Point <dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double Solution<dim>::value(const Point <dim> &p, const unsigned int) const {
    double radius_sqrd = 0;
    for (unsigned int i = 0; i < dim; ++i) {
        radius_sqrd += p[i] * p[i];
    }
    return radius_sqrd * radius_sqrd / 4. - 2. * radius_sqrd;
}

class Integrator {
public:
    Integrator();

    void run();

    void output_results();

    void refine();

    double integrate_bulk() const;

    double integrate_outward_flux() const;

    double real_bulk() const;

    double real_outward_flux() const;

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
    int cycle;
    ConvergenceTable convergence_table;
};

Integrator::Integrator() :
        fe(1),
        dof_handler(triangulation),
        cycle(0) {
    make_grid();
}

void Integrator::make_grid() {
    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(4);
    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}

void Integrator::refine() {
    triangulation.refine_global(1);
    std::cout << "Refinement: Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}

void Integrator::setup_system() {
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

void Integrator::assemble_system() {
    const int dim = 2;
    int integration_order = 2;
    QGauss<dim> quadrature_formula(integration_order);
    QGauss<dim - 1> face_quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_gradients | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_normal_vectors | update_quadrature_points |
                                     update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
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
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, Solution<2>(), boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

}

void Integrator::solve() {
    SolverControl solver_control(10000, 1e-10);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());
}

double Integrator::integrate_bulk() const {
    const int dim = 2;
    const int integration_order = 3;
    // computed as: f(x) = \int_Y \rho(x,y)dy
    double integral = 0.;
    auto rhs = Solution<dim>();
    QGauss<dim> quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
    const unsigned int n_q_points = quadrature_formula.size();
//  const unsigned int dofs_per_cell = fe.dofs_per_cell;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        std::vector<double> interp_solution(n_q_points);
        fe_values.get_function_values(solution, interp_solution);
//      for (unsigned int i=0;i<dofs_per_cell;i++) {
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
//          integral += fe_values.shape_value(i,q_index)*interp_solution[q_index] * fe_values.JxW(q_index); // Both possible.
            integral += interp_solution[q_index] * fe_values.JxW(q_index);
        }
//      }
    }
    return integral;
}

double Integrator::integrate_outward_flux() const {
    const int dim = 2;
    const int integration_order = 3;
    // computed as: f(x) = \int_Y \rho(x,y)dy
    double integral = 0;
    QGauss<dim - 1> quadrature_formula(integration_order);
    FEFaceValues<dim> fe_face_values(fe, quadrature_formula,
                                     update_values | update_quadrature_points | update_JxW_values |
                                     update_normal_vectors | update_gradients);
    const unsigned int n_q_face_points = quadrature_formula.size();
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                std::vector<Tensor<1, dim>> solution_gradient(n_q_face_points);
                fe_face_values.get_function_gradients(solution, solution_gradient);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    double neumann = solution_gradient[q_index] * fe_face_values.normal_vector(q_index);
                    integral += neumann * fe_face_values.JxW(q_index);
                }
            }
        }
    }
    return integral;
}

double Integrator::real_bulk() const {
    // Computed analytically for this case
    return -3.1415926536 * (11. / 12);
}

double Integrator::real_outward_flux() const {
    return -3 * 2 * 3.1415926536; // Not so sure about this one yet.
}

void Integrator::process_solution() {
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
    printf("Real outward: %.3f\n", real_outward_flux());
    printf("Computed outward: %.3f\n", integrate_outward_flux());
    convergence_table.add_value("int", integrate_bulk() - real_bulk());
    convergence_table.add_value("normal", integrate_outward_flux() - real_outward_flux());
    cycle++;
}

void Integrator::output_results() {
    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output("results/solution.gpl");
    data_out.write_gnuplot(output);

    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L2", true);
    convergence_table.set_precision("int", 3);
    convergence_table.set_scientific("int", true);
    convergence_table.set_precision("normal", 3);
    convergence_table.set_scientific("normal", true);
//    convergence_table.set_tex_caption("cells", "\\# cells");
//    convergence_table.set_tex_caption("dofs", "\\# dofs");
//    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
//    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
//    convergence_table.set_tex_format("cells", "r");
//    convergence_table.set_tex_format("dofs", "r");
    std::ofstream ofs("results/integrator.txt");
    convergence_table.write_text(ofs);
    ofs.close();
}

void Integrator::run() {
    setup_system();
    assemble_system();
    solve();
    process_solution();
}

int main() {
    deallog.depth_console(2);
    Integrator poisson_problem;
    for (int i = 0; i < 4; i++) {
        poisson_problem.refine();
        poisson_problem.run();
    }
    poisson_problem.output_results();
    return 0;
}

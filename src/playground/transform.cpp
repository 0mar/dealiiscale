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
#include <deal.II/base/tensor_function.h>
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
    return 3 * p[1] * p[1] - 2 * p[0];
}


template<int dim>
class DomainMapping {
public:
    DomainMapping();

    Tensor<2, dim> map_coef;
    SymmetricTensor<2, dim> bilin_coef;

//    const double det_jac(const Point<dim> &p);
    void get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt);

    Point<dim> map(const Point<dim> &p) const;

private:
    const double PI = 3.14159265358979323;
    const double theta = 0.25 * PI;
    Tensor<1, dim> offset;
    const double scaling_x_factor = 0.7;
    const double scaling_y_factor = 0.7;
};

template<int dim>
DomainMapping<dim>::DomainMapping() {
    for (unsigned int i = 0; i < dim; i++) {
        offset[i] = 0;
    }
    Tensor<2, dim> scaling;
    scaling[0][0] = scaling_x_factor;
    scaling[1][1] = scaling_y_factor;
    Tensor<2, dim> rotation;
    rotation[0][0] = std::cos(theta);
    rotation[0][1] = -std::sin(theta);
    rotation[1][0] = std::sin(theta);
    rotation[1][1] = std::cos(theta);
    map_coef = rotation * scaling;
    bilin_coef = SymmetricTensor<2, dim>(invert(map_coef) * transpose(invert(map_coef))) * determinant(map_coef);
}

template<int dim>
Point<dim> DomainMapping<dim>::map(const Point<dim> &p) const {
    return Point<dim>(map_coef * p + offset);
}


template<int dim>
void DomainMapping<dim>::get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt) {
    AssertDimension(kkt.dimension, bilin_coef.dimension);
    AssertDimension(kkt.rank, bilin_coef.rank);
    kkt = bilin_coef;
}

template<int dim>
class NonLinDomainMapping {
public:
    NonLinDomainMapping();

    Tensor<2, dim> map_coef;
    SymmetricTensor<2, dim> bilin_coef;

    void get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt);
//    const double det_jac(const Point<dim> &p);

    Point<dim> map(const Point<dim> &p) const;

private:
    const double PI = 3.14159265358979323;
    const double theta = 0.25 * PI;
    Tensor<1, dim> offset;
};

template<int dim>
NonLinDomainMapping<dim>::NonLinDomainMapping() {
    for (unsigned int i = 0; i < dim; i++) {
        offset[i] = 0;
    }
    Tensor<2, dim> scaling;
    scaling[0][0] = 1;
    scaling[1][1] = 2;
    Tensor<2, dim> rotation;
    rotation[0][0] = std::cos(theta);
    rotation[0][1] = -std::sin(theta);
    rotation[1][0] = std::sin(theta);
    rotation[1][1] = std::cos(theta);
    map_coef = rotation * scaling;
    bilin_coef = SymmetricTensor<2, dim>(invert(map_coef) * transpose(invert(map_coef))) * determinant(map_coef);
}

template<int dim>
void NonLinDomainMapping<dim>::get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt) {
    AssertDimension(kkt.dimension, bilin_coef.dimension);
    AssertDimension(kkt.rank, bilin_coef.rank);
    Tensor<2, dim> x_part;
    x_part[0][0] = 1. / (2 * p[0]);
    x_part[1][1] = 1. / (2 * p[1]);
    kkt = SymmetricTensor<2, dim>(x_part * bilin_coef * x_part) * determinant(x_part);
}

template<int dim>
Point<dim> NonLinDomainMapping<dim>::map(const Point<dim> &p) const {
    Point<dim> p2;
    for (unsigned int i = 0; i < dim; i++) {
        p2[i] = p[i] * p[i];
    }
    return Point<dim>(map_coef * p2 + offset);
}

template<int dim>
class Solution : public Function<dim> {
public:
    Solution() : Function<dim>() {}

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

};

template<int dim>
double Solution<dim>::value(const Point<dim> &p, const unsigned int) const {
    return std::pow(p[0], 3) / 3 - std::pow(p[1], 4) / 4;
}

class RobinSolver {
public:
    RobinSolver();

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
    DomainMapping<2> dm;
    Vector<double> solution;
    Vector<double> system_rhs;
    int cycle;
    ConvergenceTable convergence_table;
};

RobinSolver::RobinSolver() :
        fe(1),
        dof_handler(triangulation),
        cycle(0) {
    make_grid();

}

void RobinSolver::make_grid() {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);
    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}

void RobinSolver::refine() {
    triangulation.refine_global(1);
    std::cout << "Refinement: Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}

void RobinSolver::setup_system() {
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

void RobinSolver::assemble_system() {
    int integration_order = 2;
    const int dim = 2;
    QGauss<dim> quadrature_formula(integration_order);
    QGauss<dim - 1> face_quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_gradients | update_JxW_values);
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
        SymmetricTensor<2, dim> kkt;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            dm.get_kkt(fe_values.quadrature_point(q_index), kkt);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          kkt *
                                          fe_values.shape_grad(j, q_index) *
                                          fe_values.JxW(q_index));
                }
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                rhs.value(fe_values.quadrature_point(q_index)) *
                                determinant(dm.map_coef) *
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

void RobinSolver::solve() {
    SolverControl solver_control(10000, 1e-10);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
                 PreconditionIdentity());
}

void RobinSolver::process_solution() {
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

void RobinSolver::output_results() {
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/solution.gpl");
        data_out.write_gnuplot(output);
        output.close();
    }
    {
        const int dim = 2;
        const int spacedim = 2;
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        const FE_Q<dim, spacedim> feq(1);
        const FESystem<dim, spacedim> fesystem(feq, spacedim);
        DoFHandler<dim, spacedim> dhq(triangulation);
        dhq.distribute_dofs(fesystem);
        const ComponentMask mask(spacedim, true);
        Vector<double> eulerq(dhq.n_dofs());
        Vector<double> mappedq(dhq.n_dofs());
        VectorTools::get_position_vector(dhq, eulerq, mask);
        for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
            const Point<2> p(eulerq[2 * i], eulerq[2 * i + 1]);
            const auto mapped_point = dm.map(p);
            mappedq[2 * i] = mapped_point[0];
            mappedq[2 * i + 1] = mapped_point[1];

        }
        MappingFEField<dim, spacedim> map(dhq, mappedq, mask);
        std::vector<Point<spacedim>> support_points(dhq.n_dofs());
        DoFTools::map_dofs_to_support_points(map, dhq, support_points);
        data_out.build_patches(map);
        std::ofstream output("results/solution2.gpl");
        data_out.write_gnuplot(output);
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

void RobinSolver::run() {
    setup_system();
    assemble_system();
    solve();
    process_solution();
}

int main() {
    deallog.depth_console(2);
    RobinSolver poisson_problem;
    for (unsigned int i = 0; i < 4; i++) {
        poisson_problem.refine();
        poisson_problem.run();
    }
    poisson_problem.output_results();
    return 0;
}

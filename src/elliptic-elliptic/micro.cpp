/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "micro.h"

using namespace dealii;

template<int dim>
double MicroBoundary<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = 0;
    for (unsigned int i = 0; i < dim; i++) {
        val += p(i) * p(i) * macro_sol[macro_cell_index];
    }
    return val;
}

template<int dim>
Tensor<1, dim> MicroBoundary<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;

    return_val[0] = 2 * p(0) * macro_sol[macro_cell_index];
    return_val[1] = 2 * p(1) * macro_sol[macro_cell_index];
    return return_val;
}

template<int dim>
void MicroBoundary<dim>::set_macro_solution(const Vector<double> &macro_solution) {
    this->macro_sol = macro_solution;
}


template<int dim>
void MicroBoundary<dim>::set_macro_cell_index(const unsigned int index) {
    macro_cell_index = index;
}

template<int dim>
MicroSolver<dim>::MicroSolver():  dof_handler(triangulation), boundary(), fe(1), macro_solution(nullptr),
                                  macro_dof_handler(nullptr) {
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    refine_level = 1;
    num_grids = 1;
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
    triangulation.refine_global(refine_level);

    std::cout << "   Number of active cells: " // toto: Make sense of the output messages
              << triangulation.n_active_cells()
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
void MicroSolver<dim>::set_refine_level(int refinement_level) {
    this->refine_level = refinement_level;
}

template<int dim>
void MicroSolver<dim>::refine_grid() {
    triangulation.refine_global(1);
    setup_system();
    setup_scatter();
}

template<int dim>
void MicroSolver<dim>::setup_scatter() {
    solutions.clear();
    righthandsides.clear();
    system_matrices.clear();
    compute_macroscopic_contribution();
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int i = 0; i < num_grids; i++) {
        Vector<double> solution(n_dofs);
        solutions.push_back(solution);

        Vector<double> rhs(n_dofs);
        righthandsides.push_back(rhs);

        SparseMatrix<double> system_matrix;
        system_matrices.push_back(system_matrix);
    }
}


template<int dim>
void MicroSolver<dim>::set_macro_solution(Vector<double> *_solution, DoFHandler<dim> *_dof_handler) {
    this->macro_solution = _solution;
    this->macro_dof_handler = _dof_handler;
}

template<int dim>
void MicroSolver<dim>::set_macro_boundary_condition(const Vector<double> &macro_condition) {
    this->boundary.set_macro_solution(macro_condition);
}

template<int dim>
void MicroSolver<dim>::compute_macroscopic_contribution() {
    // Nothing needs to happen in this simple case
}

template<int dim>
void MicroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);


    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (unsigned int k = 0; k < num_grids; k++) {
        righthandsides.at(k) = 0;
        solutions.at(k) = 0;
        system_matrices.at(k).reinit(sparsity_pattern);

    }
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;

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
                for (unsigned int k = 0; k < num_grids; k++) {
                    system_matrices.at(k).add(local_dof_indices[i],
                                              local_dof_indices[j],
                                              cell_matrix(i, j));
                }
            }
        }
        for (unsigned int k = 0; k < num_grids; k++) {
            cell_rhs = 0;
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
                    cell_rhs(i) += -laplacian * (*macro_solution)(k) * fe_values.shape_value(i, q_index) *
                                   fe_values.JxW(q_index);
                }
                righthandsides.at(k)(local_dof_indices[i]) += cell_rhs(i);
            }
        }
    }
    for (unsigned int k = 0; k < num_grids; k++) {
        this->boundary.set_macro_cell_index(k);
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
    for (unsigned int k = 0; k < num_grids; k++) {
        solver.solve(system_matrices.at(k), solutions.at(k), righthandsides.at(k), PreconditionIdentity());
    }
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
}


template<int dim>
void MicroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    Vector<double> macro_domain_l2_error(num_grids);
    Vector<double> macro_domain_h1_error(num_grids);
    for (unsigned int k = 0; k < num_grids; k++) {
        boundary.set_macro_cell_index(k); // Todo: Change when we separate boundary and exact solution.
        const unsigned int n_active = triangulation.n_active_cells();
        Vector<double> difference_per_cell(n_active);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell, QGauss<dim>(3),
                                          VectorTools::L2_norm);
        double micro_l2_error = difference_per_cell.l2_norm();
        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell, QGauss<dim>(3),
                                          VectorTools::H1_seminorm);
        double micro_h1_error = difference_per_cell.l2_norm();
        macro_domain_l2_error(k) = micro_l2_error;
        macro_domain_h1_error(k) = micro_h1_error;
    }
//    Vector<double> macro_integral(num_grids);
//    VectorTools::integrate_difference(*macro_dof_handler,macro_domain_l2_error,Functions::ZeroFunction<dim>(),macro_integral,QGauss<dim>(3),VectorTools::L2_norm);
    l2_error = macro_domain_l2_error.l2_norm() / macro_domain_l2_error.size(); // Is this the most correct norm?
    h1_error = macro_domain_h1_error.l2_norm() / macro_domain_h1_error.size();
}

template<int dim>
void MicroSolver<dim>::run() {
    assemble_system();
    solve();
}

template<int dim>
unsigned int MicroSolver<dim>::get_num_grids() const {
    return num_grids;
}

template<int dim>
void MicroSolver<dim>::set_num_grids(unsigned int _num_grids) {
    this->num_grids = _num_grids;
}

// Explicit instantiation

template
class MicroBoundary<1>;

template
class MicroBoundary<2>;

template
class MicroBoundary<3>;

template
class MicroSolver<1>;

template
class MicroSolver<2>;

template
class MicroSolver<3>;
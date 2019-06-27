/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "rho_solver.h"

using namespace dealii;

template<int dim>
double MicroInitCondition<dim>::value(const Point<dim> &p, const unsigned int component) const {
    double val = 0; // Todo: This is not dependent on the macroscopic solution.
    for (int i = 0; i < dim; i++) {
        val += 1 - p[i] * p[i];
    }
    return val;
}

template<int dim>
void MicroInitCondition<dim>::set_macro_solution(const Vector<double> &macro_solution) {
    this->macro_sol = macro_solution;
}

template<int dim>
void MicroInitCondition<dim>::set_macro_cell_index(unsigned int index) {
    macro_cell_index = index;
}


template<int dim>
RhoSolver<dim>::RhoSolver():  dof_handler(triangulation), fe(1), macro_solution(nullptr),
                              macro_dof_handler(nullptr), old_macro_solution(nullptr) {
    std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
    refine_level = 1;
    num_grids = 1;
    integration_order = fe.degree + 1;
}

template<int dim>
void RhoSolver<dim>::setup() {
    make_grid();
    setup_system();
    setup_scatter();
}

template<int dim>
void RhoSolver<dim>::make_grid() {
    std::cout << "Setting up micro grid" << std::endl;
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);
    // If we ever use refinement, we have to remark every time we refine the grid.
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (std::fabs(cell->face(face_number)->center()(0) < 0)) {
                cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
            } // Else: Robin by default.
        }
    }

    std::cout << " Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

template<int dim>
void RhoSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);

    std::cout << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(integration_order), mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
}

template<int dim>
void RhoSolver<dim>::set_refine_level(int refinement_level) {
    this->refine_level = refinement_level;
}

template<int dim>
void RhoSolver<dim>::refine_grid() {
    triangulation.refine_global(1);
    setup_system();
    setup_scatter();
}

template<int dim>
void RhoSolver<dim>::setup_scatter() {
    solutions.clear();
    old_solutions.clear();
    righthandsides.clear();
    system_matrices.clear();
    compute_macroscopic_contribution();
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int i = 0; i < num_grids; i++) {
        Vector<double> solution(n_dofs);
        VectorTools::interpolate(dof_handler, MicroInitCondition<dim>(), solution);
        solutions.push_back(solution);
        Vector<double> old_solution(n_dofs);
        VectorTools::interpolate(dof_handler, MicroInitCondition<dim>(), old_solution);
        old_solutions.push_back(old_solution);

        Vector<double> rhs(n_dofs);
        righthandsides.push_back(rhs);

        SparseMatrix<double> system_matrix;
        system_matrices.push_back(system_matrix);
        intermediate_vector.reinit(dof_handler.n_dofs());
    }
}


template<int dim>
void RhoSolver<dim>::set_macro_solutions(Vector<double> *_solution, Vector<double> *_old_solution,
                                         DoFHandler<dim> *_dof_handler) {
    this->macro_solution = _solution;
    this->old_macro_solution = _old_solution;
    this->macro_dof_handler = _dof_handler;
}

template<int dim>
void RhoSolver<dim>::compute_macroscopic_contribution() {
    // Nothing needs to happen in this simple case
}


template<int dim>
void RhoSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(integration_order);
    QGauss<dim - 1> face_quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_normal_vectors | update_quadrature_points |
                                     update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_face_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (unsigned int k = 0; k < num_grids; k++) {
        righthandsides.at(k) = 0;
        solutions.at(k) = 0;
        system_matrices.at(k).reinit(sparsity_pattern);
        mass_matrix.vmult(righthandsides.at(k), old_solutions.at(k));
        laplace_matrix.vmult(intermediate_vector, old_solutions.at(k));
        righthandsides.at(k).add(-dt * (1 - scheme_theta), intermediate_vector); // scalar factor, matrix

        system_matrices.at(k).copy_from(mass_matrix);
        system_matrices.at(k).add(dt * scheme_theta * D, laplace_matrix);
    }
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary() and cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                fe_face_values.reinit(cell, face_number);
                cell_matrix = 0;
                cell->get_dof_indices(local_dof_indices);
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                    for (unsigned int j = 0; j < dofs_per_cell; j++) {
                        for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                            cell_matrix(i, j) +=
                                    kappa * dt * scheme_theta * R * fe_face_values.shape_value(i, q_index) *
                                    fe_face_values.shape_value(j, q_index) * fe_face_values.JxW(q_index);
                        }
                    }
                }
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                    for (unsigned int j = 0; j < dofs_per_cell; j++) {
                        for (unsigned int k = 0; k < num_grids; k++) {
                            system_matrices.at(k).add(local_dof_indices[i],
                                                      local_dof_indices[j],
                                                      cell_matrix(i, j));
                        }
                    }
                }
            }
        }

        for (unsigned int k = 0; k < num_grids; k++) {
            for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
                if (cell->face(face_number)->at_boundary()) { // Todo: Switch between Neumann and Robin
                    fe_face_values.reinit(cell, face_number);
                    cell->get_dof_indices(local_dof_indices);
                    cell_rhs = 0;
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                            cell_rhs(i) += (fe_face_values.shape_value(i, q_index)) * dt * kappa *
                                           (scheme_theta * (*macro_solution)(k) +
                                            (1 - scheme_theta) * (*old_macro_solution)(k) + p_F -
                                            R * (1 - scheme_theta) * old_solutions.at(k)(i)) *
                                           fe_face_values.JxW(q_index); // Todo: Is this a consistent approximation?
                        }
                    }
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        righthandsides.at(k)(local_dof_indices[i]) += cell_rhs(i);
                    }
                }
            }
        }
    }
}


template<int dim>
void RhoSolver<dim>::solve_time_step() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    for (unsigned int k = 0; k < num_grids; k++) {
        solver.solve(system_matrices.at(k), solutions.at(k), righthandsides.at(k), PreconditionIdentity());
    }
    std::cout << "   " << solver_control.last_step()
              << " CG iterations needed to obtain convergence."
              << std::endl;
    compute_residual();
    old_solutions = solutions; // todo: Does this work as expected? Consider swap
}


template<int dim>
void RhoSolver<dim>::compute_residual() {
    Vector<double> macro_domain_l2_error(num_grids);
//    for (unsigned int k = 0; k < num_grids; k++) { // Todo: Update with residual
//        boundary.set_macro_cell_index(k);
//        const unsigned int n_active = triangulation.n_active_cells();
//        Vector<double> difference_per_cell(n_active);
//        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell,
//                                          QGauss<dim>(3),
//                                          VectorTools::L2_norm);
//        double micro_l2_error = difference_per_cell.l2_norm();
//        VectorTools::integrate_difference(dof_handler, solutions.at(k), boundary, difference_per_cell,
//                                          QGauss<dim>(3),
//                                          VectorTools::H1_seminorm);
//        double micro_h1_error = difference_per_cell.l2_norm();
//        macro_domain_l2_error(k) = micro_l2_error;
//        macro_domain_h1_error(k) = micro_h1_error;
//    }
//    l2_error = macro_domain_l2_error.l2_norm() / macro_domain_l2_error.size(); // Is this the most correct norm?
//    h1_error = macro_domain_h1_error.l2_norm() / macro_domain_h1_error.size();

}

template<int dim>
void RhoSolver<dim>::iterate(const double &time_step) {
    dt = time_step;
    assemble_system();
    solve_time_step();
}

template<int dim>
void RhoSolver<dim>::set_num_grids(unsigned int _num_grids) {
    this->num_grids = _num_grids;
}

// Explicit instantiation

template
class RhoSolver<1>;

template
class RhoSolver<2>;

template
class RhoSolver<3>;

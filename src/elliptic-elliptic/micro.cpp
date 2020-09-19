/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "micro.h"

using namespace dealii;

template<int dim>
MicroSolver<dim>::MicroSolver(EllipticMicroData<dim> &micro_data, unsigned int refine_level):  dof_handler(
        triangulation),
                                                                                               refine_level(
                                                                                                       refine_level),
                                                                                               fe(1),
                                                                                               macro_solution(nullptr),
                                                                                               macro_dof_handler(
                                                                                                       nullptr),
                                                                                               pde_data(micro_data),
                                                                                               fem_objects{&solutions,
                                                                                                           &dof_handler,
                                                                                                           &mapmap,
                                                                                                           &fem_q_deg,
                                                                                                           nullptr} {
    printf("Solving micro problem in %d space dimensions\n", dim);
    num_grids = 1;
    fem_q_deg = 12;
}

template<int dim>
void MicroSolver<dim>::setup() {
    make_grid();
    setup_system();
    setup_scatter();
    compute_pullback_objects();
}

template<int dim>
void MicroSolver<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);
    printf("%d active micro cells\n", triangulation.n_active_cells());
}

template<int dim>
void MicroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d micro DoFs\n", dof_handler.n_dofs());
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
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
void MicroSolver<dim>::compute_pullback_objects() {
    QGauss<dim> quadrature_formula(fem_q_deg);
    FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
    SymmetricTensor<2, dim> kkt;
    double det_jac;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        for (unsigned int k = 0; k < num_grids; k++) {
            for (unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
                Tensor<2, dim> jacobian = pde_data.map_jac.mtensor_value(grid_locations.at(k),
                                                                         fe_values.quadrature_point(q_index));
                Tensor<2, dim> inv_jacobian = invert(jacobian);
                det_jac = determinant(jacobian);
                kkt = SymmetricTensor<2, dim>(inv_jacobian * transpose(inv_jacobian));
                mapmap.set(grid_locations.at(k), fe_values.quadrature_point(q_index), det_jac, kkt);
            }
        }
    }
}

template<int dim>
void MicroSolver<dim>::set_macro_solution(Vector<double> *_solution, DoFHandler<dim> *_dof_handler) {
    this->macro_solution = _solution;
    this->macro_dof_handler = _dof_handler;
}

template<int dim>
void MicroSolver<dim>::compute_macroscopic_contribution() {
    // Nothing needs to happen in this simple case
}

template<int dim>
void
MicroSolver<dim>::integrate_cell(const typename DoFHandler<dim>::active_cell_iterator &cell, const unsigned int &k,
                                 FEValues<dim> &fe_values,
                                 const unsigned int &n_q_points, FullMatrix<double> &cell_matrix,
                                 Vector<double> &cell_rhs) {
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    SymmetricTensor<2, dim> kkt;
    double det_jac;
    fe_values.reinit(cell);
    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        mapmap.get(grid_locations.at(k), fe_values.quadrature_point(q_index), det_jac, kkt);
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                cell_matrix(i, j) += fe_values.shape_grad(i, q_index) * kkt
                                     * fe_values.shape_grad(j, q_index) * fe_values.JxW(q_index) * det_jac;
            }
        }
    }
    for (unsigned int i = 0; i < dofs_per_cell; i++) {
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            mapmap.get_det_jac(grid_locations.at(k), fe_values.quadrature_point(q_index), det_jac);
            const Point<dim> mapped_p = pde_data.mapping.mmap(grid_locations.at(k),
                                                              fe_values.quadrature_point(q_index));
            double rhs_val = pde_data.rhs.mvalue(grid_locations.at(k), mapped_p);
            cell_rhs(i) += ((*macro_solution)(k) + rhs_val) *
                           fe_values.shape_value(i, q_index) * fe_values.JxW(q_index) * det_jac;
        }
    }
}


template<int dim>
void MicroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(fem_q_deg);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (unsigned int k = 0; k < num_grids; k++) {
        righthandsides.at(k) = 0;
        solutions.at(k) = 0;
        system_matrices.at(k).reinit(sparsity_pattern);
        for (const auto &cell: dof_handler.active_cell_iterators()) {
            cell_matrix = 0;
            cell_rhs = 0;
            integrate_cell(cell, k, fe_values, quadrature_formula.size(), cell_matrix, cell_rhs);
            cell->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    system_matrices[k].add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
                }
                righthandsides[k](local_dof_indices[i]) += cell_rhs(i);
            }
        }
        std::map<types::global_dof_index, double> boundary_values;
        pde_data.bc.set_macro_point(grid_locations.at(k));
        VectorTools::interpolate_boundary_values(dof_handler, 0, pde_data.bc, boundary_values);
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
    printf("\t %d CG iterations to convergence (micro)\n", solver_control.last_step());
}


template<int dim>
void MicroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    Vector<double> macro_domain_l2_error(num_grids);
    Vector<double> macro_domain_h1_error(num_grids);
    for (unsigned int k = 0; k < num_grids; k++) {
        pde_data.solution.set_macro_point(grid_locations.at(k));
        const unsigned int n_active = triangulation.n_active_cells();
        Vector<double> difference_per_cell(n_active);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), pde_data.solution, difference_per_cell,
                                          QGauss<dim>(fem_q_deg),
                                          VectorTools::L2_norm);
        double micro_l2_error = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                  VectorTools::L2_norm);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), pde_data.solution, difference_per_cell,
                                          QGauss<dim>(fem_q_deg),
                                          VectorTools::H1_seminorm);
        double micro_h1_error = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                  VectorTools::H1_seminorm);
        macro_domain_l2_error(k) = micro_l2_error;
        macro_domain_h1_error(k) = micro_h1_error;
    }
    Vector<double> macro_integral(num_grids);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_l2_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(fem_q_deg), VectorTools::L2_norm);
    l2_error = VectorTools::compute_global_error(macro_dof_handler->get_triangulation(), macro_integral,
                                                 VectorTools::L2_norm);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_h1_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(fem_q_deg), VectorTools::L2_norm);
    h1_error = VectorTools::compute_global_error(macro_dof_handler->get_triangulation(), macro_integral,
                                                 VectorTools::L2_norm); //todo: Not sure about this norm, although output is consistent
}

template<int dim>
void MicroSolver<dim>::run() {
    assemble_system();
    solve();
}

template<int dim>
void MicroSolver<dim>::set_exact_solution() {
    std::cout << "Exact micro-solution set" << std::endl;
    MappingQ1<dim> mapping;
    AffineConstraints<double> constraints; // Object that is necessary to use `Vectortools::project`
    constraints.close();
    for (unsigned int k = 0; k < num_grids; k++) {
        pde_data.solution.set_macro_point(grid_locations.at(k));
        VectorTools::project(mapping, dof_handler, constraints, QGauss<dim>(fem_q_deg), pde_data.solution,
                             solutions.at(k));
    }
}

template<int dim>
unsigned int MicroSolver<dim>::get_num_grids() const {
    return num_grids;
}

template<int dim>
void MicroSolver<dim>::set_grid_locations(const std::vector<Point<dim>> &locations) {
    grid_locations = locations;
    num_grids = locations.size();
}

// Explicit instantiation

template
class MicroSolver<1>;

template
class MicroSolver<2>;

template
class MicroSolver<3>;

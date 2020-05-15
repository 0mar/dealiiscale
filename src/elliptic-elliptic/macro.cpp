/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "macro.h"

using namespace dealii;


template<int dim>
MacroSolver<dim>::MacroSolver(MacroData<dim> &macro_data, unsigned int refine_level):dof_handler(triangulation),
                                                                                     pde_data(macro_data), fe(1),
                                                                                     micro_dof_handler(nullptr),
                                                                                     micro_solutions(nullptr),
                                                                                     refine_level(refine_level) {
    printf("Solving macro problem in %d space dimensions\n", dim);
}

template<int dim>
void MacroSolver<dim>::setup() {
    make_grid();
    setup_system();
}

template<int dim>
void MacroSolver<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);
    printf("%d active macro cells\n", triangulation.n_active_cells());
}

template<int dim>
void MacroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d macro DoFs\n", dof_handler.n_dofs());
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    micro_contribution.reinit(dof_handler.n_dofs());
    get_dof_locations(micro_grid_locations);
}

template<int dim>
Vector<double> MacroSolver<dim>::get_exact_solution() const {
    Vector<double> exact_values(dof_handler.n_dofs());
    MappingQ1<dim> mapping;
    std::vector<Point<dim>> dof_locations(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_locations);
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        exact_values[i] = pde_data.solution.value(dof_locations[i]);
    }
    return exact_values;
}

template<int dim>
void MacroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(8);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    compute_microscopic_contribution();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    system_matrix = 0;
    system_rhs = 0;
    std::vector<double> local_micro_cont(n_q_points);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.get_function_values(micro_contribution, local_micro_cont);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index)) *
                                         fe_values.JxW(q_index);


                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                (local_micro_cont[q_index] + pde_data.rhs.value(fe_values.quadrature_point(q_index))) *
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
    VectorTools::interpolate_boundary_values(dof_handler, 0, pde_data.bc, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template<int dim>
void MacroSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    printf("\t %d CG iterations to convergence (macro)\n", solver_control.last_step());
}

template<int dim>
void MacroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    const unsigned int n_active = triangulation.n_active_cells();
    Vector<double> difference_per_cell(n_active);
    VectorTools::integrate_difference(dof_handler, solution, pde_data.solution, difference_per_cell, QGauss<dim>(8),
                                      VectorTools::L2_norm);
    l2_error = difference_per_cell.l2_norm();
    VectorTools::integrate_difference(dof_handler, solution, pde_data.solution, difference_per_cell, QGauss<dim>(8),
                                      VectorTools::H1_seminorm);
    h1_error = difference_per_cell.l2_norm();
}

template<int dim>
void MacroSolver<dim>::set_micro_objects(std::vector<Vector<double>> *_solutions, DoFHandler<dim> *_dof_handler,
                                         MapMap<dim, dim> *_micro_mapmap) {
    this->micro_solutions = _solutions;
    this->micro_dof_handler = _dof_handler;
    this->micro_mapmap = micro_mapmap;

}

template<int dim>
double MacroSolver<dim>::get_micro_bulk(unsigned int cell_index) const {
    // manufactured as: f(x) = \int_Y \rho(x,y)dy
    double integral = 0;
    QGauss<dim> quadrature_formula(8);
    FEValues<dim> fe_values(micro_dof_handler->get_fe(), quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = micro_dof_handler->get_fe().dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        std::vector<double> interp_solution(n_q_points);
        fe_values.get_function_values(micro_solutions->at(cell_index), interp_solution);
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            double det_jac;
            micro_mapmap->get_det_jac(micro_grid_locations.at(cell_index), fe_values.quadrature_point(q_index),
                                      det_jac);
            integral += interp_solution[q_index] / det_jac * fe_values.JxW(q_index);
        }
    }
    return integral;
}


template<int dim>
double MacroSolver<dim>::get_micro_flux(unsigned int micro_index) const {
    // Computed as: f(x) = \int_\Gamma_R \nabla_y \rho(x,y) \cdot n_y d_\sigma_y
    const int integration_order = 8;
    double integral = 0;
    QGauss<dim - 1> quadrature_formula(integration_order); // Not necessarily the same dim
    FEFaceValues<dim> fe_face_values(micro_dof_handler->get_fe(),
                                     quadrature_formula,
                                     update_values | update_quadrature_points | update_JxW_values |
                                     update_normal_vectors | update_gradients);
    const unsigned int n_q_face_points = quadrature_formula.size();
    std::vector<Tensor<1, dim>> solution_gradient(n_q_face_points);
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                fe_face_values.get_function_gradients(micro_solutions->at(micro_index), solution_gradient);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    double det_jac;
                    Assert(false,ExcNotImplemented("Flux integrals with mappings not implemented yet"))
                    micro_mapmap->get_det_jac(micro_grid_locations.at(micro_index), fe_face_values.quadrature_point(q_index),
                                              det_jac);
                    double neumann = solution_gradient[q_index] * fe_face_values.normal_vector(q_index);
                    integral += neumann / det_jac * fe_face_values.JxW(q_index);
                }
            }
        }
    }
    return integral;
}


template<int dim>
void MacroSolver<dim>::compute_microscopic_contribution() {
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        micro_contribution[i] = get_micro_bulk(i);
    }
}

template<int dim>
void MacroSolver<dim>::get_dof_locations(std::vector<Point<dim>> &locations) {
    MappingQ1<dim> mapping;
    locations.clear();
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, locations);

}

template<int dim>
void MacroSolver<dim>::run() {
    assemble_system();
    solve();
}

// Explicit instantiation

template
class MacroSolver<1>;

template
class MacroSolver<2>;

template
class MacroSolver<3>;
/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "macro.h"

using namespace dealii;


template<int dim>
MacroSolver<dim>::MacroSolver(BioMacroData<dim> &macro_data, unsigned int refine_level):dof_handler(triangulation),
                                                                                        pde_data(macro_data), fe(1),
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
    const double EPS = 1E-4;
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double abs_x = std::fabs(cell->face(face_number)->center()(0));
                const double abs_y = std::fabs(cell->face(face_number)->center()(1));
                if (std::fabs(abs_y - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
                } else if (std::fabs(abs_x - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(DIRICHLET_BOUNDARY);
                } else {
                    Assert(false, ExcMessage("Part of the boundary is not initialized correctly"))
                }
            }
        }
    }
}

template<int dim>
void MacroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d macro DoFs\n", dof_handler.n_dofs());
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix_u.reinit(sparsity_pattern);
    system_matrix_w.reinit(sparsity_pattern);
    sol_u.reinit(dof_handler.n_dofs());
    sol_w.reinit(dof_handler.n_dofs());
    system_rhs_u.reinit(dof_handler.n_dofs());
    system_rhs_w.reinit(dof_handler.n_dofs());
    micro_contribution_u.reinit(dof_handler.n_dofs());
    micro_contribution_w.reinit(dof_handler.n_dofs());
    get_dof_locations(micro_grid_locations);
}

template<int dim>
void MacroSolver<dim>::set_exact_solution() {
    std::cout << "Exact macro-solution set" << std::endl;
    MappingQ1<dim> mapping;
    AffineConstraints<double> constraints;
    constraints.close();
    VectorTools::project(mapping, dof_handler, constraints, QGauss<dim>(8), pde_data.solution_u, sol_u);
    VectorTools::project(mapping, dof_handler, constraints, QGauss<dim>(8), pde_data.solution_w, sol_w);
}

template<int dim>
void MacroSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(8);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
    QGauss<dim - 1> face_quadrature_formula(8);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_quadrature_points | update_values | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_face_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix_u(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix_w(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs_u(dofs_per_cell);
    Vector<double> cell_rhs_w(dofs_per_cell);

    compute_microscopic_contribution();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    system_matrix_u = 0;
    system_matrix_w = 0;
    system_rhs_u = 0;
    system_rhs_w = 0;
    std::vector<double> u_micro_cont(n_q_points);
    std::vector<double> w_micro_cont(n_q_points);
    const double &k_1 = pde_data.params.get_double("kappa_1");
    const double &k_4 = pde_data.params.get_double("kappa_4");
    const double &D_1 = pde_data.params.get_double("D_1");
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix_u = 0;
        cell_matrix_w = 0;
        cell_rhs_u = 0;
        cell_rhs_w = 0;
        fe_values.get_function_values(micro_contribution_u, u_micro_cont);
        fe_values.get_function_values(micro_contribution_w, w_micro_cont);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            const Point<dim> &q_point = fe_values.quadrature_point(q_index);
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const double laplace_term = fe_values.shape_grad(i, q_index) *
                                                fe_values.shape_grad(j, q_index) *
                                                fe_values.JxW(q_index);
                    const double mass_term =
                            fe_values.shape_value(i, q_index) * fe_values.shape_value(j, q_index) *
                            fe_values.JxW(q_index);
                    cell_matrix_u(i, j) += laplace_term - mass_term * k_1 * pde_data.inflow_measure.value(q_point);
                    cell_matrix_w(i, j) +=
                            D_1 * laplace_term + mass_term * k_4 * pde_data.outflow_measure.value(q_point);
                }
                cell_rhs_u(i) += (u_micro_cont[q_index] + pde_data.bulk_rhs_u.value(q_point)) *
                                 fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
                cell_rhs_w(i) += (w_micro_cont[q_index] + pde_data.bulk_rhs_w.value(q_point)) *
                                 fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
            }
        }

        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            const auto &face = cell->face(face_number);
            if (face->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {

                        if (face->boundary_id() == NEUMANN_BOUNDARY) {

                            cell_rhs_u(i) += fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index) *
                                             pde_data.bc_u_2.value(fe_face_values.quadrature_point(q_index));
                            cell_rhs_w(i) += fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index) *
                                             pde_data.bc_w_1.value(fe_face_values.quadrature_point(q_index));
                        } else {
                            cell_rhs_w(i) += fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index) *
                                             pde_data.bc_w_2.value(fe_face_values.quadrature_point(q_index));
                        }
                    }
                }
            }
        }
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                system_matrix_u.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_u(i, j));
                system_matrix_w.add(local_dof_indices[i], local_dof_indices[j], cell_matrix_w(i, j));
            }
            system_rhs_u(local_dof_indices[i]) += cell_rhs_u(i);
            system_rhs_w(local_dof_indices[i]) += cell_rhs_w(i);
        }
    }
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, DIRICHLET_BOUNDARY, pde_data.bc_u_1,
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix_u, sol_u, system_rhs_u);
}

template<int dim>
void MacroSolver<dim>::solve() {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix_u, sol_u, system_rhs_u, PreconditionIdentity());
    solver.solve(system_matrix_w, sol_w, system_rhs_w, PreconditionIdentity());
    printf("\t %d CG iterations to convergence (macro)\n", solver_control.last_step());
}

template<int dim>
void MacroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    const unsigned int n_active = triangulation.n_active_cells();
    Vector<double> difference_per_cell(n_active);
    VectorTools::integrate_difference(dof_handler, sol_u, pde_data.solution_u, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::L2_norm);
    l2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler, sol_w, pde_data.solution_w, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::L2_norm);
    l2_error += VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler, sol_u, pde_data.solution_u, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::H1_seminorm);
    h1_error = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                 VectorTools::H1_seminorm);
    VectorTools::integrate_difference(dof_handler, sol_w, pde_data.solution_w, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::H1_seminorm);
    h1_error += VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                  VectorTools::H1_seminorm);
}

template<int dim>
void MacroSolver<dim>::set_micro_objects(const MicroFEMObjects<dim> &micro_fem_objects) {
    this->micro = micro_fem_objects;
}

template<int dim>
void
MacroSolver<dim>::integrate_micro_cells(unsigned int micro_index, const Point<dim> &macro_point, double &u_contribution,
                                        double &w_contribution) {
    // Computed as: f(x) = \int_\Gamma_R \nabla_y \rho(x,y) \cdot n_y d_\sigma_y
    u_contribution = 0;
    w_contribution = 0;
    QGauss<dim - 1> quadrature_formula(*(micro.q_degree)); // Not necessarily the same dim
    FEFaceValues<dim> fe_face_values(micro.dof_handler->get_fe(),
                                     quadrature_formula, update_values | update_quadrature_points | update_JxW_values |
                                                         update_normal_vectors);
    const unsigned int n_q_face_points = quadrature_formula.size();
    const unsigned int dofs_per_cell = micro.dof_handler->get_fe().dofs_per_cell;
    std::vector<double> interp_solution(n_q_face_points);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const double &k_2 = micro.data->params.get_double("kappa_2");
    const double &k_3 = micro.data->params.get_double("kappa_3");
    double det_jac;
    Tensor<2, dim> rotation_matrix; // Todo: Move
    rotation_matrix[0][1] = -1;
    rotation_matrix[1][0] = 1;
    for (const auto &cell: micro.dof_handler->active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                fe_face_values.get_function_values(micro.solutions->at(micro_index), interp_solution);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    const double &jxw = fe_face_values.JxW(q_index);
                    const Point<dim> &q_point = fe_face_values.quadrature_point(q_index);
//                    const double y0 = q_point(0);
//                    const double y1 = q_point(1);
//                    const double symb_val = micro.data->params.get_double("D_2") *std::sqrt(2)/2 * (y0*y1 - y0*(1-y1) - y1*(1-y1));

                    det_jac = (micro.data->map_jac.mtensor_value(macro_point, q_point) * rotation_matrix *
                               fe_face_values.normal_vector(q_index)).norm();
//                    const double num_val = micro.data->bc_v_1.mvalue(macro_point, q_point) - k_2*interp_solution[q_index] + micro.data->params.get_double("kappa_1")*pde_data.solution_u.value(macro_point);
//                    printf("(%.2f, %.2f)x(%.2f, %.2f) -> %.2f (exact %.2f)\n", macro_point(0), macro_point(1), y0,y1, num_val, symb_val);
                    switch (cell->face(face_number)->boundary_id()) {
                        case 0: // INFLOW_BOUNDARY // Todo: Not clean, should be micro enums
                            u_contribution += (-k_2 * interp_solution[q_index] +
                                               micro.data->bc_v_1.mvalue(macro_point, q_point)) * jxw * det_jac;
//                                u_contribution += (symb_val - ) * jxw * det_jac;
                            break;
                        case 1: // OUTFLOW_BOUNDARY
                            w_contribution += (k_3 * interp_solution[q_index] +
                                               micro.data->bc_v_2.mvalue(macro_point, q_point)) * jxw * det_jac;
                            break;
                        }
                }
            }
        }
    }
}


template<int dim>
void MacroSolver<dim>::compute_microscopic_contribution() {
    std::vector<Point<dim>> locations;
    get_dof_locations(locations);
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        integrate_micro_cells(i, locations[i], micro_contribution_u(i),
                              micro_contribution_w(i)); // Todo: does this work?
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

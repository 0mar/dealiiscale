/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "macro.h"

using namespace dealii;

template<int dim>
MacroSolver<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe)
        : fe_values(fe, QGauss<dim>(8),
                    update_values | update_gradients | update_quadrature_points | update_JxW_values),
          fe_face_values(fe, QGauss<dim - 1>(8), update_quadrature_points | update_values | update_JxW_values),
          rhs_values(fe_values.get_quadrature().size()) { // todo: optimize contents
}

template<int dim>
MacroSolver<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
        : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                    update_values | update_gradients | update_quadrature_points | update_JxW_values),
          fe_face_values(scratch_data.fe_face_values.get_fe(), scratch_data.fe_face_values.get_quadrature(),
                         update_quadrature_points | update_values | update_JxW_values),
          rhs_values(scratch_data.rhs_values.size()) {

}

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
    GridGenerator::subdivided_hyper_cube(triangulation, refine_level,-1, 1);
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
    u_constraints.clear();
//    DoFTools::make_hanging_node_constraints(dof_handler, u_constraints);
    u_constraints.close();
    w_constraints.clear();
//    DoFTools::make_hanging_node_constraints(dof_handler, w_constraints);
    w_constraints.close();
    printf("%d macro DoFs\n", dof_handler.n_dofs());
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, w_constraints, false);
    sparsity_pattern.copy_from(dsp);
    system_matrix_u.reinit(sparsity_pattern);
    system_matrix_w.reinit(sparsity_pattern);
    sol_u.reinit(dof_handler.n_dofs());
    sol_w.reinit(dof_handler.n_dofs());
    old_sol_u.reinit(dof_handler.n_dofs());
    old_sol_w.reinit(dof_handler.n_dofs());
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
void MacroSolver<dim>::local_assemble_system(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             MacroSolver::AssemblyScratchData &scratch_data,
                                             MacroSolver::AssemblyCopyData &copy_data) {
    const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();
    const unsigned int n_q_face_points = scratch_data.fe_face_values.get_quadrature().size();
    copy_data.cell_matrix_u.reinit(fe.dofs_per_cell, fe.dofs_per_cell);
    copy_data.cell_rhs_u.reinit(fe.dofs_per_cell);
    copy_data.cell_matrix_w.reinit(fe.dofs_per_cell, fe.dofs_per_cell);
    copy_data.cell_rhs_w.reinit(fe.dofs_per_cell);
    copy_data.local_dof_indices.resize(fe.dofs_per_cell);
    scratch_data.fe_values.reinit(cell);
    const auto &sd = scratch_data;
    std::vector<double> u_micro_cont(n_q_points);
    std::vector<double> w_micro_cont(n_q_points);
    sd.fe_values.get_function_values(micro_contribution_u, u_micro_cont);
    sd.fe_values.get_function_values(micro_contribution_w, w_micro_cont);
    const double &k_1 = pde_data.params.get_double("kappa_1");
    const double &k_3 = pde_data.params.get_double("kappa_3");
    const double &D_1 = pde_data.params.get_double("D_1");
    for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
        const Point<dim> &q_point = sd.fe_values.quadrature_point(q_index);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
                const double laplace_term = sd.fe_values.shape_grad(i, q_index) *
                                            sd.fe_values.shape_grad(j, q_index) *
                                            sd.fe_values.JxW(q_index);
                const double mass_term =
                        sd.fe_values.shape_value(i, q_index) * sd.fe_values.shape_value(j, q_index) *
                        sd.fe_values.JxW(q_index);
                copy_data.cell_matrix_u(i, j) +=
                        laplace_term + mass_term * k_1 * pde_data.inflow_measure.value(q_point);
                copy_data.cell_matrix_w(i, j) +=
                        D_1 * laplace_term + mass_term * k_3 * pde_data.outflow_measure.value(q_point);
            }
            copy_data.cell_rhs_u(i) += (-u_micro_cont[q_index] + pde_data.bulk_rhs_u.value(q_point)) *
                                       sd.fe_values.shape_value(i, q_index) * sd.fe_values.JxW(q_index);
            copy_data.cell_rhs_w(i) += (-w_micro_cont[q_index] + pde_data.bulk_rhs_w.value(q_point)) *
                                       sd.fe_values.shape_value(i, q_index) * sd.fe_values.JxW(q_index);
        }
    }

    for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
        const auto &face = cell->face(face_number);
        if (face->at_boundary()) {
            scratch_data.fe_face_values.reinit(cell, face_number);
            for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                for (unsigned int i = 0; i < fe.dofs_per_cell; i++) {

                    if (face->boundary_id() == NEUMANN_BOUNDARY) {

                        copy_data.cell_rhs_u(i) +=
                                sd.fe_face_values.shape_value(i, q_index) * sd.fe_face_values.JxW(q_index) *
                                pde_data.bc_u_2.value(sd.fe_face_values.quadrature_point(q_index));
                        copy_data.cell_rhs_w(i) +=
                                sd.fe_face_values.shape_value(i, q_index) * sd.fe_face_values.JxW(q_index) *
                                pde_data.bc_w_2.value(sd.fe_face_values.quadrature_point(q_index));
                    } else {
                        copy_data.cell_rhs_w(i) +=
                                sd.fe_face_values.shape_value(i, q_index) * sd.fe_face_values.JxW(q_index) *
                                pde_data.bc_w_1.value(sd.fe_face_values.quadrature_point(q_index));
                    }
                }
            }
        }
    }
    cell->get_dof_indices(copy_data.local_dof_indices);
}

template<int dim>
void MacroSolver<dim>::copy_local_to_global(const MacroSolver::AssemblyCopyData &copy_data) {
    u_constraints.distribute_local_to_global(copy_data.cell_matrix_u, copy_data.cell_rhs_u,
                                             copy_data.local_dof_indices, system_matrix_u, system_rhs_u);
    w_constraints.distribute_local_to_global(copy_data.cell_matrix_w, copy_data.cell_rhs_w,
                                             copy_data.local_dof_indices, system_matrix_w, system_rhs_w);

}

template<int dim>
void MacroSolver<dim>::assemble_system() {
    system_matrix_u = 0;
    system_matrix_w = 0;
    system_rhs_u = 0;
    system_rhs_w = 0;
//    std::cout << "Computing micro assembly" << std::endl;
    compute_microscopic_contribution();
//    std::cout << "Starting assembly" << std::endl;
    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), *this, &MacroSolver::local_assemble_system,
                    &MacroSolver::copy_local_to_global, AssemblyScratchData(fe), AssemblyCopyData());
//    std::cout << "Finishing assembly" << std::endl;
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, DIRICHLET_BOUNDARY, pde_data.bc_u_1, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix_u, sol_u, system_rhs_u);
}

template<int dim>
void MacroSolver<dim>::solve() {
    old_sol_u = sol_u;
    old_sol_w = sol_w;
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix_u, sol_u, system_rhs_u, PreconditionIdentity());
    solver.solve(system_matrix_w, sol_w, system_rhs_w, PreconditionIdentity());
}

template<int dim>
void MacroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    const unsigned int n_active = triangulation.n_active_cells();
    Vector<double> difference_per_cell(n_active);
    VectorTools::integrate_difference(dof_handler, sol_u, pde_data.solution_u, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::L2_norm);
    l2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    printf("U error: %.4e\n", l2_error);
    VectorTools::integrate_difference(dof_handler, sol_w, pde_data.solution_w, difference_per_cell,
                                      QGauss<dim>(8),
                                      VectorTools::L2_norm);
    l2_error += VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    printf("U + W error: %.4e\n", l2_error);
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
void MacroSolver<dim>::compute_residual(double &l2_residual) {
    Vector<double> error(dof_handler.n_dofs());
    error += sol_u;
    error -= old_sol_u;
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, error, Functions::ZeroFunction<dim>(), difference_per_cell,
                                      QGauss<dim>(8), VectorTools::L2_norm);
    l2_residual = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    error = 0;
    error += sol_w;
    error -= old_sol_w;
    VectorTools::integrate_difference(dof_handler, error, Functions::ZeroFunction<dim>(), difference_per_cell,
                                      QGauss<dim>(8), VectorTools::L2_norm);
    l2_residual = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
}

template<int dim>
void MacroSolver<dim>::set_micro_objects(const MicroFEMObjects<dim> &micro_fem_objects) {
    this->micro = micro_fem_objects;
}

template<int dim>
void
MacroSolver<dim>::integrate_micro_cells(unsigned int micro_index, const Point<dim> &macro_point, double &u_contribution,
                                        double &w_contribution) {
    // Todo: Could be parallel
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
    const double &k_4 = micro.data->params.get_double("kappa_4");
    double det_jac;
    for (const auto &cell: micro.dof_handler->active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                fe_face_values.get_function_values(micro.solutions->at(micro_index), interp_solution);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    const double &jxw = fe_face_values.JxW(q_index);
                    const Point<dim> &q_point = fe_face_values.quadrature_point(q_index);
                    const Point<dim> mq_point = micro.data->mapping.mmap(macro_point, q_point);
                    micro.get_map_det_jac_bc(macro_point, fe_face_values.quadrature_point(q_index),
                                             fe_face_values.normal_vector(q_index), det_jac);
                    switch (cell->face(face_number)->boundary_id()) {
                        case 0: // INFLOW_BOUNDARY // Not clean, should be micro enums
                            u_contribution += (-k_2 * interp_solution[q_index] +
                                               micro.data->bc_v_1.mvalue(macro_point, mq_point)) * jxw * det_jac;
                            break;
                        case 1: // OUTFLOW_BOUNDARY
                            w_contribution += (-k_4 * interp_solution[q_index] +
                                               micro.data->bc_v_2.mvalue(macro_point, mq_point)) * jxw * det_jac;
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
        integrate_micro_cells(i, locations[i], micro_contribution_u(i), micro_contribution_w(i));
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
void MacroSolver<dim>::assemble_and_solve() {
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

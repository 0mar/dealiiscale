/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "micro.h"

using namespace dealii;

template<int dim>
MicroSolver<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe)
        :  fe_values(fe, QGauss<dim>(12), update_values | update_gradients | // todo Fix this constant
                                          update_quadrature_points | update_JxW_values),
           fe_face_values(fe, QGauss<dim - 1>(12), update_quadrature_points | update_values | update_JxW_values),
           rhs_values(fe_values.get_quadrature().size()) {


}

template<int dim>
MicroSolver<dim>::AssemblyScratchData::AssemblyScratchData(const MicroSolver::AssemblyScratchData &scratch_data)
        : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                    update_values | update_gradients | update_quadrature_points | update_JxW_values),
          fe_face_values(scratch_data.fe_face_values.get_fe(), scratch_data.fe_face_values.get_quadrature(),
                         update_quadrature_points | update_values | update_JxW_values),
          rhs_values(scratch_data.rhs_values.size()) {

}

template<int dim>
MicroSolver<dim>::AssemblyCopyData::AssemblyCopyData(unsigned int num_grids) {
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        cell_matrices.push_back(FullMatrix<double>());
        cell_rhs.push_back(Vector<double>());
    }
}

template<int dim>
MicroSolver<dim>::MicroSolver(BioMicroData<dim> &micro_data, unsigned int refine_level):
        dof_handler(triangulation),
        refine_level(refine_level),
        fe(1),
        fem_q_deg(12),
        quadrature_formula(fem_q_deg),
        face_quadrature_formula(fem_q_deg),
        sol_u(nullptr),
        sol_w(nullptr),
        macro_dof_handler(nullptr),
        pde_data(micro_data),
        fem_objects{&solutions, &dof_handler, &mapmap, &fem_q_deg, &pde_data} {
    printf("Solving micro problem in %d space dimensions\n", dim);
    num_grids = 1;
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
    const double EPS = 1E-4;
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double x = cell->face(face_number)->center()(0);
                const double y = cell->face(face_number)->center()(1);
                if (std::fabs(x - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(OUTFLOW_BOUNDARY);
                } else if (std::fabs(x + 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(INFLOW_BOUNDARY);
                } else if (std::fabs(y - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(TOP_NEUMANN);
                } else if (std::fabs(y + 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(BOTTOM_NEUMANN);
                } else {
                    Assert(false, ExcMessage("Part of the boundary is not initialized correctly"))
                }
            }
        }
    }
}

template<int dim>
void MicroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    constraints.close();
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
    FEValues<dim> fe_values(fe, quadrature_formula, update_quadrature_points);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula, update_quadrature_points | update_normal_vectors);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
    SymmetricTensor<2, dim> kkt;
    Tensor<2, dim> rotation_matrix; // inits to zero
    rotation_matrix[0][1] = -1;
    rotation_matrix[1][0] = 1;
    double det_jac;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
            for (unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
                Tensor<2, dim> jacobian = pde_data.map_jac.mtensor_value(grid_locations[grid_num],
                                                                         fe_values.quadrature_point(q_index));
                Tensor<2, dim> inv_jacobian = invert(jacobian);
                det_jac = determinant(jacobian);
                Assert(det_jac > 1E-4, ExcMessage("Determinant of jacobian of mapping is not positive!"))
                kkt = SymmetricTensor<2, dim>(inv_jacobian * transpose(inv_jacobian));
                mapmap.set(grid_locations[grid_num], fe_values.quadrature_point(q_index), det_jac, kkt);
            }
            for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
                if (cell->face(face_number)->at_boundary()) {
                    fe_face_values.reinit(cell, face_number);
                    for (unsigned int q_index = 0; q_index < face_quadrature_formula.size(); q_index++) {
                        det_jac = (pde_data.map_jac.mtensor_value(grid_locations[grid_num],
                                                                  fe_face_values.quadrature_point(q_index)) *
                                   rotation_matrix * fe_face_values.normal_vector(q_index)).norm();
                        // Dummy kkt object. None is used on boundary.
                        mapmap.set(grid_locations[grid_num], fe_face_values.quadrature_point(q_index), det_jac, kkt);
                    }
                }
            }
        }
    }
}

template<int dim>
void
MicroSolver<dim>::set_macro_solution(Vector<double> *_sol_u, Vector<double> *_sol_w, DoFHandler<dim> *_dof_handler) {
    this->sol_u = _sol_u;
    this->sol_w = _sol_w;
    this->macro_dof_handler = _dof_handler;
}

template<int dim>
void MicroSolver<dim>::compute_macroscopic_contribution() {
    // Nothing needs to happen in this coupling structure
}

template<int dim>
void MicroSolver<dim>::local_assemble_system(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             MicroSolver::AssemblyScratchData &scratch_data,
                                             MicroSolver::AssemblyCopyData &copy_data) {
    const double &k_1 = pde_data.params.get_double("kappa_1");
    const double &k_2 = pde_data.params.get_double("kappa_2");
    const double &k_3 = pde_data.params.get_double("kappa_3");
    const double &k_4 = pde_data.params.get_double("kappa_4");
    const double &D_2 = pde_data.params.get_double("D_2");
    const auto &sd = scratch_data; //abbr
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    SymmetricTensor<2, dim> kkt;
    double det_jac;
    copy_data.local_dof_indices.resize(fe.dofs_per_cell);
    scratch_data.fe_values.reinit(cell);
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        copy_data.cell_matrices[grid_num].reinit(fe.dofs_per_cell, fe.dofs_per_cell);
        copy_data.cell_rhs[grid_num].reinit(fe.dofs_per_cell);

        for (unsigned int q_index = 0; q_index < quadrature_formula.size(); ++q_index) {
            mapmap.get(grid_locations[grid_num], sd.fe_values.quadrature_point(q_index), det_jac, kkt);
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                    copy_data.cell_matrices[grid_num](i, j) += D_2 * sd.fe_values.shape_grad(i, q_index) * kkt
                                                               * sd.fe_values.shape_grad(j, q_index) *
                                                               sd.fe_values.JxW(q_index) * det_jac;
                }
            }
        }
        for (unsigned int q_index = 0; q_index < quadrature_formula.size(); q_index++) {
            mapmap.get_det_jac(grid_locations[grid_num], sd.fe_values.quadrature_point(q_index), det_jac);
            const Point<dim> mapped_p = pde_data.mapping.mmap(grid_locations[grid_num],
                                                              sd.fe_values.quadrature_point(q_index));
            double rhs_val = pde_data.bulk_rhs_v.mvalue(grid_locations[grid_num], mapped_p);
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                copy_data.cell_rhs[grid_num](i) +=
                        rhs_val * sd.fe_values.shape_value(i, q_index) * sd.fe_values.JxW(q_index) *
                        det_jac;
            }
        }
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                scratch_data.fe_face_values.reinit(cell, face_number);
                for (unsigned int q_index = 0; q_index < face_quadrature_formula.size(); q_index++) {
                    mapmap.get_det_jac(grid_locations[grid_num], sd.fe_face_values.quadrature_point(q_index),
                                       det_jac);
                    Point<dim> mp = pde_data.mapping.mmap(grid_locations[grid_num],
                                                          sd.fe_face_values.quadrature_point(q_index));
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        switch (cell->face(face_number)->boundary_id()) {
                            case INFLOW_BOUNDARY:
                                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                                    copy_data.cell_matrices[grid_num](i, j) +=
                                            sd.fe_face_values.shape_value(i, q_index) * det_jac * k_2 *
                                            sd.fe_face_values.shape_value(j, q_index) *
                                            sd.fe_face_values.JxW(q_index);
                                }
                                copy_data.cell_rhs[grid_num](i) +=
                                        (pde_data.bc_v_1.mvalue(grid_locations[grid_num], mp) +
                                         k_1 * (*sol_u)(grid_num)) *
                                        sd.fe_face_values.shape_value(i, q_index) *
                                        det_jac * sd.fe_face_values.JxW(q_index);
                                break;
                            case OUTFLOW_BOUNDARY:
                                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                                    copy_data.cell_matrices[grid_num](i, j) +=
                                            sd.fe_face_values.shape_value(i, q_index) * det_jac * k_4 *
                                            sd.fe_face_values.shape_value(j, q_index) *
                                            sd.fe_face_values.JxW(q_index);
                                }
                                copy_data.cell_rhs[grid_num](i) +=
                                        (pde_data.bc_v_2.mvalue(grid_locations[grid_num], mp) +
                                         k_3 * (*sol_w)(grid_num)) *
                                        sd.fe_face_values.shape_value(i, q_index) *
                                        det_jac * sd.fe_face_values.JxW(q_index);
                                break;
                            case TOP_NEUMANN:
                                copy_data.cell_rhs[grid_num](i) +=
                                        pde_data.bc_v_3.mvalue(grid_locations[grid_num], mp) *
                                        sd.fe_face_values.shape_value(i, q_index) *
                                        det_jac * sd.fe_face_values.JxW(q_index);
                                break;
                            case BOTTOM_NEUMANN:
                                copy_data.cell_rhs[grid_num](i) +=
                                        pde_data.bc_v_4.mvalue(grid_locations[grid_num], mp) *
                                        sd.fe_face_values.shape_value(i, q_index) *
                                        det_jac * sd.fe_face_values.JxW(q_index);
                                break;
                            default: Assert(false, ExcMessage("Part of the boundary is not initialized correctly"))
                        }
                    }
                }
            }
        }
    }
    cell->get_dof_indices(copy_data.local_dof_indices);
}

template<int dim>
void MicroSolver<dim>::copy_local_to_global(const MicroSolver::AssemblyCopyData &copy_data) {
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        constraints.distribute_local_to_global(copy_data.cell_matrices[grid_num], copy_data.cell_rhs[grid_num],
                                               copy_data.local_dof_indices,
                                               system_matrices[grid_num], righthandsides[grid_num]);
    }
}


template<int dim>
void MicroSolver<dim>::assemble_and_solve_all() {
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        system_matrices[grid_num].reinit(sparsity_pattern);
        righthandsides[grid_num] = 0;
    }
    WorkStream::run(dof_handler.begin_active(), dof_handler.end(), *this, &MicroSolver::local_assemble_system,
                    &MicroSolver::copy_local_to_global, AssemblyScratchData(fe), AssemblyCopyData(num_grids));
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        solve(grid_num);
    }
}


template<int dim>
void MicroSolver<dim>::solve(int grid_num) {
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrices.at(grid_num), solutions.at(grid_num), righthandsides.at(grid_num),
                 PreconditionIdentity());
    printf("\t %d CG iterations to convergence (micro)\n", solver_control.last_step());
}


template<int dim>
void MicroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    Vector<double> macro_domain_l2_error(num_grids);
    Vector<double> macro_domain_h1_error(num_grids);
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        pde_data.solution_v.set_macro_point(grid_locations[grid_num]);
        const unsigned int n_active = triangulation.n_active_cells();
        Vector<double> difference_per_cell(n_active);
        VectorTools::integrate_difference(dof_handler, solutions[grid_num], pde_data.solution_v, difference_per_cell,
                                          QGauss<dim>(fem_q_deg),
                                          VectorTools::L2_norm);
        double micro_l2_error = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                  VectorTools::L2_norm);
        VectorTools::integrate_difference(dof_handler, solutions[grid_num], pde_data.solution_v, difference_per_cell,
                                          QGauss<dim>(fem_q_deg),
                                          VectorTools::H1_seminorm);
        double micro_h1_error = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                  VectorTools::H1_seminorm);
        macro_domain_l2_error(grid_num) = micro_l2_error;
        macro_domain_h1_error(grid_num) = micro_h1_error;
    }
    Vector<double> macro_integral(num_grids);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_l2_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(fem_q_deg), VectorTools::L2_norm);
    l2_error = VectorTools::compute_global_error(macro_dof_handler->get_triangulation(), macro_integral,
                                                 VectorTools::L2_norm);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_h1_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(fem_q_deg), VectorTools::L2_norm);
    h1_error = VectorTools::compute_global_error(macro_dof_handler->get_triangulation(), macro_integral,
                                                 VectorTools::L2_norm); //Not sure about this norm, although output is consistent
}

template<int dim>
void MicroSolver<dim>::set_exact_solution() {
    std::cout << "Exact micro-solution set" << std::endl;
    MappingQ1<dim> mapping;
    AffineConstraints<double> constraints; // Object that is necessary to use `Vectortools::project`
    constraints.close();
    for (unsigned int grid_num = 0; grid_num < num_grids; grid_num++) {
        pde_data.solution_v.set_macro_point(grid_locations[grid_num]);
        VectorTools::project(mapping, dof_handler, constraints, QGauss<dim>(fem_q_deg), pde_data.solution_v,
                             solutions[grid_num]);
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

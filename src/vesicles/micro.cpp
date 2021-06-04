/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "micro.h"

using namespace dealii;

template<int dim>
MicroSolver<dim>::MicroSolver(ParabolicMicroData<dim> &micro_data, unsigned int h_inv):  dof_handler(triangulation),
                                                                                         fe(1),
                                                                                         h_inv(h_inv),
                                                                                         num_grids(1),
                                                                                         macro_solution(nullptr),
                                                                                         old_macro_solution(nullptr),
                                                                                         macro_dof_handler(nullptr),
                                                                                         pde_data(micro_data),
                                                                                         integration_order(
                                                                                                 fe.degree + 1) {
    printf("Solving micro problem in %d space dimensions\n", dim);
    init_macro_field.reinit(num_grids);
    init_macro_field = 1;
    time = 0;
}

template<int dim>
void MicroSolver<dim>::setup() {
    make_grid();
    setup_system();
    setup_scatter();
}

template<int dim>
void MicroSolver<dim>::make_grid() {
    GridGenerator::hyper_ball(triangulation);
    triangulation.refine_global(3);
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double y1_abs = std::fabs(cell->face(face_number)->center()(1));
                if (y1_abs < 0) {
                    cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
                } // Else: Robin by default.
                // Note that this arrangement is implicitly coupled with ./prepare_two_scale.py
            }
        }
    }
    printf("%d active micro cells\n", triangulation.n_active_cells());
}

template<int dim>
void MicroSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d micro DoFs\n", dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(integration_order), mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
}

template<int dim>
void MicroSolver<dim>::setup_scatter() {
    solutions.clear();
    old_solutions.clear();
    solutions_w.clear();
    old_solutions_w.clear();
    righthandsides.clear();
    system_matrices.clear();
    compute_macroscopic_contribution();
    constraints.close();
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int k = 0; k < num_grids; k++) {
        pde_data.init_v.set_macro_point(grid_locations[k]);
        pde_data.init_w.set_macro_point(grid_locations[k]);
        Vector<double> solution_v(n_dofs), solution_w(n_dofs);
        VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.init_v, solution_v);
        solutions.push_back(solution_v);
        solutions_w.push_back(solution_w);
        Vector<double> old_solution_v(n_dofs), old_solution_w(n_dofs);
        VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.init_w, solution_w);
        old_solutions.push_back(old_solution_v);
        old_solutions_w.push_back(old_solution_w);

        Vector<double> rhs(n_dofs);
        righthandsides.push_back(rhs);
        SparseMatrix<double> system_matrix;
        system_matrices.push_back(system_matrix);
        intermediate_vector.reinit(dof_handler.n_dofs());
    }
}


template<int dim>
void MicroSolver<dim>::set_macro_solutions(Vector<double> *_solution, Vector<double> *_old_solution,
                                           DoFHandler<dim> *_dof_handler) {
    this->macro_solution = _solution;
    this->old_macro_solution = _old_solution;
    this->macro_dof_handler = _dof_handler;
}

template<int dim>
void MicroSolver<dim>::compute_macroscopic_contribution() {
    // Nothing needs to happen in this case because it is local in x
}


template<int dim>
void MicroSolver<dim>::assemble_system() {
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
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    Vector<double> local_w(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    // Todo: this is a break from Crank Nicholson implementation
    // Fix by: Storing a copy of the previously projected RHS in a persistent vector.
    const double alpha = pde_data.params.get_double("alpha");
    const double beta = pde_data.params.get_double("beta");
    const double H = pde_data.params.get_double("H");
    const double R = pde_data.params.get_double("dt");
    const double D = pde_data.params.get_double("D");
    const double k1 = pde_data.params.get_double("k1");
    const double k2 = pde_data.params.get_double("k2");
    const double k3 = pde_data.params.get_double("k3");
    const double k4 = pde_data.params.get_double("k4");
    const double euler = pde_data.params.get_double("euler");
    const double dt = pde_data.params.get_double("dt");
    for (unsigned int k = 0; k < num_grids; k++) {
        righthandsides.at(k) = 0;
        solutions.at(k) = 0;
        system_matrices.at(k).reinit(sparsity_pattern);
        mass_matrix.vmult(righthandsides.at(k), old_solutions.at(k));
        laplace_matrix.vmult(intermediate_vector, old_solutions.at(k));
        righthandsides.at(k).add(-dt * D * (1 - euler), intermediate_vector); // scalar factor, matrix

        // For now, implicit euler only
        Vector<double> rhs_func(dof_handler.n_dofs());
        pde_data.rhs.set_macro_point(grid_locations[k]);
        VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(fe.degree + 1), pde_data.rhs, rhs_func);
        righthandsides.at(k).add(dt * (euler), rhs_func);
        // Missing: rhs_func * dt * (1- euler) from the previous time step
        system_matrices.at(k).add(1 + k1 * dt, mass_matrix);
        system_matrices.at(k).add(dt * D * euler, laplace_matrix);
    }
    std::vector<double> old_sol_v;
    std::vector<double> old_sol_w;
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        for (unsigned int k = 0; k < num_grids; k++) {
            cell_rhs = 0;
            local_w = 0;
            fe_values.get_function_values(old_solutions[k], old_sol_v);
            fe_values.get_function_values(old_solutions_w[k], old_sol_w);
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
                for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
                    const double w_sq = old_sol_w[q_index] * old_sol_w[q_index]; // w^2
                    const double g1 = k2 * w_sq;
                    const double g2 = -k4 * old_sol_w[q_index] + 2 * k1 * old_sol_v[q_index] - k2 * w_sq -
                                      k3 * old_sol_v[q_index] * old_sol_w[q_index];
                    cell_rhs(i) += fe_values.shape_value(i, q_index) * g1 * euler * dt;
                    local_w(i) = fe_values.shape_value(i, q_index) * g2 * euler * dt;
                }
                righthandsides[k](local_dof_indices[i]) += cell_rhs(i);
                //soft forward euler
                solutions_w[k](local_dof_indices[i]) = old_solutions[k](local_dof_indices[i]) + local_w(i);
            }
        }

        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary() and cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                fe_face_values.reinit(cell, face_number);
                cell_matrix = 0;
                cell->get_dof_indices(local_dof_indices);
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                    for (unsigned int j = 0; j < dofs_per_cell; j++) {
                        for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                            cell_matrix(i, j) +=
                                    alpha * H * dt * fe_face_values.shape_value(i, q_index) *
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
                fe_face_values.reinit(cell, face_number); // pretty sure this can be moved inside the if body
                if (cell->face(face_number)->at_boundary()) {
                    cell->get_dof_indices(local_dof_indices);
                    cell_rhs = 0;
                    std::vector<double> old_interpolated_solution(n_q_face_points);
                    fe_face_values.get_function_values(old_solutions.at(k), old_interpolated_solution);
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                            const double dtxvixJxW =
                                    dt * fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index);
                            const double macro_part =
                                    euler * (*macro_solution)(k) + (1 - euler) * (*old_macro_solution)(k);
                            cell_rhs(i) += alpha * beta * macro_part * dtxvixJxW;
                        }
                    }
                }
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                    righthandsides.at(k)(local_dof_indices[i]) += cell_rhs(i);
                }
            }
        }
    }
}


template<int dim>
void MicroSolver<dim>::solve_time_step() {
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    for (unsigned int k = 0; k < num_grids; k++) {
        solver.solve(system_matrices.at(k), solutions.at(k), righthandsides.at(k), PreconditionIdentity());
    }
    printf("\t %d CG iterations to convergence (micro)\n", solver_control.last_step());
}

template<int dim>
void MicroSolver<dim>::iterate() {
    assemble_system();
    solve_time_step();
    old_solutions = solutions;
}

template<int dim>
Point<dim> MicroSolver<dim>::get_micro_grid_size(const std::vector<Point<dim>> &locations) const {
    // Only implemented for rectangular non-refined grids!
    // For more complex meshes we will have to think of something smart
    double side_length = std::pow(locations.size(), 1.0 / dim);
    Assert(std::fmod(side_length, 1) < 1E-5, ExcNotMultiple(side_length, 1))
    Point<dim> size;
    for (unsigned int i = 0; i < dim; i++) {
        size(i) = 1. / (int(side_length) - 1);
    }
    return size;
}


template<int dim>
void MicroSolver<dim>::write_solutions_to_file(const std::vector<Vector<double>> &sols,
                                               const DoFHandler<dim> &corr_dof_handler) {
    const std::string filename = "results/test_rho_" + std::to_string(h_inv) + ".txt";
    std::ofstream output(filename);
    output << h_inv << std::endl;
    std::vector<Point<dim>> locations;
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
    for (unsigned long i = 0; i < locations.size(); i++) {
        for (const Vector<double> &sol: sols) {
            AssertDimension(locations.size(), sol.size())
            output << sol(i) << " ";
        }
        for (unsigned int j = 0; j < dim; j++) {
            output << " " << locations[i](j);
        }
        output << std::endl;
    }
    output.close();
}

template<int dim>
void MicroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    Vector<double> macro_domain_l2_error(num_grids);
    Vector<double> macro_domain_h1_error(num_grids);
    Vector<double> macro_domain_mass(num_grids);
    for (unsigned int k = 0; k < num_grids; k++) {
        pde_data.solution.set_macro_point(grid_locations.at(k));
        const unsigned int n_active = triangulation.n_active_cells();
        Vector<double> difference_per_cell(n_active);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), pde_data.solution, difference_per_cell,
                                          QGauss<dim>(3),
                                          VectorTools::L2_norm);
        macro_domain_l2_error(k) = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                     VectorTools::L2_norm);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), pde_data.solution, difference_per_cell,
                                          QGauss<dim>(3),
                                          VectorTools::H1_norm);
        macro_domain_h1_error(k) = VectorTools::compute_global_error(triangulation, difference_per_cell,
                                                                     VectorTools::H1_seminorm);
        VectorTools::integrate_difference(dof_handler, solutions.at(k), Functions::ZeroFunction<dim>(),
                                          difference_per_cell,
                                          QGauss<dim>(3),
                                          VectorTools::L1_norm);

        macro_domain_mass(k) = difference_per_cell.l1_norm();
    }
    Vector<double> macro_integral(num_grids);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_l2_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(3), VectorTools::L2_norm);
    l2_error = macro_integral.l2_norm();
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_h1_error, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(3), VectorTools::L2_norm);
    h1_error = macro_integral.l2_norm();
    printf("Micro error: %.3e\n", l2_error);
    VectorTools::integrate_difference(*macro_dof_handler, macro_domain_mass, Functions::ZeroFunction<dim>(),
                                      macro_integral, QGauss<dim>(3), VectorTools::L2_norm);
    printf("Micro mass:  %.3e\n", macro_integral.l1_norm());
}

template<int dim>
void MicroSolver<dim>::set_grid_locations(const std::vector<Point<dim>> &locations) {
    grid_locations = locations;
    num_grids = locations.size();
}


template<int dim>
void MicroSolver<dim>::set_exact_solution() {
    std::cout << "Exact rho solutions set" << std::endl;
    for (unsigned int k = 0; k < num_grids; k++) {
        pde_data.solution.set_macro_point(grid_locations[k]);
        VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.solution, solutions[k]);
    }
}

template<int dim>
unsigned int MicroSolver<dim>::get_num_grids() {
    return num_grids;
}
//
//template<int dim>
//void MicroSolver<dim>::read_solutions_from_file(const std::string &filename, std::vector<Vector<double>> &sols,
//                                              DoFHandler<dim> &corr_dof_handler) {
//    std::ifstream input(filename);
//    int refine_lvl;
//    std::string line;
//    std::getline(input, line);
//    std::istringstream iss(line);
//    iss >> refine_lvl;
//    Triangulation<dim> tria;
//    DoFHandler<dim> new_dof_handler(tria);
//    GridGenerator::hyper_cube(tria, -1, 1);
//    tria.refine_global(refine_lvl);
//    new_dof_handler.distribute_dofs(fe);
//    std::vector<Point<dim>> locations(new_dof_handler.n_dofs());
//    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
//    std::vector<Point<dim>> check_locations(locations.size());
//    sols.resize(num_grids);
//    for (unsigned int k = 0; k < num_grids; k++) {
//        sols.at(k) = Vector<double>(locations.size());
//    }
//    for (unsigned long i = 0; i < locations.size(); i++) {
//        std::getline(input, line);
//        std::istringstream iss1(line);
//        for (unsigned int k = 0; k < num_grids; k++) {
//            iss1 >> sols.at(k)(i);
//        }
//        Point<dim> point;
//        for (unsigned int j = 0; j < dim; j++) {
//            iss1 >> point(j);
//        }
//        check_locations.at(i) = point;
//    }
//    std::getline(input, line);
//    Assert(line.empty(), ExcInternalError("Too many locations in file"))
//    input.close();
//    // Check if the points match
//    bool is_correct = true;
//    double eps = 1E-4;
//    for (unsigned long i = 0; i < check_locations.size(); i++) {
//        for (unsigned int j = 0; j < dim; j++) {
//            is_correct &= std::fabs(check_locations.at(i)[j] - locations.at(i)[j]) < eps;
//        }
//    }
//}


// Explicit instantiation

template
class MicroSolver<1>;

template
class MicroSolver<2>;

template
class MicroSolver<3>;

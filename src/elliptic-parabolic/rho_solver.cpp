/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "rho_solver.h"

using namespace dealii;

template<int dim>
double MicroInitCondition<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = macro_field[macro_cell_index];
    double pi = 3.141592;
    for (int i = 0; i < dim; i++) {
        val *= 1 + std::cos(pi * p[i]);
    }
    return val;
}

template<int dim>
void MicroInitCondition<dim>::set_macro_field(const Vector<double> &field) {
    this->macro_field = field;
}

template<int dim>
void MicroInitCondition<dim>::set_macro_cell_index(unsigned int index) {
    macro_cell_index = index;
}


template<int dim>
RhoSolver<dim>::RhoSolver():  dof_handler(triangulation), fe(1), macro_solution(nullptr),
                              old_macro_solution(nullptr), macro_dof_handler(nullptr),
                              diffusion_coefficient(1.),
                              R(2.),
                              kappa(1.),
                              p_F(4.),
                              theta(1),
                              integration_order(2) {
    printf("Solving micro problem in %d space dimensions\n",dim);
    refine_level = 1;
    num_grids = 1;
    integration_order = fe.degree + 1;
    init_macro_field.reinit(num_grids);
    init_macro_field = 1;
}

template<int dim>
void RhoSolver<dim>::setup() {
    make_grid();
    setup_system();
    setup_scatter();
}

template<int dim>
void RhoSolver<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);
    // If we ever use refinement, we have to remark every time we refine the grid.
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary() and
                std::fabs(cell->face(face_number)->center()(0) < 0)) { // Todo: Play with this
                cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
            } // Else: Robin by default.
        }
    }
    printf("%d active micro cells\n",triangulation.n_active_cells());
}

template<int dim>
void RhoSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d micro DoFs\n",dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(integration_order), mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
}

template<int dim>
void RhoSolver<dim>::set_refine_level(const int &refinement_level) {
    this->refine_level = refinement_level;
}

template<int dim>
void RhoSolver<dim>::set_initial_condition(const Vector<double> &initial_condition) {
    AssertDimension(initial_condition.size(), num_grids)
    init_macro_field = initial_condition;
}

template<int dim>
void RhoSolver<dim>::setup_scatter() {
    solutions.clear();
    old_solutions.clear();
    righthandsides.clear();
    system_matrices.clear();
    compute_macroscopic_contribution();
    MicroInitCondition<dim> mic;
    mic.set_macro_field(init_macro_field);
    unsigned int n_dofs = dof_handler.n_dofs();
    for (unsigned int i = 0; i < num_grids; i++) {
        mic.set_macro_cell_index(i);
        Vector<double> solution(n_dofs);

        VectorTools::interpolate(dof_handler, mic, solution);
        solutions.push_back(solution);
        Vector<double> old_solution(n_dofs);
        VectorTools::interpolate(dof_handler, mic, old_solution);
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
    // Nothing needs to happen in this simple case because it is local in x
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
        righthandsides.at(k).add(-dt * (1 - theta), intermediate_vector); // scalar factor, matrix

        system_matrices.at(k).copy_from(mass_matrix);
        system_matrices.at(k).add(dt * theta * diffusion_coefficient, laplace_matrix);
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
                                    kappa * dt * theta * R * fe_face_values.shape_value(i, q_index) *
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
                if (cell->face(face_number)->at_boundary() and
                    cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                    fe_face_values.reinit(cell, face_number);
                    cell->get_dof_indices(local_dof_indices);
                    cell_rhs = 0;
                    std::vector<double> old_interpolated_solution(n_q_face_points);
                    fe_face_values.get_function_values(old_solutions.at(k), old_interpolated_solution);
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                            cell_rhs(i) += (fe_face_values.shape_value(i, q_index)) * dt * kappa *
                                           (theta * (*macro_solution)(k) +
                                            (1 - theta) * (*old_macro_solution)(k) + p_F -
                                            R * (1 - theta) * old_interpolated_solution[q_index]) *
                                           fe_face_values.JxW(q_index);
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
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    for (unsigned int k = 0; k < num_grids; k++) {
        solver.solve(system_matrices.at(k), solutions.at(k), righthandsides.at(k), PreconditionIdentity());
    }
    printf("\t %d CG iterations to convergence (micro)\n",solver_control.last_step());
    compute_residual();
    old_solutions = solutions;
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
void RhoSolver<dim>::patch_micro_solutions(const std::vector<Point<dim>> &locations) const {
    std::ofstream output("results/patched-micro-solution.gpl");
    Point<dim> micro_size = 2 * get_micro_grid_size(locations);
    Point<dim> down_left;
    Point<dim> up_right;
    for (unsigned int i = 0; i < dim; i++) {
        down_left(i) = -0.5 * micro_size(i);
        up_right(i) = 0.5 * micro_size(i);
    }
    for (unsigned int i = 0; i < locations.size(); i++) {
        Triangulation<dim> mapped_tria;
        GridGenerator::hyper_rectangle(mapped_tria, down_left + locations.at(i), up_right + locations.at(i));
        mapped_tria.refine_global(refine_level);
        DoFHandler<dim> mapped_dof_handler(mapped_tria);
        mapped_dof_handler.distribute_dofs(fe);
        DataOut<dim> data_out;

        data_out.attach_dof_handler(mapped_dof_handler);
        data_out.add_data_vector(solutions.at(i), "solution");
        data_out.build_patches();
        data_out.write_gnuplot(output);
    }
    output.close();
}

template<int dim>
void RhoSolver<dim>::set_num_grids(unsigned int _num_grids) {
    this->num_grids = _num_grids;
}

template<int dim>
Point<dim> RhoSolver<dim>::get_micro_grid_size(const std::vector<Point<dim>> &locations) const {
    // Only implemented for rectangular non-refined grids!
    // For more complex meshes we will have to think of something smart
    double side_length = std::pow(locations.size(), 1.0 / dim);
    double eps = 1E-5;
    Assert(std::fmod(side_length, 1) < eps, ExcNotMultiple(side_length, 1))
    Point<dim> size;
    for (unsigned int i = 0; i < dim; i++) {
        size(i) = 1. / (int(side_length) - 1);
    }
    return size;
}


template<int dim>
void RhoSolver<dim>::write_solutions_to_file(const std::vector<Vector<double>> &sols,
                                             const DoFHandler<dim> &corr_dof_handler) {
    const std::string filename="results/test_rho_"+std::to_string(refine_level)+".txt";
    std::ofstream output(filename);
    output << refine_level << std::endl;
    std::vector<Point<dim>> locations;
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
    for (unsigned int i = 0; i < locations.size(); i++) {
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
void RhoSolver<dim>::read_solutions_from_file(const std::string &filename, std::vector<Vector<double>> &sols,
                                              DoFHandler<dim> &corr_dof_handler) {
    std::ifstream input(filename);
    int refine_lvl;
    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> refine_lvl;
    Triangulation<dim> tria;
    DoFHandler<dim> new_dof_handler(tria);
    GridGenerator::hyper_cube(tria, -1, 1);
    tria.refine_global(refine_lvl);
    new_dof_handler.distribute_dofs(fe);
    std::vector<Point<dim>> locations(new_dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
    std::vector<Point<dim>> check_locations(locations.size());
    sols.resize(num_grids);
    for (unsigned int k = 0; k < num_grids; k++) {
        sols.at(k) = Vector<double>(locations.size());
    }
    for (unsigned int i = 0; i < locations.size(); i++) {
        std::getline(input, line);
        std::istringstream iss1(line);
        for (unsigned int k = 0; k < num_grids; k++) {
            iss1 >> sols.at(k)(i);
        }
        Point<dim> point;
        for (unsigned int j = 0; j < dim; j++) {
            iss1 >> point(j);
        }
        check_locations.at(i) = point;
    }
    std::getline(input, line);
    Assert(line.empty(), ExcInternalError("Too many locations in file"))
    input.close();
    // Check if the points match
    bool is_correct = true;
    double eps = 1E-4;
    for (unsigned int i = 0; i < check_locations.size(); i++) {
        for (unsigned int j = 0; j < dim; j++) {
            is_correct &= std::fabs(check_locations.at(i)[j] - locations.at(i)[j]) < eps;
        }
    }
}


// Explicit instantiation

template
class RhoSolver<1>;

template
class RhoSolver<2>;

template
class RhoSolver<3>;

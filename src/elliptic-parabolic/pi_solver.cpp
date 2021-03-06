/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "pi_solver.h"

using namespace dealii;

template<int dim>
PiSolver<dim>::PiSolver(MacroData<dim> &macro_data, unsigned int h_inv):dof_handler(triangulation), fe(1),
                                                                        micro_dof_handler(nullptr),
                                                                        micro_solutions(nullptr),
                                                                        pde_data(macro_data),
                                                                        integration_order(fe.degree + 1),
                                                                        h_inv(h_inv) {
    residual = 1;
    printf("Solving macro problem in %d space dimensions\n", dim);
    if (pde_data.params.get_bool("nonlinear")) {
        printf("Running nonlinear right hand side in macroscopic part\n");
    } else {
        printf("Running linear right hand side in macroscopic part\n");
    }
}


template<int dim>
void PiSolver<dim>::setup() {
    make_grid();
    setup_system();
}

template<int dim>
void PiSolver<dim>::make_grid() {
    GridGenerator::subdivided_hyper_cube(triangulation, h_inv, -1, 1);
    printf("%d active macro cells\n", triangulation.n_active_cells());

}

template<int dim>
void PiSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
    printf("%d macro DoFs\n", dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    macro_contribution.reinit(dof_handler.n_dofs());
    old_solution = 5; // Todo: How to choose the initial value?
    laplace_matrix.reinit(sparsity_pattern);
    MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
    constraints.close();
}

template<int dim>
void
PiSolver<dim>::get_pi_contribution_rhs(const Vector<double> &pi, Vector<double> &out_vector, bool nonlinear) const {
    AssertDimension(pi.size(), out_vector.size())
    for (unsigned int i = 0; i < pi.size(); i++) {
        if (nonlinear) {
            const double abs_pi = std::fabs(pi(i));
            out_vector(i) = pde_data.params.get_double("theta") * std::fmin(abs_pi, std::sqrt(abs_pi));
        } else {
            out_vector(i) = pde_data.params.get_double("theta") * pi(i);
        }
    }
}

template<int dim>
void PiSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(integration_order);

    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    system_matrix = 0;
    system_rhs = 0;
    // Todo: This can be moved out, it only happens once.
    system_matrix.add(pde_data.params.get_double("A"), laplace_matrix);
    Vector<double> micro_contribution(dof_handler.n_dofs());
    const bool is_nonlinear = pde_data.params.get_bool("nonlinear");
    get_microscopic_contribution(micro_contribution, is_nonlinear);
    get_pi_contribution_rhs(old_solution, macro_contribution, is_nonlinear);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        std::vector<double> rho_rhs_points(n_q_points);
        std::vector<double> pi_rhs_points(n_q_points);
        fe_values.get_function_values(macro_contribution, pi_rhs_points);
        fe_values.get_function_values(micro_contribution, rho_rhs_points);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double functional = rho_rhs_points[q_index] * pi_rhs_points[q_index] +
                                          pde_data.rhs.value(fe_values.quadrature_point(q_index));
                cell_rhs(i) += fe_values.shape_value(i, q_index) * functional * fe_values.JxW(q_index);
            }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
//    std::cout << "Macro rhs: " << system_rhs << std::endl;
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, pde_data.bc, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template<int dim>
void PiSolver<dim>::interpolate_function(const Vector<double> &func, Vector<double> &interp_func) {
    interp_func = 0;
    if (func.size() != dof_handler.n_dofs()) {
        throw std::invalid_argument(
                "func lengths:" + std::to_string(dof_handler.n_dofs()) + "/" + std::to_string(func.size()));
    } else if (interp_func.size() != triangulation.n_active_cells()) {
        throw std::invalid_argument("func lengths:" + std::to_string(interp_func.size()) + "/" +
                                    std::to_string(triangulation.n_active_cells()));
    }
    FEValues<dim> fe_value(fe, QMidpoint<dim>(), update_values | update_quadrature_points | update_JxW_values);
    std::vector<double> mid_point_value(1);
    for (const auto &cell:dof_handler.active_cell_iterators()) {
        fe_value.reinit(cell);
        fe_value.get_function_values(func, mid_point_value);
        interp_func[cell->active_cell_index()] = mid_point_value[0];
    }
}

template<int dim>
void PiSolver<dim>::solve() {
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    printf("\t %d CG iterations to convergence (macro)\n", solver_control.last_step());
}

template<int dim>
void PiSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    Vector<double> mass_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, solution, pde_data.solution, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::L2_norm);
    l2_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler, solution, pde_data.solution, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::H1_seminorm);
    h1_error = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::H1_seminorm);
    VectorTools::integrate_difference(dof_handler, solution, Functions::ZeroFunction<dim>(), mass_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L1_norm);

    const double l2_mass = mass_per_cell.l1_norm();
    printf("Macro error: %.3e\n", l2_error);
    printf("Macro mass:  %.3e\n", l2_mass);
}

template<int dim>
void PiSolver<dim>::set_micro_solutions(std::vector<Vector<double>> *_solutions, DoFHandler<dim> *_dof_handler) {
    this->micro_solutions = _solutions;
    this->micro_dof_handler = _dof_handler;

}

template<int dim>
double PiSolver<dim>::get_micro_mass(unsigned int micro_index) const {
    // computed as: f(x) = \int_Y \rho(x,y) dy
    double integral = 0;
    QGauss<dim> quadrature_formula(integration_order);
    FEValues<dim> fe_values(micro_dof_handler->get_fe(), quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        std::vector<double> interp_solution(n_q_points);
        fe_values.get_function_values(micro_solutions->at(micro_index), interp_solution);
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            integral += interp_solution[q_index] * fe_values.JxW(q_index);
        }
    }
    return integral;
}

template<int dim>
double PiSolver<dim>::get_micro_flux(unsigned int micro_index) const {
    const int ROBIN_BOUNDARY = 0; //Not linked, a bit ugly
    // Computed as: f(x) = \int_\Gamma_R \nabla_y \rho(x,y) \cdot n_y d_\sigma_y
    double integral = 0;
    QGauss<dim - 1> quadrature_formula(integration_order);
    FEFaceValues<dim> fe_face_values(micro_dof_handler->get_fe(),
                                     quadrature_formula, // Todo: Check flags (update_values), also in other places.
                                     update_values | update_quadrature_points | update_JxW_values |
                                     update_normal_vectors | update_gradients);
    const unsigned int n_q_face_points = quadrature_formula.size();
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary() && cell->face(face_number)->boundary_id() == ROBIN_BOUNDARY) {
                fe_face_values.reinit(cell, face_number);
                std::vector<Tensor<1, dim>> solution_gradient(n_q_face_points);
                fe_face_values.get_function_gradients(micro_solutions->at(micro_index), solution_gradient);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    double neumann = solution_gradient[q_index] * fe_face_values.normal_vector(q_index);
                    integral += neumann * fe_face_values.JxW(q_index);
                }
            }
        }
    }
    return integral;
}

template<int dim>
void PiSolver<dim>::get_dof_locations(std::vector<Point<dim>> &locations) {
    MappingQ1<dim> mapping;
    locations.clear();
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, locations);

}

template<int dim>
void PiSolver<dim>::get_microscopic_contribution(Vector<double> &micro_contribution, bool nonlinear) {
    AssertDimension(micro_contribution.size(), dof_handler.n_dofs())
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        if (nonlinear) {
            micro_contribution[i] = std::fmin(std::fabs(get_micro_mass(i)), 1);
        } else {
            micro_contribution[i] = get_micro_mass(i);
        }
    }
//    std::cout << "Micro Mass " << micro_contribution << std::endl;
}

template<int dim>
void PiSolver<dim>::iterate() {
    assemble_system();
    solve();
//    std::cout << "Macro: " << solution << std::endl;
    if (count == 0) {
        double diff = 1;
        while (diff > 1E-9) {
            printf("Difference: %.3e, reiterating first step\n", diff);
            get_solution_difference(diff);
            old_solution = solution;
            assemble_system();
            solve();
        }
    }
    count++;
    old_solution = solution;
}

template<int dim>
void PiSolver<dim>::get_solution_difference(double &diff) {
    Vector<double> tmp_diff(solution);
    tmp_diff -= old_solution;
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, tmp_diff, Functions::ZeroFunction<dim>(), difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    diff = difference_per_cell.l2_norm();
}


template<int dim>
void PiSolver<dim>::set_exact_solution() {
    std::cout << "Exact pi solution set" << std::endl;
    VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.solution, solution);
}


template<int dim>
void PiSolver<dim>::write_solution_to_file(const Vector<double> &sol,
                                           const DoFHandler<dim> &corr_dof_handler) {
    const std::string filename = "results/test_pi_" + std::to_string(h_inv) + ".txt";

    std::ofstream output(filename);
    output << h_inv << std::endl;
    std::vector<Point<dim>> locations;
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
    AssertDimension(locations.size(), sol.size())
    for (unsigned int i = 0; i < sol.size(); i++) {
        output << sol(i);
        for (unsigned int j = 0; j < dim; j++) {
            output << " " << locations[i](j);
        }
        output << std::endl;
    }
    output.close();

}

template<int dim>
void PiSolver<dim>::read_solution_from_file(const std::string &filename, Vector<double> &sol,
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
    sol.reinit(locations.size());
    std::vector<Point<dim>> check_locations(sol.size());
    for (unsigned int i = 0; i < sol.size(); i++) {
        std::getline(input, line);
        std::istringstream iss1(line);
        iss1 >> sol(i);
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
class PiSolver<1>;

template
class PiSolver<2>;

template
class PiSolver<3>;
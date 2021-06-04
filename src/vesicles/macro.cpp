/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "macro.h"

using namespace dealii;

template<int dim>
MacroSolver<dim>::MacroSolver(ParabolicMacroData <dim> &macro_data, unsigned int h_inv):dof_handler(triangulation),
                                                                                        fe(1),
                                                                                        micro_dof_handler(nullptr),
                                                                                        micro_solutions(nullptr),
                                                                                        pde_data(macro_data),
                                                                                        integration_order(
                                                                                                fe.degree + 1),
                                                                                        h_inv(h_inv) {
    residual = 1;
    printf("Solving macro problem in %d space dimensions\n", dim);
}


template<int dim>
void MacroSolver<dim>::setup() {
    make_grid();
    setup_system();
}

template<int dim>
void MacroSolver<dim>::make_grid() {
    const double length = 5;
    const double height = 1;
    Point <dim> p1(0, 0);
    Point <dim> p2(length, height);
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    triangulation.refine_global(2);
    const double EPS = 1E-4;
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double x_abs = std::fabs(cell->face(face_number)->center()(0));
                if (x_abs < EPS or x_abs > length - EPS) {
                    cell->face(face_number)->set_boundary_id(DIRICHLET_BOUNDARY);
                } else {
                    cell->face(face_number)->set_boundary_id(NEUMANN_BOUNDARY);
                }
                // Note that this arrangement is implicitly coupled with ./prepare_two_scale.py
            }
        }
    } // no need to lable
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
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    constraints.close();
//    VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.init_u, old_solution);
    laplace_matrix.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
    MatrixTools::create_mass_matrix(dof_handler, QGauss<dim>(integration_order), mass_matrix);
}

template<int dim>
void
MacroSolver<dim>::get_pi_contribution_rhs(const Vector<double> &pi, Vector<double> &out_vector, bool nonlinear) const {
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
void MacroSolver<dim>::assemble_system() {
    QGauss <dim> quadrature_formula(integration_order);

    FEValues <dim> fe_values(fe, quadrature_formula,
                             update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector <types::global_dof_index> local_dof_indices(dofs_per_cell);
    Point <dim> _point;
    const double D = pde_data.diffusion.value(_point);
    const double dt = pde_data.params.get_double("dt");
    const double euler = pde_data.params.get_double("euler");
    Vector<double> aux_vector;
    aux_vector.reinit(dof_handler.n_dofs());
    system_matrix = 0;
    system_rhs = 0; // superfluous I think
    mass_matrix.vmult(system_rhs, old_solution);
    laplace_matrix.vmult(aux_vector, old_solution);
    system_rhs.add(-dt * D * (1 - euler), aux_vector);

    system_matrix.copy_from(mass_matrix);
    system_matrix.add(dt * D * euler, laplace_matrix);
    Vector<double> micro_contribution(dof_handler.n_dofs());
    get_microscopic_contribution(micro_contribution);
    std::vector<double> rho_rhs_points(n_q_points);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.get_function_values(micro_contribution, rho_rhs_points);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double functional = rho_rhs_points[q_index];
                cell_rhs(i) += fe_values.shape_value(i, q_index) * functional * fe_values.JxW(q_index) * euler * dt;
            }
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
//    std::cout << "Macro rhs: " << system_rhs << std::endl;
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, DIRICHLET_BOUNDARY, pde_data.bc, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);
}

template<int dim>
void MacroSolver<dim>::interpolate_function(const Vector<double> &func, Vector<double> &interp_func) {
    interp_func = 0;
    if (func.size() != dof_handler.n_dofs()) {
        throw std::invalid_argument(
                "func lengths:" + std::to_string(dof_handler.n_dofs()) + "/" + std::to_string(func.size()));
    } else if (interp_func.size() != triangulation.n_active_cells()) {
        throw std::invalid_argument("func lengths:" + std::to_string(interp_func.size()) + "/" +
                                    std::to_string(triangulation.n_active_cells()));
    }
    FEValues <dim> fe_value(fe, QMidpoint<dim>(), update_values | update_quadrature_points | update_JxW_values);
    std::vector<double> mid_point_value(1);
    for (const auto &cell:dof_handler.active_cell_iterators()) {
        fe_value.reinit(cell);
        fe_value.get_function_values(func, mid_point_value);
        interp_func[cell->active_cell_index()] = mid_point_value[0];
    }
}

template<int dim>
void MacroSolver<dim>::solve() {
    solution = old_solution;
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    printf("\t %d CG iterations to convergence (macro)\n", solver_control.last_step());
}

template<int dim>
void MacroSolver<dim>::compute_error(double &l2_error, double &h1_error) {
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
void MacroSolver<dim>::compute_residual(double &l2_residual) {
    Vector<double> error(dof_handler.n_dofs());
    error += solution;
    error -= old_solution;
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, error, Functions::ZeroFunction<dim>(), difference_per_cell,
                                      QGauss<dim>(8), VectorTools::L2_norm);
    l2_residual = VectorTools::compute_global_error(triangulation, difference_per_cell, VectorTools::L2_norm);
}

template<int dim>
void MacroSolver<dim>::set_micro_solutions(std::vector <Vector<double>> *_solutions, DoFHandler <dim> *_dof_handler) {
    this->micro_solutions = _solutions;
    this->micro_dof_handler = _dof_handler;
}

template<int dim>
double MacroSolver<dim>::get_micro_mass(unsigned int micro_index) const {
    // computed as: f(x) = \int_Y \rho(x,y) dy
    double integral = 0;
    QGauss <dim> quadrature_formula(integration_order);
    FEValues <dim> fe_values(micro_dof_handler->get_fe(), quadrature_formula,
                             update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector <types::global_dof_index> local_dof_indices(dofs_per_cell);
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
double MacroSolver<dim>::get_micro_flux(unsigned int micro_index) const {
    double integral = 0;
    QGauss < dim - 1 > quadrature_formula(integration_order);
    FEFaceValues <dim> fe_face_values(micro_dof_handler->get_fe(),
                                      quadrature_formula, update_values | update_quadrature_points | update_JxW_values |
                                                          update_normal_vectors | update_gradients);
    const unsigned int n_q_face_points = quadrature_formula.size();
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                std::vector <Tensor<1, dim>> solution_gradient(n_q_face_points);
                fe_face_values.get_function_gradients(micro_solutions->at(micro_index), solution_gradient);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    double neumann = solution_gradient[q_index] * fe_face_values.normal_vector(q_index);
                    integral += -pde_data.params.get_double("d") * neumann * fe_face_values.JxW(q_index);
                }
            }
        }
    }
    return integral;
}

template<int dim>
void MacroSolver<dim>::get_dof_locations(std::vector <Point<dim>> &locations) {
    MappingQ1 <dim> mapping;
    locations.clear();
    locations.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, locations);

}

template<int dim>
void MacroSolver<dim>::get_microscopic_contribution(Vector<double> &micro_contribution) {
    AssertDimension(micro_contribution.size(), dof_handler.n_dofs())
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        micro_contribution[i] = get_micro_flux(i);
    }
//    std::cout << "Micro Mass " << micro_contribution << std::endl;
}

template<int dim>
void MacroSolver<dim>::iterate() {
    assemble_system();
    solve();
//    std::cout << "Macro: " << solution << std::endl;
    count++;
}

template<int dim>
void MacroSolver<dim>::get_solution_difference(double &diff) {
    Vector<double> tmp_diff(solution);
    tmp_diff -= old_solution;
    Vector<double> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler, tmp_diff, Functions::ZeroFunction<dim>(), difference_per_cell,
                                      QGauss<dim>(3),
                                      VectorTools::L2_norm);
    diff = difference_per_cell.l2_norm();
}


template<int dim>
void MacroSolver<dim>::set_exact_solution() {
    std::cout << "Exact pi solution set" << std::endl;
    VectorTools::project(dof_handler, constraints, QGauss<dim>(3), pde_data.solution, solution);
}


template<int dim>
void MacroSolver<dim>::write_solution_to_file(const Vector<double> &sol,
                                              const DoFHandler <dim> &corr_dof_handler) {
    const std::string filename = "results/test_pi_" + std::to_string(h_inv) + ".txt";

    std::ofstream output(filename);
    output << h_inv << std::endl;
    std::vector <Point<dim>> locations;
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
void MacroSolver<dim>::read_solution_from_file(const std::string &filename, Vector<double> &sol,
                                               DoFHandler <dim> &corr_dof_handler) {
    std::ifstream input(filename);
    int refine_lvl;
    std::string line;
    std::getline(input, line);
    std::istringstream iss(line);
    iss >> refine_lvl;
    Triangulation <dim> tria;
    DoFHandler <dim> new_dof_handler(tria);
    GridGenerator::hyper_cube(tria, -1, 1);
    tria.refine_global(refine_lvl);
    new_dof_handler.distribute_dofs(fe);
    std::vector <Point<dim>> locations(new_dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), corr_dof_handler, locations);
    sol.reinit(locations.size());
    std::vector <Point<dim>> check_locations(sol.size());
    for (unsigned int i = 0; i < sol.size(); i++) {
        std::getline(input, line);
        std::istringstream iss1(line);
        iss1 >> sol(i);
        Point <dim> point;
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
class MacroSolver<1>;

template
class MacroSolver<2>;

template
class MacroSolver<3>;
/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "pi_solver.h"

using namespace dealii;

template<int dim>
double MacroBoundary<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = 0;
    return val;
}

template<int dim>
Tensor<1, dim> MacroBoundary<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;
    return_val = 0;
    return return_val;
}


template<int dim>
PiSolver<dim>::PiSolver():dof_handler(triangulation), fe(1), micro_dof_handler(nullptr), micro_solutions(nullptr),
                          boundary(), integration_order(2), diffusion_coefficient(1), max_support(10) {
    refine_level = 1;
    residual = 1;
    integration_order = fe.degree + 1;
}

template<int dim>
void PiSolver<dim>::setup() {
    make_grid();
    setup_system();
}

template<int dim>
void PiSolver<dim>::set_refine_level(int num_bisections) {
    this->refine_level = num_bisections;
}

template<int dim>
void PiSolver<dim>::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refine_level);

//    std::cout << "   Number of active cells: "
//              << triangulation.n_active_cells()
//              << std::endl
}

template<int dim>
void PiSolver<dim>::setup_system() {
    dof_handler.distribute_dofs(fe);
//    std::cout << "   Number of degrees of freedom: "
//              << dof_handler.n_dofs()
//              << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    macro_contribution.reinit(dof_handler.n_dofs());
    old_solution = 1.; // Todo: How to choose the initial value?
    micro_contribution.reinit(dof_handler.n_dofs());
    laplace_matrix.reinit(sparsity_pattern);
    MatrixTools::create_laplace_matrix(dof_handler, QGauss<dim>(integration_order), laplace_matrix);
}

template<int dim>
void PiSolver<dim>::get_pi_contribution_rhs(const Vector<double> &pi, Vector<double> &out_vector) {
    Assert(pi.size() == out_vector.size(), ExcDimensionMismatch(pi.size(), out_vector.size()))
    for (unsigned int i = 0; i < pi.size(); i++) {
        out_vector(i) = std::fmax(1. - 2. * pi(i) / max_support, 0);
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

    compute_microscopic_contribution();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    system_matrix = 0;
    system_rhs = 0;
    system_matrix.add(diffusion_coefficient, laplace_matrix); // Todo: This can be moved out, it only happens once.

    get_pi_contribution_rhs(old_solution, macro_contribution); // Todo: Change name
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
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                rho_rhs_points[q_index] *
                                pi_rhs_points[q_index] *
                                fe_values.JxW(q_index));
            }

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, 0, ZeroFunction<dim>(), boundary_values);
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
    SolverControl solver_control(1000, 1e-12);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    printf("Convergence after %d CG iterations\n", solver_control.last_step());
    compute_residual();
    old_solution = solution;
}

template<int dim>
void PiSolver<dim>::compute_residual() {
    const unsigned int n_active = triangulation.n_active_cells();
    // Todo: Compute the residual
}

template<int dim>
void PiSolver<dim>::set_micro_solutions(std::vector<Vector<double>> *_solutions, DoFHandler<dim> *_dof_handler) {
    this->micro_solutions = _solutions;
    this->micro_dof_handler = _dof_handler;

}

template<int dim>
double PiSolver<dim>::get_micro_mass(unsigned int micro_index) { // todo: Make const
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
double PiSolver<dim>::get_micro_flux(unsigned int micro_index) {
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
void PiSolver<dim>::compute_microscopic_contribution() {
    for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
        micro_contribution[i] = get_micro_mass(i);
    }
}

template<int dim>
void PiSolver<dim>::iterate() {
    assemble_system();
    solve();
}

// Explicit instantiation

template
class MacroBoundary<1>;

template
class MacroBoundary<2>;

template
class MacroBoundary<3>;

template
class PiSolver<1>;

template
class PiSolver<2>;

template
class PiSolver<3>;
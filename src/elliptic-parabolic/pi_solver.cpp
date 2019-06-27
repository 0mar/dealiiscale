/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "pi_solver.h"

using namespace dealii;

template<int dim>
double MacroBoundary<dim>::value(const Point<dim> &p, const unsigned int) const {
    double val = 0; // todo: remove
    val = std::sin(lambda * p(0)) + std::cos(lambda * p(1));
    return val;
}

template<int dim>
Tensor<1, dim> MacroBoundary<dim>::gradient(const Point<dim> &p, const unsigned int) const {
    Tensor<1, dim> return_val;

    return_val[0] = lambda * std::cos(lambda * p(0));
    return_val[1] = -lambda * std::sin(lambda * p(1));
    return return_val;
}


template<int dim>
PiSolver<dim>::PiSolver():dof_handler(triangulation), fe(1), micro_dof_handler(nullptr), micro_solutions(nullptr),
                                boundary() {
    refine_level = 1;
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
//              << "   Total number of cells: "
//              << triangulation.n_cells()
//              << std::endl;
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
    interpolated_solution.reinit(triangulation.n_active_cells());
    micro_contribution.reinit(triangulation.n_active_cells());
}

template<int dim>
Vector<double> PiSolver<dim>::get_exact_solution() {
    Vector<double> exact_values(triangulation.n_active_cells());
    FEValues<dim> fe_value(fe, QMidpoint<dim>(), update_quadrature_points);
    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_value.reinit(cell);
        std::vector<Point<dim>> quad_points = fe_value.get_quadrature_points();
        exact_values[cell->active_cell_index()] = boundary.value(quad_points[0]);
    }
    return exact_values;
}

template<int dim>
double PiSolver<dim>::get_pi_contribution_rhs(double s) {
    const double g_s = std::fmax(1. - 2. * s / theta, 0);
    return g_s;
}

template<int dim>
void PiSolver<dim>::assemble_system() {
    QGauss<dim> quadrature_formula(2);


//    const RightHandSide<dim> right_hand_side;


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

    for (const auto &cell: dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += (A * fe_values.shape_grad(i, q_index) *
                                          fe_values.shape_grad(j, q_index)) *
                                         fe_values.JxW(q_index);


                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                micro_contribution[cell->active_cell_index()] * // Interpolate solution data
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
    interpolate_function(solution, interpolated_solution);
}

template<int dim>
void PiSolver<dim>::compute_error(double &l2_error, double &h1_error) {
    const unsigned int n_active = triangulation.n_active_cells();
    Vector<double> difference_per_cell(n_active);
    VectorTools::integrate_difference(dof_handler, solution, boundary, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::L2_norm);
    l2_error = difference_per_cell.l2_norm();
    VectorTools::integrate_difference(dof_handler, solution, boundary, difference_per_cell, QGauss<dim>(3),
                                      VectorTools::H1_seminorm);
    h1_error = difference_per_cell.l2_norm();
    std::swap(old_solution, solution);
}

template<int dim>
void PiSolver<dim>::set_micro_solutions(std::vector<Vector<double>> *_solutions, DoFHandler<dim> *_dof_handler) {
    this->micro_solutions = _solutions;
    this->micro_dof_handler = _dof_handler;

}

template<int dim>
double PiSolver<dim>::integrate_micro_grid(unsigned int cell_index) {
    // computed as: f(x) = \int_Y \rho(x,y)dy
    double integral = 0;
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell: micro_dof_handler->active_cell_iterators()) {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        std::vector<double> interp_solution(n_q_points);
        fe_values.get_function_values(micro_solutions->at(cell_index), interp_solution);
        for (unsigned int q_index = 0; q_index < n_q_points; q_index++) {
            integral += interp_solution[q_index] * fe_values.JxW(q_index);
        }
    }
    return integral;
}

template<int dim>
void PiSolver<dim>::compute_microscopic_contribution() {
    for (const auto &cell: triangulation.active_cell_iterators()) {
        micro_contribution[cell->active_cell_index()] = integrate_micro_grid(cell->active_cell_index()); // optimize
    }
}

template<int dim>
void PiSolver<dim>::run() {
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
/* ---------------------------------------------------------------------
 *
 * Author: Omar Richardson, 2020
 */
#include <deal.II/grid/tria.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <iostream>
#include <memory>

using namespace dealii;

template<int dim>
class ParsedTensorFunction : TensorFunction<2, dim> {
public:
    ParsedTensorFunction(const FunctionParser<dim> &parsed_function)
            : TensorFunction<2, dim>(), parsed_function(parsed_function) {}

    virtual Tensor<2, dim> value(const Point<dim> &p) const;

    virtual void value_list(const std::vector<Point<dim>> &points, std::vector<Tensor<2, dim>> &values) const;

    const FunctionParser<dim> &parsed_function;

};

template<int dim>
Tensor<2, dim> ParsedTensorFunction<dim>::value(const Point<dim> &p) const {
    Tensor<2, dim> tensor;
    Vector<double> tmp(dim * dim);
    parsed_function.vector_value(p, tmp);
    for (unsigned int i = 0; i < dim; i++) {
        for (unsigned int j = 0; j < dim; j++) {
            tensor[i][j] = tmp[i * dim + j];
        }
    }
    return tensor;
}

template<int dim>
void ParsedTensorFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                           std::vector<Tensor<2, dim>> &values) const {
    AssertDimension(points.size(), values.size());
    Vector<double> tmp(dim * dim);
    for (unsigned int i = 0; i < points.size(); i++) {
        parsed_function.vector_value(points[i], tmp);
        for (unsigned int d1 = 0; d1 < dim; d1++) {
            for (unsigned int d2 = 0; d2 < dim; d2++) {
                values[i][d1][d2] = tmp[d1 * dim + d2];
            }
        }
    }
}

template<int dim>
class ProblemData {
public:
    ProblemData(const std::string &param_file);

    FunctionParser<dim> solution;
    FunctionParser<dim> ref_solution;
    FunctionParser<dim> rhs;
    FunctionParser<dim> map;
    FunctionParser<dim> left_robin;
    FunctionParser<dim> right_robin;
    FunctionParser<dim> up_neumann;
    FunctionParser<dim> down_neumann;
    std::unique_ptr<ParsedTensorFunction<dim>> map_jac;
    ParameterHandler params;
private:
    FunctionParser<dim> map_jac_vector;

};

template<int dim>
ProblemData<dim>::ProblemData(const std::string &param_file) : map(dim), map_jac_vector(dim * dim) {
    params.declare_entry("solution", "sin(x)*y", Patterns::Anything());
    params.declare_entry("rhs", "sin(x)*y",
                         Patterns::Anything());
    params.declare_entry("ref_solution",
                         "(x/2 + sqrt(3)*y/2)*sin(sqrt(3)*x/2 - y/2)",
                         Patterns::Anything());
    params.declare_entry("jac_mapping",
                         "cos(pi/6);-sin(pi/6);sin(pi/6);cos(pi/6)",
                         Patterns::Anything());
    params.declare_entry("mapping",
                         "x*cos(pi/6) - y*sin(pi/6);x*sin(pi/6) + y*cos(pi/6)",
                         Patterns::Anything());
    params.declare_entry("left_robin", "y*sin(x) - sqrt(3)*y*cos(x)/2 - sin(x)/2", Patterns::Anything());
    params.declare_entry("right_robin", "y*sin(x) + sqrt(3)*y*cos(x)/2 + sin(x)/2", Patterns::Anything());
    params.declare_entry("up_neumann", "-y*cos(x)/2 + sqrt(3)*sin(x)/2", Patterns::Anything());
    params.declare_entry("down_neumann", "y*cos(x)/2 - sqrt(3)*sin(x)/2", Patterns::Anything()); // Todo: Add Dirichlet
    params.parse_input(param_file);
    std::map<std::string, double> constants;
    constants["pi"] = numbers::PI;
    rhs.initialize(FunctionParser<dim>::default_variable_names(), params.get("rhs"),
                   constants);
    solution.initialize(FunctionParser<dim>::default_variable_names(), params.get("solution"),
                        constants);
    ref_solution.initialize(FunctionParser<dim>::default_variable_names(), params.get("ref_solution"),
                            constants);
    map_jac_vector.initialize(FunctionParser<dim>::default_variable_names(), params.get("jac_mapping"),
                              constants);
    map_jac = std::make_unique<ParsedTensorFunction<dim>>(map_jac_vector);
    left_robin.initialize(FunctionParser<dim>::default_variable_names(), params.get("left_robin"),
                          constants);
    right_robin.initialize(FunctionParser<dim>::default_variable_names(), params.get("right_robin"),
                           constants);
    up_neumann.initialize(FunctionParser<dim>::default_variable_names(), params.get("up_neumann"),
                          constants);
    down_neumann.initialize(FunctionParser<dim>::default_variable_names(), params.get("down_neumann"),
                            constants);
    map.initialize(FunctionParser<dim>::default_variable_names(), params.get("mapping"),
                   constants);

}

template<int dim>
class NonLinDomainMapping {
public:
    NonLinDomainMapping(const ProblemData<dim> &solution_base) : solution_base(solution_base) {}

    void get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt, double &det_jac) const;

    Point<dim> map(const Point<dim> &p) const;

public:
    const ProblemData<dim> &solution_base;
};

template<int dim>
void NonLinDomainMapping<dim>::get_kkt(const Point<dim> &p, SymmetricTensor<2, dim> &kkt, double &det_jac) const {
    Tensor<2, dim> jacobian = solution_base.map_jac->value(p);
    Tensor<2, dim> inv_jacobian = invert(jacobian);
    det_jac = determinant(jacobian);
    kkt = SymmetricTensor<2, dim>(inv_jacobian * transpose(inv_jacobian));
}

template<int dim>
Point<dim> NonLinDomainMapping<dim>::map(const Point<dim> &p) const {
    Point<dim> p2;
    Vector<double> mapped_point(dim);
    solution_base.map.vector_value(p, mapped_point);
    for (unsigned int i = 0; i < dim; i++) {
        p2[i] = mapped_point(i);
    }
    return p2;
}


class RobinSolver {
public:
    RobinSolver(const std::string &id);

    void run();

    void output_results();

    void refine();

    void set_exact_solution();

private:
    void make_grid();

    void setup_system();

    void assemble_system();

    void solve();

    void process_solution();

    Triangulation<2> triangulation;
    FE_Q<2> fe;
    ProblemData<2> solution_base;
    DoFHandler<2> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    NonLinDomainMapping<2> dm;
    Vector<double> solution;
    Vector<double> system_rhs;
    int cycle;
    ConvergenceTable convergence_table;
    static constexpr unsigned int LEFT_ROBIN = 0;
    static constexpr unsigned int RIGHT_ROBIN = 1;
    static constexpr unsigned int UP_NEUMANN = 2;
    static constexpr unsigned int DOWN_NEUMANN = 3;
};


RobinSolver::RobinSolver(const std::string &id) :
        fe(1),
        solution_base("input/" + id + ".prm"),
        dof_handler(triangulation),
        dm(solution_base),
        cycle(0) {
    make_grid();
}

void RobinSolver::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(1);
    const double EPS = 1E-4;
    for (const auto &cell: triangulation.active_cell_iterators()) {
        for (unsigned int face_number = 0; face_number < GeometryInfo<2>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                const double x = cell->face(face_number)->center()(0);
                const double y = cell->face(face_number)->center()(1);
                if (std::fabs(x - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(RIGHT_ROBIN);
                } else if (std::fabs(x + 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(LEFT_ROBIN);
                } else if (std::fabs(y - 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(UP_NEUMANN);
                } else if (std::fabs(y + 1) < EPS) {
                    cell->face(face_number)->set_boundary_id(DOWN_NEUMANN);
                } else {
                    Assert(false, ExcMessage("Part of the boundary is not initialized correctly"))
                }
            }
        }
    }
}

void RobinSolver::refine() {
    triangulation.refine_global(1);
}

void RobinSolver::setup_system() {
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}

void RobinSolver::assemble_system() {
    const int integration_order = 8;
    const int dim = 2;
    QGauss<dim> quadrature_formula(integration_order);
    QGauss<dim - 1> face_quadrature_formula(integration_order);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_quadrature_points | update_gradients | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                     update_values | update_normal_vectors | update_quadrature_points |
                                     update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_q_face_points = face_quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    double det_jac;
    SymmetricTensor<2, dim> kkt;
    for (const DoFHandler<dim>::active_cell_iterator &cell:dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index) {
            dm.get_kkt(fe_values.quadrature_point(q_index), kkt, det_jac);
//            std::cout << kkt << "\t" << det_jac << std::endl;
            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    cell_matrix(i, j) += (fe_values.shape_grad(i, q_index) *
                                          kkt *
                                          fe_values.shape_grad(j, q_index) *
                                          fe_values.JxW(q_index)) * det_jac;
//                    std::cout << cell_matrix(i,j) << std::endl;
                }
//                double debug_info = solution_base.rhs.value(dm.map(fe_values.quadrature_point(q_index)));
//                std::cout << i << "\t" << q_index << "\t" << debug_info << std::endl;
                cell_rhs(i) += (fe_values.shape_value(i, q_index) *
                                solution_base.rhs.value(dm.map(fe_values.quadrature_point(q_index))) *
                                det_jac *
                                fe_values.JxW(q_index));
            }
        }
        for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell; face_number++) {
            if (cell->face(face_number)->at_boundary()) {
                fe_face_values.reinit(cell, face_number);
                for (unsigned int q_index = 0; q_index < n_q_face_points; q_index++) {
                    Tensor<2, dim> jac = dm.solution_base.map_jac->value(fe_face_values.quadrature_point(q_index));
                    Tensor<2, dim> rot_mat;
                    rot_mat[0][1] = -1;
                    rot_mat[1][0] = 1;
                    det_jac = (jac * rot_mat * fe_face_values.normal_vector(q_index)).norm();
//                    std::cout << det_jac << "\t" << dm.map(fe_face_values.quadrature_point(q_index)) << std::endl;
                    Point<dim> mapped_point = dm.map(fe_face_values.quadrature_point(q_index));
                    for (unsigned int i = 0; i < dofs_per_cell; i++) {
                        switch (cell->face(face_number)->boundary_id()) {
                            case RIGHT_ROBIN:
//                                std::cout << "Right " << det_jac << std::endl;
                                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                                    cell_matrix(i, j) += fe_face_values.shape_value(i, q_index) *
                                                         fe_face_values.shape_value(j, q_index) * det_jac *
                                                         fe_face_values.JxW(q_index);
                                }
                                cell_rhs(i) += solution_base.right_robin.value(mapped_point) * det_jac *
                                               fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index);
                                break;
                            case LEFT_ROBIN:
//                                std::cout << "left " << det_jac << std::endl;
                                for (unsigned int j = 0; j < dofs_per_cell; j++) {
                                    cell_matrix(i, j) += fe_face_values.shape_value(i, q_index) *
                                                         fe_face_values.shape_value(j, q_index) * det_jac *
                                                         fe_face_values.JxW(q_index);
                                }
                                cell_rhs(i) += solution_base.left_robin.value(mapped_point) * det_jac *
                                               fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index);
                                break;
                            case UP_NEUMANN:
//                                std::cout << "up " << det_jac << std::endl;
                                cell_rhs(i) += solution_base.up_neumann.value(mapped_point) * det_jac *
                                               fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index);
                                break;
                            case DOWN_NEUMANN:
//                                std::cout << "down " << det_jac << std::endl;
                                cell_rhs(i) += solution_base.down_neumann.value(mapped_point) * det_jac *
                                               fe_face_values.shape_value(i, q_index) * fe_face_values.JxW(q_index);
                                break;
                            default: Assert(false, ExcMessage("Part of the boundary is not initialized correctly"))
                        }
                    }
                }
            }
        }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
            }
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
}

void RobinSolver::solve() {
    SolverControl solver_control(10000, 1e-10);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}


void RobinSolver::set_exact_solution() {
    MappingQ1<2> mapping;
    AffineConstraints<double> constraints;
    constraints.close();
    VectorTools::project(mapping, dof_handler, constraints, QGauss<2>(5), solution_base.ref_solution, solution);
}

void RobinSolver::process_solution() {
    const int dim = 2;
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      solution_base.ref_solution,
                                      difference_per_cell,
                                      QGauss<dim>(5),
                                      VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();
    std::cout << "Cycle " << cycle << ':'
              << std::endl
              << "   Number of active cells:       "
              << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs
              << std::endl;
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    cycle++;
}

void RobinSolver::output_results() {
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/solution.gpl");
        data_out.write_gnuplot(output);
        output.close();
    }
    {
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        Vector<double> interpolated_solution(dof_handler.n_dofs());
        VectorTools::interpolate(dof_handler, solution_base.ref_solution, interpolated_solution);
        data_out.add_data_vector(interpolated_solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/exact_solution.gpl");
        data_out.write_gnuplot(output);
        output.close();
    }
    {
        Vector<double> error(dof_handler.n_dofs());
        Vector<double> interpolated_solution(dof_handler.n_dofs());
        VectorTools::interpolate(dof_handler, solution_base.ref_solution, interpolated_solution);
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        error = 0;
        error += solution;
        error -= interpolated_solution;
        data_out.add_data_vector(error, "error");
        data_out.build_patches();
        std::ofstream output("results/error_solution.gpl");
        data_out.write_gnuplot(output);
        output.close();
    }
    {
        const int dim = 2;
        const int spacedim = 2;
        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        const FE_Q<dim, spacedim> feq(1);
        const FESystem<dim, spacedim> fesystem(feq, spacedim);
        DoFHandler<dim, spacedim> dhq(triangulation);
        dhq.distribute_dofs(fesystem);
        const ComponentMask mask(spacedim, true);
        Vector<double> eulerq(dhq.n_dofs());
        Vector<double> mappedq(dhq.n_dofs());
        VectorTools::get_position_vector(dhq, eulerq, mask);
        for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) {
            const Point<2> p(eulerq[2 * i], eulerq[2 * i + 1]);
            const auto mapped_point = dm.map(p);
            mappedq[2 * i] = mapped_point[0];
            mappedq[2 * i + 1] = mapped_point[1];

        }
        MappingFEField<dim, spacedim> map(dhq, mappedq, mask);
        std::vector<Point<spacedim>> support_points(dhq.n_dofs());
        DoFTools::map_dofs_to_support_points(map, dhq, support_points);
        data_out.build_patches(map);
        std::ofstream output("results/solution2.gpl");
        data_out.write_gnuplot(output);
    }

    convergence_table.set_precision("L2", 3);
    convergence_table.set_scientific("L2", true);
//    convergence_table.set_tex_caption("cells", "\\# cells");
//    convergence_table.set_tex_caption("dofs", "\\# dofs");
//    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
//    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
//    convergence_table.set_tex_format("cells", "r");
//    convergence_table.set_tex_format("dofs", "r");
    std::ofstream ofs("results/robin.txt");
    convergence_table.write_text(ofs);
    ofs.close();
}

void RobinSolver::run() {
    setup_system();
    assemble_system();
    solve();
    process_solution();
}

int main(int argc, char *argv[]) {
    deallog.depth_console(2);
    std::string id = "parsed_mapping";
    if (argc == 2) {
        id = argv[1];
    }
    RobinSolver poisson_problem(id);
    for (unsigned int i = 2; i < 7; i++) {
        poisson_problem.refine();
        poisson_problem.run();
    }
    poisson_problem.output_results();
    return 0;
}


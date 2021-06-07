/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef RHO_SOLVER_H
#define RHO_SOLVER_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/affine_constraints.h>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <deal.II/base/logstream.h>
#include "../tools/multiscale_function_parser.h"
#include "../tools/pde_data.h"
using namespace dealii;

template<int dim>
class MicroSolver {
public:

    /**
     * Create a MicroSolver that resolves the microscopic systems.
     */
    MicroSolver(ParabolicMicroData<dim> &micro_data, unsigned int refine_level);

    /**
     * Collection method for setting up all necessary tools for the microsolver
     */
    void setup();

    /**
     * Collection method for solving the micro systems for one time step
     * @param time_step Time step size.
     */
    void iterate();

    /**
     * Set the macroscopic solution so that the solver can compute its contribution from it.
     * @param _solution Pointer to the macroscopic solution (so that the content is always up to date).
     * @param _old_solution Pointer to the old macroscopic solution (so that the content is always up to date).
     * @param _dof_handler pointer to the DoF handler.
     */
    void set_macro_solutions(Vector<double> *_solution, Vector<double> *_old_solution, DoFHandler<dim> *_dof_handler);

    /**
     * Post-process the solution (write convergence estimates and other stuff)
     */
    void compute_error(double &l2_error, double &h1_error);

    /**
     * Compute residuals of the microscopic grid
     * @param micro_residual
     */
    void compute_all_residuals(double &l2_residual);

    /**
   * Set the locations of the microgrids with respect to the macrogrids.
   * in practice, these are the locations of the macroscopic degrees of freedom, although other options are possible.
   */
    void set_grid_locations(const std::vector<Point<dim>> &locations);

    /**
     * Prescribe the number of microscopic systems.
     * @param _num_grids int with the number of microscopic systems.
     */

    void set_exact_solution();

    unsigned int get_num_grids();

    void get_color(Vector<double> &color);

    void write_solutions_to_file(const std::vector<Vector<double>> &sols,
                                 const DoFHandler<dim> &corr_dof_handler);

//    void read_solutions_from_file(const std::string &filename, std::vector<Vector<double>> &sols,
//                                  DoFHandler<dim> &corr_dof_handler);


    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> old_solutions;
    std::vector<Vector<double>> solutions_w;
    std::vector<Vector<double>> old_solutions_w;
    std::vector<unsigned int> grid_indicator;
    double time;
    double residual = 1;
private:

    /**
     * Create a domain and make a triangulation.
     * Mark half of the boundary (the left half) as a Neumann boundary condition.
     * Mark the other half as Robin.
     */
    void make_grid();

    /**
     * Common data structures are created in this method
     * Distribute the degrees of freedom
     * Create a common sparsity pattern based on the mesh
     */
    void setup_system();

    /**
     * Specific data structures are created in this method.
     * Create specific right hand sides and solution vectors.
     */
    void setup_scatter();

    /**
     * Actual important method: Create the system matrices and create the right hand side vectors
     * by looping over all the cells and computing the discrete weak forms
     */
    void assemble_system();

    /**
     * Solve the system we obtained in `assemble_system`.
     */
    void solve_time_step();

    /**
     * Use the (probably updated) macroscopic data to compute new elements of the microscopic system.
     */
    void compute_macroscopic_contribution();

    void print(Vector<double> vec);

    double get_residual(unsigned int grid_num);

    Point<dim> get_micro_grid_size(const std::vector<Point<dim>> &locations) const;

    FE_Q<dim> fe;
    unsigned int fem_q_deg = 8;
    unsigned int h_inv;
    int iteration = 0;
    unsigned int num_grids;
    SparsityPattern sparsity_pattern;
    Vector<double> *macro_solution;
    Vector<double> *old_macro_solution;
    Vector<double> init_macro_field;
    DoFHandler<dim> *macro_dof_handler;
    std::vector<SparseMatrix<double>> system_matrices;
    std::vector<Vector<double>> righthandsides;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    std::vector<Point<dim>> grid_locations;
    ParabolicMicroData<dim> &pde_data;
    int integration_order;
    Vector<double> intermediate_vector;
    AffineConstraints<double> constraints;
    const unsigned int DIRICHLET_BOUNDARY=1;
    const unsigned int NEUMANN_BOUNDARY=2;
    const unsigned int ROBIN_BOUNDARY=3;
};

#endif //RHO_SOLVER_H

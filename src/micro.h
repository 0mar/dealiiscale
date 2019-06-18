/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef MICRO_H
#define MICRO_H

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

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim>
class MicroBoundary : public Function<dim> {
public:
    MicroBoundary() : Function<dim>() {

    }

    /**
     * Compute the value of the microscopic boundary at a given point
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

    /**
     * Compute the analytic gradient of the boundary at point p. Necessary for Robin/Neumann boundary conditions and
     * exact evaluation of the error.
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return gradient of the microscopic boundary condition at p
    */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const;

    /**
     * Set a precomputed macroscopic solution for the boundary.
     * After this value is set, individual microboundaries can be imposed by simply setting the macroscopic cell index.
     * @param macro_solution Value of the macroscopic solution for the corresponding microscopic system.
     */
    void set_macro_solution(const Vector<double> &macro_solution);

    /**
     *
     * Set the macroscopic cell index so that the microboundary has the appropriate macroscopic value.
     * @param index
     */
    void set_macro_cell_index(unsigned int index);

private:

    /**
     * Contains the macroscopic exact solution
     */
    Vector<double> macro_sol;

    /**
     * The macroscopic cell this boundary needs to work on.
     */
    unsigned int macro_cell_index = 0;
};

template<int dim>
class MicroSolver {
public:

    /**
     * Create a Microsolver that resolves the microscopic systems.
     * @param macro_dof_handler The macroscopic degrees of freedom
     * @param macro_solution The macroscopic solution
     */
    MicroSolver();

    /**
     * Collection method for setting up all necessary tools for the microsolver
     */
    void setup();

    /**
     * Collection method for solving the micro systems
     */
    void run();

    /**
     * Set the refinement level of the grid (i.e. h = 1/2^refinement_level)
     * @param refine_level number of bisections of the grid.
     */
    void set_refine_level(int refinement_level);

    /**
     * Refine the grid by splitting each cell in four new cells.
     */
    void refine_grid();

    /**
     * Compute the right hand side of the Macroscopic function;
     * a contribution from the microscopic solution.
     * @param dof_index Degree of freedom corresponding to the microscopic grid.
     * @return double with the value of the integral/other RHS function
     */
    double integrate_micro_grid(unsigned int cell_index);

    void set_macro_solution(Vector<double> *_solution, DoFHandler<dim> *_dof_handler);

    void set_macro_boundary_condition(const Vector<double> &macro_condition);

    /**
     * Output the results/write them to file/something
     */
    void output_results();

    /**
     * Post-process the solution (write convergence estimates and other stuff)
     */
    void process_solution();

    /**
     * Getter for the number of microgrids.
     * @return number of microgrids based on the number of macroscopic cells.
     */
    unsigned int get_num_grids() const;

    void set_num_grids(unsigned int _num_grids);

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    std::vector<Vector<double>> solutions;
    MicroBoundary<dim> boundary;
private:

    /**
     * Create a domain and make a triangulation
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
    void solve();

    /**
     * Use the (probably updated) macroscopic data to compute new elements of the microscopic system.
     */
    void compute_macroscopic_contribution();

    const double laplacian = 4.;
    /**
     * The level of refinement (every +1 means a bisection)
     */
    unsigned int refine_level;
    FE_Q<dim> fe;
    ConvergenceTable convergence_table;
    unsigned int cycle;
    unsigned int num_grids;
    SparsityPattern sparsity_pattern;
    Vector<double> *macro_solution;
    Vector<double> macro_contribution;
    DoFHandler<dim> *macro_dof_handler;
    std::vector<SparseMatrix<double>> system_matrices;
    std::vector<Vector<double>> righthandsides;
};

#endif //MICRO_H

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

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <deal.II/base/logstream.h>

using namespace dealii;


template<int dim>
class MicroInitCondition : public Function<dim> {
public:
    MicroInitCondition() : Function<dim>() {

    }

    virtual double value(const Point<dim> &p, const unsigned int component = 0) const;

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
    unsigned int macro_cell_index = 0; // Todo: Create a MicroFunction base class to derive this from

};

template<int dim>
class RhoSolver {
public:

    /**
     * Create a RhoSolver that resolves the microscopic systems.
     */
    RhoSolver();

    /**
     * Collection method for setting up all necessary tools for the microsolver
     */
    void setup();

    /**
     * Collection method for solving the micro systems
     */
    void iterate(const double &time_step);

    /**
     * Set the refinement level of the grid (i.e. h = 1/2^refinement_level)
     * @param refine_level number of bisections of the grid.
     */
    void set_refine_level(int refinement_level);

    /**
     * Refine the grid by splitting each cell in 2^d new cells.
     */
    void refine_grid();

    /**
     * Set the macroscopic solution so that the solver can compute its contribution from it.
     * @param _solution Pointer to the macroscopic solution (so that the content is always up to date).
     * @param _dof_handler pointer to the DoF handler.
     */
    void set_macro_solutions(Vector<double> *_solution, Vector<double> *_old_solution, DoFHandler<dim> *_dof_handler);

    /**
     * Post-process the solution (write convergence estimates and other stuff)
     */
    void compute_residual();

    /**
     * Getter for the number of microgrids.
     * @return number of microgrids based on the number of macroscopic cells.
     */
    unsigned int get_num_grids() const;

    void set_num_grids(unsigned int _num_grids);

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> old_solutions;
    double dt = 0.1;
    double residual = 1;
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
    void solve_time_step();

    /**
     * Use the (probably updated) macroscopic data to compute new elements of the microscopic system.
     */
    void compute_macroscopic_contribution();

    /**
     * The level of refinement (every +1 means a bisection)
     */
    unsigned int refine_level;
    FE_Q<dim> fe;
    unsigned int num_grids;
    SparsityPattern sparsity_pattern;
    Vector<double> *macro_solution;
    Vector<double> *old_macro_solution; // Todo: Add old macro solution
    Vector<double> macro_contribution;
    DoFHandler<dim> *macro_dof_handler;
    std::vector<SparseMatrix<double>> system_matrices;
    std::vector<Vector<double>> righthandsides;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    const double D = 1;
    const double R = 2;
    const double kappa = 1;
    const double p_F = 1;
    const double scheme_theta = 1;
    int integration_order = 2;
    Vector<double> intermediate_vector;
};

#endif //RHO_SOLVER_H

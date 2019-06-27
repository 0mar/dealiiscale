/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef PISOLVER_H
#define PISOLVER_H

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
class MacroBoundary : public Function<dim> {
public:
    MacroBoundary() : Function<dim>() {

    }

    /**
     * Creates a macroscopic boundary (only Dirichlet at this point)
     * @param p The point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary condition at p
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

private:
    const double lambda = std::sqrt(8. / 3.); // Coming from manufactured problem
};


template<int dim>
class PiSolver {
public:
    /**
     * Create and run a macroscopic solver with the given resolution (number of unit square bisections)
     * @param refine_level Number of macroscopic and microscopic bisections.
     */
    PiSolver();

    /**
     * All the methods that setup the system.
     * 1. Create the grid.
     * 2. Set the FE degree and compute the DoFs
     * 3. Initialize FEM matrices and vectors
     */
    void setup();

    /**
     * All the methods (after setup()) necessary to compute a solution
     * 4. Assemble the system matrix and right hand side
     * 5. Solve the system
     * 6. Process the solution (numerically)
     */
    void run();

    /**
     * Compute residual/convergence estimates.
     */
    void compute_error(double &l2_error, double &h1_error);

    /**
     * Set the microscopic solutions pointers, so that this solver can compute its contribution from it.
     * @param _solutions Pointer to a vector of microscopic solution vectors.
     * @param _dof_handler Pointer to the microscopic DoF handler.
     */
    void set_micro_solutions(std::vector<Vector<double>> *_solutions, DoFHandler<dim> *_dof_handler);

    /**
     * Compute the exact solution value based on the analytic solution present in the boundary condition
     * @param exact_values Output vector
     */
    Vector<double> get_exact_solution();

    /**
     * Set the refinement level of the grid (i.e. h = 1/2^refinement_level)
     * @param refine_level number of bisections of the grid.
     */
    void set_refine_level(int num_bisections);

    Vector<double> solution;
    Vector<double> old_solution;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    Vector<double> interpolated_solution;
    double residual = 1;

private:
    /**
     * Create the grid and triangulation
     * @param refine_level Number of bisections done on the unit square
     */
    void make_grid();

    /**
     * Compute the degrees of freedom, set up the structure of the system matrix and right hand side/solution vectors.
     */
    void setup_system();

    /**
     * Compute the actual system matrix and right hand side based on the discrete weak form of the macro PDE.
     */
    void assemble_system();

    /**
     * Interpolate a finite element function defined on the present grid to a single value per cell.
     * @param fe Finite element function corresponding to the finite element space
     * @return Vector (finite element function) with length `#cells`
     */
    void interpolate_function(const Vector<double> &func, Vector<double> &interp_func);

    /**
     * Use the (probably updated) microscopic data to compute new elements of the macroscopic system.
     */
    void compute_microscopic_contribution();

    /**
    * Compute the right hand side of the Macroscopic function;
    * a contribution from the microscopic solution.
    * @param dof_index Degree of freedom corresponding to the microscopic grid.
    * @return double with the value of the integral/other RHS function
    */
    double integrate_micro_grid(unsigned int cell_index);

    /**
     * Compute the pi-dependent factor of the right hand side of the elliptic equation
     * f(s) = g(s)*\int_Y \rho(x,y)dy.
     * This method computes g(s): a hat function with support from 0 to some theta
     */
    double get_pi_contribution_rhs(double s);

    /**
     * Apply an (iterative) solver for the linear system made in `assemble_system` and obtain a solution
     */
    void solve();

    FE_Q<dim> fe;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    DoFHandler<dim> *micro_dof_handler;
    Vector<double> system_rhs;
    Vector<double> micro_contribution;
    std::vector<Vector<double>> *micro_solutions;
    MacroBoundary<dim> boundary;
    int refine_level;
    const double theta = 10;
    const double A = 1; // Move all parameters to initializer

};

#endif //PISOLVER

/* --------------------------------------------------------------------
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef MACRO_H
#define MACRO_H

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
#include "micro.h"
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
class MacroSolver {
public:
    /**
     * Create and run a macroscopic solver with the given resolution (number of unit square bisections)
     * @param refine_level Number of macroscopic and microscopic bisections.
     */
    MacroSolver(unsigned int refine_level);

    /**
     * Assemble the system, solve the system and process the solution.
     */
    void run();

    /**
     * Compute residual/convergence estimates and add them to the convergence table.
     */
    void process_solution();

    /**
     * Compute the exact solution value based on the analytic solution present in the boundary condition
     * @param exact_values Output vector
     */
    void compute_exact_value(Vector<double> &exact_values);

private:
    /**
     * Create the grid and triangulation
     * @param refine_level Number of bisections done on the unit square
     */
    void make_grid(unsigned int refine_level);

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
     * Apply an (iterative) solver for the linear system made in `assemble_system` and obtain a solution
     */
    void solve();

    /**
     * Write the results to file, both convergence results as .gpl files for gnuplot or whatever.
     */
    void output_results();

    Triangulation<dim> triangulation;
    FE_Q<dim> fe;
    DoFHandler<dim> dof_handler;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> interpolated_solution;
    Vector<double> system_rhs;
    MicroSolver<dim> micro;
    MacroBoundary<dim> boundary;
    ConvergenceTable convergence_table;
    int cycle;

};

#endif //MACRO_H

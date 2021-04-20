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
#include <deal.II/lac/affine_constraints.h>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <cstdlib>
#include "../tools/multiscale_function_parser.h"
#include "../tools/pde_data.h"
#include <deal.II/base/logstream.h>

using namespace dealii;

template<int dim>
class MacroSolver {
public:
    /**
     * Create and run a macroscopic solver with the given resolution (number of unit square bisections)
     * @param refine_level Number of macroscopic and microscopic bisections.
     */
    MacroSolver(MacroData<dim> &macro_data, unsigned int refine_level);

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
    void iterate();

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

    void get_dof_locations(std::vector<Point<dim>> &locations);

    void write_solution_to_file(const Vector<double> &sol,
                                const DoFHandler<dim> &corr_dof_handler);

    void read_solution_from_file(const std::string &filename, Vector<double> &sol, DoFHandler<dim> &corr_dof_handler);

    Vector<double> solution;
    Vector<double> old_solution;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    double residual;

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
    void get_microscopic_contribution(Vector<double> &micro_contribution, bool nonlinear);

    /**
    * Compute the right hand side of the Macroscopic function;
    * a contribution from the microscopic solution.
     * In this case, the contribution is an integral over the mass of the microscopic grids.
    * @param dof_index Degree of freedom corresponding to the microscopic grid.
    * @return double with the value of the integral/other RHS function
    */
    double get_micro_mass(unsigned int micro_index) const;

    /**
    * Compute the right hand side of the Macroscopic function;
    * a contribution from the microscopic solution.
     * In this case, the contribution is a boundary integral over the normal derivative: the total outflux of the microscopic grids.
    * @param dof_index Degree of freedom corresponding to the microscopic grid.
    * @return double with the value of the integral/other RHS function
    */
    double get_micro_flux(unsigned int_micro_index) const;

    /**
     * Compute the pi-dependent factor of the right hand side of the elliptic equation
     * f(s) = g(s)*\int_Y \rho(x,y)dy.
     * This method computes g(s): a hat function with support from 0 to some theta
     */
    void get_pi_contribution_rhs(const Vector<double> &pi, Vector<double> &out_vector, bool nonlinear) const;

    /**
     * Apply an (iterative) solver for the linear system made in `assemble_system` and obtain a solution
     */
    void solve();

    void set_exact_solution();

    void get_solution_difference(double &diff);

    FE_Q<dim> fe;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> laplace_matrix;
    Vector<double> macro_contribution;
    DoFHandler<dim> *micro_dof_handler;
    Vector<double> system_rhs;
    std::vector<Vector<double>> *micro_solutions;
    MacroData<dim> &pde_data;
    int integration_order;
    unsigned int h_inv;
    double diffusion_coefficient;
    double max_support;
    AffineConstraints<double> constraints;
    int count = 0; //debug

};

#endif //PISOLVER_H

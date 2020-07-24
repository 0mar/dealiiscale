/*
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
#include <cmath>
#include <cstdlib>
#include <deal.II/base/logstream.h>
#include "../tools/pde_data.h"
#include "../tools/mapping.h"

using namespace dealii;


template<int dim>
class MacroSolver {
public:
    /**
     * Create and run a macroscopic solver with the given resolution (number of unit square bisections)
     * @param refine_level Number of macroscopic and microscopic bisections.
     */
    MacroSolver(BioMacroData<dim> &macro_data, unsigned int refine_level);

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
    void set_micro_objects(const MicroFEMObjects<dim> &micro_fem_objects);

    /**
     * Compute the exact solution value based on the analytic solution present in the boundary condition
     * @param exact_values Output vector
     */
    void set_exact_solution();

    /**
     * Obtain the physical locations of the degrees of freedom
     * @param locations output vector
     */
    void get_dof_locations(std::vector<Point<dim>> &locations);

    Vector<double> sol_u, sol_w;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    static constexpr unsigned int DIRICHLET_BOUNDARY = 0;
    static constexpr unsigned int NEUMANN_BOUNDARY = 1;

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
     * Use the (probably updated) microscopic data to compute new elements of the macroscopic system.
     */
    void compute_microscopic_contribution();

    /**
    * Compute the right hand side of the Macroscopic function;
    * a contribution from the microscopic solution.
    * @param dof_index Degree of freedom corresponding to the microscopic grid.
    * @return double with the value of the integral/other RHS function
    */
    void integrate_micro_cells(unsigned int macro_index, const Point<dim> &macro_point, double &u_contribution,
                               double &w_contribution);

    /**
     * Apply an (iterative) solver for the linear system made in `assemble_system` and obtain a solution
     */
    void solve();

    /**
     * Struct containing all data and macroscopic functions.
     */
    BioMacroData<dim> &pde_data;

    const FE_Q<dim> fe;
    MicroFEMObjects<dim> micro;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix_u, system_matrix_w;
    Vector<double> system_rhs_u, system_rhs_w;
    Vector<double> micro_contribution_u, micro_contribution_w;
    const int refine_level;
    std::vector<Point<dim>> micro_grid_locations;
};

#endif //MACRO_H

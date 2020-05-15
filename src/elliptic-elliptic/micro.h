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
#include <cmath>
#include <stdlib.h>
#include <deal.II/base/logstream.h>
#include "../tools/multiscale_function_parser.h"
#include "../tools/pde_data.h"
using namespace dealii;


template<int dim>
class MicroSolver {
public:

    /**
     * Create a Microsolver that resolves the microscopic systems.
     */
    MicroSolver(MicroData<dim> &micro_data, unsigned int refine_level);

    /**
     * Collection method for setting up all necessary tools for the microsolver
     */
    void setup();

    /**
     * Collection method for solving the micro systems
     */
    void run();

    /**
     * Set the macroscopic solution so that the solver can compute its contribution from it.
     * @param _solution Pointer to the macroscopic solution (so that the content is always up to date).
     * @param _dof_handler pointer to the DoF handler.
     */
    void set_macro_solution(Vector<double> *_solution, DoFHandler<dim> *_dof_handler); // todo: make reference here.

    /**
     * Post-process the solution (write convergence estimates and other stuff)
     */
    void compute_error(double &l2_error, double &h1_error);

    /**
     * Getter for the number of microgrids.
     * @return number of microgrids based on the number of macroscopic cells.
     */
    unsigned int get_num_grids() const;

    /** debug **/
    void set_exact_solution();

    /**
     * Set the locations of the microgrids with respect to the macrogrids.
     * in practice, these are the locations of the macroscopic degrees of freedom, although other options are possible.
     */
    void set_grid_locations(const std::vector<Point<dim>> &locations);

    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    std::vector<Vector<double>> solutions;
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

    /**
     * Compute pullback objects
     */

    void compute_pullback_objects();


    void get_pullback_objects(const Point<dim> &px, const Point<dim> &py, SymmetricTensor<2, dim> &kkt,
                              double &det_jac) const;

    // The level of refinement (every +1 means a bisection)
    const unsigned int refine_level;
    unsigned int fem_quadrature;
    const FE_Q<dim> fe;
    // Number of microscopic grids
    unsigned int num_grids;
    SparsityPattern sparsity_pattern;
    // Pointer to the solution of the macroscopic equation (readonly)
    const Vector<double> *macro_solution;
    // Macroscopic contribution to the microscopic equation [unused]
    Vector<double> macro_contribution;
    // Macroscopic degree of freedom handler
    const DoFHandler<dim> *macro_dof_handler;
    // System matrices for each microscopic domain
    std::vector<SparseMatrix<double>> system_matrices;
    // Right hand side vectors for each microscopic domain
    std::vector<Vector<double>> righthandsides;
    // Macroscopic locations of individual microscopic domains
    std::vector<Point<dim>> grid_locations;
    // Precomputed inverse jacobians of mappings for each microscopic and macroscopic degree of freedom
    std::vector<std::vector<SymmetricTensor<2,dim>>> kkts;
    // Precomputed determinants of the jacobians for each microscopic and macroscopic degree of freedom
    std::vector<std::vector<double>> det_jacs;
    // Object containing microscopic problem data
    MicroData<dim> &pde_data;
};

#endif //MICRO_H

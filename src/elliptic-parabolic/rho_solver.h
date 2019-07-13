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
     * Set a precomputed macroscopic scalar field for the boundary.
     * After this value is set, individual micro boundaries can be imposed by simply setting the macroscopic cell index.
     * @param macro_solution Value of the macroscopic field for the corresponding microscopic system.
     */
    void set_macro_field(const Vector<double> &field);

    /**
     *
     * Set the macroscopic cell index so that the micro boundary has the appropriate macroscopic value.
     * @param index
     */
    void set_macro_cell_index(unsigned int index);

private:

    /**
     * Contains the macroscopic field
     */
    Vector<double> macro_field;

    /**
     * The macroscopic cell this boundary needs to work on.
     */
    unsigned int macro_cell_index = 0;

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
     * Collection method for solving the micro systems for one time step
     * @param time_step Time step size.
     */
    void iterate(const double &time_step);

    /**
     * Set the refinement level of the grid (i.e. h = 1/2^refinement_level)
     * @param refine_level number of bisections of the grid.
     */
    void set_refine_level(const int &refinement_level);

    void set_initial_condition(const Vector<double> &initial_condition);

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
    void compute_residual();

    /**
     * Prescribe the number of microscopic systems.
     * @param _num_grids int with the number of microscopic systems.
     */
    void set_num_grids(const unsigned int _num_grids);

    void patch_micro_solutions(const std::vector<Point<dim>> &locations) const;

    void write_solutions_to_file(const std::vector<Vector<double>> &sols,
                                 const DoFHandler<dim> &corr_dof_handler);

    void read_solutions_from_file(const std::string &filename, std::vector<Vector<double>> &sols,
                                  DoFHandler<dim> &corr_dof_handler);


    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler;
    std::vector<Vector<double>> solutions;
    std::vector<Vector<double>> old_solutions;
    double dt = 0.1;
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

    Point<dim> get_micro_grid_size(const std::vector<Point<dim>> &locations) const;

    const unsigned int ROBIN_BOUNDARY = 0;
    const unsigned int NEUMANN_BOUNDARY = 1;
    unsigned int refine_level;
    FE_Q<dim> fe;
    unsigned int num_grids;
    SparsityPattern sparsity_pattern;
    Vector<double> *macro_solution;
    Vector<double> *old_macro_solution;
    Vector<double> macro_contribution;
    Vector<double> init_macro_field;
    DoFHandler<dim> *macro_dof_handler;
    std::vector<SparseMatrix<double>> system_matrices;
    std::vector<Vector<double>> righthandsides;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    double diffusion_coefficient = 1;
    double R;
    double kappa;
    double p_F;
    double theta; // Todo: Test thorougly if theta<1 works as well
    int integration_order;
    Vector<double> intermediate_vector;
};

#endif //RHO_SOLVER_H

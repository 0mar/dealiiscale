/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_MANAGER_H
#define DEALIISCALE_MANAGER_H

#include "pi_solver.h"
#include "rho_solver.h"



class Manager {

public:
    const static int MACRO_DIMENSIONS = 2;
    const static int MICRO_DIMENSIONS = 2;
    PiSolver<MACRO_DIMENSIONS> pi_solver;
    RhoSolver<MICRO_DIMENSIONS> rho_solver;
    int repetitions;

    /**
     * Class that facilitates the interaction between the microscopic and macroscopic solvers.
     * @param macro_refinement Resolution of the macro solver.
     * @param micro_refinement Resolution of the micro solver.
     */
    Manager(int macro_refinement, int micro_refinement);

    /**
     * Run all the methods that setup the solvers of the two scales.
     */
    void setup();

    /**
     * Run iterations of the microscopic solvers until the result (of a single time step) is sufficiently close
     * to the solution.
     */
    void run();

    /**
     * Print the error computations/estimates in a convergence table.
     */
    void output_results();

    void write_plot();
    /**
     * Set a custom name for the file containing the convergence table.
     * @param file_name Name of the convergence table file.
     */
    void set_ct_file_name(std::string &file_name);

    double eps = 1E-4;
    double max_iterations = 1E4;
    double time_step = 0.1;
    double time = 0; // Todo: Put all in initializer list
    double final_time = 5;

private:
    int it;

    /**
     * One (Banach-like) fixed point iteration. The multiscale system is operator-splitted into two single-scale problems.
     */
    void iterate();

    /**
     * Compute the multiscale residual by adding the macroscopic and the microscopic error.
     * Analysis shows that this is bounded.
     * @param old_residual The residual in the previous operator splitting iteration.
     * @param residual The residual in the current operator splitting iteration.
     */
    void compute_residuals(double &old_residual, double &residual);

    std::string ct_file_name = "convergence.txt";
    ConvergenceTable convergence_table;

};





#endif //DEALIISCALE_MANAGER_H

/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_MANAGER_H
#define DEALIISCALE_MANAGER_H

#include "micro.h"
#include "macro.h"
#include "../tools/pde_data.h"


class Manager {

public:
    const static int MACRO_DIMENSIONS = 2;
    const static int MICRO_DIMENSIONS = 2;
    BioData<MACRO_DIMENSIONS> data;
    MacroSolver<MACRO_DIMENSIONS> macro_solver;
    MicroSolver<MICRO_DIMENSIONS> micro_solver;

    /**
     * Class that facilitates the interaction between the microscopic and macroscopic solvers.
     * @param macro_refinement Resolution of the macro solver.
     * @param micro_refinement Resolution of the micro solver.
     * @param data_file String that contains the name of a data file.
     */
    Manager(unsigned int macro_refinement, unsigned int micro_refinement, const std::string &data_file,
            const std::string &output_file);

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

    double eps = 1E-2;
    double max_iterations = 1E4;


private:
    int cycle;
    const bool parallel;

    /**
     * One (Banach-like) fixed point iteration. The multiscale system is operator-splitted into two single-scale problems.
     */
    void fixed_point_iterate();

    /**
     * Compute the multiscale residual by adding the macroscopic and the microscopic error.
     * Analysis shows that this is bounded.
     * @param old_error The residual in the previous operator splitting iteration.
     * @param error The residual in the current operator splitting iteration.
     */
    void compute_errors(double &old_error, double &error);

    void compute_residuals(double &old_residual, double &residual);

    void patch_and_write_solutions();

    const std::string &ct_file_name = "convergence.txt";
    ConvergenceTable convergence_table;

};

#endif //DEALIISCALE_MANAGER_H

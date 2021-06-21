/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_TIME_MANAGER_H
#define DEALIISCALE_TIME_MANAGER_H

#include "macro.h"
#include "micro.h"
#include "../tools/pde_data.h" // todo: fix imports


class Manager {

public:
    const static int MACRO_DIMENSIONS = 2;
    const static int MICRO_DIMENSIONS = 2;
    VesicleData<MACRO_DIMENSIONS> data;
    MacroSolver<MACRO_DIMENSIONS> macro;
    MicroSolver<MICRO_DIMENSIONS> micro;

    /**
     * Class that facilitates the interaction between the microscopic and macroscopic solvers.
     * @param macro_refinement Resolution of the macro solver.
     * @param micro_refinement Resolution of the micro solver.
     */
    Manager(unsigned int macro_h_inv, unsigned int micro_h_inv, const std::string &data_file,
                const std::string &out_dir);

    /**
     * Run all the methods that setup the solvers of the two scales and couple the data structures.
     */
    void setup();

    /**
     * Run iterations of the macro and micro problems until we reached the final time.
     * Currently, we use explicit (Picard) iterations for the macroscopic (elliptic) problem and
     * an implicit time stepping scheme (Backward Euler) for the microscopic (parabolic) problem.
     * Plots are written on each time step.
     */
    void run();

    /**
     * Print the error computations/estimates in a convergence table.
     */
    void output_results();

    /**
     * Write plots to VTK format so they can be opened with Paraview.
     */
    void write_plot();

    // Time variable
    double time = 0;

    // Time until done
    double final_time = 5;

private:
    int it = 0;

    /**
     * One forward iteration in time. The multiscale system is operator-splitted into two single-scale problems.
     */
    void iterate();

    /**
     * Compute the multiscale residual by adding the macroscopic and the microscopic error.
     * Analysis shows that this is bounded.
     * @param old_residual The residual in the previous operator splitting iteration.
     * @param residual The residual in the current operator splitting iteration.
     */
    void compute_residuals(double &old_residual, double &residual);
    void patch_and_write_solutions();

    void write_micro_grid_locations(const std::string &filename);

    std::string results_dir = "convergence.txt";
    ConvergenceTable convergence_table;

};

#endif //DEALIISCALE_TIME_MANAGER_H

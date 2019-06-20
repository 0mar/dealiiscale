/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_MACRO_MANAGER_H
#define DEALIISCALE_MACRO_MANAGER_H

#include "micro.h"
#include "macro.h"


class MacroManager {

public:
    const static int DIMENSIONS = 2;
    MacroSolver<DIMENSIONS> macro_solver;
    int repetitions;


    MacroManager(int micro_refinement);

    void setup();

    void run();

    void output_results();

    void set_ct_file_name(std::string &file_name);

    double eps = 1E-4;
    double max_iterations = 1E4;


private:
    int cycle;

    void fixed_point_iterate();

    void compute_residuals(double &old_residual, double &residual);

    void compute_virtual_micros(std::vector<Vector<double>> &virtual_micros);

    std::string ct_file_name = "convergence.txt";
    ConvergenceTable convergence_table;
    std::vector<Vector<double>> micro_solutions;
};


#endif //DEALIISCALE_MACRO_MANAGER_H

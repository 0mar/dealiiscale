/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_MICRO_MANAGER_H
#define DEALIISCALE_MICRO_MANAGER_H

#include "micro.h"
#include "macro.h"



class MicroManager {

public:
    const static int DIMENSIONS = 2;
    MacroSolver<DIMENSIONS> macro_solver;
    MicroSolver<DIMENSIONS> micro_solver;
    int repetitions;


    MicroManager(int micro_refinement);

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

    std::string ct_file_name = "convergence.txt";
    ConvergenceTable convergence_table;
    Vector<double> macro_solution;
};


#endif //DEALIISCALE_MICRO_MANAGER_H

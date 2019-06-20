/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_MANAGER_H
#define DEALIISCALE_MANAGER_H

#include "micro.h"
#include "macro.h"



class Manager {

public:
    const static int MACRO_DIMENSIONS = 2;
    const static int MICRO_DIMENSIONS = 2;
    MacroSolver<MACRO_DIMENSIONS> macro_solver;
    MicroSolver<MICRO_DIMENSIONS> micro_solver;
    int repetitions;


    Manager(int macro_refinement, int micro_refinement);

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

};





#endif //DEALIISCALE_MANAGER_H

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


private:
    int cycle;

    void fixed_point_iterate();

    void compute_error(double &error_norm);


    ConvergenceTable convergence_table;

};





#endif //DEALIISCALE_MANAGER_H

/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "time_manager.h"

TimeManager::TimeManager(const std::string &data_file, const std::string &out_file) : data(data_file),
                                                                                      pi_solver(data.macro),
                                                                                      rho_solver(data.micro),
                                                                                      time_step(0.25),
                                                                                      final_time(0.25),
                                                                                      ct_file_name(out_file) {
    time_step /= std::pow(2, 2);
    printf("Using a time step of %.2e\n", time_step);
}

void TimeManager::setup() {
    // Create the grids and solution data structures for each grid
    pi_solver.setup();
    std::vector<Point<MACRO_DIMENSIONS>> dof_locations;
    pi_solver.get_dof_locations(dof_locations);
    rho_solver.set_grid_locations(dof_locations);
    rho_solver.setup();
    // Couple the macro structures with the micro structures.

    rho_solver.set_macro_solutions(&pi_solver.solution, &pi_solver.solution,
                                   &pi_solver.dof_handler);
    pi_solver.set_micro_solutions(&rho_solver.solutions, &rho_solver.dof_handler);
//    std::vector<std::string> out_file_names = {"macro_vals.txt", "micro_vals.txt", "macro_convergence.txt",
//                                               "micro_convergence.txt"};
//    for (const std::string &out_file_name: out_file_names) {
//        std::ofstream ofs;
//        ofs.open("results/" + out_file_name, std::ofstream::out | std::ofstream::trunc);
//        ofs.close();
//    }
    it = 0;
}

void TimeManager::run() {
    time += time_step; 
    iterate();
}


void TimeManager::iterate() {
    data.set_time(time);
    pi_solver.iterate();
    rho_solver.iterate(time_step);
}

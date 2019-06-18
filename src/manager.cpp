/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "manager.h"

Manager::Manager(int macro_refinement, int micro_refinement) : macro_solver(), micro_solver() {
    macro_solver.set_refine_level(macro_refinement);
    micro_solver.set_refine_level(micro_refinement);
}

void Manager::setup() {
    // Create the grids and solution data structures for each grid
    macro_solver.setup();
    micro_solver.set_num_grids(macro_solver.triangulation.n_active_cells());
    micro_solver.setup();
    micro_solver.set_macro_boundary_condition(macro_solver.get_exact_solution());
    // Couple the macro structures with the micro structures.
    micro_solver.set_macro_solution(&macro_solver.solution, &macro_solver.dof_handler);
    macro_solver.set_micro_solutions(&micro_solver.solutions, &micro_solver.dof_handler);

}

void Manager::run() {
    double eps = 1E-4;
    double error_norm = 1.;
    // while (std::fabs(error_norm) > eps) {
    for (int i = 0; i < 10; i++) {
        // Todo: Interpolate from midpoint to Gaussian
        fixed_point_iterate();
        error_norm = fem_error();
    }
    output_results();
}

double Manager::fem_error() {
    return 0; // Todo: Implement based on some macro-micro error estimate
}

void Manager::output_results() {
    macro_solver.output_results();
    micro_solver.output_results(); // todo: Fix to move all into one combined file.
}

void Manager::fixed_point_iterate() {
    macro_solver.run();
    micro_solver.run();
}

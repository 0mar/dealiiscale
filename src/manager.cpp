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

}

void Manager::run() {
    double eps = 1E-4;
    double error_norm = 1.;
    Vector<double> macro_rhs(macro_solver.triangulation.n_active_cells());
    Vector<double> micro_rhs(macro_solver.triangulation.n_active_cells());
    while (std::fabs(error_norm) > eps) {
        micro_solver.set_macro_solution(
                macro_solver.get_solution()); // Todo: This does not need to happen each step if we fix the reference.
        macro_solver.set_micro_solutions(micro_solver.get_solutions());
        macro_solver.set_micro_contribution(
                micro_solver.get_contribution(macro_solver.triangulation)); // Todo: Also better fixed with pointers
        micro_solver.set_macro_contribution(macro_solver.get_contribution()); // Maybe not yet used here
        // Todo: Interpolate from midpoint to Gaussian
        fixed_point_iterate();
        error_norm = fem_error();
    }
    output_results();
}

double Manager::fem_error() {
    return 0;
}

void Manager::output_results() {
    macro_solver.output_results();
    micro_solver.output_results(); // todo: Fix to move all into one combined file.
}

void Manager::fixed_point_iterate() {
    macro_solver.run();
    micro_solver.run();
}

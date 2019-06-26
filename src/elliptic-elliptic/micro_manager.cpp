/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "micro_manager.h"

MicroManager::MicroManager(int micro_refinement) : macro_solver(), micro_solver() {
    macro_solver.set_refine_level(3);
    micro_solver.set_refine_level(micro_refinement);
    repetitions = 0;
    cycle = 0;
}

void MicroManager::setup() {
    // Create the grids and solution data structures for each grid
    macro_solver.setup();
    micro_solver.set_num_grids(macro_solver.triangulation.n_active_cells());
    micro_solver.setup();
    macro_solution = macro_solver.get_exact_solution();
    micro_solver.set_macro_boundary_condition(macro_solution);
    // Couple the macro structures with the micro structures.
    micro_solver.set_macro_solution(&macro_solution, &macro_solver.dof_handler); // Todo: Introduce smart pointers!
    // Now the dofhandler is not necessary of course
}

void MicroManager::run() {
    double old_residual = 0;
    double residual = 0;
    fixed_point_iterate();
    compute_residuals(old_residual, residual);
    printf("Residual %.2e\n", residual);
    cycle++;
    output_results();
}

void MicroManager::fixed_point_iterate() {
    micro_solver.run();
}

void MicroManager::compute_residuals(double &old_residual, double &residual) {
    double micro_l2 = 0;
    double micro_h1 = 0;
    micro_solver.compute_error(micro_l2, micro_h1);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", micro_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", micro_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("mH1", micro_h1);
    old_residual = residual;
    residual = micro_l2;
}

void MicroManager::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void MicroManager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "mH1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<DIMENSIONS> data_out;

    data_out.attach_dof_handler(micro_solver.dof_handler);
    data_out.add_data_vector(micro_solver.solutions.at(0), "solution");

    data_out.build_patches();

    std::ofstream output("results/micro_only-solution.gpl");
    data_out.write_gnuplot(output);
}
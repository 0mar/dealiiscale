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
    std::vector<Point<MACRO_DIMENSIONS>> dof_locations;
    macro_solver.get_dof_locations(dof_locations);
    micro_solver.set_grid_locations(dof_locations);
    micro_solver.setup();
    // Couple the macro structures with the micro structures.
    micro_solver.set_macro_solution(&macro_solver.solution, &macro_solver.dof_handler);
    macro_solver.set_micro_solutions(&micro_solver.solutions, &micro_solver.dof_handler);
//    std::vector<std::string> out_file_names = {"macro_vals.txt", "micro_vals.txt", "macro_convergence.txt",
//                                               "micro_convergence.txt"};
//    for (const std::string &out_file_name: out_file_names) {
//        std::ofstream ofs;
//        ofs.open("results/" + out_file_name, std::ofstream::out | std::ofstream::trunc);
//        ofs.close();
//    }
    cycle = 0;
}

void Manager::run() {
    double old_residual = 1;
    double residual = 0;
    while (std::fabs(old_residual - residual) > eps) {
        // Todo: Interpolate from midpoint to Gaussian
        fixed_point_iterate();
        compute_residuals(old_residual, residual);
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
        cycle++;
        if (cycle > max_iterations) {
            std::cout << "Can't get the residual small enough..." << std::endl;
            break;
        }
    }
    output_results();
}

void Manager::fixed_point_iterate() {
    macro_solver.run();
    micro_solver.run();
}

void Manager::compute_residuals(double &old_residual, double &residual) {
    double macro_l2 = 0;
    double macro_h1 = 0;
    double micro_l2 = 0;
    double micro_h1 = 0;
    macro_solver.compute_error(macro_l2, macro_h1);
    micro_solver.compute_error(micro_l2, micro_h1);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", macro_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", macro_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("mH1", micro_h1);
    convergence_table.add_value("ML2", macro_l2);
    convergence_table.add_value("MH1", macro_h1);
    old_residual = residual;
    residual = micro_l2 + macro_l2;
}

void Manager::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void Manager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "mH1", "ML2", "MH1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<MACRO_DIMENSIONS> data_out;

    data_out.attach_dof_handler(macro_solver.dof_handler);
    data_out.add_data_vector(macro_solver.solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/macro-solution.gpl");
    data_out.write_gnuplot(output);
}
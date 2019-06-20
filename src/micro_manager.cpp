/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "micro_manager.h"

MicroManager::MicroManager(int micro_refinement) : macro_solver(), micro_solver() {
    macro_solver.set_refine_level(3);
    micro_solver.set_refine_level(micro_refinement);
}

void MicroManager::setup() {
    // Create the grids and solution data structures for each grid
    macro_solver.setup();
    micro_solver.set_num_grids(macro_solver.triangulation.n_active_cells());
    micro_solver.setup();
    micro_solver.set_macro_boundary_condition(macro_solver.get_exact_solution());
    // Couple the macro structures with the micro structures.
    micro_solver.set_macro_solution(&macro_solver.interpolated_solution, &macro_solver.dof_handler);
    // Now the dofhandler is not necessary of course
//    std::vector<std::string> out_file_names = {"macro_vals.txt", "micro_vals.txt", "macro_convergence.txt",
//                                               "micro_convergence.txt"};
//    for (const std::string &out_file_name: out_file_names) {
//        std::ofstream ofs;
//        ofs.open("results/" + out_file_name, std::ofstream::out | std::ofstream::trunc);
//        ofs.close();
//    }
    cycle = 0;
}

void MicroManager::run() {
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

void MicroManager::fixed_point_iterate() {
    micro_solver.run();
}

void MicroManager::compute_residuals(double &old_residual, double &residual) {
    double micro_l2 = 0;
    double micro_h1 = 0;
    micro_solver.compute_error(micro_l2, micro_h1);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", macro_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", macro_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("mH1", micro_h1);
    old_residual = residual;
    residual = micro_l2;
}

void MicroManager::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void MicroManager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "mH1", "ML2", "MH1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<MACRO_DIMENSIONS> data_out;

    data_out.attach_dof_handler(micro_solver.dof_handler);
    data_out.add_data_vector(micro_solver.solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/micro_only-solution.gpl");
    data_out.write_gnuplot(output);
}
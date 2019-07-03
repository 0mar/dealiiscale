/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "time_manager.h"

TimeManager::TimeManager(int macro_refinement, int micro_refinement) :
        pi_solver(),
        rho_solver(),
        time_step(0.1),
        final_time(5) {
    pi_solver.set_refine_level(macro_refinement);
    rho_solver.set_refine_level(micro_refinement);
}

void TimeManager::setup() {
    // Create the grids and solution data structures for each grid
    pi_solver.setup();
    rho_solver.set_num_grids(pi_solver.dof_handler.n_dofs());
    Vector<double> init_condition;
    pi_solver.get_initial_condition(init_condition);
    rho_solver.set_initial_condition(init_condition);
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
    double old_residual = 1;
    double residual = 0;
    while (time < final_time) {
        compute_residuals(old_residual, residual);
        iterate();
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
        time += time_step;
        it++;
    }
    output_results();
}

void TimeManager::iterate() {
    pi_solver.iterate();
    rho_solver.iterate(time_step);
    write_plot();
}

void TimeManager::compute_residuals(double &old_residual, double &residual) {
    double macro_l2 = pi_solver.residual;
    double micro_l2 = rho_solver.residual;
    convergence_table.add_value("iteration", it);
    convergence_table.add_value("time", time);
    convergence_table.add_value("cells", pi_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", pi_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("ML2", macro_l2);
    old_residual = residual;
    residual = micro_l2 + macro_l2;
}

void TimeManager::write_plot() {
    {
        DataOut<MICRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(rho_solver.dof_handler);
        data_out.add_data_vector(rho_solver.solutions.at(0), "solution");
        data_out.build_patches();
        std::ofstream output("results/micro-solution" + Utilities::int_to_string(it, 3) + ".vtk");
        data_out.write_vtk(output);
    }
    {
        DataOut<MACRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(pi_solver.dof_handler);
        data_out.add_data_vector(pi_solver.solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/macro-solution" + Utilities::int_to_string(it, 3) + ".vtk");
        data_out.write_vtk(output);
    }
}

void TimeManager::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void TimeManager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "ML2"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<MACRO_DIMENSIONS> data_out;

    data_out.attach_dof_handler(pi_solver.dof_handler);
    data_out.add_data_vector(pi_solver.solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/macro-solution.gpl");
    data_out.write_gnuplot(output);
}
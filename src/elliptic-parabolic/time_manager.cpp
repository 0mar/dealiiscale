/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "time_manager.h"

TimeManager::TimeManager(unsigned int macro_refinement, unsigned int micro_refinement, const std::string &data_file,
                         const std::string &out_file) : data(data_file),
                                                        pi_solver(data.macro, macro_refinement),
                                                        rho_solver(data.micro, micro_refinement),
                                                        time_step(0.1),
                                                        final_time(5),
                                                        ct_file_name(out_file) {
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
    double old_residual = 1;
    double residual = 0;
    while (time < final_time) {
        compute_residuals(old_residual, residual);
        iterate();
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
        time += time_step; // todo: Update to it*dt
        it++;
        if (it==10) {
            printf("Storing patched micro and corresponding macro solutions at time %.2f\n",time);
            std::vector<Point<2>> locations;
            pi_solver.get_dof_locations(locations);
            rho_solver.patch_micro_solutions(locations);
            output_results();
        }
    }
    output_results();
//    pi_solver.write_solution_to_file(pi_solver.solution, pi_solver.dof_handler);
//    rho_solver.write_solutions_to_file(rho_solver.solutions, rho_solver.dof_handler);
}


void TimeManager::iterate() {
    data.set_time(time);
    pi_solver.iterate();
    rho_solver.iterate(time_step);
    write_plot();
}

void TimeManager::compute_residuals(double &old_residual, double &residual) {
    double macro_l2 = 0;
    double micro_l2 = 0;
    pi_solver.compute_error(macro_l2);
    rho_solver.compute_error(micro_l2);
    convergence_table.add_value("iteration", it);
    convergence_table.add_value("time", time);
    convergence_table.add_value("cells", pi_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", pi_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("ML2", macro_l2);
    const double res = residual;
    old_residual = res;
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

void TimeManager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "ML2"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output(ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<MACRO_DIMENSIONS> data_out;

    data_out.attach_dof_handler(pi_solver.dof_handler);
    data_out.add_data_vector(pi_solver.solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/final-macro-solution.gpl");
    data_out.write_gnuplot(output);
}
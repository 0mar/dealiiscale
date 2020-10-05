/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "manager.h"

Manager::Manager(unsigned int macro_refinement, unsigned int micro_refinement, const std::string &data_file,
                 const std::string &out_file) :
        data(data_file),
        macro_solver(data.macro, macro_refinement),
        micro_solver(data.micro, micro_refinement),
        parallel(true),
        ct_file_name(out_file) {
    printf("Running elliptic-elliptic solver with data from %s, storing results in %s\n", data_file.c_str(),
           out_file.c_str());
    MultithreadInfo::set_thread_limit(4);
    std::cout<<"Running on max " << MultithreadInfo::n_cores() << std::endl;
    if (MultithreadInfo::is_running_single_threaded()) {
        std::cout << "Running single threaded" << std::endl;
    }
}

void Manager::setup() {
    // Create the grids and solution data structures for each grid
    printf("Starting setup\n");
    macro_solver.setup();
    std::vector<Point<MACRO_DIMENSIONS>> dof_locations;
    macro_solver.get_dof_locations(dof_locations);
    printf("Finished macro setup\n");
    micro_solver.set_grid_locations(dof_locations);
    micro_solver.setup();
    printf("Finished micro setup\n");
    // Couple the macro structures with the micro structures.
    micro_solver.set_macro_solution(&macro_solver.sol_u, &macro_solver.sol_w, &macro_solver.dof_handler);
    macro_solver.set_micro_objects(micro_solver.fem_objects);
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
    double old_residual = 2;
    double residual = 1;
    double old_error = 2;
    double error = 1;
    while (std::fabs(old_error - error) > eps) {
        fixed_point_iterate();
//        compute_residuals(old_residual, residual);
        compute_errors(old_error, error);
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
        printf("Old error %.2e, new error %.2e\n", old_error, error);
        cycle++;
        if (cycle > max_iterations) {
            std::cout << "Can't get the residual small enough..." << std::endl;
            break;
        }
    }
    output_results();
}

void Manager::fixed_point_iterate() {
    macro_solver.assemble_and_solve();
    micro_solver.assemble_and_solve_all();
}

void Manager::compute_errors(double &old_error, double &error) {
    double macro_l2 = 0;
    double macro_h1 = 0;
    double micro_l2 = 0;
    double micro_h1 = 0;
    macro_solver.compute_error(macro_l2, macro_h1);
    micro_solver.compute_all_errors(micro_l2, micro_h1);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", macro_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", macro_solver.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("mH1", micro_h1);
    convergence_table.add_value("ML2", macro_l2);
    convergence_table.add_value("MH1", macro_h1);
    old_error = error;
    printf("Macro error: %.2e\tMicro error: %.2e\n", macro_l2, micro_l2);
    error = micro_l2 + macro_l2;
}

void Manager::compute_residuals(double &old_residual, double &residual) {
    double macro_residual = 0;
    double micro_residual = 0;
    macro_solver.compute_residual(macro_residual);
    micro_solver.compute_all_residuals(micro_residual);
    old_residual = residual;
    printf("Macro residual: %.2e\tMicro residual: %.2e\n", macro_residual, micro_residual);
    residual = macro_residual + micro_residual;
}

void Manager::output_results() {
    std::vector<std::string> error_classes = {"mL2", "mH1", "ML2", "MH1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output(ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    patch_and_write_solutions();
}

void Manager::patch_and_write_solutions() {
    {
        DataOut<MACRO_DIMENSIONS> macro_data_out;
        macro_data_out.attach_dof_handler(macro_solver.dof_handler);
        macro_data_out.add_data_vector(macro_solver.sol_u, "solution");
        macro_data_out.build_patches();
        std::ofstream macro_output("results/u-solution.gpl");
        macro_data_out.write_gnuplot(macro_output);
    }
    {
        DataOut<MACRO_DIMENSIONS> macro_data_out;
        macro_data_out.attach_dof_handler(macro_solver.dof_handler);
        macro_data_out.add_data_vector(macro_solver.sol_w, "solution");
        macro_data_out.build_patches();
        std::ofstream macro_output("results/w-solution.gpl");
        macro_data_out.write_gnuplot(macro_output);
    }
    {
        DataOut<MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro_solver.dof_handler);
        const unsigned int some_int = (int) (micro_solver.get_num_grids() / 2);
        micro_data_out.add_data_vector(micro_solver.solutions.at(some_int), "solution");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/v-computed.gpl");
        micro_data_out.write_gnuplot(micro_output);
    }
    {
        DataOut<MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro_solver.dof_handler);
        const unsigned int some_int = (int) (micro_solver.get_num_grids() / 2);
        Vector<double> error(micro_solver.dof_handler.n_dofs());
        error += micro_solver.solutions[some_int];
        micro_solver.set_exact_solution();
        error -= micro_solver.solutions[some_int];
        micro_data_out.add_data_vector(error, "solution");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-error.gpl");
        micro_data_out.write_gnuplot(micro_output);
    }
    {
        DataOut<MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro_solver.dof_handler);
        micro_solver.set_exact_solution(); // Superfluous but okay
        const unsigned int some_int = (int) (micro_solver.get_num_grids() / 2);
        micro_data_out.add_data_vector(micro_solver.solutions.at(some_int), "solution");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-exact.gpl");
        micro_data_out.write_gnuplot(micro_output);
    }

}

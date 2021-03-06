/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "manager.h"

Manager::Manager(unsigned int macro_refinement, unsigned int micro_refinement, const std::string &data_file,
                 const std::string &out_file) :
        data(data_file),
        macro_solver(data.macro, macro_refinement),
        micro_solver(data.micro, micro_refinement),
        ct_file_name(out_file) {
    printf("Running elliptic-elliptic solver with data from %s, storing results in %s\n", data_file.c_str(),
           out_file.c_str());
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
    double old_residual = 1;
    double residual = 0;
    while (std::fabs(old_residual - residual) > eps) { // Todo: make relative
        fixed_point_iterate();
        compute_errors(old_residual, residual);
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
//    macro_solver.solution = macro_solver.get_exact_solution();
    micro_solver.run();
//    micro_solver.set_exact_solution();
}

void Manager::compute_errors(double &old_residual, double &residual) {
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
    old_residual = residual;
    printf("Macro residual: %.2e\tMicro residual: %.2e\n", macro_l2, micro_l2);
    residual = micro_l2 + macro_l2;
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
    DataOut<MACRO_DIMENSIONS> macro_data_out;
    macro_data_out.attach_dof_handler(macro_solver.dof_handler);
    macro_data_out.add_data_vector(macro_solver.solution, "solution");
    macro_data_out.build_patches();
    std::ofstream macro_output("results/macro-solution.gpl");
    macro_data_out.write_gnuplot(macro_output);
    {
        DataOut<MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro_solver.dof_handler);
        const unsigned int some_int = (int) (micro_solver.get_num_grids() / 2);
        micro_data_out.add_data_vector(micro_solver.solutions.at(some_int), "solution");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-computed.gpl");
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

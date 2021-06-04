/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "manager.h"

Manager::Manager(unsigned int macro_h_inv, unsigned int micro_h_inv, unsigned int t_inv,
                 const std::string &data_file, const std::string &out_file) : data(data_file),
                                                                              macro(data.macro,
                                                                                    macro_h_inv),
                                                                              micro(data.micro,
                                                                                    micro_h_inv),
                                                                              final_time(0.5),
                                                                              ct_file_name(out_file) {
}

void Manager::setup() {
    // Create the grids and solution data structures for each grid
    macro.setup();
    std::vector<Point<MACRO_DIMENSIONS>> dof_locations;
    macro.get_dof_locations(dof_locations);
    micro.set_grid_locations(dof_locations); // todo: fix a small number of structures
    micro.setup();
    // Couple the macro structures with the micro structures.

    micro.set_macro_solutions(&macro.solution, &macro.solution,
                              &macro.dof_handler);
    macro.set_micro_solutions(&micro.solutions, &micro.dof_handler);
//    std::vector<std::string> out_file_names = {"macro_vals.txt", "micro_vals.txt", "macro_convergence.txt",
//                                               "micro_convergence.txt"};
//    for (const std::string &out_file_name: out_file_names) {
//        std::ofstream ofs;
//        ofs.open("results/" + out_file_name, std::ofstream::out | std::ofstream::trunc);
//        ofs.close();
//    }
    it = 0;
}

void Manager::run() {
    double old_residual = 1;
    double residual = 0;
    while (time < final_time) {
        time += data.params.get_double("dt");
        it++;
        printf("\nSolving for t = %f\n", time);
        iterate();
        compute_residuals(old_residual, residual);
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
//        if (it==10) {
//            printf("Storing patched micro and corresponding macro solutions at time %.2f\n",time);
//            std::vector<Point<2>> locations;
//            macro.get_dof_locations(locations);
//            micro.patch_micro_solutions(locations);
//            output_results();
//        }
        break;
    }
    output_results();
//    macro.write_solution_to_file(macro.solution, macro.dof_handler);
//    micro.write_solutions_to_file(micro.solutions, micro.dof_handler);
}


void Manager::iterate() {
    data.set_time(time);
    macro.iterate();
    micro.iterate();
}

void Manager::compute_residuals(double &old_residual, double &residual) {
    double macro_l2 = 0;
    double macro_h1 = 0;
    double micro_l2 = 0;
    double micro_h1 = 0;
    macro.compute_error(macro_l2, macro_h1);
    micro.compute_error(micro_l2, micro_h1);
    convergence_table.add_value("iteration", it);
    convergence_table.add_value("time", time);
    convergence_table.add_value("cells", macro.triangulation.n_active_cells());
    convergence_table.add_value("dofs", macro.dof_handler.n_dofs());
    convergence_table.add_value("mL2", micro_l2);
    convergence_table.add_value("ML2", macro_l2);
    convergence_table.add_value("mH1", micro_h1);
    convergence_table.add_value("MH1", macro_h1);
    const double res = residual;
    old_residual = res;
    residual = micro_l2 + macro_l2;
}

void Manager::write_plot(double time) {
    {
        DataOut<MICRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(micro.dof_handler);
        data_out.add_data_vector(micro.solutions.at(0), "solution");
        data_out.build_patches();
        std::ofstream output("results/micro-solution" + Utilities::int_to_string(it, 3) + ".vtk");
        data_out.write_vtk(output);
    }
    {
        DataOut<MACRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(macro.dof_handler);
        data_out.add_data_vector(macro.solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/macro-solution" + Utilities::int_to_string(it, 3) + ".vtk");
        data_out.write_vtk(output);
    }
}

void Manager::output_results() {
    if (false) {
        std::vector<std::string> error_classes = {"mL2", "mH1", "ML2", "MH1"};
        for (const std::string &error_class: error_classes) {
            convergence_table.set_precision(error_class, 3);
            convergence_table.set_scientific(error_class, true);
        }
        std::ofstream convergence_output(ct_file_name, std::iostream::app);
        convergence_table.write_text(convergence_output);
        convergence_output.close();
    }
    patch_and_write_solutions();
}

void Manager::patch_and_write_solutions() {
    {
        DataOut<MACRO_DIMENSIONS> macro_data_out;
        macro_data_out.attach_dof_handler(macro.dof_handler);
        macro_data_out.add_data_vector(macro.solution, "u");
        macro_data_out.build_patches();
        std::ofstream macro_output("results/u-solution.vtk");
        macro_data_out.write_vtk(macro_output);
    }
    write_micro_grid_locations("results/micro-solutions/grid_locations.txt");
    for (unsigned int grid_num = 0; grid_num < micro.get_num_grids(); grid_num++) {
        DataOut<MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro.dof_handler);
        micro_data_out.add_data_vector(micro.solutions.at(grid_num), "v");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-solutions/v-solution-" + std::to_string(grid_num) + ".vtk");
        micro_data_out.write_vtk(micro_output);
    }
    if (false) {
        {
            DataOut<MICRO_DIMENSIONS> micro_data_out;
            micro_data_out.attach_dof_handler(micro.dof_handler);
            const unsigned int some_int = (int) (micro.get_num_grids() / 2);
            Vector<double> error_(micro.dof_handler.n_dofs());
            error_ += micro.solutions[some_int];
            micro.set_exact_solution();
            error_ -= micro.solutions[some_int];
            micro_data_out.add_data_vector(error_, "v");
            micro_data_out.build_patches();
            std::ofstream micro_output("results/micro-error.vtk");
            micro_data_out.write_vtk(micro_output);
        }
    }
    if (false) {
        {
            DataOut<MICRO_DIMENSIONS> micro_data_out;
            micro_data_out.attach_dof_handler(micro.dof_handler);
            micro.set_exact_solution(); // Superfluous but okay
            const unsigned int some_int = (int) (micro.get_num_grids() / 2);
            micro_data_out.add_data_vector(micro.solutions.at(some_int), "v");
            micro_data_out.build_patches();
            std::ofstream micro_output("results/micro-exact.vtk");
            micro_data_out.write_vtk(micro_output);
        }
    }
}

void Manager::write_micro_grid_locations(const std::string &filename) {
    std::cout << "Writing grid locations to file" << std::endl;
    std::ofstream loc_file;
    loc_file.open(filename);
    auto grid_locations = std::vector<Point<MACRO_DIMENSIONS>>();
    macro.get_dof_locations(grid_locations);
    for (unsigned int grid_num = 0; grid_num < micro.get_num_grids(); grid_num++) {
        for (unsigned int dim = 0; dim < MACRO_DIMENSIONS; dim++) {
            loc_file << grid_locations[grid_num][dim] << " ";
        }
        loc_file << std::endl;
    }
    loc_file << "\n";
    loc_file.close();

}

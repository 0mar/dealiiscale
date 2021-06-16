/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "manager.h"

Manager::Manager(unsigned int macro_h_inv, unsigned int micro_h_inv,
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
    std::vector <Point<MACRO_DIMENSIONS>>
            dof_locations;
    macro.get_dof_locations(dof_locations);
    micro.set_grid_locations(dof_locations, macro.micro_indicator);
    micro.setup();
    // Couple the macro structures with the micro structures.

    micro.set_macro_solutions(&macro.solution, &macro.solution,
                              &macro.dof_handler);
    macro.set_micro_solutions(&micro.solutions, &micro.dof_handler);
    std::vector <std::string> out_file_names = {"w-color.txt"};
    for (const std::string &out_file_name: out_file_names) {
        std::ofstream ofs;
        ofs.open("results/" + out_file_name, std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }
    it = 0;
}

void Manager::run() {
    double old_residual = -1;
    double residual = 0;
    while (time < final_time) {
        time += data.params.get_double("dt");
        it++;
        printf("\nSolving for t = %f\n", time);
        iterate();
        compute_residuals(old_residual, residual);
        printf("Old residual %.2e, new residual %.2e\n", old_residual, residual);
        if (time > 0.5) {
            output_results();
            exit(0);
        }
    }
    output_results();
}


void Manager::iterate() {
    write_plot();
    data.set_time(time);
    macro.iterate();
    micro.iterate();
}

void Manager::compute_residuals(double &old_residual, double &residual) {
    double macro_residual = 0;
    double micro_residual = 0;
    printf("Macro residual: %.2f\nMicro residual: %.2f", macro_residual, micro_residual);
    macro.compute_residual(macro_residual);
    micro.compute_all_residuals(micro_residual);
    const double res = residual;
    old_residual = res;
    residual = macro_residual + micro_residual;
    convergence_table.add_value("iteration", it);
    convergence_table.add_value("time", time);
    convergence_table.add_value("residual", residual);
}

void Manager::write_plot() {
    for (unsigned int k: micro.grid_indicator) {
        DataOut <MICRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(micro.dof_handler);
        data_out.add_data_vector(micro.solutions.at(k), "v");
        data_out.build_patches();
        std::ofstream output(
                "results/v(" + Utilities::int_to_string(k) + ")-solution-slice" + Utilities::int_to_string(it, 3) +
                ".vtk");
        data_out.write_vtk(output);
    }
    for (unsigned int k: micro.grid_indicator) {
        DataOut <MICRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(micro.dof_handler);
        data_out.add_data_vector(micro.solutions_w.at(k), "w");
        data_out.build_patches();
        std::ofstream output(
                "results/w(" + Utilities::int_to_string(k) + ")-solution-slice" + Utilities::int_to_string(it, 3) +
                ".vtk");
        data_out.write_vtk(output);
    }
    {
        DataOut <MACRO_DIMENSIONS> data_out;
        data_out.attach_dof_handler(macro.dof_handler);
        data_out.add_data_vector(macro.solution, "solution");
        data_out.build_patches();
        std::ofstream output("results/macro-solution-slice" + Utilities::int_to_string(it, 3) + ".vtk");
        data_out.write_vtk(output);
    }
    {
        Vector<double> color(micro.get_num_grids());
        micro.get_color(color);
        std::ofstream output("results/w-color.txt", std::ofstream::app);
        output << color(0);
        for (unsigned int k: micro.grid_indicator) {
            output << "," << color(k);
        }
        output << std::endl;
        output.close();
    }
}

void Manager::output_results() {
    patch_and_write_solutions();
}

void Manager::patch_and_write_solutions() {
    {
        DataOut <MACRO_DIMENSIONS> macro_data_out;
        macro_data_out.attach_dof_handler(macro.dof_handler);
        macro_data_out.add_data_vector(macro.solution, "u");
        macro_data_out.build_patches();
        std::ofstream macro_output("results/u-solution.vtk");
        macro_data_out.write_vtk(macro_output);
    }
    write_micro_grid_locations("results/micro-solutions/grid_locations.txt");
    for (unsigned int grid_num = 0; grid_num < micro.get_num_grids(); grid_num++) {
        DataOut <MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro.dof_handler);
        micro_data_out.add_data_vector(micro.solutions.at(grid_num), "v");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-solutions/v-solution-" + std::to_string(grid_num) + ".vtk");
        micro_data_out.write_vtk(micro_output);
    }
    for (unsigned int grid_num = 0; grid_num < micro.get_num_grids(); grid_num++) {
        DataOut <MICRO_DIMENSIONS> micro_data_out;
        micro_data_out.attach_dof_handler(micro.dof_handler);
        micro_data_out.add_data_vector(micro.solutions_w.at(grid_num), "w");
        micro_data_out.build_patches();
        std::ofstream micro_output("results/micro-solutions/w-solution-" + std::to_string(grid_num) + ".vtk");
        micro_data_out.write_vtk(micro_output);
    }
}

void Manager::write_micro_grid_locations(const std::string &filename) {
    std::cout << "Writing grid locations to file" << std::endl;
    std::ofstream loc_file;
    loc_file.open(filename);
    auto grid_locations = std::vector < Point < MACRO_DIMENSIONS >> ();
    macro.get_dof_locations(grid_locations);
    for (unsigned int k: micro.grid_indicator) {
        loc_file << k << " ";
        for (unsigned int dim = 0; dim < MACRO_DIMENSIONS; dim++) {
            loc_file << grid_locations[k][dim] << " ";
        }
        loc_file << std::endl;
    }
    loc_file << "\n";
    loc_file.close();

}

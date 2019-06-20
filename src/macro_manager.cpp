/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include "macro_manager.h"

MacroManager::MacroManager(int macro_refinement) : macro_solver() {
    macro_solver.set_refine_level(macro_refinement);
    repetitions = 0;
    cycle = 0;
}

void MacroManager::setup() {
    // Create the grids and solution data structures for each grid
    macro_solver.setup();
    compute_virtual_micros(micro_solutions);
    macro_solver.set_micro_solutions(&micro_solutions, &macro_solver.dof_handler);
}

void MacroManager::run() {
    double old_residual = 0;
    double residual = 0;
    fixed_point_iterate();
    compute_residuals(old_residual, residual);
    printf("Residual %.2e\n", residual);
    cycle++;
    output_results();
}

void MacroManager::compute_virtual_micros(std::vector<Vector<double>> &virtual_micros) {
    virtual_micros.clear();
    Vector<double> values = macro_solver.get_exact_solution(); // Is this possible?
    Vector<double> micros(macro_solver.dof_handler.n_dofs()); // Todo: generalize for other manufactured ones.
    for (unsigned int i = 0; i < macro_solver.triangulation.n_active_cells(); i++) {
        micros = 2. / 3. * values(i);
        virtual_micros.push_back(micros);
    }
}

void MacroManager::fixed_point_iterate() {
    macro_solver.run();
}

void MacroManager::compute_residuals(double &old_residual, double &residual) {
    double macro_l2 = 0;
    double macro_h1 = 0;
    macro_solver.compute_error(macro_l2, macro_h1);
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", macro_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", macro_solver.dof_handler.n_dofs());
    convergence_table.add_value("ML2", macro_l2);
    convergence_table.add_value("MH1", macro_h1);
    old_residual = residual;
    residual = macro_l2;
}

void MacroManager::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void MacroManager::output_results() {
    std::vector<std::string> error_classes = {"ML2", "MH1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    DataOut<DIMENSIONS> data_out;

    data_out.attach_dof_handler(macro_solver.dof_handler);
    data_out.add_data_vector(macro_solver.solution, "solution");

    data_out.build_patches();

    std::ofstream output("results/micro_only-solution.gpl");
    data_out.write_gnuplot(output);
}
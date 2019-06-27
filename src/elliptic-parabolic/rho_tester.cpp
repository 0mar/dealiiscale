/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#ifndef DEALIISCALE_RHO_TESTER_H
#define DEALIISCALE_RHO_TESTER_H

#include "pi_solver.h"
#include "rho_solver.h"


class RhoTester {

public:
    const static int MICRO_DIMENSIONS = 2;
    const int macro_num = 3;
    RhoSolver<MICRO_DIMENSIONS> rho_solver;
    double time = 0;
    double time_step = 0.1;
    double final_time = 8;

    /**
     * Class that facilitates the interaction between the microscopic and macroscopic solvers.
     * @param macro_refinement Resolution of the macro solver.
     * @param micro_refinement Resolution of the micro solver.
     */
    RhoTester(int macro_refinement, int micro_refinement);

    /**
     * Run all the methods that setup the solvers of the two scales.
     */
    void setup();

    /**
     * Run iterations of the microscopic solvers until the result (of a single time step) is sufficiently close
     * to the solution.
     */
    void run();

    /**
     * Print the error computations/estimates in a convergence table.
     */
    void output_results();

    /**
     * Set a custom name for the file containing the convergence table.
     * @param file_name Name of the convergence table file.
     */
    void set_ct_file_name(std::string &file_name);

    double eps = 1E-4;
    double max_iterations = 1E4;
    Vector<double> virtual_pi;
    int it = 0; // Todo: Put all in intializer list


private:
    int cycle;

    /**
     * One iteration in time. The multiscale system is operator-splitted into two single-scale problems.
     */
    void iterate();

    /**
     * Compute the multiscale residual by adding the macroscopic and the microscopic error.
     * Analysis shows that this is bounded.
     * @param old_residual The residual in the previous operator splitting iteration.
     * @param residual The residual in the current operator splitting iteration.
     */
    void compute_residuals(double &old_residual, double &residual);

    std::string ct_file_name = "rho-only-convergence.txt";
    ConvergenceTable convergence_table;

};


RhoTester::RhoTester(int macro_refinement, int micro_refinement) : rho_solver() {
    rho_solver.set_refine_level(micro_refinement);
    virtual_pi.reinit(macro_num);
    for (int i = 0; i < macro_num; i++) {
        virtual_pi(i) = i;
    }
}

void RhoTester::setup() {
    // Create the grids and solution data structures for each grid
    rho_solver.set_num_grids(3);
    rho_solver.setup();
    // Couple the macro structures with the micro structures.
    rho_solver.set_macro_solutions(&virtual_pi, &virtual_pi, nullptr);
    // Now the dofhandler is not necessary of course
    cycle = 0;
}

void RhoTester::run() {
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

void RhoTester::iterate() {
    rho_solver.iterate(time_step);
}

void RhoTester::compute_residuals(double &old_residual, double &residual) {
    double micro_l2 = 0;
    double micro_h1 = 0;
    convergence_table.add_value("time", time);
    convergence_table.add_value("cells", rho_solver.triangulation.n_active_cells());
    convergence_table.add_value("dofs", rho_solver.dof_handler.n_dofs());
    convergence_table.add_value("L2", micro_l2);
    convergence_table.add_value("H1", micro_h1);

    old_residual = residual;
    residual = micro_l2;
    DataOut<MICRO_DIMENSIONS> data_out;
    data_out.attach_dof_handler(rho_solver.dof_handler);
    data_out.add_data_vector(rho_solver.solutions.at(0), "solution");
    data_out.build_patches();
    std::ofstream output("results/micro-solution" + Utilities::int_to_string(it, 3) + ".vtk");
    data_out.write_vtk(output);
}

void RhoTester::set_ct_file_name(std::string &file_name) {
    ct_file_name = file_name;

}

void RhoTester::output_results() {
    std::vector<std::string> error_classes = {"L2", "H1"};
    for (const std::string &error_class: error_classes) {
        convergence_table.set_precision(error_class, 3);
        convergence_table.set_scientific(error_class, true);
    }
    std::ofstream convergence_output("results/" + ct_file_name, std::iostream::app);
    convergence_table.write_text(convergence_output);
    convergence_output.close();
    std::cout << " debug " << rho_solver.solutions.at(0)(0) << std::endl;
}

int main() {
    RhoTester rho_tester(2, 4);
    rho_tester.setup();
    rho_tester.run();
    return 0;
}

#endif //DEALIISCALE_RHO_TESTER_H
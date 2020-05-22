/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include <regex>
#include "time_manager.h"
#include <cmath>
#include <cstdio>

/**
 * Run that solver
 * @return 0
 */

void run(const std::string &input_path) {
    const std::regex r("input/(.+).prm");
    std::smatch m;
    std::regex_search(input_path,m,r);
    const std::string id = m[1];
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (int i = 0; i < 7; i++) {
        auto macro_h_inv = (unsigned int) std::round(8 * std::pow(2, i / 2.));
        auto micro_h_inv = (unsigned int) std::round(8 * std::pow(2, i / 2.));
        auto t_inv = (unsigned int) std::round(4 * std::pow(2, i));
        TimeManager manager(macro_h_inv, micro_h_inv, t_inv, input_path, output_path);
        manager.setup();
        manager.run();
    }
}

void plot(const std::string &input_path) {
    const std::regex r("input/(.+).prm");
    std::smatch m;
    std::regex_search(input_path,m,r);
    const std::string id = m[1];
    const std::string output_path = "/dev/null";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    {
        unsigned int macro_h_inv = 4;
        unsigned int micro_h_inv = 16;
        unsigned int t_inv = 16;
        TimeManager manager(macro_h_inv, micro_h_inv, t_inv, input_path, output_path);
        manager.setup();
        manager.run();
        const int succeeded = std::rename("results/patched_micro_solution.gpl", "results/patched_plot.gpl");
        printf("Moving micro plot. Succeeded = %d\n", succeeded);
    }
    {
        unsigned int macro_h_inv = 4;
        unsigned int micro_h_inv = 32;
        unsigned int t_inv = 16;
        TimeManager manager(macro_h_inv, micro_h_inv, t_inv, input_path, output_path);
        manager.setup();
        manager.run();
        const int succeeded = std::rename("results/final_micro_solution.gpl", "results/micro_plot.gpl");
        printf("Moving micro plot. Succeeded = %d\n", succeeded);
    }
    {
        unsigned int macro_h_inv = 32;
        unsigned int micro_h_inv = 32;
        unsigned int t_inv = 16;
        TimeManager manager(macro_h_inv, micro_h_inv, t_inv, input_path, output_path);
        manager.setup();
        manager.run();
        const int succeeded = std::rename("results/final_macro_solution.gpl", "results/macro_plot.gpl");
        printf("Moving macro plot. Succeeded = %d\n", succeeded);
    }

}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    const std::string input_path = "input/full_nonlinear.prm";
    if (argc == 2) {
        input_path = argv[1];
    } else if (argc > 2) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    if (input_path == "input/paper_plot.prm") {
        plot(input_path);
    } else {
        run(input_path);
    }

    return 0;
}
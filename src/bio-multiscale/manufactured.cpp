/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "manager.h"
#include <regex>
#include <deal.II/base/timer.h>

/**
 * Run that solver
 * @return 0
 */


std::string get_id(const std::string &path) {
    const std::regex r("input/(.+).prm");
    std::smatch m;
    std::regex_search(path, m, r);
    std::string id = m[1];
    return id;
}

void conv_test(const std::string &input_path, int num_threads) {
    const std::string id = get_id(input_path);
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 0; i < 5; i++) {
        auto macro_refinement = (unsigned int) std::round(8 * std::pow(2, i / 2.));
        auto micro_refinement = (unsigned int) std::round(8 * std::pow(2, i / 2.));
        Manager manager(macro_refinement, micro_refinement, input_path, output_path, num_threads);
        manager.setup();
        manager.run();
    }
}

void run(const std::string &input_path, int macro_refinement, int micro_refinement, int num_threads) {
    dealii::Timer timer;
    timer.start();
    const std::string output_path = "results/out.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    Manager manager(macro_refinement, micro_refinement, input_path, output_path, num_threads);
    manager.setup();
    manager.run();
    timer.stop();
    printf("Results: 'macro_refinement', 'micro_refinement', 'num_threads', 'wall_time', 'cpu_time'\n");
    printf("%d, %d, %d, %.3f, %.3f\n", macro_refinement, micro_refinement, num_threads, timer.wall_time(), timer.cpu_time());
    if (num_threads != 0) {
        const std::string id = get_id(input_path);
        const std::string time_path = "results/" + id + "_timings.txt";
        std::cout << "Storing timing results in " << time_path << std::endl;
        std::ofstream ofs;
        ofs.open(time_path, std::ofstream::app);
        ofs << num_threads << "\t" << timer.wall_time() << "\t" << timer.cpu_time() << std::endl;
        ofs.close();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string input_path = "input/linear.prm";
    int macro_refinement = 2;
    int micro_refinement = 4;
    int num_threads = 0;
    if (argc >= 2) {
        input_path = argv[1];
    }
    if (argc >= 3) {
        macro_refinement = std::stoi(argv[2]);
    }
    if (argc >= 4) {
        micro_refinement = std::stoi(argv[3]);
    }
    if (argc >= 5) {
        num_threads = std::stoi(argv[4]);
    } else if (argc > 5) {
        std::cout << "Too many arguments" << std::endl
                  << "Supply (1) input file, (2) macro refinement, (3) micro refinement, (4) number of cores"
                  << std::endl;
        return 1;
    }
    conv_test(input_path, num_threads);
    return 0;
}

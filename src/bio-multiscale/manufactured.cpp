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

void run(const std::string &input_path, int num_threads) {
    const std::string id = get_id(input_path);
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 4; i < 5; i++) {
        Manager manager(i - 1, i, input_path, output_path, num_threads);
        manager.setup();
        manager.run();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string input_path = "input/linear.prm";
    int num_threads = 0;
    if (argc >= 2) {
        input_path = argv[1];
    }
    if (argc >= 3) {
        num_threads = std::stoi(argv[2]);
    } else if (argc > 3) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    dealii::Timer timer;
    timer.start();
    run(input_path, num_threads);
    timer.stop();
    printf("Ran in %.2f seconds with %.2f CPU time\n", timer.wall_time(), timer.cpu_time());
    if (num_threads != 0) {
        const std::string id = get_id(input_path);
        const std::string time_path = "results/" + id + "_timings.txt";
        std::cout << "Storing timing results in " << time_path << std::endl;
        std::ofstream ofs;
        ofs.open(time_path, std::ofstream::out);
        ofs << num_threads << "\t" << timer.wall_time() << "\t" << timer.cpu_time() << std::endl;
        ofs.close();
    }
    return 0;
}

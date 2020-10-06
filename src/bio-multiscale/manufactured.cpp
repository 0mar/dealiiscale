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

void run(const std::string &input_path) {
    const std::regex r("input/(.+).prm");
    std::smatch m;
    std::regex_search(input_path, m, r);
    const std::string id = m[1];
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 4; i < 5; i++) {
        Manager manager(i - 1, i, input_path, output_path);
        manager.setup();
        manager.run();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string id = "input/linear.prm";
    if (argc == 2) {
        id = argv[1];
    } else if (argc > 2) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    dealii::Timer timer;
    timer.start();
    run(id);
    timer.stop();
    printf("Ran in %.2f seconds with %.2f CPU time\n", timer.wall_time(), timer.cpu_time());
    return 0;
}

/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "manager.h"
#include <cmath>

/**
 * Run that solver
 * @return 0
 */

void run(const std::string &id) {
    const std::string input_path = "input/" + id + ".prm";
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 7; i++) {
        Manager manager(i, i, input_path, output_path);
        manager.setup();
        manager.run();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string id = "map_test";
    if (argc == 2) {
        id = argv[1];
    } else if (argc > 2) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    run(id);
    return 0;
}
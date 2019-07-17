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

void run(std::string id) {
    std::string file_name = "multi-convergence.txt";
    std::ofstream ofs;
    ofs.open("results/" + id + file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 5; i++) {
        Manager manager(i, i, "input/" + id + ".prm");
        manager.set_ct_file_name(file_name);
        manager.setup();
        manager.run();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string test = "test";
    if (argc == 2) {
        test = argv[1];
    } else if (argc > 2) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    run(test);
    return 0;
}
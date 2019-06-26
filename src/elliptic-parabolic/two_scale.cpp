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
int main() {
    dealii::deallog.depth_console(0);
    std::cout << "Warning: Not yet correctly implemented!" << std::endl;
    std::string file_name = "two-scale-convergence.txt";
    std::ofstream ofs;
    ofs.open("results/" + file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 8; i++) {
        Manager manager(i, i);
        manager.set_ct_file_name(file_name);
        manager.setup();
        manager.run();
    }
    return 0;
}
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
    std::string file_name = "two-scale-convergence.txt";
    std::ofstream ofs;
    ofs.open("results/" + file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    Manager manager(4, 4);
    manager.set_ct_file_name(file_name);
    manager.setup();
    manager.run();
    manager.write_plot();
    return 0;
}
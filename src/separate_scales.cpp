/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "manager.h"
#include "micro_manager.h"
#include <cmath>


void micro_only() {
    dealii::deallog.depth_console(0);
    std::string file_name = "micro-convergence.txt";
    std::ofstream ofs;
    ofs.open("results/" + file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 10; i++) {
        MicroManager manager(i);
        manager.set_ct_file_name(file_name);
        manager.setup();
        manager.run();
    }
}


/**
 * Run those solvers
 * @return 0
 */
int main() {
    dealii::deallog.depth_console(0);
    micro_only();
    return 0;
}
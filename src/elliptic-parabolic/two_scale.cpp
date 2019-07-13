/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "time_manager.h"
#include <cmath>

/**
 * Run a multiscale elliptic-parabolic solver
 * @return 0
 */
int main(int argc, char *argv[]) {
    int micro_refine = 3;
    int macro_refine = 3;
    if (argc >= 2) {
        macro_refine = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        micro_refine = std::stoi(argv[2]);
    }
    dealii::deallog.depth_console(0);
    std::string file_name = "two-scale-convergence.txt";
    std::ofstream ofs;
    ofs.open("results/" + file_name, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    TimeManager manager(macro_refine, micro_refine);
    manager.set_ct_file_name(file_name);
    manager.setup();
    manager.run();
    return 0;
}

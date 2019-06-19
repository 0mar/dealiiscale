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
    for (unsigned int i = 2; i < 5; i++) {
        Manager manager(i, i);
        manager.setup();
        manager.run();
        manager.output_results();
    }
    return 0;
}
/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "manager.h"


/**
 * Run that solver
 * @return 0
 */
int main() {
    dealii::deallog.depth_console(0);
    Manager manager(4, 4);
    manager.setup();
    manager.run();
    manager.output_results();
    return 0;
}


#include <deal.II/base/logstream.h>
#include "macro.h"


/**
 * Run that solver
 * @return 0
 */
int main() {
    dealii::deallog.depth_console(0);
    for (int i = 2; i < 10; i++) {
        MacroSolver<2> macro(i);
    }
    return 0;
}
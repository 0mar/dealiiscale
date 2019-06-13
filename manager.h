//
// Created by omar on 6/13/19.
//

#ifndef DEALIISCALE_MANAGER_H
#define DEALIISCALE_MANAGER_H

#include "manufactured.h"

class Manager {


};


/**
 * Run that solver
 * @return 0
 */
int main() {
    deallog.depth_console(0);
    for (int i = 2; i < 10; i++) {
        MacroSolver<2> macro(i);
    }
    return 0;
}


#endif //DEALIISCALE_MANAGER_H

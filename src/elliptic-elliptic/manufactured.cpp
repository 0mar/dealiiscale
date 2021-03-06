/*
 * Author: Omar Richardson, Karlstad University, 2019
 */

#include <deal.II/base/logstream.h>
#include "manager.h"
#include <regex>
/**
 * Run that solver
 * @return 0
 */

void run(const std::string &input_path) {
    const std::regex r("input/(.+).prm");
    std::smatch m;
    std::regex_search(input_path,m,r);
    const std::string id = m[1];
    const std::string output_path = "results/" + id + "_" + "convergence_table.txt";
    std::ofstream ofs;
    ofs.open(output_path, std::ofstream::out | std::ofstream::trunc);
    ofs.close();
    for (unsigned int i = 2; i < 6; i++) {
        Manager manager(i-1, i, input_path, output_path);
        manager.setup();
        manager.run();
    }
}

int main(int argc, char *argv[]) {
    dealii::deallog.depth_console(0);
    std::string id = "input/map_test.prm";
    if (argc == 2) {
        id = argv[1];
    } else if (argc > 2) {
        std::cout << "Too many arguments" << std::endl;
        return 1;
    }
    run(id);
    return 0;
}
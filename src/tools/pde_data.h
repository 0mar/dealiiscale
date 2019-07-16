/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef PDE_DATA_H
#define PDE_DATA_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <string>
#include <stdlib.h>
#include "multiscale_function_parser.h"

using namespace dealii;

template<int dim>
struct MacroData {
    FunctionParser<dim> solution;
    FunctionParser<dim> rhs;
    FunctionParser<dim> bc;
    ParameterHandler &params;

    MacroData(ParameterHandler &params) : solution(), rhs(), bc(), params(params) {
        // Needed because this is a reference
    }
};

template<int dim>
struct MicroData {
    MultiscaleFunctionParser<dim> solution;
    MultiscaleFunctionParser<dim> rhs;
    MultiscaleFunctionParser<dim> bc;
    ParameterHandler &params;

    MicroData(ParameterHandler &params) : solution(), rhs(), bc(), params(params) {
        // Needed because this is a reference
    }
};

template<int dim>
class MultiscaleData {
public:
    MultiscaleData(const std::string &param_file);

    ParameterHandler params;
    MacroData<dim> macro;
    MicroData<dim> micro;
};


#endif
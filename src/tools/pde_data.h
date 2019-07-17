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

/**
 * Struct containing all the macroscopic functions and parameters
 * @tparam dim dimension of functions
 */
template<int dim>
struct MacroData {
    FunctionParser<dim> solution;
    FunctionParser<dim> rhs;
    FunctionParser<dim> bc;
    ParameterHandler &params;

    /**
     * Initialize struct with parameter object
     * @param params parameterhandler object
     */
    MacroData(ParameterHandler &params) : solution(), rhs(), bc(), params(params) {
        // Needed because this is a reference
    }
};

/**
 * Struct containing all multiscale functions and parameters
 * @tparam dim dimension of the multiscale functions.
 * Caveat: currently, due to the implementation of MultiscaleFunctionParser,
 * the dimensions of the macroscale and microscale must be equal.
 */
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

/**
 * Multiscale data class, composed of a macrodata and microdata struct.
 * This data class reads data from a file and initialized the corresponding structs with the relevant parameters.
 * @tparam dim dimensions of functions.
 */
template<int dim>
class MultiscaleData {
public:
    MultiscaleData(const std::string &param_file);

    ParameterHandler params;
    MacroData<dim> macro;
    MicroData<dim> micro;

    const std::string multiscale_variables();

    const std::string macro_variables();
};

template<int dim>
const std::string MultiscaleData<dim>::multiscale_variables() {
    switch (dim) {
        case 1:
            return "x0,y0";
        case 2:
            return "x0,x1,y0,y1";
        case 3:
            return "x0,x1,x2,y0,y1,y2,";
        default: Assert(false, ExcNotImplemented())
    }
    return "";
}

template<int dim>
const std::string MultiscaleData<dim>::macro_variables() {
    switch (dim) {
        case 1:
            return "x0";
        case 2:
            return "x0,x1";
        case 3:
            return "x0,x1,x2";
        default: Assert(false, ExcNotImplemented())
    }
    return "";
}


#endif
/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef PDE_DATA_H
#define PDE_DATA_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <string>
#include <cstdlib>
#include "multiscale_function_parser.h"
#include "mapping.h"
#include <memory>

using namespace dealii;

template<int dim>
struct MicroFEMObjects { // Can be moved to a different file if imports would give trouble at some point
    const std::vector<Vector<double>> *solutions;
    const DoFHandler<dim> *dof_handler;
    const MapMap<dim, dim> *mapmap;
    const unsigned int *q_degree;
};

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
struct MicroData { // todo: Rename to EllipticMicroData
    MicroData(ParameterHandler &params) : solution(), rhs(), bc(), mapping(dim), map_jac(dim * dim), params(params) {}

    MultiscaleFunctionParser<dim> solution;
    MultiscaleFunctionParser<dim> rhs;
    MultiscaleFunctionParser<dim> bc;
    MultiscaleFunctionParser<dim> mapping;
    MultiscaleFunctionParser<dim> map_jac;
    ParameterHandler &params;

};

template<int dim>
struct ParabolicMicroData {
    MultiscaleFunctionParser<dim> solution;
    MultiscaleFunctionParser<dim> rhs;
    MultiscaleFunctionParser<dim> neumann_bc;
    MultiscaleFunctionParser<dim> robin_bc;
    MultiscaleFunctionParser<dim> init_rho;
    ParameterHandler &params;

    ParabolicMicroData(ParameterHandler &params)
            : solution(), rhs(), neumann_bc(), robin_bc(), init_rho(), params(params) {
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

    static std::string multiscale_variables();

    static std::string macro_variables();
};

template<int dim>
class TwoPressureData {
public:
    TwoPressureData(const std::string &param_file);

    ParameterHandler params;
    MacroData<dim> macro;
    ParabolicMicroData<dim> micro;

    void set_time(const double time);

    static std::string multiscale_variables();

    static std::string macro_variables();
};

template<int dim>
std::string MultiscaleData<dim>::multiscale_variables() {
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
std::string MultiscaleData<dim>::macro_variables() {
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


template<int dim>
std::string TwoPressureData<dim>::multiscale_variables() {
    switch (dim) {
        case 1:
            return "x0,y0,t";
        case 2:
            return "x0,x1,y0,y1,t";
        case 3:
            return "x0,x1,x2,y0,y1,y2,t";
        default: Assert(false, ExcNotImplemented())
    }
    return "";
}

template<int dim>
std::string TwoPressureData<dim>::macro_variables() {
    switch (dim) {
        case 1:
            return "x0,t";
        case 2:
            return "x0,x1,t";
        case 3:
            return "x0,x1,x2,t";
        default: Assert(false, ExcNotImplemented())
    }
    return "";
}


#endif

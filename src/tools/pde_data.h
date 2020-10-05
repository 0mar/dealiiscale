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


/**
 * Struct containing all the macroscopic functions and parameters for the biomath problem
 * @tparam dim dimension of functions
 */
template<int dim>
struct BioMacroData {
    FunctionParser<dim> solution_u;
    FunctionParser<dim> bulk_rhs_u;
    FunctionParser<dim> bc_u_1;
    FunctionParser<dim> bc_u_2;
    FunctionParser<dim> inflow_measure;
    FunctionParser<dim> outflow_measure;

    FunctionParser<dim> solution_w;
    FunctionParser<dim> bulk_rhs_w;
    FunctionParser<dim> bc_w_1;
    FunctionParser<dim> bc_w_2;

    ParameterHandler &params;

    /**
     * Initialize struct with parameter object
     * @param params parameterhandler object
     */
    BioMacroData(ParameterHandler &params) : params(params) {
        // Needed because this is a reference
    }
};

template <int dim>
struct BioMicroData {
    MultiscaleFunctionParser<dim> solution_v;
    MultiscaleFunctionParser<dim> bulk_rhs_v;
    MultiscaleFunctionParser<dim> bc_v_1;
    MultiscaleFunctionParser<dim> bc_v_2;
    MultiscaleFunctionParser<dim> bc_v_3;
    MultiscaleFunctionParser<dim> bc_v_4;
    MultiscaleFunctionParser<dim> mapping;
    MultiscaleFunctionParser<dim> map_jac;

    ParameterHandler &params;

    /**
     * Initialize struct with parameter object
     * @param params parameterhandler object
     */
    BioMicroData(ParameterHandler &params) : mapping(dim), map_jac(dim * dim), params(params) {}

};

template<int dim>
struct MicroFEMObjects { // Can be moved to a different file if imports would give trouble at some point
    const std::vector<Vector<double>> *solutions;
    const DoFHandler<dim> *dof_handler;
    const MapMap<dim, dim> *mapmap;
    const unsigned int *q_degree;
    const BioMicroData<dim> *data; // Not really nice for the other problems but perhaps we can take a polymorphic solution later.
    bool cache_map_data;

    void get_map_data(const Point<dim> &px, const Point<dim> &py, double &det_jac,
                      SymmetricTensor<2, dim> &kkt);
    void get_map_det_jac(const Point<dim> &px, const Point<dim> &py, double &det_jac);
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
    MacroData(ParameterHandler &params) : params(params) {
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
struct EllipticMicroData {
    EllipticMicroData(ParameterHandler &params) : mapping(dim), map_jac(dim * dim), params(params) {}

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
            :  params(params) {
        // Needed because this is a reference
    }
};

/**
 * Multiscale data class, composed of a macrodata and microdata struct.
 * This data class reads data from a file and initializes the corresponding struct with the relevant parameters.
 * @tparam dim dimensions of functions.
 */
template<int dim>
class MultiscaleData {
public:
    MultiscaleData(const std::string &param_file);

    ParameterHandler params;
    MacroData<dim> macro;
    EllipticMicroData<dim> micro;

    static std::string multiscale_variables();

    static std::string macro_variables();
};

/**
 * Data required for Two-pressures model (elliptic-parabolic system of equations)
 * This data class reads data from a file and initializes the corresponding struct with the relevant parameters.
 * @tparam dim dimensions of functions
 */
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

/**
 * Data required for the biomathematical system of equations.
 * This data class reads data from a file and initializes the corresponding struct with the relevant parameters.
 * @tparam dim dimensions of functions
 */

template<int dim>
class BioData {
public:
    BioData(const std::string &param_file);

    ParameterHandler params;
    BioMacroData<dim> macro;
    BioMicroData<dim> micro;

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
std::string BioData<dim>::multiscale_variables() {
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
std::string BioData<dim>::macro_variables() {
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


#include "pde_data.h"

using namespace dealii;

template<int dim>
MultiscaleData<dim>::MultiscaleData(const std::string &param_file) : macro(params), micro(params) {
    params.declare_entry("macro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("macro_rhs", " x0^2*sin(x0*x1) + x1^2*sin(x0*x1) + 2*cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("macro_solution", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("macro_bc", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());

    params.declare_entry("micro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("micro_rhs", "-sin(x0*x1) - cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("micro_solution", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("micro_bc", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());

    params.parse_input(param_file);

    std::map<std::string, double> constants;
    macro.rhs.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_rhs"), constants);
    macro.bc.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_bc"), constants);
    macro.solution.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_solution"), constants);

    micro.rhs.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_rhs"), constants);
    micro.bc.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_bc"), constants);
    micro.solution.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_solution"),
                              constants);
}


template<int dim>
TwoPressureData<dim>::TwoPressureData(const std::string &param_file) : macro(params), micro(params) {
    params.declare_entry("macro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("macro_rhs",
                         "-A*(2*(2*x0^2 + 1)*exp(t^2 + x0^2 + x1^2) + 2*(2*x1^2 + 1)*exp(t^2 + x0^2 + x1^2)) - 12*theta*exp(t^2 + x0^2 + x1^2)",
                         Patterns::Anything());
    params.declare_entry("macro_solution", "exp(t^2 + x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("macro_bc", "exp(t^2 + x0^2 + x1^2)", Patterns::Anything());

    params.declare_entry("micro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("micro_rhs", "-D*(2*(sin(y0)^2 - cos(y0)^2) + 2*(-sin(y1)^2 + cos(y1)^2))",
                         Patterns::Anything());
    params.declare_entry("micro_solution", "sin(y1)^2 + cos(y0)^2 + 2", Patterns::Anything());
    params.declare_entry("micro_bc_neumann", "-2*D*y0*sin(y0)*cos(y0)", Patterns::Anything());
    params.declare_entry("micro_bc_robin",
                         "2*D*y1*sin(y1)*cos(y1) - kappa*(-R*(sin(y1)^2 + cos(y0)^2 + 2) + p_F + exp(t^2 + x0^2 + x1^2))",
                         Patterns::Anything());
    params.declare_entry("init_rho", "sin(y1)^2 + cos(y0)^2 + 2", Patterns::Anything());

    std::map<std::string, double> constants = {{"A",     0.8},
                                               {"D",     1},
                                               {"theta", 0.5},
                                               {"kappa", 1},
                                               {"p_F",   4},
                                               {"R",     2}};
    // All the constants will be declared in the file
    for (const auto &pair: constants) {
        params.declare_entry(pair.first, std::to_string(pair.second), Patterns::Double());
    }
    params.parse_input(param_file);
    // Override the default constants declared above
    for (auto &pair: constants) {
        pair.second = params.get_double(pair.first);
    }


    macro.rhs.initialize(TwoPressureData<dim>::macro_variables(), params.get("macro_rhs"), constants, true);
    macro.bc.initialize(TwoPressureData<dim>::macro_variables(), params.get("macro_bc"), constants, true);
    macro.solution.initialize(TwoPressureData<dim>::macro_variables(), params.get("macro_solution"), constants, true);

    micro.rhs.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_rhs"), constants, true);
    micro.neumann_bc.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_bc_neumann"), constants,
                                true);
    micro.robin_bc.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_bc_robin"), constants,
                              true);
    micro.solution.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_solution"), constants,
                              true);
    micro.init_rho.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("init_rho"), constants, false);
}

template<int dim>
void TwoPressureData<dim>::set_time(const double time) {
    macro.rhs.set_time(time);
    macro.bc.set_time(time);
    macro.solution.set_time(time);

    micro.rhs.set_time(time);
    micro.neumann_bc.set_time(time);
    micro.robin_bc.set_time(time);
    micro.solution.set_time(time);
}

// Explicit instantiation

template
struct MacroData<1>;
template
struct MacroData<2>;
template
struct MacroData<3>;

template
struct MicroData<1>;
template
struct MicroData<2>;
template
struct MicroData<3>;


template
struct ParabolicMicroData<1>;
template
struct ParabolicMicroData<2>;
template
struct ParabolicMicroData<3>;

template
class MultiscaleData<1>;

template
class MultiscaleData<2>;

template
class MultiscaleData<3>;

template
class TwoPressureData<1>;

template
class TwoPressureData<2>;

template
class TwoPressureData<3>;

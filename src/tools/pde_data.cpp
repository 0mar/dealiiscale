
#include "pde_data.h"

using namespace dealii;

template<int dim>
MultiscaleData<dim>::MultiscaleData(const std::string &param_file) : macro(params), micro(params) {
    params.declare_entry("macro_geometry", "[0,1]x[0,1]", Patterns::Anything());
    params.declare_entry("macro_rhs", "0", Patterns::Anything());
    params.declare_entry("macro_solution", "sin(lambda*x) + cos(lambda*y)", Patterns::Anything());
    params.declare_entry("macro_bc", "sin(lambda*x) + cos(lambda*y)", Patterns::Anything());

    params.declare_entry("lambda", "1.63299316", Patterns::Double(), "Boundary constant");
    params.declare_entry("micro_geometry", "[0,1]x[0,1]", Patterns::Anything());
    params.declare_entry("micro_rhs", "0", Patterns::Anything());
    params.declare_entry("micro_solution", "cos(x0)*cos(x1)*(y0*y0+y1*y1)", Patterns::Anything());
    params.declare_entry("micro_bc", "cos(x0)*cos(x1)*(y0*y0+y1*y1)", Patterns::Anything());

    params.parse_input(param_file);

    std::map<std::string, double> constants;
    constants["lambda"] = params.get_double("lambda");
    macro.rhs.initialize(FunctionParser<dim>::default_variable_names(), params.get("macro_rhs"), constants);
    macro.bc.initialize(FunctionParser<dim>::default_variable_names(), params.get("macro_bc"), constants);
    macro.solution.initialize(FunctionParser<dim>::default_variable_names(), params.get("macro_solution"),
                              constants);

    micro.rhs.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), params.get("micro_rhs"), constants);
    micro.bc.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), params.get("micro_bc"), constants);
    micro.solution.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), params.get("micro_solution"),
                              constants);
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
class MultiscaleData<1>;

template
class MultiscaleData<2>;

template
class MultiscaleData<3>;

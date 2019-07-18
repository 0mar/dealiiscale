
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
    macro.rhs.initialize(macro_variables(), params.get("macro_rhs"), constants);
    macro.bc.initialize(macro_variables(), params.get("macro_bc"), constants);
    macro.solution.initialize(macro_variables(), params.get("macro_solution"), constants);

    micro.rhs.initialize(multiscale_variables(), params.get("micro_rhs"), constants);
    micro.bc.initialize(multiscale_variables(), params.get("micro_bc"), constants);
    micro.solution.initialize(multiscale_variables(), params.get("micro_solution"),
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

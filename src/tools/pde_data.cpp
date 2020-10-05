
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

    params.declare_entry("mapping", "(3+x0+x1)*(1.6*y0 - 1.6*y1);(3+x0+x1)*(2.4*y0 + 2.4*y1);", Patterns::Anything());
    params.declare_entry("jac_mapping", "1.6*(3+x0+x1);-1.6*(3+x0+x1);2.4*(3+x0+x1);2.4*(3+x0+x1);",
                         Patterns::Anything());


    params.parse_input(param_file);

    std::map<std::string, double> constants;
    macro.rhs.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_rhs"), constants);
    macro.bc.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_bc"), constants);
    macro.solution.initialize(MultiscaleData<dim>::macro_variables(), params.get("macro_solution"), constants);
    micro.mapping.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("mapping"), constants);

    micro.rhs.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_rhs"), constants);
    micro.bc.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_bc"), constants, &micro.mapping);
    micro.solution.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("micro_solution"),
                              constants, &micro.mapping);
    micro.map_jac.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("jac_mapping"), constants);
}

template<int dim>
BioData<dim>::BioData(const std::string &param_file) : macro(params), micro(params) {
    params.declare_entry("macro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("solution_u", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bulk_rhs_u", " x0^2*sin(x0*x1) + x1^2*sin(x0*x1) + 2*cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bc_u_1", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bc_u_2", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("inflow_measure", "x0 + x1", Patterns::Anything());
    params.declare_entry("outflow_measure", "x0 + x1", Patterns::Anything());

    params.declare_entry("solution_w", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bulk_rhs_w", " x0^2*sin(x0*x1) + x1^2*sin(x0*x1) + 2*cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bc_w_1", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bc_w_2", "sin(x0*x1) + cos(x0 + x1)", Patterns::Anything());

    params.declare_entry("micro_geometry", "[-1,1]x[-1,1]", Patterns::Anything());
    params.declare_entry("solution_v", "-D_1 * sin(x0*x1) - cos(x0 + x1)", Patterns::Anything());
    params.declare_entry("bulk_rhs_v", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("bc_v_1", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("bc_v_2", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("bc_v_3", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());
    params.declare_entry("bc_v_4", "y0*y1 + exp(x0^2 + x1^2)", Patterns::Anything());

    params.declare_entry("mapping", "(3+x0+x1)*(1.6*y0 - 1.6*y1);(3+x0+x1)*(2.4*y0 + 2.4*y1);", Patterns::Anything());
    params.declare_entry("jac_mapping", "1.6*(3+x0+x1);-1.6*(3+x0+x1);2.4*(3+x0+x1);2.4*(3+x0+x1);",
                         Patterns::Anything());

    std::map<std::string, double> constants = {{"D_1",     2},
                                               {"D_2",     2},
                                               {"kappa_1", 1},
                                               {"kappa_2", 1},
                                               {"kappa_3", 1},
                                               {"kappa_4", 1}};
    // All the constants will be declared in the file
    for (const auto &pair: constants) {
        params.declare_entry(pair.first, std::to_string(pair.second), Patterns::Double());
    }
    params.parse_input(param_file);
    // Override the default constants declared above
    for (auto &pair: constants) {
        pair.second = params.get_double(pair.first);
    }

    macro.solution_u.initialize(BioData < dim > ::macro_variables(), params.get("solution_u"), constants);
    macro.bulk_rhs_u.initialize(BioData < dim > ::macro_variables(), params.get("bulk_rhs_u"), constants);
    macro.bc_u_1.initialize(BioData < dim > ::macro_variables(), params.get("bc_u_1"), constants);
    macro.bc_u_2.initialize(BioData < dim > ::macro_variables(), params.get("bc_u_2"), constants);
    macro.inflow_measure.initialize(BioData < dim > ::macro_variables(), params.get("inflow_measure"), constants);
    macro.outflow_measure.initialize(BioData < dim > ::macro_variables(), params.get("outflow_measure"), constants);

    macro.solution_w.initialize(BioData < dim > ::macro_variables(), params.get("solution_w"), constants);
    macro.bulk_rhs_w.initialize(BioData < dim > ::macro_variables(), params.get("bulk_rhs_w"), constants);
    macro.bc_w_1.initialize(BioData < dim > ::macro_variables(), params.get("bc_w_1"), constants);
    macro.bc_w_2.initialize(BioData < dim > ::macro_variables(), params.get("bc_w_2"), constants);

    // All microfunctions that require full dealii evaluation (i.e. diriclet BC and solution functions) need a mapping
    micro.solution_v.initialize(BioData < dim > ::multiscale_variables(), params.get("solution_v"), constants,
                                &micro.mapping);
    micro.bulk_rhs_v.initialize(BioData < dim > ::multiscale_variables(), params.get("bulk_rhs_v"), constants);
    micro.bc_v_1.initialize(BioData < dim > ::multiscale_variables(), params.get("bc_v_1"), constants);
    micro.bc_v_2.initialize(BioData < dim > ::multiscale_variables(), params.get("bc_v_2"), constants);
    micro.bc_v_3.initialize(BioData < dim > ::multiscale_variables(), params.get("bc_v_3"), constants);
    micro.bc_v_4.initialize(BioData < dim > ::multiscale_variables(), params.get("bc_v_4"), constants);

    micro.mapping.initialize(BioData<dim>::multiscale_variables(), params.get("mapping"), constants);
    micro.map_jac.initialize(BioData<dim>::multiscale_variables(), params.get("jac_mapping"), constants);
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
    params.declare_entry("nonlinear", "false", Patterns::Bool());

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

    micro.rhs.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_rhs"), constants, nullptr,
                         true);
    micro.neumann_bc.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_bc_neumann"), constants,
                                nullptr, true);
    micro.robin_bc.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_bc_robin"), constants,
                              nullptr, true);
    micro.solution.initialize(TwoPressureData<dim>::multiscale_variables(), params.get("micro_solution"), constants,
                              nullptr, true);
    micro.init_rho.initialize(MultiscaleData<dim>::multiscale_variables(), params.get("init_rho"), constants, nullptr,
                              false);
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

template<int dim>
void MicroFEMObjects<dim>::get_map_data(const Point<dim> &px, const Point<dim> &py, double &det_jac,
                                        SymmetricTensor<2, dim> &kkt) {
    if (cache_map_data) {
        mapmap->get(px, py, det_jac, kkt);
    } else {
        Tensor<2, dim> jacobian = data->map_jac.mtensor_value(px, py);
        Tensor<2, dim> inv_jacobian = invert(jacobian);
        det_jac = determinant(jacobian);
        Assert(det_jac > 1E-4, ExcMessage("Determinant of jacobian of mapping is not positive!"))
        kkt = SymmetricTensor<2, dim>(inv_jacobian * transpose(inv_jacobian));
    }

}

template<int dim>
void MicroFEMObjects<dim>::get_map_det_jac(const Point<dim> &px, const Point<dim> &py, double &det_jac) {
    if (cache_map_data) {
        mapmap->get_det_jac(px, py, det_jac);
    } else {
        Tensor<2, dim> jacobian = data->map_jac.mtensor_value(px, py);
        det_jac = determinant(jacobian);
        Assert(det_jac > 1E-4, ExcMessage("Determinant of jacobian of mapping is not positive!"))
    }
}

// Explicit instantiation

template
struct MicroFEMObjects<1>;

template
struct MicroFEMObjects<2>;

template
struct MicroFEMObjects<3>;

template
struct MacroData<1>;
template
struct MacroData<2>;
template
struct MacroData<3>;

template
struct BioMacroData<1>;
template
struct BioMacroData<2>;
template
struct BioMacroData<3>;

template
struct BioMicroData<1>;
template
struct BioMicroData<2>;
template
struct BioMicroData<3>;


template
class BioData<1>;

template
class BioData<2>;

template
class BioData<3>;

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

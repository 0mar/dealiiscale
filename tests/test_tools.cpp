#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include "../src/tools/multiscale_function_parser.h"
#include "../src/tools/pde_data.h"
#include <cmath>
#include <cstdio>

using namespace dealii;
BOOST_AUTO_TEST_SUITE(test_functions)
    double eps = 1E-9;

    template<int dim>
    void load_function(MultiscaleFunctionParser<dim> &function, const std::string &definition) {
        std::map<std::string, double> constants;
        constants["pi"] = numbers::PI;
        function.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), definition, constants, false);
    }

    BOOST_AUTO_TEST_CASE(test_function_mvalue) {
        const int dim = 2;
        MultiscaleFunctionParser<dim> function;
        load_function(function, "x0*y1 - x1^2 - 2*y0*y0");
        Point<dim> x({0.2, 0.4});
        Point<dim> y({0., 1});
        double result = 0.2 - 0.16;
        BOOST_CHECK_CLOSE(function.mvalue(x, y), result, eps);
    }

    BOOST_AUTO_TEST_CASE(test_function_set_x) {
        const int dim = 2;
        MultiscaleFunctionParser<dim> function;
        load_function(function, "cos(pi*x0*y1) - sin(x1*2*pi) - 2");
        Point<dim> x({0.5, -2});
        Point<dim> y({-3, 1});
        double result = 0. - 2;
        function.set_macro_point(x);
        BOOST_CHECK_CLOSE(function.value(y), result, eps);
    }

BOOST_AUTO_TEST_SUITE_END()
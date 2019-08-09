#define BOOST_TEST_MODULE MyTest

#include <boost/test/unit_test.hpp>
#include "../src/tools/multiscale_function_parser.h"
#include <cmath>

using namespace dealii;
BOOST_AUTO_TEST_SUITE(test_functions)
    double eps = 1E-9;

    template<int dim>
    void load_function(MultiscaleFunctionParser<dim> &function, const std::string &definition) {
        std::map<std::string, double> constants;
        function.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), definition, constants, false);
    }

    template<int dim>
    Point<dim> get_point(const std::vector<double> &values) {
        AssertDimension(values.size(), dim)
        Point<dim> point;
        for (unsigned int i = 0; i < dim; i++) {
            point[i] = values[i];
        }
        return point;
    }

    BOOST_AUTO_TEST_CASE(test_simulation_creation) {
        const int dim = 2;
        MultiscaleFunctionParser<dim> function;
        load_function(function, "x0*y1 - x1^2 - 2*y0*y0");
        Point<dim> get_pointx({0.2, 0.4}); // Todo: Can we remove the method up here?
        Point<dim> get_pointy({0., 1}); // Todo: Can we remove the method up here?
        double result = 0.2 - 0.16;
        BOOST_CHECK_CLOSE(function.mvalue(get_pointx, get_pointy), result, eps);
    }


BOOST_AUTO_TEST_SUITE_END()
#define BOOST_TEST_MODULE dealiiscale_tests

#include <boost/test/unit_test.hpp>
#include "../src/tools/multiscale_function_parser.h"
#include "../src/tools/pde_data.h"
#include "../src/tools/mapping.h"
#include <cmath>
#include <cstdio>

using namespace dealii;
BOOST_AUTO_TEST_SUITE(test_functions)
    double eps = 1E-9;

    template<int dim>
    void load_function(MultiscaleFunctionParser<dim> &function, const std::string &definition) {
        std::map<std::string, double> constants;
        constants["pi"] = numbers::PI;
        function.initialize(MultiscaleFunctionParser<dim>::default_variable_names(), definition, constants);
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


BOOST_AUTO_TEST_SUITE(test_mapmap)

    void get_set_map(MapMap<3, 2> &map) {
        Point<3> px(2.1415, 2.71812, 4.532);
        Point<2> py(3.1415, 2.71812);
        SymmetricTensor<2, 2> kkt;
        kkt[0][0] = 2;
        kkt[1][0] = 1;
        kkt[1][1] = 2;
        map.set(px, py, 3, kkt);
    }

    BOOST_AUTO_TEST_CASE(test_setting) {
        MapMap<3, 2> mapmap;
        get_set_map(mapmap);
        BOOST_CHECK_EQUAL(mapmap.size(), 1);
    }

    BOOST_AUTO_TEST_CASE(test_duplicate_setting) {
        MapMap<3, 2> mapmap;
        get_set_map(mapmap);
        get_set_map(mapmap);
        BOOST_CHECK_EQUAL(mapmap.size(), 1);

    }

    BOOST_AUTO_TEST_CASE(test_getting) {
        MapMap<3, 2> mapmap;
        get_set_map(mapmap);
        Point<3> px(2.1415, 2.71812, 4.532);
        Point<2> py(3.1415, 2.71812);
        SymmetricTensor<2, 2> kkt;
        double det_jac;
        mapmap.get(px, py, det_jac, kkt);
        BOOST_CHECK_EQUAL(kkt[0][0], 2);
    }

    BOOST_AUTO_TEST_CASE(test_invalid_getting) {
        MapMap<3, 2> mapmap;
        Point<3> px(2.1415, 2.71812, 4.532);
        Point<2> py(3.1415, 2.71812);
        SymmetricTensor<2, 2> kkt;
        double det_jac;
        BOOST_CHECK_THROW(mapmap.get(px, py, det_jac, kkt), ExcMessage);
    }

    BOOST_AUTO_TEST_CASE(test_string_robustness) {
        MapMap<3, 2> mapmap;
        get_set_map(mapmap);
        Point<3> px(2.14150, 2.7181200, 4.53200000000);
        Point<2> py(3.1415, 2.71812);
        SymmetricTensor<2, 2> kkt;
        kkt[0][0] = 2;
        kkt[1][0] = 1;
        kkt[1][1] = 2;
        mapmap.set(px, py, 3, kkt);

        BOOST_CHECK_EQUAL(mapmap.size(), 1);
    }

BOOST_AUTO_TEST_SUITE_END()

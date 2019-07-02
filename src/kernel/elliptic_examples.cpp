#include "elliptic_examples.h" // I need to think of a good structure.

using namespace dealii;


double TrigoSolution::value(const Point<2> &p, const unsigned int) const {
    double val = std::sin(lambda * p(0)) + std::cos(lambda * p(1));
    return val;
}

Tensor<1, 2> TrigoSolution::gradient(const Point<2> &p, const unsigned int) const {
    Tensor<1, 2> return_val;
    return_val[0] = lambda * std::cos(lambda * p(0));
    return_val[1] = -lambda * std::sin(lambda * p(1));
    return return_val;
}

double TrigoRHS::value(const Point<2> &, unsigned int) const {
    return 1;
}

double TrigoBoundary::value(const Point<2> &p, unsigned int) const {
    double val = std::sin(lambda * p(0)) + std::cos(lambda * p(1));
    return val;
}

Tensor<1, 2> TrigoBoundary::gradient(const Point<2> &p, unsigned int) const {
    Tensor<1, 2> return_val;
    return_val[0] = lambda * std::cos(lambda * p(0));
    return_val[1] = -lambda * std::sin(lambda * p(1));
    return return_val;
}

double PolySolution::value(const Point<2> &p, unsigned int) const {
    double val = 0;
    for (unsigned int i = 0; i < 2; i++) {
        val += p(i) * p(i) * macro_values[macro_index];
    }
    return val;
}

Tensor<1, 2> PolySolution::gradient(const Point<2> &p, unsigned int) const {
    Tensor<1, 2> return_val;
    return_val[0] = 2 * p(0) * macro_values[macro_index];
    return_val[1] = 2 * p(1) * macro_values[macro_index];
    return return_val;
}

double PolyBoundary::value(const Point<2> &p, unsigned int) const {
    double val = 0;
    for (unsigned int i = 0; i < 2; i++) {
        val += p(i) * p(i) * macro_values[macro_index];
    }
    return val;
}

Tensor<1, 2> PolyBoundary::gradient(const Point<2> &p, unsigned int) const {
    Tensor<1, 2> return_val;
    return_val[0] = 2 * p(0) * macro_values[macro_index];
    return_val[1] = 2 * p(1) * macro_values[macro_index];
    return return_val;
}

double PolyRHS::value(const Point<2> &, unsigned int) const {
    return -4.;
}

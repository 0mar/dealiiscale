#include "oracle.h" // I need to think of a good structure.

using namespace dealii;


class TrigoSolution : public Solution<2> {
public:
    const double lambda = std::sqrt(8. / 3.);

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int component = 0) const override;
};

double TrigoSolution::value(const Point<2> &p, const unsigned int) const {
    double val = std::sin(lambda * p(0)) + std::cos(lambda * p(1));
    return val;
}

Tensor<1, 2> TrigoSolution::gradient(const Point<2> &p, const unsigned int component) const {
    Tensor<1, 2> return_val;
    return_val[0] = lambda * std::cos(lambda * p(0));
    return_val[1] = -lambda * std::sin(lambda * p(1));
    return return_val;
}

class PolySolution : public Solution<2> {
public:

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int component = 0) const override;

    Vector<double> macro_function;
    unsigned int macro_cell_index = 0;
};

double PolySolution::value(const Point<2> &p, unsigned int) const {
    double val = 0;
    for (unsigned int i = 0; i < 2; i++) {
        val += p(i) * p(i) * macro_function[macro_cell_index];
    }
    return val;
}


Tensor<1, 2> PolySolution::gradient(const Point<2> &p, unsigned int component) const {
    Tensor<1, 2> return_val;
    return_val[0] = 2 * p(0) * macro_function[macro_cell_index];
    return_val[1] = 2 * p(1) * macro_function[macro_cell_index];
    return return_val;
}

class TrigoPoly : public MultiScaleOracle<2> {
public:

    TrigoPoly();

    const double lambda = std::sqrt(8. / 3.);
    const double micro_laplacian = -4;

    TrigoSolution macro_boundary;
    PolySolution micro_boundary;


};

TrigoPoly::TrigoPoly() {

}

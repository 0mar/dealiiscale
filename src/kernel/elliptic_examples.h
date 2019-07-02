/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef ELLIPTIC_EXAMPLES_H
#define ELLIPTIC_EXAMPLES_H

#include <deal.II/base/function.h>
#include <stdlib.h>
#include "base.h"

using namespace dealii;


class TrigoSolution : public Solution<2> {
public:
    TrigoSolution() : Solution<2>() {

    }

    const double lambda = std::sqrt(8. / 3.);

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int) const override;
};


class TrigoRHS : public RightHandSide<2> {
public:
    TrigoRHS() : RightHandSide<2>() {

    }

    const double lambda = std::sqrt(8. / 3.);

    double value(const Point<2> &p, unsigned int) const override;
};

class TrigoBoundary : public BoundaryCondition<2> {
public:
    TrigoBoundary() : BoundaryCondition<2>() {

    }

    const double lambda = std::sqrt(8. / 3.);

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int) const override;

};

class Trigo : public Oracle<2> {
public:

    Trigo();

    TrigoSolution boundary;
    TrigoRHS rhs;
    TrigoBoundary bc;

};

class PolySolution : public Solution<2>, public MicroObject<2> {
public:

    PolySolution() : Solution<2>(), MicroObject<2>() {

    }

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int) const override;

};

class PolyRHS : public RightHandSide<2>, public MicroObject<2> {
public:
    PolyRHS() : RightHandSide<2>(), MicroObject<2>() {

    }

    double value(const Point<2> &p, unsigned int) const override;
};


class PolyBoundary : public BoundaryCondition<2>, public MicroObject<2> {
public:

    PolyBoundary() : BoundaryCondition<2>(), MicroObject<2>() {

    }

    double value(const Point<2> &p, unsigned int) const override;

    Tensor<1, 2> gradient(const Point<2> &p, unsigned int) const override;
};

class Poly : public Oracle<2>, MicroData<2> {
public:
    Poly();

    PolySolution solution;
    PolyRHS rhs;
    PolyBoundary bc;
};

#endif //ELLIPTIC_EXAMPLES_H
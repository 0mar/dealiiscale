/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#ifndef BASE_H
#define BASE_H

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <cmath>
#include <stdlib.h>
#include <deal.II/base/logstream.h>

using namespace dealii;


template<int dim>
class MicroObject {
public:
    MicroObject() : macro_index(0) {
        macro_values.reinit(1);
        macro_values = 1;

    };

    void set_macro_index(const unsigned int &index);

    void set_macro_values(const Vector<double> &values);

protected:

    unsigned int macro_index;

    Vector<double> macro_values;
};


template<int dim>
class BoundaryCondition : public Function<dim>, public MicroObject<dim> {
public:
    BoundaryCondition() : Function<dim>(), MicroObject<dim>() {

    };

    /**
     * Creates a macroscopic boundary (only Dirichlet at this point)
     * @param p The point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary condition at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const = 0;


    /**
     * Compute the analytic gradient of the boundary at point p. Necessary for Robin/Neumann boundary conditions and
     * exact evaluation of the error.
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return gradient of the microscopic boundary condition at p
    */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const = 0;

};


template<int dim>
class RightHandSide : public Function<dim>, public MicroObject<dim> {
public:
    RightHandSide() : Function<dim>(), MicroObject<dim>() {

    };

    /**
     * Creates a macroscopic boundary (only Dirichlet at this point)
     * @param p The point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary condition at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const = 0;
};


template<int dim>
class Solution : public Function<dim>, public MicroObject<dim> {
public:
    Solution() : Function<dim>(), MicroObject<dim>() {

    };

    /**
     * Creates a macroscopic boundary (only Dirichlet at this point)
     * @param p The point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return Value of the microscopic boundary condition at p
     */
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const = 0;


    /**
     * Compute the analytic gradient of the boundary at point p. Necessary for Robin/Neumann boundary conditions and
     * exact evaluation of the error.
     * @param p The nD point where the boundary condition is evaluated
     * @param component Component of the vector: not used in this case
     * @return gradient of the microscopic boundary condition at p
    */
    virtual Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component = 0) const = 0;

};

template<int dim>
class BaseData {
public :
    BaseData() {

    };

    RightHandSide<dim> rhs;
    BoundaryCondition<dim> bc;
    const static int ROBIN_BOUNDARY = 0;
    const static int NEUMANN_BOUNDARY = 1;
    const static int DIRICHLET_BOUNDARY = 2;

    int boundary_indicator = -1;
};

template<int dim>
class Oracle : public BaseData<dim> {
public:
    Oracle() : BaseData<dim>() {

    };

    Solution<dim> solution;
};

#include "base.tpp"

#endif //BASE_H

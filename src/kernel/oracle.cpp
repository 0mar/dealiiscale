/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "oracle.h"

using namespace dealii;


template<int dim>
MultiScaleOracle<dim>::MultiScaleOracle() = default;

template<int dim>
void MultiScaleData<dim>::set_macro_index(const unsigned int index) {
    macro_index = index;

}

template<int dim>
void MultiScaleData<dim>::set_macro_values(const Vector<double> &values) {
    macro_values = values;
}


template<int dim>
MultiScaleData<dim>::MultiScaleData() : macro_index(0) {

}

/*
 * Author: Omar Richardson, Karlstad University, 2019
 */


#include "base.h"

using namespace dealii;


template<int dim>
BaseData<dim>::BaseData() {

}

template<int dim>
Oracle<dim>::Oracle() {

}

template<int dim>
MicroObject<dim>::MicroObject() : macro_index(0) {

}


template<int dim>
void MicroObject<dim>::set_macro_index(const unsigned int &index) {
    macro_index = index;
}


template<int dim>
void MicroObject<dim>::set_macro_values(const Vector<double> &values) {
    macro_values = values;
}

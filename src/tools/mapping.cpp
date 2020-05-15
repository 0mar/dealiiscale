/* ---------------------------------------------------------------------
 *
 * Author: Omar Richardson, 2020
 */


#include "mapping.h"

template<int macro_dim, int micro_dim>
MapMap<macro_dim, micro_dim>::MapMap() : tensor_map(), determinant_map() {

}

template<int macro_dim, int micro_dim>
void MapMap<macro_dim, micro_dim>::set(const Point<macro_dim> &px, const Point<micro_dim> &py, const double &det_jac,
                                       const KKT<micro_dim> &kkt) {
    std::string repr;
    to_strimg(px, py, repr);
    determinant_map[repr] = det_jac;
    tensor_map[repr] = kkt;
}

template<int macro_dim, int micro_dim>
void MapMap<macro_dim, micro_dim>::get(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac,
                                       KKT<micro_dim> &kkt) {
    std::string repr;
    to_string(px,py,repr);
    det_jac = determinant[repr];
    kkt = tensor_map[repr];
}

template<int macro_dim, int micro_dim>
void
MapMap<macro_dim, micro_dim>::get_det_jac(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac) {
    std::string repr;
    to_string(px,py,repr);
    det_jac = determinant[repr];
}

template<int macro_dim, int micro_dim>
void
MapMap<macro_dim, micro_dim>::to_string(const Point<macro_dim> &px, const Point<micro_dim> &py, std::string &repr) {
    std::ostringstream repr_stream;
    repr_stream << px[0];
    for (unsigned int i=1;i<macro_dim;i++) {
        repr_stream << "," << px[i];
    }
    repr_stream << ";" << py[0];
    for (unsigned int j=0;j<micro_dim;j++) {
        repr_stream << "," << py[j];
    }
    repr = repr_stream.str();
}

template<int macro_dim, int micro_dim>
void MapMap<macro_dim, micro_dim>::from_string(const std::string &repr, Point<micro_dim> &px, Point<macro_dim> &py) {
    std::string px_string = repr.substr(0,repr.find(';'));
    std::string py_string = repr.substr(1,repr.find(';'));
    for (unsigned long i=0;i<macro_dim;i++) {
        px(i) = px_string.substr(i,px_string.find(','));
    }
    for (unsigned long j=0;j<micro_dim;j++ ) {
        py(j) = py_string.substr(j,py_string.find(','));
    }
}


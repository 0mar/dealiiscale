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
    to_string(px, py, repr);
    // If key is present, it should have same value as we want to put in.
    Assert(det_jac > 0, ExcZero())
    Assert(tensor_map.count(repr) == 0 || tensor_map[repr] == kkt,
           ExcMessage("New value for " + repr + " differs from already present value"))
    determinant_map[repr] = det_jac;
    tensor_map[repr] = kkt;
}

template<int macro_dim, int micro_dim>
 void MapMap<macro_dim, micro_dim>::get(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac,
                                       KKT<micro_dim> &kkt) const {
    std::string repr;
    to_string(px, py, repr);
    AssertThrow(determinant_map.count(repr), ExcMessage("Coordinates not present"));
    det_jac = determinant_map.at(repr); // Apparently using [] creates default values if map not found.
    kkt = tensor_map.at(repr); // Could be used for a dynamic mapper, should that ever be convenient.
}

template<int macro_dim, int micro_dim>
 void MapMap<macro_dim, micro_dim>::get_det_jac(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac) const{
    std::string repr;
    to_string(px, py, repr);
    AssertThrow(determinant_map.count(repr), ExcMessage("Coordinates not present"));
    det_jac = determinant_map.at(repr);
}

template<int macro_dim, int micro_dim>
 void MapMap<macro_dim, micro_dim>::to_string(const Point<macro_dim> &px, const Point<micro_dim> &py, std::string &repr) const{
    std::ostringstream repr_stream;
    repr_stream << px[0];
    for (unsigned int i = 1; i < macro_dim; i++) {
        repr_stream << "," << px[i];
    }
    repr_stream << ";" << py[0];
    for (unsigned int j = 0; j < micro_dim; j++) {
        repr_stream << "," << py[j];
    }
    repr = repr_stream.str();
}

template<int macro_dim, int micro_dim>
 void MapMap<macro_dim, micro_dim>::from_string(const std::string &repr, Point<micro_dim> &px, Point<macro_dim> &py) const{
    std::string px_string = repr.substr(0, repr.find(';'));
    std::string py_string = repr.substr(1, repr.find(';'));
    for (unsigned long i = 0; i < macro_dim; i++) {
        px(i) = std::stod(px_string.substr(i, px_string.find(',')));
    }
    for (unsigned long j = 0; j < micro_dim; j++) {
        py(j) = std::stod(py_string.substr(j, py_string.find(',')));
    }
}

template<int macro_dim, int micro_dim>
void MapMap<macro_dim, micro_dim>::list_all() const {
    for (const auto pair: determinant_map) {
        std::cout << pair.first << "\t" << pair.second << "\t" << tensor_map.at(pair.first) << std::endl;
    }

}

template<int macro_dim, int micro_dim>
 unsigned long MapMap<macro_dim, micro_dim>::size() const{
    return tensor_map.size();
}

// Explicit instantiation

template
class MapMap<1, 1>;

template
class MapMap<2, 1>;

template
class MapMap<3, 1>;

template
class MapMap<1, 2>;

template
class MapMap<2, 2>;

template
class MapMap<3, 2>;

template
class MapMap<1, 3>;

template
class MapMap<2, 3>;

template
class MapMap<3, 3>;

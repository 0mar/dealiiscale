//
// Created by omar on 2020-05-14.
//

#ifndef DEALIISCALE_MAPPING_H
#define DEALIISCALE_MAPPING_H

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/point.h>
#include <map>
#include <sstream>

using namespace dealii;
template<int micro_dim>
using KKT = SymmetricTensor<2, micro_dim>;

template<int macro_dim, int micro_dim>
/**
 * Object that stores information on the mapping for each point in the micro and macro domain.
 * In lieu of a descriptive name, I chose for a funny one. Apologies.
 * The MapMap internally has a map which assigns to any relevant combinations of micro and macro point
 * an inverse Jacobian tensor and determinant.
 * This serves two purposes: Tensor and determinant do not need to be recomputed each time they are used,
 * and the information can be shared easily with the macroscopic scale when microscopic contributions need to be made.
 *
 * @tparam macro_dim Dimension of macroscopic domain
 * @tparam micro_dim Dimension of microscopic domain
 */
class MapMap {
public:
    /**
     * Create a new MapMap. Empty on initialization.
     */
    MapMap();

    /**
     * Set mapping information for (px,py). In debug mode, checks if information is already present and if so, the same.
     *
     * @param px Macroscopic point
     * @param py Microscopic point
     * @param det_jac Value of the determinant of the Jacobian
     * @param kkt Symmetric inverse Jacobian tensor.
     */
    void set(const Point<macro_dim> &px, const Point<micro_dim> &py, const double &det_jac, const KKT<micro_dim> &kkt);

    /**
     * Get mapping information for (px,py). In debug mode, throws error if information is not present.
     * @param px Macroscopic point
     * @param py Microscopic point
     * @param det_jac Value of the determinant of the Jacobian [output]
     * @param kkt Symmetric inverse Jacobian tensor [output]
     */
     void get(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac, KKT<micro_dim> &kkt) const;

    /**
     * Get determinant of the Jacobian of the mapping. Although usually this information is used in conjunction with the
     * symmetric tensor, it is not always the case. This method saves a lookup, I guess.
     *
     * @param px Macroscopic point
     * @param py Microscopic point
     * @param det_jac Value of the determinant of the Jacobian [output]
     */
    void get_det_jac(const Point<macro_dim> &px, const Point<micro_dim> &py, double &det_jac) const ;

    /**
     * Print the coordinates and the corresponding mapping objects present in the map.
     */
    void list_all() const;

    /**
     * Return the number of maps present
     * @return number of computed maps
     */
    unsigned long size() const;

private:
    /**
     * Internally, we are not able to use a Point as a key for a map, let alone a combination of them.
     * For this reason we convert the points to a string representation and use that.
     * This relies strongly on the assumption that the string representation of identical points is the same.
     * So no leading or trailing zeros, or even small roundoff errors.
     * Whatever engine generates the points for setting should be the one that generates the points for getting.
     * Luckily, we are able to output errors when it fails. Improvements are very welcome.
     *
     * @param px Macroscopic point
     * @param py Microscopic point
     * @param repr representation: "px1,...,pxn;py1,...,pyn" [output]
     */
    void to_string(const Point<macro_dim> &px, const Point<micro_dim> &py, std::string &repr) const;

    /**
     * Convert the string representation of the points product back to points.
     * Same caveat as the `to_string` method
     * @param repr representation
     * @param px Macroscopic point
     * @param py Microscopic point
     */
    void from_string(const std::string &repr, Point<micro_dim> &px, Point<macro_dim> &py) const;

    std::map<std::string, KKT<micro_dim>> tensor_map;
    std::map<std::string, double> determinant_map;

};

#endif //DEALIISCALE_MAPPING_H

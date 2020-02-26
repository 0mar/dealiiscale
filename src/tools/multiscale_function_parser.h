//
// Created by Omar Richardson on 2019-07-15.
//

#ifndef DEALIISCALE_MULTISCALEFUNCTIONPARSER_H
#define DEALIISCALE_MULTISCALEFUNCTIONPARSER_H

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/auto_derivative_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/point.h>
#include <deal.II/base/thread_local_storage.h>
#include <vector>
#include <map>
#include <memory>

namespace mu {
    class Parser;
}

DEAL_II_NAMESPACE_OPEN


    template<typename>
    class Vector;


/**
 * This class has been lifted from the standard function parser
 */
    template<int dim>
    class MultiscaleFunctionParser : public AutoDerivativeFunction<dim> {
    public:
        /**
         * Constructor for Parsed functions. Its arguments are the same of
         * the base class Function, with the additional parameter @p h, used
         * for the computation of gradients using finite differences. This
         * object needs to be initialized with the initialize() method
         * before you can use it. If an attempt to use this function is made
         * before the initialize() method has been called, then an exception
         * is thrown.
         */
        MultiscaleFunctionParser(const unsigned int n_components = 1,
                                 const double initial_time = 0.0,
                                 const double h = 1e-8);

        /**
         * Destructor. Explicitly delete the MultiscaleFunctionParser objects (there is one
         * for each component of the function).
         */
        ~MultiscaleFunctionParser();

        /**
         * Type for the constant map. Used by the initialize() method.
         */
        typedef std::map<std::string, double> ConstMap;

        /**
         * Iterator for the constants map. Used by the initialize() method.
         */
        typedef ConstMap::iterator ConstMapIterator;

        /**
         * Initialize the function.  This methods accepts the following parameters:
         *
         * <b>vars</b>: a string with the variables that will be used by the
         * expressions to be evaluated. Note that the variables can have any name
         * (of course different from the function names defined above!), but the
         * order IS important. The first variable will correspond to the first
         * component of the point in which the function is evaluated, the second
         * variable to the second component and so forth. If this function is also
         * time dependent, then it is necessary to specify it by setting the
         * <tt>time_dependent</tt> parameter to true.  An exception is thrown if the
         * number of variables specified here is different from dim (if this
         * function is not time-dependent) or from dim+1 (if it is time- dependent).
         *
         * <b>expressions</b>: a list of strings containing the expressions that
         * will be byte compiled by the internal parser (MultiscaleFunctionParser). Note that
         * the size of this vector must match exactly the number of components of
         * the MultiscaleFunctionParser, as declared in the constructor. If this is not the
         * case, an exception is thrown.
         *
         * <b>constants</b>: a map of constants used to pass any necessary constant
         * that we want to specify in our expressions (in the example above the
         * number pi). An expression is valid if and only if it contains only
         * defined variables and defined constants (other than the functions
         * specified above). If a constant is given whose name is not valid (eg:
         * <tt>constants["sin"] = 1.5;</tt>) an exception is thrown.
         *
         * <b>time_dependent</b>. If this is a time dependent function, then the
         * last variable declared in <b>vars</b> is assumed to be the time variable,
         * and this->get_time() is used to initialize it when evaluating the
         * function. Naturally the number of variables parsed by the initialize()
         * method in this case is dim+1. The value of this parameter defaults to
         * false, i.e. do not consider time.
         */
        void initialize(const std::string &vars,
                        const std::vector<std::string> &expressions,
                        const ConstMap &constants,
                        const bool time_dependent = false);

        /**
         * Initialize the function. Same as above, but accepts a string rather than
         * a vector of strings. If this is a vector valued function, its components
         * are expected to be separated by a semicolon. An exception is thrown if
         * this method is called and the number of components successfully parsed
         * does not match the number of components of the base function.
         */
        void initialize(const std::string &vars,
                        const std::string &expression,
                        const ConstMap &constants,
                        const bool time_dependent = false);

        /**
         * A function that returns default names for variables, to be used in the
         * first argument of the initialize() functions: it returns "x" in 1d, "x,y"
         * in 2d, and "x,y,z" in 3d.
         */
        static std::string default_variable_names();

        /**
         * Set the macroscopic grid point to use the microscopic evaluation.
         * Required for (at least) the smooth evaluation of Dirichlet bcs.
         * @param point Point on the macroscopic grid.
         */
        void set_macro_point(const Point<dim> &point);

        /**
         * Return the value of the function at the given point. Unless there is only
         * one component (i.e. the function is scalar), you should state the
         * component you want to have evaluated; it defaults to zero, i.e. the first
         * component.
         */
        double mvalue(const Point<dim> &px, const Point<dim> &py, const unsigned int component = 0) const;

        Point<dim> mmap(const Point<dim> &px, const Point<dim> &py) const;

        /**
         * Return the tensor value of a function with dim*dim components
         * @param px macro point
         * @param py micro point
         * @param value vector with output
         */
        Tensor<2, dim> mtensor_value(const Point<dim> &px, const Point<dim> &py) const;

        /**
         * Return the value of the function at the given point. Unless there is only
         * one component (i.e. the function is scalar), you should state the
         * component you want to have evaluated; it defaults to zero, i.e. the first
         * component.
         */
        virtual double value(const Point<dim> &py, const unsigned int component = 0) const;

        /**
         * @addtogroup Exceptions
         * @{
         */
        DeclException2 (ExcParseError,
                        int,std::string,
                        << "Parsing Error at Column " << arg1
                                << ". The parser said: " << arg2);

        DeclException2 (ExcInvalidExpressionSize,
                        int, int,
                        << "The number of components (" << arg1
                                << ") is not equal to the number of expressions ("
                                << arg2 << ").");

        //@}

    private:
#ifdef DEAL_II_WITH_MUPARSER
        /**
         * Place for the variables for each thread
         */
        mutable Threads::ThreadLocalStorage <std::vector<double>> vars;

        /**
         * The muParser objects for each thread (and one for each component). We are
         * storing a unique_ptr so that we don't need to include the definition of
         * mu::Parser in this header.
         */
#if TBB_VERSION_MAJOR >= 4
        mutable Threads::ThreadLocalStorage <std::vector<std::unique_ptr<mu::Parser> >> fp;
#else
        // older TBBs have a bug in which they want to return thread-local
  // objects by value. this doesn't work for std::unique_ptr, so use a
  // std::shared_ptr
  mutable Threads::ThreadLocalStorage<std::vector<std::shared_ptr<mu::Parser> > > fp;
#endif

        /**
         * An array to keep track of all the constants, required to initialize fp in
         * each thread.
         */
        std::map<std::string, double> constants;

        /**
         * An array for the variable names, required to initialize fp in each
         * thread.
         */
        std::vector<std::string> var_names;

        /**
         * An array of function expressions (one per component), required to
         * initialize fp in each thread.
         */
        std::vector<std::string> expressions;

        /**
         * Macroscopic grid point.
         */
        Point <dim> macro_point;
        /**
         * Initialize fp and vars on the current thread. This function may only be
         * called once per thread. A thread can test whether the function has
         * already been called by testing whether 'fp.get().size()==0' (not
         * initialized) or >0 (already initialized).
         */
        void init_muparser() const;

#endif

        /**
         * State of usability. This variable is checked every time the function is
         * called for evaluation. It's set to true in the initialize() methods.
         */
        bool initialized;

        /**
         * Assert that we touched the macro before using `value`. Not necessary for computations, just a security check.
         */
        bool macro_set;

        /**
         * Number of variables. If this is also a function of time, then the number
         * of variables is dim+1, otherwise it is dim. In the case that this is a
         * time dependent function, the time is supposed to be the last variable. If
         * #n_vars is not identical to the number of the variables parsed by the
         * initialize() method, then an exception is thrown.
         */
        unsigned int n_vars;
    };

    template<int dim>
    std::string
    MultiscaleFunctionParser<dim>::default_variable_names() {
        switch (dim) {
            case 1:
                return "x0,y0";
            case 2:
                return "x0,x1,y0,y1";
            case 3:
                return "x0,x1,x2,y0,y1,y2,";
            default: Assert(false, ExcNotImplemented());
        }
        return "";
    }

DEAL_II_NAMESPACE_CLOSE

#endif



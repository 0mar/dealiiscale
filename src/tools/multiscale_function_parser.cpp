//
// Created by Omar Richardson on 2019-07-15.
//

#include "multiscale_function_parser.h"
#include <deal.II/base/utilities.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/lac/vector.h>
#include <cmath>
#include <map>
#include <boost/random.hpp>
#include <boost/math/special_functions/erf.hpp>

#ifdef DEAL_II_WITH_MUPARSER

#include <muParser.h>

#else



namespace fparser
{
  class MultiscaleFunctionParser
  {};
}
#endif

DEAL_II_NAMESPACE_OPEN

// Todo: Make it so that we have dim1 and dim2
    template<int dim>
    MultiscaleFunctionParser<dim>::MultiscaleFunctionParser(const unsigned int n_components,
                                                            const double initial_time,
                                                            const double h)
            :
            AutoDerivativeFunction<dim>(h, n_components, initial_time),
            initialized(false),
            macro_set(false),
            mapper(nullptr),
            n_vars(0) {}


// We deliberately delay the definition of the default destructor
// so that we don't need to include the definition of mu::Parser
// in the header file.
    template<int dim>
    MultiscaleFunctionParser<dim>::~MultiscaleFunctionParser() = default;


#ifdef DEAL_II_WITH_MUPARSER

    template<int dim>
    void MultiscaleFunctionParser<dim>::initialize(const std::string &variables,
                                                   const std::vector<std::string> &expressions,
                                                   const std::map<std::string, double> &constants,
                                                   MultiscaleFunctionParser<dim> *mapper,
                                                   const bool time_dependent) {
        this->fp.clear(); // this will reset all thread-local objects

        this->mapper = mapper;
        this->constants = constants;
        this->var_names = Utilities::split_string_list(variables, ',');
        this->expressions = expressions;
        AssertThrow(((time_dependent) ? dim * 2 + 1 : dim * 2) == var_names.size(),
                    ExcMessage("Wrong number of variables"));

        // We check that the number of
        // components of this function
        // matches the number of components
        // passed in as a vector of
        // strings.
        AssertThrow(this->n_components == expressions.size(),
                    ExcInvalidExpressionSize(this->n_components,
                                             expressions.size()));
        if (mapper != nullptr) {
            AssertThrow(mapper->n_components == dim, ExcMessage("Please provide a dim -> dim function as mapping"))
        }

        // Now we define how many variables
        // we expect to read in.  We
        // distinguish between two cases:
        // Time dependent problems, and not
        // time dependent problems. In the
        // first case the number of
        // variables is given by the
        // dimension plus one. In the other
        // case, the number of variables is
        // equal to the dimension. Once we
        // parsed the variables string, if
        // none of this is the case, then
        // an exception is thrown.
        if (time_dependent)
            n_vars = 2 * dim + 1;
        else
            n_vars = 2 * dim;

        // create a parser object for the current thread we can then query
        // in value() and vector_value(). this is not strictly necessary
        // because a user may never call these functions on the current
        // thread, but it gets us error messages about wrong formulas right
        // away
        init_muparser();

        // finally set the initialization bit
        initialized = true;
    }


    namespace internal {
        // convert double into int
        int mu_round(double val) {
            return static_cast<int>(val + ((val >= 0.0) ? 0.5 : -0.5));
        }

        double mu_if(double condition, double thenvalue, double elsevalue) {
            if (mu_round(condition))
                return thenvalue;
            else
                return elsevalue;
        }

        double mu_or(double left, double right) {
            return (mu_round(left)) || (mu_round(right));
        }

        double mu_and(double left, double right) {
            return (mu_round(left)) && (mu_round(right));
        }

        double mu_int(double value) {
            return static_cast<double>(mu_round(value));
        }

        double mu_ceil(double value) {
            return ceil(value);
        }

        double mu_floor(double value) {
            return floor(value);
        }

        double mu_cot(double value) {
            return 1.0 / tan(value);
        }

        double mu_csc(double value) {
            return 1.0 / sin(value);
        }

        double mu_sec(double value) {
            return 1.0 / cos(value);
        }

        double mu_log(double value) {
            return log(value);
        }

        double mu_pow(double a, double b) {
            return std::pow(a, b);
        }

        double mu_erfc(double value) {
            return boost::math::erfc(value);
        }

        // returns a random value in the range [0,1] initializing the generator
        // with the given seed
        double mu_rand_seed(double seed) {
            static Threads::Mutex rand_mutex;
            std::lock_guard<std::mutex> lock(rand_mutex);

            static boost::random::uniform_real_distribution<> uniform_distribution(0, 1);

            // for each seed an unique random number generator is created,
            // which is initialized with the seed itself
            static std::map<double, boost::random::mt19937> rng_map;

            if (rng_map.find(seed) == rng_map.end())
                rng_map[seed] = boost::random::mt19937(static_cast<unsigned int>(seed));

            return uniform_distribution(rng_map[seed]);
        }

        // returns a random value in the range [0,1]
        double mu_rand() {
            static Threads::Mutex rand_mutex;
            std::lock_guard<std::mutex> lock(rand_mutex);
            static boost::random::uniform_real_distribution<> uniform_distribution(0, 1);
            static boost::random::mt19937 rng(static_cast<unsigned long>(std::time(nullptr)));
            return uniform_distribution(rng);
        }

    }


    template<int dim>
    void MultiscaleFunctionParser<dim>::init_muparser() const {
        // check that we have not already initialized the parser on the
        // current thread, i.e., that the current function is only called
        // once per thread
        Assert (fp.get().size() == 0, ExcInternalError());

        // initialize the objects for the current thread (fp.get() and
        // vars.get())
        fp.get().reserve(this->n_components);
        vars.get().resize(var_names.size());
        for (unsigned int component = 0; component < this->n_components; ++component) {
            fp.get().emplace_back(new mu::Parser());

            for (std::map<std::string, double>::const_iterator constant = constants.begin();
                 constant != constants.end(); ++constant) {
                fp.get()[component]->DefineConst(constant->first, constant->second);
            }

            for (unsigned int iv = 0; iv < var_names.size(); ++iv)
                fp.get()[component]->DefineVar(var_names[iv], &vars.get()[iv]);

            // define some compatibility functions:
            fp.get()[component]->DefineFun("if", internal::mu_if, true);
            fp.get()[component]->DefineOprt("|", internal::mu_or, 1);
            fp.get()[component]->DefineOprt("&", internal::mu_and, 2);
            fp.get()[component]->DefineFun("int", internal::mu_int, true);
            fp.get()[component]->DefineFun("ceil", internal::mu_ceil, true);
            fp.get()[component]->DefineFun("cot", internal::mu_cot, true);
            fp.get()[component]->DefineFun("csc", internal::mu_csc, true);
            fp.get()[component]->DefineFun("floor", internal::mu_floor, true);
            fp.get()[component]->DefineFun("sec", internal::mu_sec, true);
            fp.get()[component]->DefineFun("log", internal::mu_log, true);
            fp.get()[component]->DefineFun("pow", internal::mu_pow, true);
            fp.get()[component]->DefineFun("erfc", internal::mu_erfc, true);
            fp.get()[component]->DefineFun("rand_seed", internal::mu_rand_seed, true);
            fp.get()[component]->DefineFun("rand", internal::mu_rand, true);

            try {
                // muparser expects that functions have no
                // space between the name of the function and the opening
                // parenthesis. this is awkward because it is not backward
                // compatible to the library we used to use before muparser
                // (the fparser library) but also makes no real sense.
                // consequently, in the expressions we set, remove any space
                // we may find after function names
                std::string transformed_expression = expressions[component];

                const char *function_names[] =
                        {
                                // functions predefined by muparser
                                "sin",
                                "cos",
                                "tan",
                                "asin",
                                "acos",
                                "atan",
                                "sinh",
                                "cosh",
                                "tanh",
                                "asinh",
                                "acosh",
                                "atanh",
                                "atan2",
                                "log2",
                                "log10",
                                "log",
                                "ln",
                                "exp",
                                "sqrt",
                                "sign",
                                "rint",
                                "abs",
                                "min",
                                "max",
                                "sum",
                                "avg",
                                // functions we define ourselves above
                                "if",
                                "int",
                                "ceil",
                                "cot",
                                "csc",
                                "floor",
                                "sec",
                                "pow",
                                "erfc",
                                "rand",
                                "rand_seed"
                        };
                for (unsigned int f = 0; f < sizeof(function_names) / sizeof(function_names[0]); ++f) {
                    const std::string function_name = function_names[f];
                    const unsigned int function_name_length = function_name.size();

                    std::string::size_type pos = 0;
                    while (true) {
                        // try to find any occurrences of the function name
                        pos = transformed_expression.find(function_name, pos);
                        if (pos == std::string::npos)
                            break;

                        // replace whitespace until there no longer is any
                        while ((pos + function_name_length < transformed_expression.size())
                               &&
                               ((transformed_expression[pos + function_name_length] == ' ')
                                ||
                                (transformed_expression[pos + function_name_length] == '\t')))
                            transformed_expression.erase(transformed_expression.begin() + pos + function_name_length);

                        // move the current search position by the size of the
                        // actual function name
                        pos += function_name_length;
                    }
                }

                // now use the transformed expression
                fp.get()[component]->SetExpr(transformed_expression);
            }
            catch (mu::ParserError &e) {
                std::cerr << "Message:  <" << e.GetMsg() << ">\n";
                std::cerr << "Formula:  <" << e.GetExpr() << ">\n";
                std::cerr << "Token:    <" << e.GetToken() << ">\n";
                std::cerr << "Position: <" << e.GetPos() << ">\n";
                std::cerr << "Errc:     <" << e.GetCode() << ">" << std::endl;
                AssertThrow(false, ExcParseError(e.GetCode(), e.GetMsg()));
            }
        }
    }


    template<int dim>
    void MultiscaleFunctionParser<dim>::initialize(const std::string &vars,
                                                   const std::string &expression,
                                                   const std::map<std::string, double> &constants,
                                                   MultiscaleFunctionParser<dim> *mapper,
                                                   const bool time_dependent) {
        initialize(vars, Utilities::split_string_list(expression, ';'),
                   constants, mapper, time_dependent);
    }

    template<int dim>
    double MultiscaleFunctionParser<dim>::mvalue(const Point<dim> &px, const Point<dim> &py,
                                                 const unsigned int component) const {
        Assert (initialized, ExcNotInitialized())
        Assert (component < this->n_components, ExcIndexRange(component, 0, this->n_components))
        Point<dim> point_y;
        if (mapper != nullptr) {
            point_y = mapper->mmap(px, py);
        } else {
            point_y = py;
        }
        // initialize the parser if that hasn't happened yet on the current thread
        if (fp.get().size() == 0)
            init_muparser();

        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[i] = px(i);
        }
        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[dim + i] = point_y(i);
        }
        if (dim * 2 != n_vars) {
            vars.get()[dim * 2] = this->get_time();
        }
        try {
            return fp.get()[component]->Eval();
        }
        catch (mu::ParserError &e) {
            std::cerr << "Message:  <" << e.GetMsg() << ">\n";
            std::cerr << "Formula:  <" << e.GetExpr() << ">\n";
            std::cerr << "Token:    <" << e.GetToken() << ">\n";
            std::cerr << "Position: <" << e.GetPos() << ">\n";
            std::cerr << "Errc:     <" << e.GetCode() << ">" << std::endl;
            AssertThrow(false, ExcParseError(e.GetCode(), e.GetMsg()));
            return 0.0;
        }
    }

    template<int dim>
    double MultiscaleFunctionParser<dim>::value(const Point<dim> &py, const unsigned int component) const {
        Assert(macro_set, ExcEmptyObject("Macro point not initialized yet. This most likely leads to erronous values"))
        return this->mvalue(macro_point, py, component);
    }

    template<int dim>
    Tensor<2, dim> MultiscaleFunctionParser<dim>::mtensor_value(const Point<dim> &px, const Point<dim> &py) const {
        Assert (initialized, ExcNotInitialized())
        AssertDimension(dim * dim, this->n_components)
        Assert(mapper == nullptr, ExcMessage("You are computing a mapped tensor. This should not be a use case."))
        if (fp.get().size() == 0) {
            init_muparser();
        }
        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[i] = px(i);
        }
        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[dim + i] = py(i);
        }
        if (dim * 2 != n_vars) {
            vars.get()[dim * 2] = this->get_time();
        }
        Tensor<2, dim> tensor;
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < dim; j++) {
                tensor[i][j] = fp.get()[i * dim + j]->Eval();
            }
        }
        return tensor;
    }

    template<int dim>
    Point<dim> MultiscaleFunctionParser<dim>::mmap(const Point<dim> &px, const Point<dim> &py) const {
        Point<dim> mapped_p;
        Assert(mapper == nullptr, ExcMessage("You are computing a mapped map. This should not be a use case."))
        Assert (initialized, ExcNotInitialized())
        AssertDimension(dim, this->n_components)
        if (fp.get().size() == 0) {
            init_muparser();
        }
        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[i] = px(i);
        }
        for (unsigned int i = 0; i < dim; i++) {
            vars.get()[dim + i] = py(i);
        }
        if (dim * 2 != n_vars) {
            vars.get()[dim * 2] = this->get_time();
        }
        Tensor<2, dim> tensor;
        for (unsigned int i = 0; i < dim; i++) {
            mapped_p[i] = fp.get()[i]->Eval();
        }
        return mapped_p;
    }

    template<int dim>
    void MultiscaleFunctionParser<dim>::set_macro_point(const Point<dim> &point) {
        macro_point = point;
        macro_set = true;
    }


#else


    template <int dim>
void
MultiscaleFunctionParser<dim>::initialize(const std::string &,
                                const std::vector<std::string> &,
                                const std::map<std::string, double> &,
                                const bool)
{
  Assert(false, ExcNeedsFunctionParser());
}

template <int dim>
void
MultiscaleFunctionParser<dim>::initialize(const std::string &,
                                const std::string &,
                                const std::map<std::string, double> &,
                                const bool)
{
  Assert(false, ExcNeedsFunctionparser());
}



template <int dim>
double MultiscaleFunctionParser<dim>::value (
  const Point<dim> &, const Point&, unsigned int) const
{
  Assert(false, ExcNeedsFunctionparser());
  return 0.;
}


#endif

// Explicit Instantiations.

    template
    class MultiscaleFunctionParser<1>;

    template
    class MultiscaleFunctionParser<2>;

    template
    class MultiscaleFunctionParser<3>;

DEAL_II_NAMESPACE_CLOSE

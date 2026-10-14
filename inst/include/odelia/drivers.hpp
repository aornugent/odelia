#ifndef ODELIA_DRIVERS_H
#define ODELIA_DRIVERS_H

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <odelia/interpolator.hpp>

namespace odelia {
namespace drivers {

// One named forcing: either a constant, or a series supplied from outside the
// model and read between its control points.
//
// The series arrives with values and no slopes, so `monotone_slopes` chooses them
// -- limited, because a driver must read back inside the range it was given. A
// smooth fit through the same points overshoots: on an intermittent series such as
// daily rainfall it evaluates negative at about 42% of points and moves the
// integral by 6.7e-06 where the limited fit moves it by 1.5e-07.
class Function {
public:
  Function() = default;

  Function(std::vector<double> const &x, std::vector<double> const &y)
    : is_variable(true) {
    variable.init(x, y, interpolator::monotone_slopes(x, y));
  }

  Function(double k) : constant(k) {}

  // Reading past the control points is a caller error rather than a value to
  // extrapolate: the series says nothing out there. `name` is the caller's,
  // because the driver's identity is what makes the message actionable and this
  // object does not hold it.
  //
  // ⚠️ The test is `u < min || u > max` and NOT the negation of an in-range test.
  // Every comparison against NaN is false, so as written a non-finite time falls
  // THROUGH and comes back non-finite, which callers rely on; negating an
  // in-range test reads as a tightening and is a behaviour change.
  double evaluate(double u, const std::string &name = std::string()) const {
    if (!is_variable) {
      return constant;
    }
    if (!extrapolate && (u < variable.min() || u > variable.max())) {
      util::stop("Driver " + (name.empty() ? std::string("series") : "'" + name + "'") +
                 " read outside its control points: u = " +
                 util::format_double(u) + " lies " +
                 util::format_double(u < variable.min() ? variable.min() - u
                                                        : u - variable.max()) +
                 " beyond the " + (u < variable.min() ? "lower" : "upper") +
                 " end of [" + util::format_double(variable.min()) + ", " +
                 util::format_double(variable.max()) + "].");
    }
    return variable.eval(u);
  }

  std::vector<double> evaluate_range(const std::vector<double> &u,
                                     const std::string &name = std::string()) const {
    std::vector<double> ret;
    ret.reserve(u.size());
    for (double ui : u) {
      ret.push_back(evaluate(ui, name));
    }
    return ret;
  }

  void set_extrapolate(bool e) { extrapolate = e; }

  // The control points in (from, to) where the series changes from one cubic to
  // the next: all but those inside a run of three equal values, flat on both sides.
  std::vector<double> knots(double from, double to) const {
    std::vector<double> out;
    if (!is_variable) {
      return out;
    }
    const std::vector<double>& x = variable.knots();
    const std::vector<double>& y = variable.values();
    for (auto i = static_cast<std::size_t>(
           std::upper_bound(x.begin(), x.end(), from) - x.begin());
         i < x.size() && x[i] < to; ++i) {
      const bool flat =
        i > 0 && i + 1 < x.size() && y[i - 1] == y[i] && y[i] == y[i + 1];
      if (!flat) {
        out.push_back(x[i]);
      }
    }
    return out;
  }

private:
  interpolator::hermite_interpolator<double> variable;
  double constant = 0.0;
  bool is_variable = false;
  bool extrapolate = false;
};

// The drivers a model reads, by name. Names are taken by reference throughout:
// these are called per unit per step, and a std::string copy per call is not free.
class Drivers {

public:
  // this will override any previously defined drivers with the same name
  void set_constant(const std::string &driver_name, double k) {
    drivers.erase(driver_name);
    drivers.insert({driver_name, Function(k)});
  }

  // initialise the series for a driver from its x, y control points
  void set_variable(const std::string &driver_name, std::vector<double> const &x,
                    std::vector<double> const &y) {
    drivers.erase(driver_name);
    drivers.insert({driver_name, Function(x, y)});
  }

  void set_extrapolate(const std::string &driver_name, bool extrapolate) {
    drivers.at(driver_name).set_extrapolate(extrapolate);
  }

  double evaluate(const std::string &driver_name, double x) const {
    return drivers.at(driver_name).evaluate(x, driver_name);
  }

  std::vector<double> evaluate_range(const std::string &driver_name,
                                     std::vector<double> x) const {
    return drivers.at(driver_name).evaluate_range(x, driver_name);
  }

  std::vector<double> knots(const std::string &driver_name, double from,
                            double to) const {
    return drivers.at(driver_name).knots(from, to);
  }

  // returns the name of each active driver - useful for R output
  std::vector<std::string> get_names() const {
    auto ret = std::vector<std::string>();
    for (auto const &driver: drivers) {
      ret.push_back(driver.first);
    }
    return ret;
  }

  void clear() {
    drivers.clear();
  }

  // Get pointer to Function for repeated use (returns nullptr if not found)
  const Function *get_function_ptr(const std::string &driver_name) const {
    auto it = drivers.find(driver_name);
    return it != drivers.end() ? &(it->second) : nullptr;
  }

private:
  std::unordered_map<std::string, Function> drivers;
};

}
}

#endif //ODELIA_DRIVERS_H

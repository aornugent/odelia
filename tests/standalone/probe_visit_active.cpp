// Which shapes visit_active opens, and which it passes over in silence.
//
// ode_interface.hpp says a member in a shape it does not list is skipped rather
// than refused, so the only signal is the slot count afterwards. That warning is
// the reason a plant type replacing a std::pair has to declare for_each_active --
// and this runs the dispatch rather than reading it, because the failure it
// describes has no error to look for.
//
//   make probe_visit_active && ./probe_visit_active

#include <odelia/ode_interface.hpp>
#include <cstdio>
#include <utility>

namespace {

// Two scalars under names visit_active knows nothing about.
struct unnamed_pair {
  double value;
  double slope;
};

// The same two, saying what they are.
struct declared_pair {
  double value;
  double slope;

  template <class F>
  void for_each_active(F&& f) {
    f(value);
    f(slope);
  }
};

// A scalar behind a name, inside a shape that is opened -- to show the skip is
// about the enclosing shape and not about the member's name.
struct holds_an_unnamed_pair {
  unnamed_pair inner;

  template <class F>
  void for_each_active(F&& f) {
    odelia::ode::visit_active(f, inner);
  }
};

template <class T>
int reached(T& x) {
  int n = 0;
  auto count = [&n](double&) { ++n; };
  odelia::ode::visit_active(count, x);
  return n;
}

}  // namespace

int main() {
  std::pair<double, double> as_pair{1.0, 2.0};
  unnamed_pair unnamed{1.0, 2.0};
  declared_pair declared{1.0, 2.0};
  holds_an_unnamed_pair nested{{1.0, 2.0}};
  double scalars[2] = {1.0, 2.0};

  std::printf("scalars expected: 2 each\n\n");
  std::printf("  std::pair<double,double>        %d\n", reached(as_pair));
  std::printf("  struct {value, slope}           %d   <-- skipped, no error\n",
              reached(unnamed));
  std::printf("  the same with for_each_active   %d\n", reached(declared));
  std::printf("  for_each_active over an unnamed %d   <-- skipped one level down\n",
              reached(nested));
  std::printf("  double[2]                       %d\n", reached(scalars));

  std::printf(
      "\nSo a named replacement for a pair on a recorded path must declare\n"
      "for_each_active, and declaring it on the holder is not enough.\n");
  return 0;
}

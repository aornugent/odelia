// Which shapes visit_active opens, and which it passes over in silence.
//
// ode_interface.hpp says a member in a shape it does not list is skipped rather
// than refused, so the only signal is the slot count afterwards. That warning is
// the reason a type standing in for a std::pair has to declare for_each_active --
// and this runs the dispatch rather than reading it, because the failure it
// describes has no error to look for.
//
// A std::pair reads out through the same aggregate path as everything else and
// gets no arm of its own; nothing in the family hands visit_active a bare pair.
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

// The same two, saying what they are -- through both walks, which is what a type
// on a recorded path owes. See declared_pair_mutable_only below for why.
struct declared_pair {
  double value;
  double slope;

  template <class F>
  void for_each_active(F&& f) {
    f(value);
    f(slope);
  }
  template <class F>
  void for_each_active(F&& f) const {
    f(value);
    f(slope);
  }
};

// A pair that says what it is, but only through a non-const walk. This is the
// shape the rewinding forms in implicit_node.hpp meet -- they take
// `const Inputs&...` -- and before visit_active refused it, the arm simply
// stopped matching and every scalar here was passed over in silence.
//
// ⚠️ THERE IS NO ROW IN THE TABLE BELOW FOR THIS TYPE, and that is the point:
// `reached_const(mutable_only)` does not compile. Uncomment the call in main()
// to read the static_assert.
struct declared_pair_mutable_only {
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

// The same walk over a const reference, which is how implicit_value and
// preaccumulate receive their inputs -- they read and clear through the tape by
// slot, so const is what they want. The visitor takes const too, or no arm
// matches for a reason that is not the one under test.
template <class T>
int reached_const(const T& x) {
  int n = 0;
  auto count = [&n](const double&) { ++n; };
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
  std::printf("  std::pair<double,double>        %d   <-- no longer opened\n",
              reached(as_pair));
  std::printf("  struct {value, slope}           %d   <-- skipped, no error\n",
              reached(unnamed));
  std::printf("  the same with for_each_active   %d\n", reached(declared));
  std::printf("  for_each_active over an unnamed %d   <-- skipped one level down\n",
              reached(nested));
  std::printf("  double[2]                       %d\n", reached(scalars));

  std::printf("\nthe same shapes through a const reference:\n\n");
  std::printf("  struct {value, slope}           %d\n", reached_const(unnamed));
  std::printf("  the same with for_each_active   %d   <-- const overload\n",
              reached_const(declared));
  std::printf("  double[2]                       %d\n", reached_const(scalars));
  // declared_pair_mutable_only mutable_only{1.0, 2.0};
  // std::printf("  non-const walk only             %d\n", reached_const(mutable_only));

  std::printf(
      "\nSo a type standing in for a pair on a recorded path must declare\n"
      "for_each_active, and declaring it on the holder is not enough. What\n"
      "reports a miss is active_system::release, counting slots the walk\n"
      "did not reach.\n"
      "\nAND IT MUST DECLARE A CONST OVERLOAD. Dispatch turns on whether the\n"
      "call compiles, so a non-const walk handed a const object drops that arm\n"
      "and the shape is passed over -- which is how a type reaches\n"
      "implicit_value contributing no rows at all. visit_active refuses that\n"
      "shape at compile time rather than walking past it; declared_pair_mutable_only\n"
      "above is what it refuses.\n");
  return 0;
}

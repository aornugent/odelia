// What a trapezium reduction costs on the tape when its widths are positions
// rather than state.
//
// The reduction is `tot += (x0 - x1) * (f1 + f0)` over a grid. The abscissae are
// positions -- a knot, a height, a birth date -- and carry no derivative, but a
// model that declares them in the scalar type it uses for everything else hands
// each width to the tape anyway. This asks what that costs, in statements, in
// operations and in slots, against the same arithmetic with the abscissae left
// as double.
//
// Built to be run by hand and not part of `all`:
//   make probe_trapezium && ./probe_trapezium
#include <XAD/XAD.hpp>

#include <cstdio>
#include <vector>

namespace {

using mode = xad::adj<double>;
using AD = mode::active_type;
using Tape = mode::tape_type;

struct cost {
  std::size_t statements, operations, slots;
};

// Every spelling records the same arithmetic against the same registered input,
// so a difference in these three numbers is the spelling's and nothing else's.
template <class F>
cost recorded(F&& record) {
  Tape tape;
  AD seed = 1.0;
  tape.registerInput(seed);
  tape.newRecording();
  const std::size_t s0 = tape.getNumStatements(), o0 = tape.getNumOperations(),
                    v0 = tape.getNumVariables();
  AD out = record(seed);
  tape.registerOutput(out);
  return {tape.getNumStatements() - s0, tape.getNumOperations() - o0,
          tape.getNumVariables() - v0};
}

void report(const char* what, cost c, int n) {
  std::printf("  %-46s %6zu %6zu %6zu   %5.1f %5.1f\n", what, c.statements,
              c.operations, c.slots, double(c.statements) / n,
              double(c.operations) / n);
}

const int N = 100;

// The abscissae as the scalar type the model uses for its state.
AD active_widths(AD seed) {
  std::vector<AD> x(N), f(N);
  for (int i = 0; i < N; ++i) {
    x[i] = double(i);           // a position, in an active
    f[i] = seed * double(i + 1);
  }
  AD tot = 0.0;
  for (int i = 1; i < N; ++i) tot += (x[i] - x[i - 1]) * (f[i - 1] + f[i]);
  return tot;
}

// The same arithmetic with the positions left as what they are.
AD passive_widths(AD seed) {
  std::vector<double> x(N);
  std::vector<AD> f(N);
  for (int i = 0; i < N; ++i) {
    x[i] = double(i);
    f[i] = seed * double(i + 1);
  }
  AD tot = 0.0;
  for (int i = 1; i < N; ++i) tot += (x[i] - x[i - 1]) * (f[i - 1] + f[i]);
  return tot;
}

// Whether an active holding a position is a slot at all, which decides what a
// release audit and a per-knot vector of these actually hold.
void a_constant_active() {
  Tape tape;
  AD seed = 1.0;
  tape.registerInput(seed);
  tape.newRecording();
  const std::size_t v0 = tape.getNumVariables();
  std::vector<AD> x(N);
  for (int i = 0; i < N; ++i) x[i] = double(i);
  std::printf("  %d actives assigned from double: slots %zu, statements %zu\n", N,
              std::size_t(tape.getNumVariables() - v0),
              std::size_t(tape.getNumStatements()));
}

// A small struct carrying one active, as a cached scan of a state vector does.
// Returning it by value copies that active, and a copy of a registered value is
// a statement -- which is what decides whether an accessor called once per knot
// should hand back a reference.
struct scan {
  AD h_max;
  bool ordered;
};

AD returned_by_value(AD seed) {
  const scan cached{seed * 2.0, true};
  auto get = [&]() -> scan { return cached; };
  AD tot = 0.0;
  for (int i = 0; i < N; ++i) {
    const scan s = get();
    tot += s.h_max;
  }
  return tot;
}

AD returned_by_reference(AD seed) {
  const scan cached{seed * 2.0, true};
  auto get = [&]() -> const scan& { return cached; };
  AD tot = 0.0;
  for (int i = 0; i < N; ++i) {
    const scan& s = get();
    tot += s.h_max;
  }
  return tot;
}

}  // namespace

int main() {
  std::printf("A trapezium reduction over %d points (XAD, adj<double>)\n\n", N);
  std::printf("  %-46s %6s %6s %6s   %5s %5s\n", "", "stmts", "ops", "slots",
              "s/pt", "o/pt");
  report("abscissae in the active scalar type", recorded(active_widths), N);
  report("abscissae left as double", recorded(passive_widths), N);
  std::printf("\n");
  a_constant_active();
  std::printf("\n  the same struct read %d times\n", N);
  std::printf("  %-46s %6s %6s %6s   %5s %5s\n", "", "stmts", "ops", "slots",
              "s/rd", "o/rd");
  report("returned by value", recorded(returned_by_value), N);
  report("returned by const reference", recorded(returned_by_reference), N);
  return 0;
}

// What a derivative width costs against the walks it replaces.
//
// A tape of width N holds N adjoint components in every slot, so one sweep
// carries N seeds. The sweep reads each recorded operation once whatever the
// width -- its inner expression is `derivatives_[slot] += mul * a`, and a
// width-N `a` is a Vec of N -- so a width shares the READING of the operations
// and never the arithmetic they drive, and it multiplies the derivative array
// the arithmetic scatters into. The two pull opposite ways and only a
// measurement decides which wins.
//
// The recording is not timed: it is the same recording either way. Built to be
// run by hand -- `make probe_width && ./probe_width` -- and not part of `all`.
#include <XAD/XAD.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

// The tape runtime's definitions live in exactly one object file by design, and
// the widths below are not among the ones that file instantiates, so this probe
// compiles its own copy and links alone.
#include "../../src/Tape.cpp"
namespace xad {
template class Tape<double, 2>;
template class Tape<double, 3>;
template class Tape<double, 4>;
template class CheckpointCallback<Tape<double, 2>>;
template class CheckpointCallback<Tape<double, 3>>;
template class CheckpointCallback<Tape<double, 4>>;
}

namespace {

using clock_type = std::chrono::steady_clock;

double seconds(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

// A reduction shaped like the ones a model records: a weighted sum over
// elements whose weight is a function of a query coordinate, with `operands`
// active operands per statement, which is what sets operations per statement.
template <class S>
void reduce(const std::vector<S>& x, std::vector<S>& y, int operands) {
  const std::size_t n = x.size();
  for (std::size_t q = 0; q < y.size(); ++q) {
    S acc = 0.0;
    const double w = 1.0 / static_cast<double>(q + 1);
    for (std::size_t i = 0; i + std::size_t(operands) <= n;
         i += std::size_t(operands)) {
      switch (operands) {
      case 2:
        acc += x[i] * w + x[i + 1] * (w * 0.5);
        break;
      case 4:
        acc += x[i] * w + x[i + 1] * (w * 0.5) + x[i + 2] * (w * 0.25) +
               x[i + 3] * (w * 0.125);
        break;
      case 8:
        acc += x[i] * w + x[i + 1] * (w * 0.5) + x[i + 2] * (w * 0.25) +
               x[i + 3] * (w * 0.125) + x[i + 4] * (w * 0.0625) +
               x[i + 5] * (w * 0.03) + x[i + 6] * (w * 0.015) +
               x[i + 7] * (w * 0.0075);
        break;
      default:
        acc += x[i];
        break;
      }
    }
    y[q] = acc;
  }
}

struct result {
  double sweep_s = 0.0;
  std::size_t statements = 0, operations = 0, slots = 0;
  std::vector<std::vector<double>> rows;
};

// `n_seed` seeds over one recording on a tape of width `N`: one sweep where the
// width carries them all, and `n_seed` sweeps where the width is one.
template <std::size_t N>
result sweep(int n_in, int n_out, int operands, int n_seed) {
  using tape_type = xad::Tape<double, N>;
  using active = typename tape_type::active_type;

  tape_type tape(false);
  tape.activate();
  std::vector<active> x(n_in);
  for (int i = 0; i < n_in; ++i) x[i] = 1.0 + 0.01 * i;
  tape.registerInputs(x);
  tape.newRecording();
  std::vector<active> y(n_out);
  reduce(x, y, operands);
  tape.registerOutputs(y);

  result r;
  r.statements = tape.getNumStatements();
  r.operations = tape.getNumOperations();
  r.slots = tape.getNumVariables();

  const clock_type::time_point t0 = clock_type::now();
  if constexpr (N == 1) {
    for (int m = 0; m < n_seed; ++m) {
      tape.clearDerivatives();
      xad::derivative(y[m % n_out]) = 1.0;
      tape.computeAdjoints();
      std::vector<double> row(n_in);
      for (int i = 0; i < n_in; ++i) row[i] = xad::derivative(x[i]);
      r.rows.push_back(std::move(row));
    }
  } else {
    tape.clearDerivatives();
    for (int m = 0; m < n_seed; ++m) xad::derivative(y[m % n_out])[m] = 1.0;
    tape.computeAdjoints();
    for (int m = 0; m < n_seed; ++m) {
      std::vector<double> row(n_in);
      for (int i = 0; i < n_in; ++i) row[i] = xad::derivative(x[i])[m];
      r.rows.push_back(std::move(row));
    }
  }
  r.sweep_s = seconds(t0, clock_type::now());
  tape.deactivate();
  return r;
}

bool identical(const std::vector<std::vector<double>>& a,
               const std::vector<std::vector<double>>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t m = 0; m < a.size(); ++m) {
    if (a[m].size() != b[m].size()) return false;
    for (std::size_t i = 0; i < a[m].size(); ++i) {
      if (a[m][i] != b[m][i]) return false;
    }
  }
  return true;
}

template <std::size_t N>
void one_width(int n_in, int n_out, int operands) {
  const int n_seed = int(N);
  double best_batched = 1e30, best_serial = 1e30;
  result batched, serial;
  for (int rep = 0; rep < 3; ++rep) {
    serial = sweep<1>(n_in, n_out, operands, n_seed);
    batched = sweep<N>(n_in, n_out, operands, n_seed);
    if (serial.sweep_s < best_serial) best_serial = serial.sweep_s;
    if (batched.sweep_s < best_batched) best_batched = batched.sweep_s;
  }
  std::printf("%6zu %10zu %9.2f %9zu %11.4f %11.4f %8.3f %9s\n", N,
              serial.statements,
              double(serial.operations) / double(serial.statements),
              serial.slots, best_serial, best_batched,
              best_serial / best_batched,
              identical(serial.rows, batched.rows) ? "bitwise" : "NO");
}

void table(const char* what, int n_in, int n_out) {
  for (const int operands : {2, 4, 8}) {
    std::printf("\n%s, %d active operands a statement\n", what, operands);
    std::printf("%6s %10s %9s %9s %11s %11s %8s %9s\n", "width", "stmts",
                "ops/stmt", "slots", "N x at 1", "1 x at N", "ratio", "agree");
    one_width<2>(n_in, n_out, operands);
    one_width<3>(n_in, n_out, operands);
    one_width<4>(n_in, n_out, operands);
  }
}

}  // namespace

int main() {
  // Two regimes, because the answer differs between them: the derivative array
  // is what a width multiplies, and whether it still fits in cache at N times
  // its size is what decides whether the shared reads pay for the scatter.
  table("derivative array in cache", 8000, 400);
  table("derivative array past cache", 1000000, 40);
  return 0;
}

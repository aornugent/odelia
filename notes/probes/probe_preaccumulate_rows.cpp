// Does preaccumulate leave a dirty adjoint on a pre-mark active its declared
// input list does not name?
//
// u = (x*y) * x   preaccumulated, declaring ONLY x.
// w = 5*u + 7*y   the consumer, which reads y itself.
// The region reads y, so y's slot is swept below the mark; nothing zeroes it,
// and the consumer's own sweep then adds to whatever was left.
#include <odelia/implicit_node.hpp>
#include <odelia/adjoint.hpp>
#include <cstdio>
#include <span>
#include <vector>

using A = odelia::ode::active_scalar<double>;
using Tape = odelia::ode::adjoint_tape<double>;

static void run(bool declare_y) {
  Tape tape;
  A x = 2.0, y = 3.0;
  tape.registerInput(x); tape.registerInput(y);
  tape.newRecording();

  A u;
  static A* outs[1];
  std::vector<double> scratch;
  auto region = [&]() -> std::span<A* const> {
    A shared = x * y;
    u = shared * x;            // u = x^2 y
    outs[0] = &u;
    return std::span<A* const>(outs, 1);
  };
  if (declare_y) odelia::preaccumulate<A>(region, scratch, x, y);
  else           odelia::preaccumulate<A>(region, scratch, x);

  A w = 5.0 * u + 7.0 * y;     // dw/dy = 5*x^2 + 7 = 27 ; dw/dx = 5*2xy = 60
  tape.registerOutput(w);
  xad::derivative(w) = 1.0;
  tape.computeAdjoints();
  std::printf("  declaring %-6s dw/dx = %6.2f (true 60)   dw/dy = %6.2f (true 27)%s\n",
              declare_y ? "x,y" : "x",
              xad::derivative(x), xad::derivative(y),
              (declare_y ? "" : "   <-- y's row is undeclared"));
}

static void contamination() {
  for (int dirty = 0; dirty < 2; ++dirty) {
    Tape tape;
    A x = 2.0, y = 3.0;
    tape.registerInput(x); tape.registerInput(y);
    tape.newRecording();

    if (dirty) {
      // An earlier sweep in the same recording, as a consumer taking a partial
      // answer would do. It leaves 9 on x's slot.
      A pre = 9.0 * x;
      tape.registerOutput(pre);
      xad::derivative(pre) = 1.0;
      tape.computeAdjoints();
    }

    A u; static A* outs[1]; std::vector<double> scratch;
    auto region = [&]() -> std::span<A* const> {
      u = x * x * y;                       // du/dx = 2xy = 12
      outs[0] = &u;
      return std::span<A* const>(outs, 1);
    };
    odelia::preaccumulate<A>(region, scratch, x, y);
    std::printf("  x's slot dirty=%d   harvested du/dx = %6.2f (true 12)%s\n",
                dirty, scratch[0], (dirty && scratch[0] != 12.0) ? "   <-- CONTAMINATED" : "");
  }
}

int main() {
  std::printf("the caller's own adjoint on a declared input slot\n");
  contamination();
  std::printf("\npreaccumulate, a region reading an active the input list omits\n");
  run(true);
  run(false);
  return 0;
}

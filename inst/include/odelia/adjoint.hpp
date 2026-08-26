// -*-c++-*-
#ifndef ODELIA_ADJOINT_HPP_
#define ODELIA_ADJOINT_HPP_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>
#include <XAD/XAD.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_util.hpp>

namespace odelia {
namespace ode {

// One row per seed, every row the same width: what a transpose is seeded with,
// and what it hands back. The shape is the object's own, so a ragged batch and a
// row of the wrong width cannot be built -- where a vector of vectors builds
// both, and every function on the path then has to test for them.
//
// The width is a run-time number on purpose. A sweep narrows as it descends, and
// how many seeds a caller wants is the caller's; neither is a property of a type.
class row_batch {
public:
  row_batch() = default;
  row_batch(std::size_t rows, std::size_t width) { assign(rows, width); }

  std::size_t rows() const { return rows_; }
  std::size_t width() const { return width_; }
  bool empty() const { return rows_ == 0; }

  // Every row zeroed at a new shape. A sweep hands back rows at the width of the
  // segment it swept, so a row the new shape does not reach must not survive as
  // a number from the last one.
  void assign(std::size_t rows, std::size_t width) {
    rows_ = rows;
    width_ = width;
    store_.assign(rows * width, 0.0);
  }

  std::span<double> operator[](std::size_t m) {
    return {store_.data() + m * width_, width_};
  }
  std::span<const double> operator[](std::size_t m) const {
    return {store_.data() + m * width_, width_};
  }

  // The rows `which` names, in the order it names them. Out of range is refused
  // rather than clamped: a caller indexing by position would otherwise be handed
  // a different row than the one it asked for.
  row_batch select(const std::vector<std::size_t>& which) const {
    row_batch ret(which.size(), width_);
    for (std::size_t m = 0; m < which.size(); ++m) {
      if (which[m] >= rows_) {
        util::stop("row_batch::select: row " +
                   util::to_string(static_cast<int>(which[m])) +
                   " is outside a batch of " +
                   util::to_string(static_cast<int>(rows_)));
      }
      const std::span<const double> from = (*this)[which[m]];
      std::copy(from.begin(), from.end(), ret[m].begin());
    }
    return ret;
  }

  // A batch of one, which is how a caller wanting a single transpose row asks for
  // it: there is no separate entry point for one seed.
  static row_batch one_row(const std::vector<double>& values) {
    row_batch ret(1, values.size());
    std::copy(values.begin(), values.end(), ret[0].begin());
    return ret;
  }

  // The seeds that ask for every row of a Jacobian: one per output, each picking
  // that output out.
  static row_batch all_rows(std::size_t rows) {
    row_batch ret(rows, rows);
    for (std::size_t m = 0; m < rows; ++m) {
      ret[m][m] = 1.0;
    }
    return ret;
  }

  // Rows of doubles for a caller that hands them on -- the R boundary, which
  // takes nothing else. Not how anything inside works.
  std::vector<std::vector<double>> to_rows() const {
    std::vector<std::vector<double>> ret;
    ret.reserve(rows_);
    for (std::size_t m = 0; m < rows_; ++m) {
      const std::span<const double> row = (*this)[m];
      ret.emplace_back(row.begin(), row.end());
    }
    return ret;
  }

private:
  // Set together and nowhere else, so the extent and the shape it is meant to
  // hold cannot disagree.
  std::size_t rows_ = 0, width_ = 0;
  std::vector<double> store_;
};

// The tape running for the duration, on every exit and exceptions included.
//
// ⚠️ ACTIVATED HERE ONLY IF NOTHING ELSE HOLDS IT, AND DEACTIVATED ONLY WHERE
// ACTIVATED. Activating a tape twice raises, so a caller holding one across
// several recordings cannot be nested inside by anything that activates; and
// deactivating one an outer scope is holding stops recording mid-recording,
// which reads as a missing derivative term and raises nothing.
template <typename Tape>
struct tape_scope {
  explicit tape_scope(Tape& tape) {
    Tape* const running = Tape::getActive();
    if (running == nullptr) {
      tape.activate();
      held_ = &tape;
    } else if (running != &tape) {
      util::stop("a tape is already active and it is not the one being "
                 "recorded on");
    }
  }
  ~tape_scope() {
    if (held_ != nullptr) {
      held_->deactivate();
    }
  }
  tape_scope(const tape_scope&) = delete;
  tape_scope& operator=(const tape_scope&) = delete;

private:
  Tape* held_ = nullptr;
};

// Hand a value's tape slot back, leaving its number. Move-assigning a freshly
// constructed scalar swaps the two slots, so the temporary carries the old one
// away and its destructor releases it, and nothing is recorded against either.
//
// ⚠️ BEFORE A TAPE RESET AND NOT AFTER. Clearing a tape returns its slot counter
// to zero, and handing back a slot the counter no longer knows about takes it
// below zero; where the library is built to reuse freed slots it then issues the
// same number twice. Released first and cleared second, both stay consistent.
template <class S>
void release_slot(S& x) {
  x = S(util::to_passive(x));
}

// A System lifted to the adjoint scalar, with the addresses of the parameters a
// recording writes through. Both come off one object, so a caller cannot pair a
// System with another System's parameter addresses.
//
// Held across the recordings a walk takes on it, where a copy per recording costs
// every allocation the System owns -- for a stand, every cohort and the light
// field with them.
//
// ⚠️ WHAT MAKES THAT LEGAL IS `release`, AND NOTHING ELSE. Writing a member does
// not refresh the slot it carries: assignment keeps the slot the target already
// had, so a member that survives a tape clear writes through a number the next
// recording issues to something else. The rows stay finite and stop being the
// model's, and only once two recordings differ in shape -- which is every sweep
// that narrows. So every active value the System holds is released before the
// clear, and the tape is asked whether any was missed.
template <class System>
struct lifted_system {
  using scalar = active_scalar<double>;
  using system_type = typename rebound_system<System, scalar>::type;

  // Declared first, so the destructor's tape is initialised before the System it
  // releases. Read back rather than handed to a transpose beside the System: the
  // two arriving separately is a pairing a caller can get wrong.
  adjoint_tape<double>* tape_;
  adjoint_tape<double>& tape() const { return *tape_; }

  lifted_system(const System& passive, adjoint_tape<double>& tape)
      : tape_(&tape), system(passive.template rebind_from<scalar>()),
        parameters(system.ad_parameters()) {
    static_assert(Rebindable<System, scalar>,
                  "a recording is taken on this System at the adjoint scalar; "
                  "it has no rebind_from()");
  }

  // Handed back on the way out, so a walk that builds one of these per width
  // leaves the tape as it found it. Without this, every width but the last stays
  // registered -- nothing is active when the object goes, so nothing hands its
  // slots back, and the tape zero-fills them on every later recording.
  //
  // Refused where another tape is active: these slots are not on it, and
  // handing them back would decrement its count instead.
  ~lifted_system() {
    adjoint_tape<double>* const running = adjoint_tape<double>::getActive();
    if (running != nullptr && running != tape_) {
      return;
    }
    tape_scope<adjoint_tape<double>> holding{*tape_};
    system.for_each_active([](scalar& x) { release_slot(x); });
  }

  // The parameter addresses point into `system`, so a copy would hand its own
  // parameters to the object it was copied from. Declaring the destructor above
  // suppresses the move operations, which would leave those addresses pointing
  // into the object moved from.
  lifted_system(const lifted_system&) = delete;
  lifted_system& operator=(const lifted_system&) = delete;

  // Every active value back to unregistered, so the clear below it is safe and
  // the next recording registers them fresh.
  //
  // The check is the point. Releasing every value `for_each_active` reaches has
  // to return the tape to the number of values it counted before this System
  // existed -- so a System holding one the walk does not reach says so here,
  // rather than in a row that is finite and no longer the model's. That is an
  // audit of the model turned into one number.
  //
  // Against zero, which the destructor above is what makes true: every clear
  // returns the count to zero, every recording registers this System's values,
  // and the temporaries a recording makes are gone by the time it ends.
  void release() {
    system.for_each_active([](scalar& x) { release_slot(x); });
    const std::size_t still = tape_->getNumVariables();
    if (still != 0) {
      util::stop(
          "lifted_system::release: " +
          util::to_string(static_cast<int>(still)) +
          " active values of this System are still registered after releasing "
          "every one for_each_active reaches, so it holds some the walk does "
          "not. A recording that reads one of those reads a slot the next clear "
          "issues to something else.");
    }
  }

  system_type system;
  std::vector<scalar*> parameters;
};

// One block of `f`, recorded ONCE on the tape handed in and swept once per seed:
// `input_adjoints[m]` receives transpose(jacobian) * output_adjoints[m], and the return
// value is the recording's size. `f` is instantiated at the active scalar here, so only
// doubles cross in and out.
//
// One seed is a batch of one, and there is no separate entry point for it. Where the
// recording is the expensive part -- which it is whenever `f` is a model evaluation
// rather than arithmetic -- a caller wanting several rows pays one recording rather than
// one per row, and a second signature over the same recording is a second place for the
// seam between the recording and the sweep to be got wrong.
//
// The batch carries its own shape, so a seed of the wrong width is not something a
// caller can hand in and not something this has to test for.
//
// Each sweep is bit-identical to the row a fresh recording of `f` would give, because
// clearDerivatives() returns the tape's derivative slots to zero while leaving the
// recorded operations alone. That is what makes one recording substitutable for many
// rather than an approximation of them.
//
// The tape is the caller's and is reused across calls, so nothing here allocates one; a
// tape costs about a fifth of this whole product and the product runs millions of times
// per gradient. Stops if a tape other than this one is active: recording onto a tape this
// product does not own would sweep the block's adjoints twice.
//
// An empty seed is swept anyway rather than skipped: the row is then zeros, which is
// what the caller's accumulator expects, and skipping would make the result depend on
// which seeds happen to vanish at this state.
template <class F>
std::size_t vector_jacobian_product(adjoint_tape<double>& tape,
                                     const std::vector<double>& x,
                                     const row_batch& output_adjoints,
                                     F&& f,
                                     row_batch& input_adjoints) {
    using ad_type = active_scalar<double>;
    using tape_type = adjoint_tape<double>;

    if (x.empty()) {
        util::stop("vector_jacobian_product: 'x' must have at least one entry");
    }
    if (output_adjoints.empty()) {
        util::stop("vector_jacobian_product: needs at least one seed");
    }
    const std::size_t n_out = output_adjoints.width();
    if (n_out == 0) {
        util::stop("vector_jacobian_product: a seed must have at least one entry");
    }

    tape_scope<tape_type> running{tape};
    tape.clearAll();

    std::vector<ad_type> x_active(x.begin(), x.end());
    tape.registerInputs(x_active);
    tape.newRecording();

    std::vector<ad_type> y_active(n_out);
    f(x_active, y_active);
    if (y_active.size() != n_out) {
        util::stop("vector_jacobian_product: 'f' resized the output buffer; it is "
                   "handed one entry per output adjoint and must write in place");
    }
    tape.registerOutputs(y_active);

    input_adjoints.assign(output_adjoints.rows(), x.size());
    for (std::size_t m = 0; m < output_adjoints.rows(); ++m) {
        // Between sweeps, or the previous seed's adjoints are still on the slots and
        // every row after the first is the running sum of the ones before it.
        tape.clearDerivatives();
        // The adjoint slots, named directly rather than through an accessor of
        // their own: this is the only place in the family that touches one, the
        // scalar is fixed two statements above, and a name for two sites inside
        // the function that owns the tape would say nothing the tape does not.
        for (std::size_t i = 0; i < y_active.size(); ++i) {
            xad::derivative(y_active[i]) = output_adjoints[m][i];
        }
        tape.computeAdjoints();
        for (std::size_t i = 0; i < x_active.size(); ++i) {
            input_adjoints[m][i] = xad::derivative(x_active[i]);
        }
    }

    return tape.getMemory();
}

// A transpose taken with respect to a System's state AND the parameters its
// active-scalar lists: one recording over both, swept once per seed.
//
// The lifted System is the caller's, held across the recordings a walk takes on
// it. It is released here rather than there, because the release has to sit
// immediately before the clear and the clear is inside the product below.
//
// `evaluate` is handed that System, the state half of the recorded inputs, and
// the output buffer, with the System's parameters ALREADY written from the other
// half. That order is why the two halves are one recording rather than two
// calls: a quantity the state determines reads the parameters while deriving it,
// so a state written first derives it at the previous values.
//
// The sweep splits back along the same seam, and the two halves are handled
// differently. `state_adjoint` is REPLACED: it is resized to one row per seed and
// each row assigned, so rows a shorter batch does not reach do not survive as
// stale numbers. `parameter_adjoint` is ADDED to, because a parameter is reached
// once per step and its gradient is the sum over every step swept, so it is the
// caller's to clear once per sweep and may carry rows past the seeds handed in.
//
// Those two are therefore different objects, and one being the other would mean
// resizing the accumulator between the check on its rows and the writes into
// them. Refused rather than documented.
//
// One width for the whole batch is what removes the other check this used to
// carry: a row of parameter adjoints cannot be a different length from its
// neighbours, so there is nothing to test per row.
template <class System, class Evaluate>
std::size_t state_and_parameter_adjoints(
    lifted_system<System>& active, const std::vector<double>& state,
    const row_batch& output_adjoints, Evaluate&& evaluate,
    row_batch& state_adjoint, row_batch& parameter_adjoint) {
    using scalar = active_scalar<double>;
    adjoint_tape<double>& tape = active.tape();
    const std::vector<scalar*>& parameters = active.parameters;
    const std::size_t n_state = state.size();
    const std::size_t n_parameter = parameters.size();
    const std::size_t n_seed = output_adjoints.rows();

    if (&state_adjoint == &parameter_adjoint) {
        util::stop("state_and_parameter_adjoints: the state adjoints are "
                   "replaced and the parameter adjoints accumulated, so they "
                   "cannot be the same batch");
    }
    if (parameter_adjoint.rows() < n_seed) {
        util::stop("state_and_parameter_adjoints: one row of parameter adjoints "
                   "per seed, to accumulate into");
    }
    util::check_length(parameter_adjoint.width(), n_parameter);

    std::vector<double> in(state);
    in.reserve(n_state + n_parameter);
    for (const scalar* p : parameters) {
        in.push_back(util::to_passive(*p));
    }

    // Held for the release and the product both, and the release comes first: a
    // slot is handed back by the destructor of the temporary that takes it, and
    // that destructor does nothing with no tape active.
    tape_scope<adjoint_tape<double>> running{tape};
    active.release();

    auto record = [&](const std::vector<scalar>& x,
                      std::vector<scalar>& y) -> void {
        std::size_t at = n_state;
        for (scalar* p : parameters) {
            *p = x[at++];
        }
        evaluate(active.system, x.begin(), y);
    };

    row_batch in_adjoint;
    const std::size_t recording =
        vector_jacobian_product(tape, in, output_adjoints, record, in_adjoint);

    state_adjoint.assign(n_seed, n_state);
    for (std::size_t m = 0; m < n_seed; ++m) {
        const std::span<const double> row = in_adjoint[m];
        std::copy(row.begin(), row.begin() + static_cast<std::ptrdiff_t>(n_state),
                  state_adjoint[m].begin());
        for (std::size_t p = 0; p < n_parameter; ++p) {
            parameter_adjoint[m][p] += row[n_state + p];
        }
    }
    return recording;
}

// The transpose of one rate evaluation: `state_adjoint[m]` receives
// transpose(d dydt / d y) * rate_adjoints[m], with the System's parameters in the
// same recording, so a rate the parameters reach carries their rows too.
//
// What is recorded is derivs() -- the call the forward pass makes -- so the
// transpose cannot drift from the rates it transposes, and a System that
// restores a recorded field restores it here as well. Everything between the
// state and the rates is an intermediate of that one recording, so nothing
// between them needs a transpose written for it.
//
// The rate is taken at the time, not at a recorded stage. The System recorded on
// is lifted for this recording and holds no recorded field, so there is no stage
// for it to read back -- a stage argument here would name a capability the lifted
// copy cannot have.
template <class System>
std::size_t rates_adjoint(
    lifted_system<System>& active,
    const std::vector<double>& state, double time,
    const row_batch& rate_adjoints, row_batch& state_adjoint,
    row_batch& parameter_adjoint) {
    using scalar = active_scalar<double>;
    const std::size_t n_state = state.size();
    auto rates = [&](auto& active_system,
                     typename std::vector<scalar>::const_iterator x,
                     std::vector<scalar>& dydt) -> void {
        // The state half of the recorded inputs; the parameters are already
        // written from the other half.
        std::vector<scalar> y(x, x + static_cast<std::ptrdiff_t>(n_state));
        ode::derivs(active_system, y, dydt, time);
    };
    return state_and_parameter_adjoints(active, state, rate_adjoints, rates,
                                        state_adjoint, parameter_adjoint);
}

}
}

#endif

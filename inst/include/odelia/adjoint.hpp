// -*-c++-*-
#ifndef ODELIA_ADJOINT_HPP_
#define ODELIA_ADJOINT_HPP_

#include <cstddef>
#include <vector>
#include <XAD/XAD.hpp>
#include <odelia/ode_interface.hpp>
#include <odelia/ode_util.hpp>

namespace odelia {
namespace ode {

// Deactivates the tape on every exit from the product below, exceptions included.
template <typename Tape>
struct tape_guard {
  Tape* tape;
  ~tape_guard() { tape->deactivate(); }
};

// One block of `f`, recorded and swept once on the tape handed in: `input_adjoints`
// receives transpose(jacobian) * output_adjoints, and the return value is the recording's
// size. `f` is instantiated at the active scalar here, so only doubles cross in and out.
//
// The tape is the caller's and is reused across calls, so nothing here allocates one; a
// tape costs about a fifth of this whole product and the product runs millions of times
// per gradient.
//
// Stops if a tape other than this one is active. Recording onto a tape this product does
// not own would sweep the block's adjoints twice, so "the tape handed in is the only one"
// is checked rather than assumed.
template <class F>
std::size_t vector_jacobian_product(xad::adj<double>::tape_type& tape,
                                    const std::vector<double>& x,
                                    const std::vector<double>& output_adjoints, F&& f,
                                    std::vector<double>& input_adjoints) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    ad::tape_type* active = ad::tape_type::getActive();
    if (active != nullptr && active != &tape) {
        util::stop("vector_jacobian_product: a tape is already active");
    }
    if (x.empty()) {
        util::stop("vector_jacobian_product: 'x' must have at least one entry");
    }
    if (output_adjoints.empty()) {
        util::stop("vector_jacobian_product: 'output_adjoints' must have at least one entry");
    }

    tape.activate();
    tape_guard<ad::tape_type> guard{&tape};

    // clearAll() returns the tape to an empty recording with its derivative-slot counter
    // back at zero. newRecording() alone leaves that counter where the previous call left
    // it -- destroying a registered input only releases its slot when the slot is the last
    // one, which it is not for a vector destroyed front to back -- so the tape's memory and
    // variable count would climb with every call while the adjoints stayed correct.
    tape.clearAll();

    // Inputs are registered before newRecording(). Registering after it leaves them outside
    // the recording, and the sweep then reports every input adjoint as zero with nothing
    // thrown.
    std::vector<ad_type> x_active(x.begin(), x.end());
    tape.registerInputs(x_active);
    tape.newRecording();

    std::vector<ad_type> y_active(output_adjoints.size());
    f(x_active, y_active);
    if (y_active.size() != output_adjoints.size()) {
        util::stop("vector_jacobian_product: 'f' resized the output buffer; it is "
                   "handed one entry per output adjoint and must write in place");
    }
    tape.registerOutputs(y_active);

    for (std::size_t i = 0; i < y_active.size(); ++i) {
        xad::derivative(y_active[i]) = output_adjoints[i];
    }
    tape.computeAdjoints();

    // The caller owns the buffer and reuses it across calls, so this resize is a no-op
    // after the first call and the product never allocates its own result.
    input_adjoints.resize(x.size());
    for (std::size_t i = 0; i < x_active.size(); ++i) {
        input_adjoints[i] = xad::derivative(x_active[i]);
    }

    return tape.getMemory();
}

// The same product for a caller with no tape to reuse: one call, one tape. Constructed
// inactive, so a tape already active here belongs to someone else and the overload above
// stops on it.
template <class F>
std::size_t vector_jacobian_product(const std::vector<double>& x,
                                    const std::vector<double>& output_adjoints, F&& f,
                                    std::vector<double>& input_adjoints) {
    xad::adj<double>::tape_type tape(false);
    return vector_jacobian_product(tape, x, output_adjoints, std::forward<F>(f), input_adjoints);
}

// The same block recorded ONCE and swept once per seed: `input_adjoints[m]` receives
// transpose(jacobian) * output_adjoints[m]. A caller wanting several rows of the same
// block pays one recording rather than one per row, and where the recording is the
// expensive part -- which is the case whenever `f` is a model evaluation rather than
// arithmetic -- that is the whole of the cost.
//
// Each sweep is bit-identical to the row a fresh recording of `f` would give, because
// clearDerivatives() returns the tape's derivative slots to zero while leaving the
// recorded operations alone. That is what makes this substitutable for the single-seed
// form above rather than an approximation of it.
//
// An empty seed is swept anyway rather than skipped: the row is then zeros, which is
// what the caller's accumulator expects, and skipping would make the result depend on
// which seeds happen to vanish at this state.
template <class F>
std::size_t vector_jacobian_products(xad::adj<double>::tape_type& tape,
                                     const std::vector<double>& x,
                                     const std::vector<std::vector<double>>& output_adjoints,
                                     F&& f,
                                     std::vector<std::vector<double>>& input_adjoints) {
    using ad = xad::adj<double>;
    using ad_type = ad::active_type;

    ad::tape_type* active = ad::tape_type::getActive();
    if (active != nullptr && active != &tape) {
        util::stop("vector_jacobian_products: a tape is already active");
    }
    if (x.empty()) {
        util::stop("vector_jacobian_products: 'x' must have at least one entry");
    }
    if (output_adjoints.empty()) {
        util::stop("vector_jacobian_products: needs at least one seed");
    }
    const std::size_t n_out = output_adjoints.front().size();
    if (n_out == 0) {
        util::stop("vector_jacobian_products: a seed must have at least one entry");
    }
    for (const std::vector<double>& s : output_adjoints) {
        if (s.size() != n_out) {
            util::stop("vector_jacobian_products: every seed must have one entry per "
                       "output; they are swept over one recording of the same block");
        }
    }

    tape.activate();
    tape_guard<ad::tape_type> guard{&tape};
    tape.clearAll();

    std::vector<ad_type> x_active(x.begin(), x.end());
    tape.registerInputs(x_active);
    tape.newRecording();

    std::vector<ad_type> y_active(n_out);
    f(x_active, y_active);
    if (y_active.size() != n_out) {
        util::stop("vector_jacobian_products: 'f' resized the output buffer; it is "
                   "handed one entry per output adjoint and must write in place");
    }
    tape.registerOutputs(y_active);

    input_adjoints.resize(output_adjoints.size());
    for (std::size_t m = 0; m < output_adjoints.size(); ++m) {
        // Between sweeps, or the previous seed's adjoints are still on the slots and
        // every row after the first is the running sum of the ones before it.
        tape.clearDerivatives();
        for (std::size_t i = 0; i < y_active.size(); ++i) {
            xad::derivative(y_active[i]) = output_adjoints[m][i];
        }
        tape.computeAdjoints();
        input_adjoints[m].resize(x.size());
        for (std::size_t i = 0; i < x_active.size(); ++i) {
            input_adjoints[m][i] = xad::derivative(x_active[i]);
        }
    }

    return tape.getMemory();
}

// A transpose taken with respect to a System's state AND the parameters its
// active twin lists: one recording over both, swept once per seed.
//
// `evaluate` is handed the state half of the recorded inputs and the output
// buffer, with the twin's parameters ALREADY written from the other half. That
// order is why the two halves are one recording rather than two calls: a
// quantity the state determines reads the parameters while deriving it, so a
// state written first derives it at the previous values. Writing the parameters
// here rather than in `evaluate` is what keeps that from being a rule each
// caller has to remember.
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
// `twin` must be rebound for this call and not carried over from a previous one.
// Its parameters are written from the recorded inputs, so a twin that was written
// that way before arrives holding scalars from a recording this one has cleared,
// and the sweep comes back wrong -- not obviously wrong: one seed's rows can stay
// exact while another's do not. Nothing here can see the difference, which is why
// it is stated. A caller wanting to amortise the rebind has to make the twin's
// scalars fresh some other way.
template <class Twin, class Evaluate>
std::size_t state_and_parameter_adjoints(
    xad::adj<double>::tape_type& tape, Twin& twin,
    const std::vector<double>& state,
    const std::vector<std::vector<double>>& output_adjoints, Evaluate&& evaluate,
    std::vector<std::vector<double>>& state_adjoint,
    std::vector<std::vector<double>>& parameter_adjoint) {
    using scalar = active_scalar<double>;
    const std::vector<scalar*> parameters = twin.ad_parameters();
    const std::size_t n_state = state.size();
    const std::size_t n_parameter = parameters.size();
    const std::size_t n_seed = output_adjoints.size();

    if (&state_adjoint == &parameter_adjoint) {
        util::stop("state_and_parameter_adjoints: the state adjoints are "
                   "replaced and the parameter adjoints accumulated, so they "
                   "cannot be the same vector");
    }
    if (parameter_adjoint.size() < n_seed) {
        util::stop("state_and_parameter_adjoints: one row of parameter adjoints "
                   "per seed, to accumulate into");
    }
    for (std::size_t m = 0; m < n_seed; ++m) {
        util::check_length(parameter_adjoint[m].size(), n_parameter);
    }

    std::vector<double> in(state);
    in.reserve(n_state + n_parameter);
    for (const scalar* p : parameters) {
        in.push_back(util::to_passive(*p));
    }

    auto record = [&](const std::vector<scalar>& x,
                      std::vector<scalar>& y) -> void {
        std::size_t at = n_state;
        for (scalar* p : parameters) {
            *p = x[at++];
        }
        evaluate(x.begin(), y);
    };

    std::vector<std::vector<double>> in_adjoint;
    const std::size_t recording =
        vector_jacobian_products(tape, in, output_adjoints, record, in_adjoint);

    state_adjoint.assign(n_seed, std::vector<double>(n_state, 0.0));
    for (std::size_t m = 0; m < n_seed; ++m) {
        for (std::size_t j = 0; j < n_state; ++j) {
            state_adjoint[m][j] = in_adjoint[m][j];
        }
        for (std::size_t p = 0; p < n_parameter; ++p) {
            parameter_adjoint[m][p] += in_adjoint[m][n_state + p];
        }
    }
    return recording;
}

}
}

#endif

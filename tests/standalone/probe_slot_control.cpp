// What the tape lets a caller do to slots directly, and what it does not.
//
// The reverse pass rebinds a System per width and releases its actives before
// every clear. The question this answers is whether slot control could replace
// either: whether a caller can write statements and adjoints into slots of its
// choosing, in an order of its choosing, and whether the counter can be rewound
// to a mark instead of to zero.
//
// Every answer here is the vendored tape's, read out of `src/Tape.cpp` and then
// run, because two of them are behaviour rather than signature.
//
//   make probe_slot_control && ./probe_slot_control

#include <XAD/XAD.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using mode = xad::adj<double>;
using AD = mode::active_type;
using Tape = mode::tape_type;
using slot_type = Tape::slot_type;

int failures = 0;

void check(const char* what, double got, double want, double tol = 1e-12) {
    const bool ok = std::fabs(got - want) <= tol * (1.0 + std::fabs(want));
    std::printf("  %-58s %14.8g  %s\n", what, got, ok ? "ok" : "WRONG");
    if (!ok) {
        std::printf("  %-58s %14.8g  expected\n", "", want);
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// 1. A statement written by hand, with its operations in any order.
//
// A statement is (where its operations start, the slot it writes) and the sweep
// reads the lhs adjoint, zeroes it, and adds multiplier * adjoint into each
// operation's slot. So the arithmetic never has to happen on an AReal at all:
// pushAll takes the multipliers and the slots, pushLhs closes the statement, and
// the order the operations were pushed in is the order they are visited.
void hand_written_statement() {
    std::printf("\n1. y = 3a + 5b, pushed by hand, b's operation first\n");

    Tape tape;
    AD a = 2.0, b = 7.0;
    tape.registerInput(a);
    tape.registerInput(b);
    tape.newRecording();

    const double multipliers[2] = {5.0, 3.0};
    const slot_type slots[2] = {b.getSlot(), a.getSlot()};
    tape.pushAll(multipliers, slots, 2u);
    const slot_type y = tape.registerVariable();
    tape.pushLhs(y);

    tape.derivative(y) = 1.0;
    tape.computeAdjoints();

    check("d/da, written as the second operation", tape.derivative(a.getSlot()), 3.0);
    check("d/db, written as the first operation", tape.derivative(b.getSlot()), 5.0);
    std::printf("  statements %zu, operations %zu -- no AReal arithmetic ran\n",
                std::size_t(tape.getNumStatements()), std::size_t(tape.getNumOperations()));
}

// ---------------------------------------------------------------------------
// 2. A callback writing adjoints into slots of its choosing, mid-sweep.
//
// insertCallback puts a marker statement on the tape. When the sweep reaches it,
// everything above has already been accumulated: the callback reads the adjoint
// of whatever slots it declared as outputs and increments whatever slots it
// declared as inputs, by hand, in any order. The sweep then carries on below it.
struct SuppliedDerivative : xad::CheckpointCallback<Tape> {
    slot_type in = 0, out = 0;
    double dout_din = 0.0;
    int calls = 0;

    void computeAdjoint(Tape* tape) override {
        ++calls;
        const double adjoint_out = tape->getAndResetOutputAdjoint(out);
        tape->incrementAdjoint(in, dout_din * adjoint_out);
    }
};

void callback_writes_slots() {
    std::printf("\n2. w = 3 * sin(2x), with sin supplied by a callback\n");

    Tape tape;
    AD x = 0.4;
    tape.registerInput(x);
    tape.newRecording();

    const AD u = 2.0 * x;               // recorded
    AD v = std::sin(xad::value(u));     // NOT recorded: a value the callback owns
    tape.registerOutput(v);

    SuppliedDerivative supplied;
    supplied.in = u.getSlot();
    supplied.out = v.getSlot();
    supplied.dout_din = std::cos(xad::value(u));
    tape.insertCallback(&supplied);

    AD w = 3.0 * v;                     // recorded
    tape.registerOutput(w);

    tape.derivative(w.getSlot()) = 1.0;
    tape.computeAdjoints();

    check("dw/dx", tape.derivative(x.getSlot()), 3.0 * std::cos(0.8) * 2.0);
    std::printf("  callback ran %d time(s)\n", supplied.calls);
}

// ---------------------------------------------------------------------------
// 3. One recording, swept once per seed -- which is what the stand gradient does
//    for each census metric.
//
// ⚠️ THIS IS THE ONE THAT DECIDES WHETHER CALLBACKS ARE AVAILABLE TO US AT ALL.
// computeAdjointsTo calls resetTo(end - 1) at every checkpoint, which truncates
// the statements above it AND erases the checkpoint from the list. So a recording
// containing a callback is consumed by the sweep that passes through it, where a
// recording of plain statements is not.
void sweep_the_same_recording_twice() {
    std::printf("\n3. the same recording swept twice, plain and with a callback\n");

    {
        Tape tape;
        AD x = 0.4;
        tape.registerInput(x);
        tape.newRecording();
        AD w = 3.0 * xad::sin(2.0 * x);
        tape.registerOutput(w);

        double first = 0.0, second = 0.0;
        for (int seed = 0; seed < 2; ++seed) {
            tape.clearDerivatives();
            tape.derivative(w.getSlot()) = 1.0;
            tape.computeAdjoints();
            (seed == 0 ? first : second) = tape.derivative(x.getSlot());
        }
        const double want = 3.0 * std::cos(0.8) * 2.0;
        check("plain statements, seed 1", first, want);
        check("plain statements, seed 2", second, want);
    }

    {
        Tape tape;
        AD x = 0.4;
        tape.registerInput(x);
        tape.newRecording();
        const AD u = 2.0 * x;
        AD v = std::sin(xad::value(u));
        tape.registerOutput(v);
        SuppliedDerivative supplied;
        supplied.in = u.getSlot();
        supplied.out = v.getSlot();
        supplied.dout_din = std::cos(xad::value(u));
        tape.insertCallback(&supplied);
        AD w = 3.0 * v;
        tape.registerOutput(w);

        const std::size_t before = tape.getNumStatements();
        double first = 0.0, second = 0.0;
        for (int seed = 0; seed < 2; ++seed) {
            tape.clearDerivatives();
            tape.derivative(w.getSlot()) = 1.0;
            tape.computeAdjoints();
            (seed == 0 ? first : second) = tape.derivative(x.getSlot());
        }
        const double want = 3.0 * std::cos(0.8) * 2.0;
        std::printf("  statements %zu before the first sweep, %zu after the second\n",
                    before, std::size_t(tape.getNumStatements()));
        check("with a callback, seed 1", first, want);
        std::printf("  with a callback, seed 2 -> %.8g (want %.8g): %s\n", second, want,
                    std::fabs(second - want) <= 1e-12 * (1.0 + std::fabs(want))
                        ? "survives"
                        : "CONSUMED -- one recording cannot serve two seeds");
        std::printf("  callback ran %d time(s) over two sweeps\n", supplied.calls);
    }
}

// ---------------------------------------------------------------------------
// 4. Rewinding the slot counter with nothing but the public API.
//
// unregisterVariable rewinds the counter only for the slot it issued last, so a
// rewind to a mark is a loop in reverse issue order -- the same order of work as
// the allocations it undoes, where clearAll() throws the counter away in one
// step. And the live count comes off either way: unregisterVariable decrements
// it unconditionally, so unregistering a slot that is not the last one is a
// rewind that did not happen and a count that moved anyway.
void rewinding_the_counter() {
    std::printf("\n4. rewinding the counter: LIFO by hand against clearAll\n");

    const int n = 200000;

    // Reading the counter without an accessor: issue one slot, note it, hand it
    // straight back. It is the last issued, so the counter lands where it was.
    auto mark = [](Tape& tape) -> slot_type {
        const slot_type s = tape.registerVariable();
        tape.unregisterVariable(s);
        return s;
    };

    {
        Tape tape;
        const slot_type start = mark(tape);
        std::vector<slot_type> issued;
        issued.reserve(std::size_t(n));
        for (int i = 0; i < n; ++i) issued.push_back(tape.registerVariable());

        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = issued.size(); i-- > 0;) tape.unregisterVariable(issued[i]);
        const auto t1 = std::chrono::steady_clock::now();

        const slot_type after = mark(tape);
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  LIFO unregister of %d slots: %.3f ms, counter %u -> %u, live %zu\n", n,
                    ms, unsigned(start), unsigned(after), std::size_t(tape.getNumVariables()));
        check("counter is back at the mark", double(after), double(start));
    }

    {
        Tape tape;
        std::vector<slot_type> issued;
        for (int i = 0; i < n; ++i) issued.push_back(tape.registerVariable());
        const auto t0 = std::chrono::steady_clock::now();
        tape.clearAll();
        const auto t1 = std::chrono::steady_clock::now();
        const slot_type after = mark(tape);
        std::printf("  clearAll of the same %d: %.3f ms, counter -> %u, live %zu\n", n,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), unsigned(after),
                    std::size_t(tape.getNumVariables()));
    }

    // Out of order, which is what a std::vector of actives does when it dies:
    // it destroys front to back, so all but the last hand back no slot.
    {
        Tape tape;
        const slot_type start = mark(tape);
        std::vector<slot_type> issued;
        for (int i = 0; i < 8; ++i) issued.push_back(tape.registerVariable());
        for (std::size_t i = 0; i < issued.size(); ++i) tape.unregisterVariable(issued[i]);
        const slot_type after = mark(tape);
        std::printf("  front-to-back unregister of 8: counter %u -> %u, so %u of 8 slots "
                    "leaked, live %zu\n",
                    unsigned(start), unsigned(after), unsigned(after - start),
                    std::size_t(tape.getNumVariables()));
        std::printf("  the count comes back to zero either way, which is what the release "
                    "audit reads:\n  it is a count of live values, not a map of which slots "
                    "are free.\n");
    }
}

}  // namespace

int main() {
    std::printf("What a caller can do to slots directly (XAD, adj<double>)\n");
    hand_written_statement();
    callback_writes_slots();
    sweep_the_same_recording_twice();
    rewinding_the_counter();
    std::printf("\n%s\n", failures == 0 ? "every checked number as expected"
                                        : "SOME NUMBERS WRONG -- see above");
    return failures == 0 ? 0 : 1;
}

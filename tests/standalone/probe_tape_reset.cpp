// What a tape reset does to a slot an object is still holding.
//
// A holder that keeps active members across successive recordings on one tape is
// the shape a sweep wants, because rebuilding one costs every allocation it
// owns. Whether it is legal depends on what the reset does to the slots those
// members hold, and the four resets below do not agree. `Sys::square` is the
// case that decides it: DERIVED, written only from an expression, never from a
// fresh value.
//
//   A   rebuild the holder,  clearAll        the shipped discipline
//   B   carry the holder,    clearAll        WRONG, and silently
//   C   carry the holder,    newRecording    right, and quadratic
//   D   carry the holder,    newRecording + re-register    a no-op
//   E1  carry the holder,    de-register then clearAll     right, and linear
//   E2  carry the holder,    clearAll then de-register     underflows the counter
//
// ⚠️ B IS RIGHT UNTIL TWO RECORDINGS DIFFER IN SHAPE. clearAll returns the slot
// counter to zero, so a carried member's slot is reissued -- and while the input
// count is constant the collision lands where the answer happens to survive it.
// Pass `grow` to make a later recording register more inputs, and the stale slot
// aliases a live one: finite, plausible, wrong. That is why the shape has to
// vary for this to be a test of anything.
//
// De-registration is a move-assignment from a fresh temporary. The swap hands
// the old slot to the temporary, whose destructor releases it, and no statement
// is recorded -- so the member comes back as if constructed. It has to happen
// BEFORE the reset: after one, the release decrements a counter that is already
// zero, and with slot reuse compiled in the freed slots are then reissued.
//
// Built to be run by hand -- `make probe_tape_reset && ./probe_tape_reset E1 6
// grow` -- and not part of `all`. XAD_NTMP=N adds N recorded temporaries per
// recording, in a vector that dies front-to-back, which is what a model
// evaluation does and what decides whether a reset that keeps slots can afford
// the leak.

#include <XAD/XAD.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ad_t     = xad::adj<double>;
using active   = ad_t::active_type;
using tape_t   = ad_t::tape_type;

// ---- the pattern under test -------------------------------------------------
template <typename T>
struct Sys
{
    using value_type = T;
    T a{2.0}, b{-3.0};              // "parameters", written from registered inputs
    T square{4.0};                  // DERIVED: only ever written from an expression
    void refresh() { square = a * a; }
};

// arms
enum Arm { ARM_A = 0, ARM_B, ARM_C, ARM_D, ARM_E1, ARM_E2 };
static const char* arm_name[] = {"A", "B", "C", "D", "E1", "E2"};

static int g_ntmp = 0;   // from env XAD_NTMP

// Explicit de-registration. Move-assigning a fresh temporary swaps slots
// (Literals.hpp:246-253), so the temporary carries the OLD slot away and its
// destructor calls unregisterVariable() on it. No statement is recorded, and the
// member is left with slot_ == INVALID_SLOT, i.e. as if freshly constructed.
static void deregister(Sys<active>& s)
{
    s.a      = active(s.a.getValue());
    s.b      = active(s.b.getValue());
    s.square = active(s.square.getValue());
}

static const int MAXIN = 32;

struct Res
{
    int    nin;
    double g[2][MAXIN];             // g[output][input]
    unsigned slot_x[MAXIN], slot_a, slot_b, slot_sq, slot_y[2];
};

// One recording. `sys` is the system to use (carried, or freshly built by caller).
static Res one_recording(tape_t& tape, Sys<active>& sys, double x0, double x1, Arm arm, int nin)
{
    std::vector<active> x;
    x.emplace_back(x0);
    x.emplace_back(x1);
    for (int i = 2; i < nin; ++i) x.emplace_back(0.5 * double(i));   // extra inputs

    tape.registerInputs(x);

    if (arm == ARM_D)
    {
        // Re-register the system's own members as inputs. NOTE: registerInput() is
        // a no-op when the variable already has a slot (shouldRecord() == true),
        // so from recording #2 onward this does nothing for a carried system.
        tape.registerInput(sys.a);
        tape.registerInput(sys.b);
        tape.registerInput(sys.square);
    }

    tape.newRecording();

    sys.a = x[0];
    sys.b = x[1];
    sys.refresh();                          // square = a*a  (assign-from-expression)

    // N recorded temporaries that die front-to-back at the end of this scope.
    if (g_ntmp > 0)
    {
        std::vector<active> tmp;
        tmp.reserve(size_t(g_ntmp));
        for (int i = 0; i < g_ntmp; ++i) tmp.emplace_back(x[0] * 2.0);
    }

    active extra = 0.0;
    for (int i = 2; i < nin; ++i) extra += x[i];    // every input is now LIVE

    std::vector<active> y;
    y.emplace_back(sys.a * x[0] + sys.b * x[1] + extra);   // y0 = x0^2 + x1^2 + sum_{i>=2} xi
    y.emplace_back(x[0] * x[1] + sys.square);              // y1 = x0*x1 + x0^2
    tape.registerOutputs(y);

    Res r{};
    r.nin = nin;
    for (int i = 0; i < nin; ++i) r.slot_x[i] = x[i].getSlot();
    r.slot_a    = sys.a.getSlot();
    r.slot_b    = sys.b.getSlot();
    r.slot_sq   = sys.square.getSlot();
    r.slot_y[0] = y[0].getSlot();
    r.slot_y[1] = y[1].getSlot();

    for (int k = 0; k < 2; ++k)
    {
        tape.clearDerivatives();
        tape.setDerivative(y[k].getSlot(), 1.0);
        tape.computeAdjoints();
        for (int j = 0; j < nin; ++j) r.g[k][j] = tape.getDerivative(x[j].getSlot());
    }
    return r;
}

static void analytic(double x0, double x1, int nin, double g[2][MAXIN])
{
    for (int o = 0; o < 2; ++o)
        for (int i = 0; i < nin; ++i) g[o][i] = 0.0;
    g[0][0] = 2.0 * x0;        // dy0/dx0
    g[0][1] = 2.0 * x1;        // dy0/dx1
    g[1][0] = x1 + 2.0 * x0;   // dy1/dx0
    g[1][1] = x0;              // dy1/dx1
    for (int i = 2; i < nin; ++i) g[0][i] = 1.0;   // dy0/dxi
}

static bool close(double a, double b) { return std::isfinite(a) && std::fabs(a - b) < 1e-9; }

// inputs for recording k = 1,2,3,...
static void inputs_for(int k, double& x0, double& x1) { x0 = double(k) + 1.0; x1 = -2.0 * double(k); }

static int run_arms(Arm arm, int nrec, bool memory_mode, int nin, bool grow)
{
    tape_t tape;
    Sys<double>  passive;                       // the "source of truth" values
    Sys<active>* carried = nullptr;
    if (arm != ARM_A)
        carried = new Sys<active>();            // built ONCE, reused

    for (int k = 1; k <= nrec; ++k)
    {
        size_t mem_top  = tape.getMemory();
        unsigned nv_top = (unsigned)tape.getNumVariables();

        // ---- the reset step: the variable under test ----
        if (arm == ARM_A || arm == ARM_B)
            tape.clearAll();
        else if (arm == ARM_E1)          // de-register FIRST, then clearAll
        {
            deregister(*carried);
            tape.clearAll();
        }
        else if (arm == ARM_E2)          // clearAll FIRST, then de-register
        {
            tape.clearAll();
            deregister(*carried);
        }
        // arms C and D: no clearAll(); newRecording() inside one_recording() only.

        Sys<active> fresh;
        Sys<active>* sys = carried;
        if (arm == ARM_A)
        {
            // fresh system per recording, copied out of the passive one
            fresh.a = passive.a; fresh.b = passive.b; fresh.square = passive.square;
            sys = &fresh;
        }

        double x0, x1;
        inputs_for(k, x0, x1);
        Res r = one_recording(tape, *sys, x0, x1, arm, (grow && k == 1) ? 2 : nin);

        if (memory_mode)
        {
            if (k == 1 || k == 10 || k == 100 || k % 20 == 0 || k == nrec)
                std::printf("arm %s  rec %4d  memory %12zu bytes  numVars %u\n", arm_name[arm], k,
                            tape.getMemory(), (unsigned)tape.getNumVariables());
        }
        else
        {
            double ga[2][MAXIN];
            analytic(x0, x1, r.nin, ga);
            std::printf("arm %s  rec %d  [top of rec: mem %zu, numVars %u]  x0=%g x1=%g  slots: x=[",
                        arm_name[arm], k, mem_top, nv_top, x0, x1);
            for (int i = 0; i < r.nin; ++i) std::printf("%s%u", i ? "," : "", r.slot_x[i]);
            std::printf("] a=%u b=%u sq=%u y=(%u,%u)\n", r.slot_a, r.slot_b, r.slot_sq,
                        r.slot_y[0], r.slot_y[1]);
            for (int o = 0; o < 2; ++o)
            {
                bool ok = true;
                std::printf("           y%d adj = [", o);
                for (int i = 0; i < r.nin; ++i)
                {
                    std::printf("%s% .6g", i ? ", " : "", r.g[o][i]);
                    ok = ok && close(r.g[o][i], ga[o][i]);
                }
                std::printf("]\n                 want [");
                for (int i = 0; i < r.nin; ++i) std::printf("%s% .6g", i ? ", " : "", ga[o][i]);
                std::printf("]  %s\n", ok ? "OK" : "WRONG");
            }
        }
    }
    delete carried;
    return 0;
}

int main(int argc, char** argv)
{
    bool memory_mode = false;
    int ai = 1;
    if (argc > 1 && std::strcmp(argv[1], "mem") == 0) { memory_mode = true; ai = 2; }
    const char* a = (argc > ai) ? argv[ai] : "A";
    Arm arm = ARM_A;
    if (*a == 'B') arm = ARM_B;
    else if (*a == 'C') arm = ARM_C;
    else if (*a == 'D') arm = ARM_D;
    else if (*a == 'E') arm = (a[1] == '2') ? ARM_E2 : ARM_E1;

    if (const char* e = std::getenv("XAD_NTMP")) g_ntmp = std::atoi(e);

    try
    {
        int n = (argc > ai + 1) ? std::atoi(argv[ai + 1]) : (memory_mode ? 200 : 2);
        bool grow = (argc > ai + 2) && std::strcmp(argv[ai + 2], "grow") == 0;
        if (memory_mode)
        {
            int mnin = (argc > ai + 2) ? std::atoi(argv[ai + 2]) : 2;
            auto t0 = std::chrono::steady_clock::now();
            int rc  = run_arms(arm, n, true, mnin < 2 ? 2 : mnin, false);
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("arm %s  ntmp %d  %d recordings in %.3f s\n", arm_name[arm], g_ntmp, n, secs);
            return rc;
        }
        std::printf("(ninputs = %d%s)\n", n < 2 ? 2 : n, grow ? ", grow" : "");
        return run_arms(arm, 3, false, n < 2 ? 2 : n, grow);
    }
    catch (const std::exception& e)
    {
        std::printf("arm %s: EXCEPTION: %s\n", arm_name[arm], e.what());
        return 2;
    }
}

// Whether a recording nested inside a checkpoint callback protects the slots a
// carried holder registered before it, and what the alternatives cost.
//
// `probe_tape_reset` asks what a reset does to a slot an object is still holding,
// and answers that de-registering before clearAll is the affordable one. This asks
// the prior question: is there a reset that leaves those slots alone. There is --
// ending a nested recording restores the enclosing sub-recording's slot counter
// along with the statement, operation and derivative arrays -- and this measures
// it against the two that do not.
//
//   1  every counter after every call of one recording, five times over
//   2  members registered once, then clearAll vs newRecording between recordings
//   3  slot growth and per-recording time under newRecording
//   4  what the de-register walk itself costs, per member
//   5  nesting with a locally patched newNestedRecording, arm by arm
//   6  getPosition/resetTo between recordings
//   7  nesting inside one real checkpoint callback, on the shipped library
//
// Selected by argv[1] (1|2|3|4|5|5scale|6|7|7scale|all); argv[2..] set the
// recording and temporary counts.
//
// The two counters that matter live in Tape::SubRecording:
//   numDerivatives_  -- the LIVE count, what getNumVariables() returns
//   iDerivative_     -- the SLOT counter, the next slot registerVariable() issues
// There is no public accessor for iDerivative_, but Tape::printStatus() is public
// and const and prints it as "next idx". This captures printStatus() into a string
// stream and parses the fields out, which registers nothing and records nothing;
// experiment 1 cross-checks it against a throwaway variable's getSlot().
//
// ⚠️ NESTING OUTSIDE A CHECKPOINT CALLBACK EATS THE MACHINE RATHER THAN FAILING.
// newNestedRecording() resizes the derivative array to prevMax_, which is
// slot_type(-1) on every frame except the one computeAdjointsTo sets around a
// callback -- 4.29e9 doubles, 32 GiB, and wrong answers from the second recording
// on. That arm is reachable only through experiment 5's patched copy, and run it
// under `ulimit -v` if at all.
//
// Built to be run by hand -- `make CXX=g++ probe_nested_recording &&
// ./probe_nested_recording 7` -- and not part of `all`.

#include <XAD/XAD.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using active = xad::AReal<double>;
using tape_t = xad::Tape<double>;

// ---------------------------------------------------------------- introspection

struct Status
{
    long stmts = -1, ops = -1, total_der = -1, der_alloc = -1, curr_der = -1, act_max = -1,
         next_idx = -1;
};

static long field_after(const std::string& s, const char* key)
{
    auto p = s.find(key);
    if (p == std::string::npos)
        return -1;
    p = s.find(':', p);
    if (p == std::string::npos)
        return -1;
    return std::strtol(s.c_str() + p + 1, nullptr, 10);
}

// Capture printStatus() -- the only public window onto iDerivative_.
static Status status(const tape_t& tape)
{
    std::ostringstream cap;
    std::streambuf* old = std::cout.rdbuf(cap.rdbuf());
    tape.printStatus();
    std::cout.rdbuf(old);
    const std::string s = cap.str();
    Status st;
    st.stmts     = field_after(s, "Statements");
    st.ops       = field_after(s, "Operations");
    st.total_der = field_after(s, "Total der");
    st.der_alloc = field_after(s, "Der alloc");
    st.curr_der  = field_after(s, "curr der");
    st.act_max   = field_after(s, "act. max");
    st.next_idx  = field_after(s, "next idx");
    return st;
}

// Independent probe of the slot counter: register one variable, read its slot,
// then let it die. With XAD_TAPE_REUSE_SLOTS off, the death rewinds the counter
// exactly (the slot was the last one issued), so state is restored -- except for
// one extra statement entry, which no counter we report depends on.
static long probe_slot(tape_t& tape)
{
    if (tape_t::getActive() != &tape)
        return -1;  // destructor would not rewind; do not perturb
    active tmp(0.0);
    tape.registerInput(tmp);
    return long(tmp.getSlot());
}

static void row(const char* label, tape_t& tape, bool with_probe = true)
{
    Status st = status(tape);
    long p    = with_probe ? probe_slot(tape) : -1;
    std::printf("  %-34s numVars=%-6ld nextSlot=%-6ld maxDer=%-6ld derAlloc=%-6ld stmts=%-7ld "
                "ops=%-8ld probeSlot=%ld\n",
                label, st.curr_der, st.next_idx, st.total_der, st.der_alloc, st.stmts, st.ops, p);
}

// ------------------------------------------------------------------ experiment 1

static void exp1()
{
    std::printf("=== EXP 1: counter semantics, 5 successive recordings on one tape ===\n");
    std::printf("(numVars = getNumVariables() = numDerivatives_; nextSlot = iDerivative_ from "
                "printStatus \"next idx\"; probeSlot = slot handed to a throwaway registerInput)\n");
    tape_t tape(false);  // do not activate in the ctor; we activate explicitly

    for (int k = 1; k <= 5; ++k)
    {
        std::printf("-- recording %d\n", k);
        row("before activate()", tape, false);
        tape.activate();
        row("after  activate()", tape);
        tape.clearAll();
        row("after  clearAll()", tape);

        std::vector<active> in;
        for (int i = 0; i < 4; ++i) in.emplace_back(1.0 + 0.25 * i + k);
        row("after  4 inputs constructed", tape);
        tape.registerInputs(in);
        row("after  registerInputs(4)", tape);

        tape.newRecording();
        row("after  newRecording()", tape);

        {
            // ~1000 recorded temporaries, dying front-to-back at scope exit.
            std::vector<active> tmp;
            tmp.reserve(1000);
            for (int i = 0; i < 1000; ++i) tmp.emplace_back(in[i % 4] * 1.0001 + 0.5);
            row("after  ~1000 temporaries (alive)", tape);
        }
        row("after  temporaries destroyed", tape);

        std::vector<active> out;
        out.emplace_back(in[0] * in[1] + in[2] * in[3]);
        row("after  output expression", tape);
        tape.registerOutputs(out);
        row("after  registerOutputs(1)", tape);

        tape.clearDerivatives();
        tape.setDerivative(out[0].getSlot(), 1.0);
        row("after  setDerivative", tape);
        tape.computeAdjoints();
        row("after  computeAdjoints()", tape);

        double g[4];
        for (int i = 0; i < 4; ++i) g[i] = tape.getDerivative(in[i].getSlot());
        double want[4] = {xad::value(in[1]), xad::value(in[0]), xad::value(in[3]),
                          xad::value(in[2])};
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && std::fabs(g[i] - want[i]) < 1e-12;
        std::printf("  adjoints = [%g %g %g %g] want [%g %g %g %g]  %s\n", g[0], g[1], g[2], g[3],
                    want[0], want[1], want[2], want[3], ok ? "OK" : "WRONG");

        // inputs/outputs die here, before deactivate, as they do in real code
        out.clear();
        in.clear();
        row("after  in/out vectors cleared", tape);
        tape.deactivate();
        row("after  deactivate()", tape, false);
    }
}

// ------------------------------------------------------------------ experiment 2

static const int NM = 8;

struct Holder
{
    active m[NM];
    active derived;  // written only from an expression, never from a fresh value
};

// y = sum_i (i+1) * m_i^2  +  (m0*m1 + m2*m3)
static void analytic2(const double v[NM], double g[NM])
{
    for (int i = 0; i < NM; ++i) g[i] = 2.0 * double(i + 1) * v[i];
    g[0] += v[1];
    g[1] += v[0];
    g[2] += v[3];
    g[3] += v[2];
}

static void exp2(bool use_clear_all)
{
    std::printf("=== EXP 2%s: pre-registered long-lived members, reset = %s ===\n",
                use_clear_all ? "a" : "b",
                use_clear_all ? "clearAll() + newRecording()" : "newRecording() only");
    tape_t tape;  // active from construction
    Holder h;

    for (int i = 0; i < NM; ++i) h.m[i] = 1.0 + 0.5 * double(i);
    tape.registerInput(h.derived);
    for (int i = 0; i < NM; ++i) tape.registerInput(h.m[i]);
    std::printf("  registered ONCE up front: derived slot=%u, member slots =", h.derived.getSlot());
    for (int i = 0; i < NM; ++i) std::printf(" %u", h.m[i].getSlot());
    std::printf("   numVars=%zu nextSlot=%ld\n", size_t(tape.getNumVariables()),
                status(tape).next_idx);

    for (int k = 1; k <= 5; ++k)
    {
        if (use_clear_all)
            tape.clearAll();

        double v[NM];
        for (int i = 0; i < NM; ++i)
        {
            v[i] = 1.0 + 0.5 * double(i) + 0.125 * double(k);
            h.m[i] = v[i];  // keeps the slot, records an independent-input statement
        }

        tape.newRecording();
        Status pre = status(tape);

        {
            std::vector<active> tmp;  // recorded temporaries, as a model evaluation makes
            tmp.reserve(200);
            for (int i = 0; i < 200; ++i) tmp.emplace_back(h.m[i % NM] * 1.0001 + 0.5);
        }

        h.derived = h.m[0] * h.m[1] + h.m[2] * h.m[3];

        active y = 0.0;
        for (int i = 0; i < NM; ++i) y += double(i + 1) * h.m[i] * h.m[i];
        y += h.derived;
        tape.registerOutput(y);

        double g[NM];
        bool threw = false;
        const char* what = "";
        try
        {
            tape.clearDerivatives();
            tape.setDerivative(y.getSlot(), 1.0);
            tape.computeAdjoints();
            for (int i = 0; i < NM; ++i) g[i] = tape.getDerivative(h.m[i].getSlot());
        }
        catch (const std::exception& e)
        {
            threw = true;
            what  = e.what();
            for (int i = 0; i < NM; ++i) g[i] = std::nan("");
        }

        double ga[NM];
        analytic2(v, ga);
        bool ok = !threw;
        for (int i = 0; i < NM; ++i)
            ok = ok && std::isfinite(g[i]) && std::fabs(g[i] - ga[i]) < 1e-9;

        std::printf("  rec %d  slots: derived=%u m=[", k, h.derived.getSlot());
        for (int i = 0; i < NM; ++i) std::printf("%s%u", i ? "," : "", h.m[i].getSlot());
        std::printf("] y=%u  atNewRecording{numVars=%ld nextSlot=%ld maxDer=%ld}  now{numVars=%zu "
                    "nextSlot=%ld}\n",
                    y.getSlot(), pre.curr_der, pre.next_idx, pre.total_der,
                    size_t(tape.getNumVariables()), status(tape).next_idx);
        if (threw)
            std::printf("           EXCEPTION: %s\n", what);
        std::printf("           adj  = [");
        for (int i = 0; i < NM; ++i) std::printf("%s% .6g", i ? ", " : "", g[i]);
        std::printf("]\n           want = [");
        for (int i = 0; i < NM; ++i) std::printf("%s% .6g", i ? ", " : "", ga[i]);
        std::printf("]  %s\n", ok ? "OK" : "WRONG");
    }
}

// ------------------------------------------------------------------ experiment 3

static void exp3(int nrec, int ntmp)
{
    std::printf("=== EXP 3: slot growth under newRecording(), %d temporaries/recording ===\n",
                ntmp);
    std::printf("  %6s %12s %12s %14s %14s %12s %12s\n", "rec", "numVars", "nextSlot", "maxDer",
                "derAlloc", "memory(B)", "cum s");
    tape_t tape;
    Holder h;
    for (int i = 0; i < NM; ++i) h.m[i] = 1.0 + 0.5 * double(i);
    for (int i = 0; i < NM; ++i) tape.registerInput(h.m[i]);

    auto t0 = std::chrono::steady_clock::now();
    double last_cum = 0.0;
    for (int k = 1; k <= nrec; ++k)
    {
        tape.newRecording();
        {
            std::vector<active> tmp;
            tmp.reserve(size_t(ntmp));
            for (int i = 0; i < ntmp; ++i) tmp.emplace_back(h.m[i % NM] * 1.0001 + 0.5);
        }
        active y = h.m[0] * h.m[1] + h.m[2] * h.m[3];
        tape.registerOutput(y);
        tape.clearDerivatives();
        tape.setDerivative(y.getSlot(), 1.0);
        tape.computeAdjoints();

        if (k == 1 || k == 50 || k == 100 || k == 200 || k == 400 || k == 800 || k == nrec)
        {
            double cum = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Status st  = status(tape);
            std::printf("  %6d %12ld %12ld %14ld %14ld %12zu %12.4f", k, st.curr_der, st.next_idx,
                        st.total_der, st.der_alloc, tape.getMemory(), cum);
            if (k > 1)
                std::printf("   (delta since previous mark: %.4f s)", cum - last_cum);
            std::printf("\n");
            last_cum = cum;
        }
    }
}

// ------------------------------------------------------------------ experiment 4

static void exp4(int nmem, int nrec)
{
    std::printf("=== EXP 4: cost of the release idiom, %d members x %d recordings = %lld "
                "releases ===\n",
                nmem, nrec, (long long)nmem * nrec);
    tape_t tape;
    std::vector<active> mem;
    mem.resize(size_t(nmem));
    for (int i = 0; i < nmem; ++i) mem[size_t(i)] = 1.0 + 0.001 * double(i);

    double t_release = 0.0, t_register = 0.0;
    auto tall0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nrec; ++k)
    {
        tape.clearAll();
        auto a = std::chrono::steady_clock::now();
        tape.registerInputs(mem);  // give every member a live slot to release
        auto b = std::chrono::steady_clock::now();
        tape.newRecording();
        auto c = std::chrono::steady_clock::now();
        // THE RELEASE IDIOM, applied to every member.
        for (int i = 0; i < nmem; ++i)
            mem[size_t(i)] = active(xad::value(mem[size_t(i)]));
        auto d = std::chrono::steady_clock::now();
        t_register += std::chrono::duration<double>(b - a).count();
        t_release += std::chrono::duration<double>(d - c).count();
    }
    double t_all = std::chrono::duration<double>(std::chrono::steady_clock::now() - tall0).count();
    long long n  = (long long)nmem * nrec;
    std::printf("  release total      %.4f s   (%.1f ns per member-release)\n", t_release,
                1e9 * t_release / double(n));
    std::printf("  registerInputs     %.4f s   (%.1f ns per member-register)\n", t_register,
                1e9 * t_register / double(n));
    std::printf("  whole loop         %.4f s   (per recording: release %.1f us, register %.1f us)\n",
                t_all, 1e6 * t_release / nrec, 1e6 * t_register / nrec);
    std::printf("  end state: numVars=%zu nextSlot=%ld memory=%zu B\n",
                size_t(tape.getNumVariables()), status(tape).next_idx, tape.getMemory());

    // Contrast: the same walk when every member is already slotless (nothing to
    // unregister), to separate the loop overhead from the tape bookkeeping.
    for (int i = 0; i < nmem; ++i) mem[size_t(i)] = active(xad::value(mem[size_t(i)]));
    auto e0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nrec; ++k)
        for (int i = 0; i < nmem; ++i) mem[size_t(i)] = active(xad::value(mem[size_t(i)]));
    double t_noslot = std::chrono::duration<double>(std::chrono::steady_clock::now() - e0).count();
    std::printf("  same walk, slotless members: %.4f s (%.1f ns per member) -- loop-only floor\n",
                t_noslot, 1e9 * t_noslot / double(n));
}

// ------------------------------------------------------------------ experiment 5
// THE NESTED-RECORDING HYPOTHESIS.
//
//   m_i = (i+1) * s[i%3] + 0.5 * p0        (i = 0..5, members WRITTEN each nest)
//   y   = sum_i m_i^2 + p1 * s0 + q * s1   (q = 3 * p0, recorded in the OUTER rec)
//
// p0,p1 are parameters registered once and never written; q probes whether a
// nested sweep reaches back past the nest mark into the outer statements.
//
// analytic, for the whole chain:
//   dy/ds_j = sum_{i : i%3 == j} 2 m_i (i+1) + (j==0 ? p1 : 0) + (j==1 ? q : 0)
//   dy/dp0  = sum_i 2 m_i * 0.5 + 3 * s1        (nest-local variant omits 3*s1)
//   dy/dp1  = s0

static const int NI = 6;
static const int NP = 2;
static const int NS = 3;

struct Model
{
    active m[NI];
    active p[NP];
    active s[NS];  // long-lived state inputs
    active q;
};

struct Exp5Want
{
    double ds[NS], dp[NP], dp0_nest_only, m[NI];
};

static Exp5Want analytic5(const double s[NS], double p0, double p1, double qv)
{
    Exp5Want w{};
    for (int i = 0; i < NI; ++i) w.m[i] = double(i + 1) * s[i % NS] + 0.5 * p0;
    for (int j = 0; j < NS; ++j) w.ds[j] = 0.0;
    for (int i = 0; i < NI; ++i) w.ds[i % NS] += 2.0 * w.m[i] * double(i + 1);
    w.ds[0] += p1;
    w.ds[1] += qv;
    double acc = 0.0;
    for (int i = 0; i < NI; ++i) acc += 2.0 * w.m[i] * 0.5;
    w.dp0_nest_only = acc;
    w.dp[0] = acc + 3.0 * s[1];
    w.dp[1] = s[0];
    return w;
}

struct Exp5Opts
{
    int nrec = 5;
    int ntmp = 200;
    bool preregister_members = true;   // members get slots in the OUTER recording
    bool inputs_in_nest = false;       // registerInputs() called after the nest opens
    bool verbose = true;
    bool throw_test = false;
    bool zero_outer = false;   // explicitly zero the pre-nest slots before each sweep
    const char* label = "";
};

// Returns number of wrong recordings.
static int exp5_run(const Exp5Opts& o)
{
    std::printf("=== EXP 5 [%s]: nested recordings, R=%d, %d temporaries/recording\n"
                "    members %s, state inputs registered %s ===\n",
                o.label, o.nrec, o.ntmp,
                o.preregister_members ? "PRE-REGISTERED in outer rec" : "NOT pre-registered",
                o.inputs_in_nest ? "INSIDE the nest" : "in the OUTER rec (once)");

    tape_t tape(false);
    tape.activate();
    Model mo;
    const double p0v = 2.0, p1v = 3.0;
    const double sv[NS] = {1.5, -0.75, 0.25};
    mo.p[0] = p0v;
    mo.p[1] = p1v;
    for (int i = 0; i < NI; ++i) mo.m[i] = 0.0;
    for (int j = 0; j < NS; ++j) mo.s[j] = sv[j];

    if (o.verbose)
        row("outer: after activate()", tape);
    if (o.preregister_members)
        for (int i = 0; i < NI; ++i) tape.registerInput(mo.m[i]);
    for (int i = 0; i < NP; ++i) tape.registerInput(mo.p[i]);
    if (!o.inputs_in_nest)
        for (int j = 0; j < NS; ++j) tape.registerInput(mo.s[j]);
    if (o.verbose)
        row("outer: after registerInput(...)", tape);

    tape.newRecording();
    if (o.verbose)
        row("outer: after newRecording()", tape);
    mo.q = 3.0 * mo.p[0];
    if (o.verbose)
        row("outer: after q = 3*p0", tape);

    // every slot issued before the first nest
    std::vector<unsigned> outer_slots;
    for (int i = 0; i < NI; ++i)
        if (mo.m[i].getSlot() != active::INVALID_SLOT)
            outer_slots.push_back(mo.m[i].getSlot());
    for (int i = 0; i < NP; ++i) outer_slots.push_back(mo.p[i].getSlot());
    for (int j = 0; j < NS; ++j)
        if (mo.s[j].getSlot() != active::INVALID_SLOT)
            outer_slots.push_back(mo.s[j].getSlot());
    outer_slots.push_back(mo.q.getSlot());

    Exp5Want w = analytic5(sv, p0v, p1v, 3.0 * p0v);

    auto t0 = std::chrono::steady_clock::now();
    double last_cum = 0.0;
    int nwrong = 0, first_wrong = -1;
    double worst = 0.0;

    for (int k = 1; k <= o.nrec; ++k)
    {
        bool show = o.verbose && (k <= 3 || k == o.nrec);
        double gs[NS] = {0, 0, 0}, gp[NP] = {0, 0}, gq = 0.0;
        double dmem[NI] = {0};
        bool threw = false;
        std::string what;

        // Set fresh state values WITHOUT recording anything (xad::value gives a
        // mutable reference to the underlying double), so nothing this recording
        // needs lands outside the nest.
        for (int j = 0; j < NS; ++j) xad::value(mo.s[j]) = sv[j];

        {
            xad::ScopedNestedRecording<tape_t> nest(&tape);
            if (show)
            {
                std::printf("-- recording %d\n", k);
                row("  nest opened", tape);
            }
            if (o.inputs_in_nest)
            {
                for (int j = 0; j < NS; ++j) tape.registerInput(mo.s[j]);
                if (show)
                    row("  after registerInput(state) in nest", tape);
            }
            {
                std::vector<active> tmp;
                tmp.reserve(size_t(o.ntmp));
                for (int i = 0; i < o.ntmp; ++i) tmp.emplace_back(mo.s[i % NS] * 1.0001 + 0.5);
                if (show)
                    row("  temporaries alive", tape);
            }
            if (show)
                row("  temporaries destroyed", tape);

            for (int i = 0; i < NI; ++i) mo.m[i] = double(i + 1) * mo.s[i % NS] + 0.5 * mo.p[0];
            if (show)
                row("  after writing members", tape);

            active y = 0.0;
            for (int i = 0; i < NI; ++i) y += mo.m[i] * mo.m[i];
            y += mo.p[1] * mo.s[0];
            y += mo.q * mo.s[1];
            tape.registerOutput(y);
            if (show)
                row("  after output + registerOutput", tape);

            try
            {
                if (o.throw_test && k == 2)
                    throw std::runtime_error("deliberate throw inside the nest");
                tape.clearDerivatives();
                if (o.zero_outer)
                    for (unsigned sl : outer_slots) tape.setDerivative(sl, 0.0);
                tape.setDerivative(y.getSlot(), 1.0);
                nest.computeAdjoints();
                if (show)
                    row("  after nest.computeAdjoints()", tape);
                for (int j = 0; j < NS; ++j) gs[j] = tape.getDerivative(mo.s[j].getSlot());
                for (int i = 0; i < NP; ++i) gp[i] = tape.getDerivative(mo.p[i].getSlot());
                gq = tape.getDerivative(mo.q.getSlot());
                for (int i = 0; i < NI; ++i) dmem[i] = tape.getDerivative(mo.m[i].getSlot());
            }
            catch (const std::exception& e)
            {
                threw = true;
                what  = e.what();
            }
        }  // fold

        bool ok = !threw;
        for (int j = 0; j < NS; ++j)
        {
            double d = std::fabs(gs[j] - w.ds[j]);
            if (!std::isfinite(gs[j]) || d > 1e-9)
                ok = false;
            worst = (std::max)(worst, std::isfinite(d) ? d : 1e300);
        }
        if (!ok)
        {
            ++nwrong;
            if (first_wrong < 0)
                first_wrong = k;
        }

        if (show)
        {
            std::printf("  slots: p=[%u,%u] q=%u s=[%u,%u,%u] m=[", mo.p[0].getSlot(),
                        mo.p[1].getSlot(), mo.q.getSlot(), mo.s[0].getSlot(), mo.s[1].getSlot(),
                        mo.s[2].getSlot());
            for (int i = 0; i < NI; ++i) std::printf("%s%u", i ? "," : "", mo.m[i].getSlot());
            std::printf("]\n");
            if (threw)
                std::printf("  EXCEPTION inside nest: %s\n", what.c_str());
            std::printf("  dy/ds = [% .10g % .10g % .10g]\n   want = [% .10g % .10g % .10g]  %s\n",
                        gs[0], gs[1], gs[2], w.ds[0], w.ds[1], w.ds[2], ok ? "OK" : "WRONG");
            std::printf("  dy/dp = [% .8g % .8g]  global-want [% .8g % .8g]  dp0-if-nest-local "
                        "% .8g   adj(q) = % .8g  (s1 = % .8g)\n",
                        gp[0], gp[1], w.dp[0], w.dp[1], w.dp0_nest_only, gq, sv[1]);
            std::printf("  member adjoints after sweep = [");
            for (int i = 0; i < NI; ++i) std::printf("%s% .6g", i ? ", " : "", dmem[i]);
            std::printf("]\n");
            row("  after fold", tape);
        }

        if (k == 1 || k == 100 || k == 400 || k == 1600 || k == 3400 || k == o.nrec)
        {
            double cum =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Status st = status(tape);
            std::printf("  [mark] rec %5d  numVars=%-6ld nextSlot=%-8ld maxDer=%-11ld "
                        "derAlloc=%-11ld memory=%-11zu cum=%.4f s",
                        k, st.curr_der, st.next_idx, st.total_der, st.der_alloc, tape.getMemory(),
                        cum);
            if (k > 1)
                std::printf("  (delta %.4f s over the interval)", cum - last_cum);
            std::printf("\n");
            last_cum = cum;
        }
    }
    std::printf("  SUMMARY [%s]: %d/%d recordings WRONG (first wrong: %d); worst |err| in dy/ds = "
                "%.4g\n",
                o.label, nwrong, o.nrec, first_wrong, worst);

    if (o.verbose && !o.throw_test)
    {
        std::printf("-- (f) two computeAdjoints() calls in one nest:\n");
        try
        {
            xad::ScopedNestedRecording<tape_t> nest(&tape);
            for (int i = 0; i < NI; ++i) mo.m[i] = double(i + 1) * mo.s[i % NS] + 0.5 * mo.p[0];
            active y = 0.0;
            for (int i = 0; i < NI; ++i) y += mo.m[i] * mo.m[i];
            y += mo.p[1] * mo.s[0];
            y += mo.q * mo.s[1];
            tape.registerOutput(y);
            tape.clearDerivatives();
            tape.setDerivative(y.getSlot(), 1.0);
            nest.computeAdjoints();
            double a1 = tape.getDerivative(mo.s[0].getSlot());
            tape.clearDerivatives();
            tape.setDerivative(y.getSlot(), 1.0);
            nest.computeAdjoints();
            double a2 = tape.getDerivative(mo.s[0].getSlot());
            std::printf("     sweep 1 dy/ds0 = % .10g   sweep 2 = % .10g   want % .10g  -> %s\n",
                        a1, a2, w.ds[0],
                        std::fabs(a2 - w.ds[0]) < 1e-9 ? "2nd sweep OK" : "2nd sweep WRONG");
        }
        catch (const std::exception& e)
        {
            std::printf("     EXCEPTION: %s\n", e.what());
        }
    }
    tape.deactivate();
    return nwrong;
}

// ------------------------------------------------------------------ experiment 6
// The same idea written with the public getPosition() / resetTo() pair instead of
// a nested recording.

static void exp6(int nrec, int ntmp)
{
    std::printf("=== EXP 6: same idea via public getPosition()/resetTo(), R=%d, %d "
                "temporaries/recording ===\n",
                nrec, ntmp);
    tape_t tape(false);
    tape.activate();
    Model mo;
    const double p0v = 2.0, p1v = 3.0;
    mo.p[0] = p0v;
    mo.p[1] = p1v;
    for (int i = 0; i < NI; ++i) mo.m[i] = 0.0;
    for (int i = 0; i < NI; ++i) tape.registerInput(mo.m[i]);
    for (int i = 0; i < NP; ++i) tape.registerInput(mo.p[i]);
    for (int j = 0; j < NS; ++j) tape.registerInput(mo.s[j]);
    tape.newRecording();
    mo.q = 3.0 * mo.p[0];

    const double sv[NS] = {1.5, -0.75, 0.25};
    Exp5Want w = analytic5(sv, p0v, p1v, 3.0 * p0v);
    std::vector<unsigned> outer_slots;
    for (int i = 0; i < NI; ++i) outer_slots.push_back(mo.m[i].getSlot());
    for (int i = 0; i < NP; ++i) outer_slots.push_back(mo.p[i].getSlot());
    for (int j = 0; j < NS; ++j) outer_slots.push_back(mo.s[j].getSlot());
    outer_slots.push_back(mo.q.getSlot());
    auto pos = tape.getPosition();
    std::printf("  mark: getPosition() = %u\n", unsigned(pos));

    auto t0 = std::chrono::steady_clock::now();
    double last_cum = 0.0;
    int nwrong = 0, first_wrong = -1;
    for (int k = 1; k <= nrec; ++k)
    {
        double gs[NS] = {0, 0, 0};
        bool threw = false;
        std::string what;
        try
        {
            for (int j = 0; j < NS; ++j) xad::value(mo.s[j]) = sv[j];
            {
                std::vector<active> tmp;
                tmp.reserve(size_t(ntmp));
                for (int i = 0; i < ntmp; ++i) tmp.emplace_back(mo.s[i % NS] * 1.0001 + 0.5);
            }
            for (int i = 0; i < NI; ++i)
                mo.m[i] = double(i + 1) * mo.s[i % NS] + 0.5 * mo.p[0];
            active y = 0.0;
            for (int i = 0; i < NI; ++i) y += mo.m[i] * mo.m[i];
            y += mo.p[1] * mo.s[0];
            y += mo.q * mo.s[1];
            tape.registerOutput(y);
            tape.clearDerivatives();
            for (unsigned sl : outer_slots) tape.setDerivative(sl, 0.0);
            tape.setDerivative(y.getSlot(), 1.0);
            tape.computeAdjoints();
            for (int j = 0; j < NS; ++j) gs[j] = tape.getDerivative(mo.s[j].getSlot());
        }
        catch (const std::exception& e)
        {
            threw = true;
            what  = e.what();
        }
        tape.resetTo(pos);

        bool ok = !threw;
        for (int j = 0; j < NS; ++j)
            ok = ok && std::isfinite(gs[j]) && std::fabs(gs[j] - w.ds[j]) < 1e-9;
        if (!ok)
        {
            ++nwrong;
            if (first_wrong < 0)
                first_wrong = k;
        }
        if (k <= 3 || k == 100 || k == 400 || k == 1600 || k == 3400 || k == nrec)
        {
            double cum =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            Status st = status(tape);
            std::printf("  rec %5d  dy/ds=[% .6g % .6g % .6g] want [% .6g % .6g % .6g] %-5s  "
                        "numVars=%-9ld nextSlot=%-10ld derAlloc=%-10ld memory=%-11zu cum=%.4f s",
                        k, gs[0], gs[1], gs[2], w.ds[0], w.ds[1], w.ds[2], ok ? "OK" : "WRONG",
                        st.curr_der, st.next_idx, st.der_alloc, tape.getMemory(), cum);
            if (k > 3)
                std::printf(" (delta %.4f)", cum - last_cum);
            std::printf("\n");
            if (threw)
                std::printf("           EXCEPTION: %s\n", what.c_str());
            last_cum = cum;
        }
    }
    std::printf("  SUMMARY: %d/%d recordings WRONG (first wrong: %d)\n", nwrong, nrec, first_wrong);
    tape.deactivate();
}


// ------------------------------------------------------------------ experiment 7
// VARIANT 3: run the whole nested loop INSIDE one real checkpoint callback, where
// prevMax_ is a valid slot number, against UNMODIFIED vendored XAD.

struct NestLoop : public xad::CheckpointCallback<tape_t>
{
    Model* mo = nullptr;
    int nrec = 5, ntmp = 200;
    bool verbose = true, zero_outer = false;
    int throw_at = 0;   // if >0, throw from inside computeAdjoint at this recording
    double p0v = 2.0, p1v = 3.0;
    double sv[NS] = {1.5, -0.75, 0.25};
    std::vector<unsigned> outer_slots;
    int nwrong = 0, first_wrong = -1;
    double worst = 0.0;
    bool ran = false;
    double loop_secs = 0.0;
    size_t mem_at_1 = 0, mem_at_end = 0;
    long slot_at_1 = -1, slot_at_end = -1;

    void computeAdjoint(tape_t* tape) override
    {
        ran = true;
        Exp5Want w = analytic5(sv, p0v, p1v, 3.0 * p0v);
        if (verbose)
        {
            std::printf("  [inside callback] entry state:\n");
            row("    callback entry", *tape);
        }
        auto t0 = std::chrono::steady_clock::now();
        double last_cum = 0.0;
        for (int k = 1; k <= nrec; ++k)
        {
            bool show = verbose && (k <= 3 || k == nrec);
            double gs[NS] = {0, 0, 0}, gp[NP] = {0, 0}, gq = 0;
            for (int j = 0; j < NS; ++j) xad::value(mo->s[j]) = sv[j];
            {
                xad::ScopedNestedRecording<tape_t> nest(tape);
                if (show)
                {
                    std::printf("  -- callback recording %d\n", k);
                    row("    nest opened", *tape);
                }
                {
                    std::vector<active> tmp;
                    tmp.reserve(size_t(ntmp));
                    for (int i = 0; i < ntmp; ++i) tmp.emplace_back(mo->s[i % NS] * 1.0001 + 0.5);
                }
                if (show)
                    row("    temporaries destroyed", *tape);
                if (throw_at > 0 && k == throw_at)
                    throw std::runtime_error(
                        "deliberate throw inside an OPEN nest inside the checkpoint callback");
                for (int i = 0; i < NI; ++i)
                    mo->m[i] = double(i + 1) * mo->s[i % NS] + 0.5 * mo->p[0];
                active y = 0.0;
                for (int i = 0; i < NI; ++i) y += mo->m[i] * mo->m[i];
                y += mo->p[1] * mo->s[0];
                y += mo->q * mo->s[1];
                tape->registerOutput(y);
                if (show)
                    row("    after output", *tape);
                tape->clearDerivatives();
                if (zero_outer)
                    for (unsigned sl : outer_slots) tape->setDerivative(sl, 0.0);
                tape->setDerivative(y.getSlot(), 1.0);
                nest.computeAdjoints();
                if (show)
                    row("    after nest.computeAdjoints()", *tape);
                for (int j = 0; j < NS; ++j) gs[j] = tape->getDerivative(mo->s[j].getSlot());
                for (int i = 0; i < NP; ++i) gp[i] = tape->getDerivative(mo->p[i].getSlot());
                gq = tape->getDerivative(mo->q.getSlot());
            }
            bool ok = true;
            for (int j = 0; j < NS; ++j)
            {
                double d = std::fabs(gs[j] - w.ds[j]);
                if (!std::isfinite(gs[j]) || d > 1e-9)
                    ok = false;
                worst = (std::max)(worst, std::isfinite(d) ? d : 1e300);
            }
            if (!ok)
            {
                ++nwrong;
                if (first_wrong < 0)
                    first_wrong = k;
            }
            if (show)
            {
                std::printf("    slots: p=[%u,%u] q=%u s=[%u,%u,%u] m=[", mo->p[0].getSlot(),
                            mo->p[1].getSlot(), mo->q.getSlot(), mo->s[0].getSlot(),
                            mo->s[1].getSlot(), mo->s[2].getSlot());
                for (int i = 0; i < NI; ++i) std::printf("%s%u", i ? "," : "", mo->m[i].getSlot());
                std::printf("]\n    dy/ds = [% .10g % .10g % .10g]  want [% .10g % .10g % .10g] "
                            " %s\n    dy/dp = [% .8g % .8g]  adj(q) = % .8g\n",
                            gs[0], gs[1], gs[2], w.ds[0], w.ds[1], w.ds[2], ok ? "OK" : "WRONG",
                            gp[0], gp[1], gq);
                row("    after fold", *tape);
            }
            if (k == 1 || k == 100 || k == 400 || k == 1600 || k == 3400 || k == nrec)
            {
                double cum =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                Status st = status(*tape);
                std::printf("    [mark] rec %5d  numVars=%-6ld nextSlot=%-8ld maxDer=%-11ld "
                            "derAlloc=%-11ld memory=%-11zu cum=%.4f s",
                            k, st.curr_der, st.next_idx, st.total_der, st.der_alloc,
                            tape->getMemory(), cum);
                if (k > 1)
                    std::printf("  (delta %.4f s over the interval)", cum - last_cum);
                std::printf("\n");
                last_cum = cum;
                if (k == 1)
                {
                    mem_at_1  = tape->getMemory();
                    slot_at_1 = st.next_idx;
                }
                mem_at_end  = tape->getMemory();
                slot_at_end = st.next_idx;
            }
        }
        loop_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

static void exp7(int nrec, int ntmp, bool verbose, bool zero_outer, bool cb_throws)
{
    std::printf("=== EXP 7 (VARIANT 3): nested recordings inside ONE checkpoint callback, "
                "UNPATCHED XAD, R=%d, %d temporaries/recording%s%s ===\n",
                nrec, ntmp, zero_outer ? ", zeroing pre-nest slots" : "",
                cb_throws ? ", callback throws" : "");
    tape_t tape(false);
    tape.activate();
    Model mo;
    NestLoop cb;
    cb.mo = &mo;
    cb.nrec = nrec;
    cb.ntmp = ntmp;
    cb.verbose = verbose;
    cb.zero_outer = zero_outer;
    cb.throw_at   = cb_throws ? 2 : 0;

    mo.p[0] = cb.p0v;
    mo.p[1] = cb.p1v;
    for (int i = 0; i < NI; ++i) mo.m[i] = 0.0;
    for (int j = 0; j < NS; ++j) mo.s[j] = cb.sv[j];
    for (int i = 0; i < NI; ++i) tape.registerInput(mo.m[i]);
    for (int i = 0; i < NP; ++i) tape.registerInput(mo.p[i]);
    for (int j = 0; j < NS; ++j) tape.registerInput(mo.s[j]);

    // a couple of ordinary outer inputs, as the checkpoint idiom wants
    active x0 = 1.25, x1 = -0.5;
    tape.registerInput(x0);
    tape.registerInput(x1);
    tape.newRecording();
    mo.q = 3.0 * mo.p[0];
    for (int i = 0; i < NI; ++i)
        if (mo.m[i].getSlot() != active::INVALID_SLOT) cb.outer_slots.push_back(mo.m[i].getSlot());
    for (int i = 0; i < NP; ++i) cb.outer_slots.push_back(mo.p[i].getSlot());
    for (int j = 0; j < NS; ++j) cb.outer_slots.push_back(mo.s[j].getSlot());
    cb.outer_slots.push_back(mo.q.getSlot());

    if (verbose)
        row("outer: before insertCallback", tape);
    tape.insertCallback(&cb);
    active out = x0 * x1;  // recorded AFTER the checkpoint, so the sweep reaches it first
    tape.registerOutput(out);
    if (verbose)
        row("outer: after output, before sweep", tape);

    bool threw = false;
    std::string what;
    try
    {
        tape.clearDerivatives();
        tape.setDerivative(out.getSlot(), 1.0);
        tape.computeAdjoints();
    }
    catch (const std::exception& e)
    {
        threw = true;
        what  = e.what();
    }
    std::printf("  callback ran: %s%s%s\n", cb.ran ? "YES" : "NO",
                threw ? "   outer computeAdjoints THREW: " : "", threw ? what.c_str() : "");
    if (cb.ran)
        std::printf("  SUMMARY (variant 3): %d/%d recordings WRONG (first wrong: %d); worst |err| "
                    "in dy/ds = %.4g; loop %.4f s; slotCounter rec1=%ld rec%d=%ld; memory rec1=%zu "
                    "rec%d=%zu\n",
                    cb.nwrong, cb.nrec, cb.first_wrong, cb.worst, cb.loop_secs, cb.slot_at_1, nrec,
                    cb.slot_at_end, cb.mem_at_1, nrec, cb.mem_at_end);
    // is the tape usable afterwards?
    try
    {
        double d0 = tape.getDerivative(x0.getSlot());
        double d1 = tape.getDerivative(x1.getSlot());
        std::printf("  after the sweep: d(out)/dx0 = %g (want %g), d/dx1 = %g (want %g)\n", d0,
                    xad::value(x1), d1, xad::value(x0));
        Status st = status(tape);
        std::printf("  tape after sweep: numVars=%ld nextSlot=%ld maxDer=%ld memory=%zu\n",
                    st.curr_der, st.next_idx, st.total_der, tape.getMemory());
        // and a fresh ordinary recording on the same tape
        tape.clearAll();
        active a = 3.0, b = 4.0;
        tape.registerInput(a);
        tape.registerInput(b);
        tape.newRecording();
        active c = a * b;
        tape.registerOutput(c);
        tape.clearDerivatives();
        tape.setDerivative(c.getSlot(), 1.0);
        tape.computeAdjoints();
        std::printf("  fresh recording after clearAll: dc/da = %g (want 4), dc/db = %g (want 3)\n",
                    tape.getDerivative(a.getSlot()), tape.getDerivative(b.getSlot()));
    }
    catch (const std::exception& e)
    {
        std::printf("  post-sweep use THREW: %s\n", e.what());
    }
    tape.deactivate();
}

// ---------------------------------------------------------------------- driver

int main(int argc, char** argv)
{
    const char* which = (argc > 1) ? argv[1] : "all";
    try
    {
        if (!std::strcmp(which, "1") || !std::strcmp(which, "all"))
            exp1();
        if (!std::strcmp(which, "2") || !std::strcmp(which, "all"))
        {
            exp2(true);
            exp2(false);
        }
        if (!std::strcmp(which, "3") || !std::strcmp(which, "all"))
            exp3(argc > 2 ? std::atoi(argv[2]) : 800, argc > 3 ? std::atoi(argv[3]) : 10000);
        if (!std::strcmp(which, "4") || !std::strcmp(which, "all"))
            exp4(argc > 2 ? std::atoi(argv[2]) : 1500, argc > 3 ? std::atoi(argv[3]) : 3400);
        if (!std::strcmp(which, "5") || !std::strcmp(which, "all"))
        {
            Exp5Opts a;
            a.label = "V2a hypothesis";
            exp5_run(a);
            Exp5Opts b;
            b.label                = "V2b control: members NOT pre-registered";
            b.preregister_members  = false;
            exp5_run(b);
            Exp5Opts c;
            c.label          = "V2c state inputs registered inside the nest";
            c.inputs_in_nest = true;
            exp5_run(c);
            Exp5Opts d;
            d.label      = "V2d exception inside the nest";
            d.nrec       = 4;
            d.throw_test = true;
            exp5_run(d);
            Exp5Opts e;
            e.label      = "V2e hypothesis + zeroing the pre-nest slots each sweep";
            e.zero_outer = true;
            exp5_run(e);
        }
        if (!std::strcmp(which, "5scale") || !std::strcmp(which, "all"))
        {
            Exp5Opts e;
            e.label   = "V2 scale";
            e.nrec    = argc > 2 ? std::atoi(argv[2]) : 3400;
            e.ntmp       = argc > 3 ? std::atoi(argv[3]) : 10000;
            e.verbose    = false;
            e.zero_outer = (argc > 4 && !std::strcmp(argv[4], "zero"));
            exp5_run(e);
        }
        if (!std::strcmp(which, "7") || !std::strcmp(which, "all"))
        {
            exp7(5, 200, true, false, false);
            exp7(5, 200, true, true, false);
            exp7(5, 200, false, true, true);
        }
        if (!std::strcmp(which, "7scale") || !std::strcmp(which, "all"))
            exp7(argc > 2 ? std::atoi(argv[2]) : 3400, argc > 3 ? std::atoi(argv[3]) : 10000, false,
                 argc > 4 && !std::strcmp(argv[4], "zero"), false);
        if (!std::strcmp(which, "6") || !std::strcmp(which, "all"))
            exp6(argc > 2 ? std::atoi(argv[2]) : 3400, argc > 3 ? std::atoi(argv[3]) : 10000);
    }
    catch (const std::exception& e)
    {
        std::printf("UNCAUGHT EXCEPTION: %s\n", e.what());
        return 2;
    }
    return 0;
}

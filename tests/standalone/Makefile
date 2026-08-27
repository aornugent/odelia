# Build and run the R-free core check as plain C++ (#43).
#
#   make          build and run
#   make build    build only
#   make clean
#
# The include path is deliberately just ../../inst/include -- no R, no Rcpp, no
# BH. If this needs any of those added back in order to build, the solver core
# has stopped being usable outside an R package and something needs fixing.
#
# src/Tape.cpp is compiled in because the XAD tape runtime is defined in exactly
# one object file by design (ARCHITECTURE.md); anything that instantiates Solver
# has to link it. It is R-free too.

CXX     ?= c++
# The core uses concepts, so this must stay at C++20 or later: below it every
# `concept` in the headers reads as `'concept' does not name a type` and the
# hundred errors after that are cascade.
CXXSTD  ?= -std=c++20
OPT     ?= -O0
WARN    ?= -Wall -Wextra -Wno-unused-parameter

INCLUDES = -I../../inst/include
HEADERS  = $(wildcard ../../inst/include/odelia/*.hpp)

.PHONY: all build selfcontained clean

all: build selfcontained
	./r_free

build: r_free

# Every core header must also compile on its own -- no leaning on whichever
# sibling header happened to be included first to have pulled in <vector> or
# <cassert>. The two R interface headers are exempt; they are meant to need Rcpp.
selfcontained:
	@for h in ../../inst/include/odelia/*.hpp; do \
	  b=$$(basename $$h); \
	  case $$b in solver_interface.hpp|rcpp_interface_helpers.hpp) continue;; esac; \
	  printf '#include <odelia/%s>\n' $$b \
	    | $(CXX) $(CXXSTD) $(INCLUDES) -fsyntax-only -x c++ - \
	    || { echo "not self-contained: $$b"; exit 1; }; \
	  echo "  self-contained: $$b"; \
	done

r_free: r_free.o Tape.o
	$(CXX) $(CXXSTD) -o $@ $^

r_free.o: r_free.cpp $(HEADERS)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(INCLUDES) -c -o $@ $<

# Vendored third-party code: build it quietly, we do not police its warnings.
Tape.o: ../../src/Tape.cpp
	$(CXX) $(CXXSTD) $(OPT) -w $(INCLUDES) -c -o $@ $<

# A measurement rather than a check, so it is not in `all`. It compiles its own
# copy of the tape runtime because it needs widths the shipped one does not
# instantiate, and therefore links without Tape.o.
probe_width: probe_width.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $<

# Links the shipped tape runtime rather than including it the way probe_width
# does: this probe needs only the width odelia already instantiates. Compiling
# $< alone left every Tape symbol undefined, so `make` never built it.
probe_implicit_tangent: probe_implicit_tangent.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $< ../../src/Tape.cpp

# Links the shipped tape runtime, as probe_implicit_tangent does. A measurement
# and a demonstration rather than a check, so it is not in `all`.
probe_tape_reset: probe_tape_reset.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $< ../../src/Tape.cpp

# The mark-rewind, measured: whether a recording nested inside a checkpoint
# callback protects the slots a carried holder registered before it, and what the
# alternatives cost. A measurement, so it is not in `all`.
probe_nested_recording: probe_nested_recording.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $< ../../src/Tape.cpp

# What a caller can do to slots directly: hand-written statements, a callback
# writing adjoints mid-sweep, whether a recording carrying one survives being
# swept twice, and what rewinding the slot counter costs without clearAll. A
# measurement, so it is not in `all`.
probe_slot_control: probe_slot_control.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $< ../../src/Tape.cpp

# What a trapezium reduction costs on the tape when its widths are positions
# rather than state. A measurement, so it is not in `all`.
probe_trapezium: probe_trapezium.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $< ../../src/Tape.cpp

clean:
	rm -f r_free r_free.o Tape.o probe_width probe_implicit_tangent \
	  probe_tape_reset probe_nested_recording probe_slot_control \
	  probe_trapezium

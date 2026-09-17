# The measurements. Each is one translation unit, no R, and none of them is a
# check: they answer a question about the automatic-differentiation library's
# behaviour that reading its source did not settle. They live here rather than
# under tests/ because a program under tests/ reads as a guard, and a guard that
# nothing runs is worse than no guard.
#
#   make CXX=g++ <name>    build one
#   make CXX=g++ all       build every one
#   make clean
#
# The checks are in ../../tests/standalone.

CXX     ?= c++
CXXSTD  ?= -std=c++20
OPT     ?= -O0
WARN    ?= -Wall -Wextra
INCLUDES = -I../../inst/include
HEADERS  = $(wildcard ../../inst/include/odelia/*.hpp)

PROBES = probe_preaccumulate_rows probe_width probe_implicit_tangent probe_tape_reset \
         probe_nested_recording probe_slot_control probe_trapezium \
         probe_visit_active

.PHONY: all clean

all: $(PROBES)

# It compiles its own
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

# Which shapes visit_active opens and which it skips in silence. A demonstration
# of the hazard its own comment names, so it is not in `all`.
probe_visit_active: probe_visit_active.cpp $(HEADERS)
	$(CXX) -std=c++20 -O2 -DNDEBUG -DXAD_NO_THREADLOCAL \
	  -DXAD_USE_STRONG_INLINE -w $(INCLUDES) -o $@ $<

clean:
	rm -f $(PROBES)

# Two questions about what a rewound region leaves behind on the caller's tape:
# whether the harvest reports the caller's own adjoint as the region's row, and
# what an input left out of the declared list actually costs.
probe_preaccumulate_rows: probe_preaccumulate_rows.cpp ../../src/Tape.cpp $(HEADERS)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(INCLUDES) $< ../../src/Tape.cpp -o $@

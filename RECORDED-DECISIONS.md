# What a passive pass records, and what it hands the active one

A study of the primitive behind four sites that all "store a solve and load it
back": the step trajectory, node introduction, the competition field, and the
leaf's operating point. They are one pattern, this library already carries both
halves of it, and one of the four gets it wrong.

## The principle

**A value computed off the tape splits in two, and the split is not about cost.**

* What **MOVES** with the inputs -- an argmax, a fixed point, a root -- is lifted
  onto the tape by `implicit_value`, which takes the number from the passive solve
  and the row from somewhere other than recording the steps that produced it.
* What **SELECTS** -- an arm, a step size, a knot position, a schedule -- is
  piecewise constant in the inputs. It stays `double`, and the active pass
  **replays** it rather than re-deciding.

Re-deciding is not merely wasteful. A selector's derivative is a sequence of zeros
and jumps, so differentiating through one manufactures a discontinuity the model
does not have. phylloptim says this of ProfitMax's normaliser in as many words:
*"its argmax is piecewise constant in the traits and its derivative is a sequence
of zeros and jumps."* The same is true of a step size, of which knot the field is
read at, and of which bound a collar is pinned to.

So the rule is: **the active pass differentiates the model at a fixed decision, not
the decision.**

## The two mechanisms, and that they are already the right two

`implicit_value` (`implicit_node.hpp`) is the moving half, and its own header
states the general case: *"how a quantity computed away from the tape gets onto it
-- a root-find, a submodel's own solve, anything whose derivative is known by some
means other than recording the steps that produced it. At a plain double the terms
all vanish and this is the value."*

`instruction` (`ode_interface.hpp`) is the selecting half:

    struct instruction { double time; double step_size; bool insertion = false; };

and it already unifies more than it looks. A NaN `step_size` is *"a time with no
size known, which is a grid point rather than a recorded step: step TO it. So one
type covers a grid a caller chose and a program a run emitted."* One type for a
decision, whether a solver or a caller made it. `step_record : instruction` then
adds the state, deriving rather than repeating -- *"written out separately, they
were two structs differing by one member, and pairing a time from one container
with a state from another was a thing that compiled."*

## The four sites

| site | the selector, held passive | the value, carried active | verdict |
|---|---|---|---|
| step trajectory | `instruction{time, step_size}` | `step_record::state` | correct |
| node introduction | `instruction::insertion` | the widened state | correct -- one flag on the same type |
| competition field | knot positions: fixed fractions x `height_max` | knot values and slopes carry `S` | correct -- *"the knot positions stay [double]"* |
| leaf operating point | the arm / kind | collar, sigma, ci | **WRONG** |

Three of four already observe the split. plant made the third one true on purpose:
the knot fractions were made fixed so *"the positions and the count depend on
height_max and on nothing else in the state"* -- which is exactly the act of
turning a per-build decision into a replayed one.

The leaf is the outlier. `phylloptim::Leaf::SolvedPoint` holds
`{collar, sigma, ci, kind, arm}` -- three moving values and two selectors in one
class, with one lifetime and one predicate over both. That is why it reads as an
invented representation: it is, but the invention is the *packaging*, not the
mechanism. `bound_at` underneath it calls `implicit_value` correctly.

## Recommendation: no third primitive

The two mechanisms are correctly factored and each is named. A combined type would
be a wrapper over two named things with one caller, which `principles.md` calls
not a boundary. What is missing is the **rule**, stated once -- this file.

The leaf memo then reduces to what the other three sites already are:

* the collar: one `double`, replayed through phylloptim's own
  `evaluate_root_collar_psi`, which *"leaves exactly the same outputs as
  find_root_collar_psi"* and re-derives sigma, ci and the outputs from it;
* the arm: one enum, recorded exactly as `instruction::insertion` is.

No class, no friend declaration, no `searched()` predicate.

## The cost side, since it is also real

The saving is not the reason for the split, but it is not small. A recording is a
model evaluation and a sweep is arithmetic, so *"a caller wanting several rows pays
one recording rather than one per row"* -- which is why `adjoint_rows` is a set and
not a vector. On the leaf, a placement avoids three nested root-finds, against a
`record_leaf_outputs` that ran 861 tape statements 2,333,500 times per gradient.

Getting the split right buys both: the selector costs nothing to replay, and the
moving value costs one supplied row instead of a recorded search.

# Draft: message to the flyBrain author

**Status: draft. Nothing has been sent.** A human sends it or does not.

**Where:** an issue or discussion on
[mehrantsi/flyBrain](https://github.com/mehrantsi/flyBrain), whichever that
repository has enabled. Not email — there is no address to use that was given for
this purpose.

**Why send it at all:** two of the three largest findings in this project came
from reading that repository, which the README and `ATTRIBUTION.md` already say.
Saying it to the author directly is the honest version of that. The second half
is a measured disagreement about refractory conductance, raised as a question,
because they may well be right and I may be reading their kernel wrong.

**What to check before sending:**

- That the discrepancy section still matches the README's "One open
  discrepancy" — if that section has been corrected since, this message must be
  corrected with it, or not sent.
- That the tone still reads as a question rather than a correction. If it does
  not, do not send it.

---

## Draft text

Hello — I have been building an MLX/Metal engine for the same Shiu et al. model,
and I would rather tell you directly than have you find it: two of the three
largest speedups in it came from reading flyBrain.

Specifically, and this is in my README and ATTRIBUTION as well:

- flyBrain got to `mx.fast.metal_kernel` for CSR propagation first. Its MLX lane
  already uses one thread per source neuron with an early exit — the same shape
  as my kernel.
- Kernel fusion came straight from your README's description of fusing
  decay/threshold work with propagation to remove a full-neuron dispatch per
  tick. Applying it took my engine from 0.75 s to 0.29 s per biological second.
  That is the second-largest step in the project and it was not my idea.
- Your benchmark setup — sugar-GRN stimulus, chunked steps, excluding pack load
  and shader compilation — is what made a comparison possible at all.

I do publish a comparison, and I would rather you saw how it is framed than hear
about it secondhand. On an M4 Pro, measured in one session: my fused lane 0.2933
against flyBrain 0.3769 s per biological second, so about 22 %. But my engine
rebuilds its MLX graph on the host every chunk, and that host work competes for
CPU while a native Rust loop does not: on the same machine under load (load
average 2.8) mine moves to 0.3341 (+13.9 %) while yours moves to 0.3726 (−1.1 %,
i.e. noise), so the margin falls to about 10 %. It also costs about 7× your peak
memory — 665 MB against 94 MB, most of it work `async_eval` has scheduled and
not finished; with a blocking `eval` the same run peaks at 273 MB. Two points is
not a curve and I make no claim about a busier host than I tested.

The reason I am writing rather than only linking: I think one of our engines is
wrong about refractory conductance, and I cannot tell which.

Your README says incoming conductance can accumulate while a neuron is
refractory, "matching the upstream Brian2 equations". As far as I can measure,
Brian2 2.10.1 with this model formulation does the opposite: `(unless
refractory)` shields the variable from every write, synaptic input included. A
spike arriving at a refractory neuron leaves `g` at exactly 0.00000, before and
after the refractory period ends — dropped, not queued.

It changes results. Measured with the same stimulus on both sides — the sugar
drive at 150 Hz, seed 20260816, 10,000 ticks — my engine fires 13,594 spikes as
shipped and 16,382 with that gate removed, against your 16,796. Removing it
closes 87 % of the gap, from 3,202 spikes to 414. The remaining 414 stay
unexplained, so there is likely a second difference I have not found — and I have
not read your kernel closely enough to claim your engine does what I think it
does. If you have a Brian2 run that shows accumulation during the refractory
period, I would like to see it, because then my float64 oracle is wrong and I
want to know.

Reproduction on my side is `python tools/refractory_gate.py`, which runs that
lane twice and will not report the second number unless the unmodified run first
reproduces my recorded spike count, plus `python -m lif.validate_brian2` and the
two-neuron case described in the source.

Either way: thank you. The project would have been slower and worse without
yours.

---

## Provenance of every number above

| number | where it comes from |
|---|---|
| 0.75 s → 0.29 s from fusion | README, "How it works" |
| 0.2933 vs 0.3769, same session | README, "Results" |
| 0.3341 vs 0.3726 under load, +13.9 % / −1.1 % | README, "How much to trust these" |
| 665 MB vs 94 MB, 273 MB with blocking `eval` | README, "Results" and "Use" |
| `g` stays 0.00000 while refractory | README, "One open discrepancy" |
| 13,594 → 16,382 vs their 16,796, 87 % of the gap closed | `tools/refractory_gate.py`, 2026-09-15 |

The 22 % and the 10 % are both stated, not just the flattering one. If only one
of them survives an edit, this message should not be sent.

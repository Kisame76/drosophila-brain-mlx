# Draft: the shuffled-wiring control

**Status: draft. Nothing here has been posted.** A human posts it or does not.

**Where it could go:** a short write-up for whatever venue fits — a blog post, a
Mastodon or Bluesky thread, or the body of a README section elsewhere. It is
deliberately not written as a claim about flies, and it should not be retitled
into one.

**Why it might be worth posting:** after the MaleCNS v1.0 paper, many projects
wired the connectome to games and demos, and hardly any compares against a
baseline that would show the connectome contributes anything. This is that
comparison, run on a simulation that is parity-gated against the published model.

**What to check before posting:** that the figure referenced below is the one
currently in the README, and that nobody reads the last section as optional.

---

## Draft text

### Does the wiring actually carry the signal?

A connectome simulation that produces output is not evidence that the connectome
did the work. The input alone can produce output. So here is the control.

Drive the 21 right sugar-sensing neurons of the FlyWire v630 brain with Poisson
input at 100 Hz, as the published model's example notebook does, and read out
MN9, the proboscis motor neuron that notebook reports, over 30 trials of one
second. Then do exactly the same thing on a shuffled copy of the connectome in
which every neuron keeps its number of incoming and outgoing connections and the
signs and sizes of its outgoing synapses, but each connection goes to a random
target.

- On the real wiring, MN9 fires at **67.30 Hz**.
- On the shuffled wiring, MN9 fires **not once**.
- Neurons firing at all: **408** on the real wiring, **96** on the shuffled one.
- Both runs get the same input spikes: the sugar neurons themselves fire at
  **99.40 Hz** on the real wiring and **99.38 Hz** on the shuffled one, so a
  silent MN9 is not a run in which nothing happened.

One shuffle could be luck, so there are five. Seeds 1 to 4 leave MN9 silent as
well, with 89 to 99 neurons firing and 73,735 to 74,342 spikes in all, against
290,963 on the real connectome.

Degrees and signs are preserved by construction, so what the shuffle destroys is
which neuron connects to which. In these runs that is what carries sugar to MN9.

Two commands reproduce it, and `--seed N` runs another shuffle:

```bash
python -m lif.shuffle_pack --pack data/pack/v630 --seed 0
python -m lif.control_demo
```

### What this does not show

It does not show that a fly's brain works this way. The model's constants are
Shiu et al.'s, fitted to FlyWire, and the simulation is a leaky integrate-and-fire
approximation with one delay and no plasticity. What it shows is narrower and
still worth having: in this model, on this connectome, the specific wiring — not
the degree distribution, not the excitatory/inhibitory mix — is what gets the
signal to the motor neuron.

Repository: https://github.com/Kisame76/drosophila-brain-mlx

---

## Provenance of every number above

| number | where it comes from |
|---|---|
| MN9 67.30 Hz, silent on shuffled | `python -m lif.control_demo`, 2026-09-15 |
| 408 vs 96 neurons firing | same run |
| 99.40 / 99.38 Hz sugar GRNs | same run |
| seeds 1–4: 89–99 neurons, 73,735–74,342 spikes, 290,963 real | ROADMAP, phase 3 item 1 |
| shuffle keeps degrees and signed out-weights | `src/lif/shuffle_pack.py`, manifest marks the pack as not the real connectome |

MN9's ID, 720575940660219265, is the one in upstream's notebook.

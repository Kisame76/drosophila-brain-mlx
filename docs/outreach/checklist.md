# Outreach checklist

Every item here is an action a **human** performs. Nothing in this directory has
been posted, sent, uploaded or opened as a pull request, and nothing in it should
be executed automatically.

State at the time of writing (2026-09-15): the work is committed on `main` and
**not pushed**. Pushing is yours.

## Order

Do them in this order, or skip any of them. Later items assume the repository is
public and current; none of them assume the earlier ones happened.

### 1. Push

```bash
git push
```

Everything below points people at the repository, so it should be current first.
Check `git log origin/main..main` to see what is going out.

### 2. GitHub repository settings — optional

Topics, if they are not already set: `apple-silicon`,
`computational-neuroscience`, `connectome`, `drosophila`, `metal`, `mlx`.
Nothing else about the repository settings needs changing for any item below.

### 3. PyPI — deliberately unfinished

**Nothing has been uploaded, and the tooling to upload is not installed.** What
is done: the metadata is filled in, and `sdist` + `wheel` build offline through
the setuptools backend and install into a clean virtualenv, where
`importlib.metadata.version("mlx-lif-engine")` reads `0.1.0`.

Rebuild the artifacts:

```bash
rm -rf dist && python -c "from setuptools import build_meta as b; b.build_sdist('dist'); b.build_wheel('dist')"
```

If you decide to publish, the remaining steps are yours: install `twine`, run
`twine check dist/*`, upload to TestPyPI first, install from there into a clean
environment, and only then upload to PyPI. Two things to decide first:

- **The name.** The distribution is `mlx-lif-engine` while the repository and the
  README are `drosophila-brain-mlx`. Both names were free on PyPI on 2026-09-15.
  They differ on purpose: `src/lif/experiment.py` stamps
  `importlib.metadata.version("mlx-lif-engine")` into every parquet `run_exp`
  writes, and under a different distribution name that field silently becomes
  `"unknown"`. Rename both together or neither.
- **Whether it should be on PyPI at all.** The package is useless without a
  compiled pack, which needs a 90 MB or 1.06 GB download and a compile step.
  `pip install` sets no expectations that this satisfies.

### 4. awesome-fly — probably already done

Read [awesome-fly-pr.md](awesome-fly-pr.md) first. A pull request
([cobanov/awesome-fly#5](https://github.com/cobanov/awesome-fly/pull/5)) was
already opened on 2026-09-15. **Check its state before acting**: still open → do
not open a second one; merged → nothing to do; closed → read why first.

### 5. MLX discussion — follow-up, not a new thread

[mlx-notes-post.md](mlx-notes-post.md) is written as a comment on the existing
[ml-explore/mlx#4512](https://github.com/ml-explore/mlx/discussions/4512), which
already presents `docs/mlx-notes.md`. Do not open a second thread on the same
material.

### 6. flyBrain

[flybrain-message.md](flybrain-message.md) — an issue or discussion on
`mehrantsi/flyBrain`, not email. It credits the two findings this project took
from that repository and asks about the refractory-conductance discrepancy.
Before sending, confirm the README's "One open discrepancy" still says what the
message says, and confirm the message still reads as a question.

### 7. The shuffled-wiring write-up

[shuffle-control-post.md](shuffle-control-post.md) has no venue chosen. It is the
one piece of outreach that makes an argument rather than an announcement, so it
is also the one most likely to be retitled into a claim it does not support.
Do not let "in this model, the wiring carries the signal" become "the connectome
explains behaviour".

## Numbers you may be asked for

All measured on an M4 Pro. Each is in the README with its conditions; none should
be quoted without them.

| | |
|---|---|
| fused lane, FlyWire v630 | 0.2933 s per biological second (2026-09-14 set), 0.2937 ±0.0004 re-measured 2026-09-15 |
| Brian2, same model | 2.07 s per biological second with cython, 8.38 s with numpy, measured 2026-09-15 |
| vs. flyBrain | ~22 % on a quiet machine, ~10 % under load; 7× the peak memory |
| one experiment, 30 trials of 1 s | 10.05 s end to end |
| MaleCNS pack compile | 86 s |
| `./tools/demo.sh`, data already fetched | 29 s to the film |
| tests | 171 passing |

## What was deliberately not done

- Nothing was posted, sent, uploaded, or opened as a pull request.
- No PyPI upload, and `twine` was not installed to make one easy.
- The README's benchmark table was **not** updated with the 2026-09-15 re-run.
  It is labelled as one same-session set measured together with the flyBrain
  comparison, and splicing a single fresh row into it would destroy that.
- Brian2 was not re-measured *while this checklist was written*. It was measured
  immediately afterwards, and the result corrected the README: 2.07 s per
  biological second, not 62.6, so the comparison is **~7×**, not ~213×. Any
  outreach draft quoting the old ratio must be fixed before it is used.
- `src/lif/` was not touched, so the distribution name was left alone rather than
  breaking the version field described in item 3.

# Draft: the awesome-fly entry

**Status: draft. No pull request has been opened from this file.** A human opens
it or does not.

**Important — this may already be done.** The repository was proposed for
[cobanov/awesome-fly](https://github.com/cobanov/awesome-fly) under Brain Models
on 2026-09-15 in
[cobanov/awesome-fly#5](https://github.com/cobanov/awesome-fly/pull/5) (ROADMAP,
phase 3 item 1). **Check that pull request's state before doing anything with
this file:**

- **Still open** → do not open a second one. If anything below is worth adding,
  add it as a comment on #5.
- **Merged** → nothing to do. The entry is in the list; this file is only a
  record of what was said.
- **Closed unmerged** → read why it was closed first. Re-proposing the same entry
  without addressing the reason is how a list maintainer gets annoyed.

**What has changed since #5 was opened:** the repository gained a second dataset's
activity film, a stimulus survey, a validation against the published notebook's
Brian2 files, and a one-command demo. None of that changes the one-line entry
below; it only changes what a comment on #5 could mention.

---

## The entry, as proposed

Under **Brain Models**, alphabetical position as the list requires:

```markdown
- [drosophila-brain-mlx](https://github.com/Kisame76/drosophila-brain-mlx) - Leaky integrate-and-fire simulation of the FlyWire v630 and MaleCNS v1.0 connectomes in Apple MLX with a custom Metal kernel, parity-gated against Brian2.
```

One line, no superlatives, no comparison against other entries in the list. That
is deliberate: the list is other people's work too.

## If a comment on #5 is warranted

Only if the maintainer asks what is new or what makes it different:

> Since opening this: the engine now also runs upstream's example notebook with
> its code cells unchanged and agrees with the five Brian2 result files upstream
> published for it, within trial-to-trial noise across all 15 comparisons.
> `./tools/demo.sh` goes from a bare checkout to the activity film in one
> command. There is also a shuffled-wiring control in the README, because a
> connectome simulation that produces output is not by itself evidence that the
> connectome did the work.

## What not to claim

- Not that it is faster than any other entry in the list. The only engine it has
  been measured against is flyBrain, that comparison is in the README with its
  caveats, and it does not belong in an awesome-list entry.
- Not that it reproduces fly behaviour. It does not.
- Not "the fastest" anything. The README's own numbers are machine- and
  load-dependent and say so.

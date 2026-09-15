"""Spike-event recording: which neuron fired in which tick.

The lanes count spikes on the device and never read them back. Recording needs
the events themselves, (tick, neuron) pairs, and how many there are is known only
once a chunk has run: the data-dependent output size a lazy MLX graph cannot
express (see engine_metal.py). A Metal kernel can. It runs once per chunk over
the chunk's stacked spike masks, one thread per (neuron, tick), and each thread
whose neuron fired claims the next slot of a fixed-size event buffer through an
atomic counter. The host reads the counter afterwards; more spikes than slots is
an overflow, and it raises rather than truncating.

Which thread gets which slot is a race, so the raw buffer order differs between
runs. The sorted events do not, and tests/test_spike_record.py checks that.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from lif import core

# Events per chunk, whatever the chunk length: 2 MiB of int32 pairs. The densest
# run measured, 100 hubs at 150 Hz, averaged ~166 spikes per 32-tick chunk.
CAP = 262_144

# Two outputs, both atomic because atomic_outputs applies to every output. The
# tick is the offset within the chunk; the host adds the chunk's first tick.
_SRC = """
    uint i = thread_position_in_grid.x;
    uint k = thread_position_in_grid.y;
    uint n_ticks = uint(spikes_shape[0]);
    uint n = uint(spikes_shape[1]);
    if (i >= n || k >= n_ticks) return;
    if (!spikes[k * n + i]) return;
    uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
    if (slot >= cap[0]) return;
    atomic_store_explicit(&events[2 * slot], int(k), memory_order_relaxed);
    atomic_store_explicit(&events[2 * slot + 1], int(i), memory_order_relaxed);
"""

_kernel = mx.fast.metal_kernel(
    name="spike_events",
    input_names=["spikes", "cap"],
    output_names=["events", "count"],
    source=_SRC,
    atomic_outputs=True,
)


class RecordOverflow(RuntimeError):
    """One chunk produced more spikes than its event buffer holds."""

    def __init__(self, tick_range: range, count: int, cap: int):
        self.tick_range = tick_range
        self.count = count
        self.cap = cap
        super().__init__(
            f"ticks {tick_range.start} to {tick_range.stop - 1} produced {count} spikes, "
            f"more than cap={cap} events per chunk; run again with a larger cap")

    def __reduce__(self):
        # The default pickles args, which hold only the message, so a run in a
        # worker process could not hand the overflow back to its parent.
        return type(self), (self.tick_range, self.count, self.cap)


class Recorder:
    """One run's events, collected chunk by chunk.

    submit() adds a chunk's extraction to the graph and returns the arrays to
    evaluate along with the chunk; it reads nothing back. drain() reads chunks
    back, oldest first. The lanes call drain(keep=1) after scheduling each chunk,
    which reads the chunk before it: the host never waits on the chunk it has
    just handed to async_eval, so recording does not turn the pipeline into a
    sync per chunk, and at most two chunks' event buffers are alive at a time.
    """

    def __init__(self, n_neurons: int, cap: int = CAP):
        # MLX shapes are int32. A cap beyond that does not raise in the kernel
        # call; it aborts the whole process.
        if not 1 <= cap <= 2**31 - 1:
            raise ValueError(f"cap must be from 1 to 2**31 - 1, got {cap}")
        self.n_neurons = n_neurons
        self.cap = cap
        self._cap = mx.array([cap], dtype=mx.uint32)
        self._pending: list[tuple[range, mx.array, mx.array]] = []
        self._drained: list[np.ndarray] = []

    def submit(self, spikes: Sequence[mx.array], t0: int) -> list[mx.array]:
        """Extract the events of ticks t0 .. t0 + len(spikes) - 1, one mask per tick."""
        stack = mx.stack(list(spikes))
        if stack.shape[1:] != (self.n_neurons,):
            raise ValueError(f"spike masks must have shape ({self.n_neurons},), "
                             f"got {tuple(stack.shape[1:])}")
        events, count = _kernel(
            inputs=[stack, self._cap],
            output_shapes=[(self.cap, 2), (1,)],
            output_dtypes=[mx.int32, mx.uint32],
            grid=(self.n_neurons, len(spikes), 1),
            threadgroup=(256, 1, 1),
            init_value=0,
        )
        self._pending.append((range(t0, t0 + len(spikes)), events, count))
        return [events, count]

    def drain(self, keep: int = 0) -> None:
        """Read back every submitted chunk except the `keep` most recent ones.

        An overfull chunk stays pending, so a caller that catches RecordOverflow
        gets it again from every later drain() or events(), never the other
        chunks' events without it.
        """
        while len(self._pending) > keep:
            ticks, events, count = self._pending[0]
            n = count.item()
            if n > self.cap:
                raise RecordOverflow(ticks, n, self.cap)
            chunk = np.array(np.asarray(events)[:n])
            chunk[:, 0] += ticks.start
            self._drained.append(chunk)
            self._pending.pop(0)

    def events(self) -> np.ndarray:
        """All events, int32 [E, 2] of (tick, neuron), sorted by tick, then neuron."""
        self.drain()
        if not self._drained:
            return np.zeros((0, 2), dtype=np.int32)
        ev = np.concatenate(self._drained)
        return ev[np.lexsort((ev[:, 1], ev[:, 0]))]


def tick_to_seconds(tick: np.ndarray) -> np.ndarray:
    """Biological time in seconds, float64, of the spikes recorded in `tick`.

    Engine tick k performs the update, threshold and reset Brian2 performs at
    clock time k * dt, and SpikeMonitor records that time, so there is no offset.
    validate_brian2 compares the result with Brian2's spike times exactly; if
    that ever fails, the offset belongs here and nowhere else.
    """
    return np.asarray(tick).astype(np.float64) * (core.DT / 1000.0)

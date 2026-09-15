"""run_exp: the published model's experiment interface on this engine.

philshiu/Drosophila_brain_model runs an experiment as n_run trials of t_run with
Poisson input on neu_exc at r_poi and on neu_exc2 at r_poi2, the neurons of
neu_slnc silenced, and writes one parquet row per spike. run_exp here takes the
same arguments and writes the same columns, so upstream's load_exps and get_rate
read its files unchanged. Trial n draws its input with seed + n and is one run of
a lane with spike recording.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lif import core, spike_record
from lif import names as lif_names
from lif.compile_pack import sha256_file

METADATA_KEY = b"mlx_lif_engine"

# Upstream's default_params with every quantity as the plain number float() gives
# for it, in seconds, volts and hertz. That is the unit the published notebook
# needs when it passes params['t_run'] to get_rate, and a Brian2 quantity may
# stand in for any of these values.
default_params = {
    "t_run": 1.0,
    "n_run": 30,
    "v_0": core.V_0 / 1000,
    "v_rst": core.V_0 / 1000,
    "v_th": core.V_TH / 1000,
    "t_mbr": core.T_MBR / 1000,
    "tau": core.TAU / 1000,
    "t_rfc": core.T_RFC / 1000,
    "t_dly": core.T_DLY / 1000,
    "w_syn": core.W_SYN / 1000,
    "r_poi": 150.0,
    "r_poi2": 0.0,
    "f_poi": core.F_POI,
    "eqs": """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
""",
    "eq_th": "v > v_th",
    "eq_rst": "v = v_rst; w = 0; g = 0 * mV",
}

# What the pack was compiled with, kept apart from default_params, which a notebook
# may change in place as upstream's example does.
_COMPILED = dict(default_params)
_TUNABLE = {"t_run", "n_run", "r_poi", "r_poi2"}
_UNITS = {"t_run": "second", "t_mbr": "second", "tau": "second", "t_rfc": "second",
          "t_dly": "second", "r_poi": "hertz", "r_poi2": "hertz", "v_0": "volt",
          "v_rst": "volt", "v_th": "volt", "w_syn": "volt"}


def run_exp(
    exp_name: str,
    neu_exc: Sequence[int | str],
    path_res: str | Path,
    path_comp: str | Path | None = None,
    path_con: str | Path | None = None,
    params: Mapping | None = None,
    neu_slnc: Sequence[int | str] = (),
    neu_exc2: Sequence[int | str] = (),
    n_proc: int = -1,
    force_overwrite: bool = False,
    *,
    pack: core.Pack | None = None,
    names: Mapping[int, str] | None = None,
    seed: int = 0,
    engine: str = "fused",
    edge_split: int = 1,
    cap: int = spike_record.CAP,
) -> Path:
    """Run n_run trials and write {path_res}/{exp_name}.parquet; returns that path.

    Upstream's positional arguments, in upstream's order. Neurons are dataset IDs,
    or names looked up in `names` (upstream's flyid2name). path_comp and path_con
    are optional; when given, their SHA-256 must match the files the pack was
    compiled from. n_proc is accepted and ignored. pack defaults to core.load_pack().

    Trial n draws its input with seed + n, so two experiments with the same seed
    and neu_exc give every trial the same input spike train and differ only by
    what else changed. edge_split goes to the lane: 1 suits the sparse drive of a
    few sensory neurons, 8 a drive that makes hubs fire (README, "Use").
    """
    path_save = Path(path_res) / f"{exp_name}.parquet"
    if path_save.is_file() and not force_overwrite:
        print(f">>> Skipping experiment {exp_name} because {path_save} exists and "
              f"force_overwrite = {force_overwrite}")
        return path_save

    t_run, n_ticks, n_run, r_poi, r_poi2 = _resolve_params(params)
    run = _lane(engine, edge_split, cap)
    pack = pack if pack is not None else _default_pack()
    _check_source(pack, "path_comp", path_comp, "completeness_csv")
    _check_source(pack, "path_con", path_con, "connectivity_parquet")

    index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
    exc = _resolve(pack, "neu_exc", neu_exc, names, index)
    exc2 = _resolve(pack, "neu_exc2", neu_exc2, names, index)
    slnc = _resolve(pack, "neu_slnc", neu_slnc, names, index)
    for what, found in (("neu_exc", exc), ("neu_exc2", exc2)):
        values, times = np.unique([f for f, _ in found], return_counts=True)
        if (times > 1).any():
            raise ValueError(f"{what} lists neurons more than once: "
                             f"{[int(v) for v in values[times > 1]]}")
    both = sorted({f for f, _ in exc} & {f for f, _ in exc2})
    if both:
        raise ValueError(f"neurons in both neu_exc and neu_exc2: {both}")
    mask = None
    if slnc:
        mask = np.zeros(pack.n_neurons, dtype=bool)
        mask[[i for _, i in slnc]] = True

    print(f">>> Experiment:     {exp_name}")
    print(f"    Output file:    {path_save}")
    print(f"    Excited neurons: {len(exc) + len(exc2)}")
    if slnc:
        print(f"    Silenced neurons: {len(slnc)}")

    start = time.time()
    ts, trials, neurons = [], [], []
    for n in range(n_run):
        stim = core.make_stimulus_for(pack, [i for _, i in exc], r_poi, n_ticks, seed + n,
                                      targets2=[i for _, i in exc2], rate2_hz=r_poi2)
        ev = run(pack, stim, mask).events
        ev = ev[np.lexsort((ev[:, 0], ev[:, 1]))]   # upstream's order: neuron, then time
        ts.append(spike_record.tick_to_seconds(ev[:, 0]))
        trials.append(np.full(len(ev), n, dtype=np.int64))
        neurons.append(ev[:, 1])
    print(f"    Elapsed time:   {int(time.time() - start)} s")

    neuron = np.concatenate(neurons)
    table = pa.table({
        "t": pa.array(np.concatenate(ts), pa.float64()),
        "trial": pa.array(np.concatenate(trials), pa.int64()),
        "flywire_id": pa.array(pack.neuron_ids[neuron], pa.int64()),
        "exp_name": pa.array([exp_name] * len(neuron), pa.string()),
    })
    meta = {
        "mlx_lif_engine_version": _version(),
        "dataset": pack.manifest.get("dataset", "unknown"),
        "pack_sha256": _pack_sha256(pack),
        "engine": engine, "edge_split": edge_split, "seed": seed,
        "t_run_s": t_run, "n_run": n_run, "dt_ms": core.DT,
        "r_poi_hz": r_poi, "r_poi2_hz": r_poi2,
        "neu_exc": [f for f, _ in exc], "neu_exc2": [f for f, _ in exc2],
        "neu_slnc": [f for f, _ in slnc],
    }
    table = table.replace_schema_metadata({METADATA_KEY: json.dumps(meta).encode()})

    # Written next to the target and moved into place, so an interrupted write
    # cannot leave a file that the skip above would then accept.
    path_save.parent.mkdir(parents=True, exist_ok=True)
    partial = path_save.with_name(path_save.name + ".partial")
    pq.write_table(table, partial, compression="brotli")
    os.replace(partial, path_save)
    return path_save


def metadata(path: str | Path) -> dict:
    """The experiment run_exp recorded in the file: parameters, seed, pack, engine."""
    meta = pq.read_schema(path).metadata or {}
    if METADATA_KEY not in meta:
        raise ValueError(f"{path} has no {METADATA_KEY.decode()} metadata; run_exp did not "
                         "write it")
    return json.loads(meta[METADATA_KEY])


def rates(path: str | Path, *, names: Mapping[int, str] | None = None) -> pa.Table:
    """Firing rate per neuron, as upstream's get_rate computes it.

    Spikes per trial divided by t_run, then mean and population standard deviation
    over n_run trials, a trial without spikes counting as 0 Hz; t_run and n_run come
    from the file. One row per neuron that fired at all, by ID: flywire_id, name
    (from names, else empty), rate_hz, std_hz.
    """
    meta = metadata(path)
    n_run, t_run = meta["n_run"], meta["t_run_s"]
    table = pq.read_table(path, columns=["trial", "flywire_id"])
    ids, neuron = np.unique(table["flywire_id"].to_numpy(), return_inverse=True)
    counts = np.bincount(neuron * n_run + table["trial"].to_numpy(),
                         minlength=ids.size * n_run).reshape(ids.size, n_run)
    r = counts / t_run
    return pa.table({
        "flywire_id": pa.array(ids, pa.int64()),
        "name": pa.array([(names or {}).get(int(i), "") for i in ids], pa.string()),
        "rate_hz": pa.array(r.mean(axis=1), pa.float64()),
        "std_hz": pa.array(r.std(axis=1), pa.float64()),
    })


@functools.cache
def _default_pack() -> core.Pack:
    return core.load_pack()


def _number(key: str, value) -> float:
    """A plain float in seconds, volts or hertz, from a number or a Brian2 quantity."""
    if hasattr(value, "dim"):
        import brian2  # a quantity exists, so Brian2 is installed
        from brian2.units.fundamentalunits import have_same_dimensions

        unit = getattr(brian2, _UNITS.get(key, ""), None)
        if unit is None or not have_same_dimensions(value, unit):
            raise ValueError(f"params['{key}'] = {value} is not in {_UNITS.get(key, 'no unit')}")
        return float(value)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"params['{key}'] must be a number, got {value!r}")
    return float(value)


def _resolve_params(params: Mapping | None):
    given = default_params if params is None else params
    unknown = sorted(set(given) - set(_COMPILED))
    if unknown:
        raise ValueError(f"params has keys upstream's default_params does not: {unknown}")
    p = {**_COMPILED, **given}
    for key, fixed in _COMPILED.items():
        if key in _TUNABLE:
            continue
        if isinstance(fixed, str):
            same = str(p[key]).split() == fixed.split()
        else:
            same = math.isclose(_number(key, p[key]), fixed, rel_tol=1e-9)
        if not same:
            raise NotImplementedError(
                f"params['{key}'] = {p[key]!r} differs from {fixed!r}: the model constants "
                "are compiled into the update coefficients and cannot change per run")

    t_run = _number("t_run", p["t_run"])
    n_ticks = round(t_run * 1000 / core.DT)
    if n_ticks < 1 or not math.isclose(n_ticks * core.DT / 1000, t_run, rel_tol=1e-9):
        raise ValueError(f"params['t_run'] = {p['t_run']} s is not a whole number of "
                         f"{core.DT} ms ticks")
    n_run = p["n_run"]
    if isinstance(n_run, bool) or not isinstance(n_run, (int, np.integer)) or n_run < 1:
        raise ValueError(f"params['n_run'] must be a whole number of trials, at least 1, "
                         f"got {n_run!r}")
    return t_run, n_ticks, int(n_run), _number("r_poi", p["r_poi"]), _number("r_poi2", p["r_poi2"])


def _lane(engine: str, edge_split: int, cap: int):
    from lif import engine_chunked, engine_fused, engine_metal, engine_naive

    lanes = {
        "naive": lambda pack, stim, mask: engine_naive.run(
            pack, stim, silenced=mask, warmup=0, record=True),
        "chunked": lambda pack, stim, mask: engine_chunked.run(
            pack, stim, silenced=mask, warmup=0, record=True, cap=cap),
        "metal": lambda pack, stim, mask: engine_metal.run(
            pack, stim, silenced=mask, warmup=0, split=edge_split, record=True, cap=cap),
        "fused": lambda pack, stim, mask: engine_fused.run(
            pack, stim, silenced=mask, warmup=0, edge_split=edge_split, record=True, cap=cap),
    }
    if engine not in lanes:
        raise ValueError(f"engine must be one of {sorted(lanes)}, got {engine!r}")
    return lanes[engine]


def _check_source(pack: core.Pack, arg: str, path, source: str) -> None:
    if path is None:
        return
    dataset = pack.manifest.get("dataset", "unknown")
    recorded = pack.manifest.get("sources", {}).get(source)
    if recorded is None:
        raise ValueError(f"{arg} was given, and the {dataset} pack records no {source} "
                         "to compare it with")
    digest = sha256_file(Path(path))
    if digest != recorded["sha256"]:
        raise ValueError(f"{arg} {path} has sha256 {digest}, and the {dataset} pack was "
                         f"compiled from {recorded['path']} with sha256 {recorded['sha256']}")


def _resolve(pack: core.Pack, what: str, entries, names, index: dict) -> list[tuple[int, int]]:
    """(dataset ID, model index) for every entry; every unknown one named at once.

    A name is looked up in names when they are given, where it must name one
    neuron, and otherwise in the pack's names sidecar, where it selects every
    neuron whose instance or, failing that, whose type it is (lif.names).
    """
    by_name: dict[str, list[int]] | None = None
    sidecar = None
    found, unknown = [], []
    for entry in entries:
        if isinstance(entry, str) and names is None:
            sidecar = sidecar or lif_names.load(pack)
            if sidecar is None:
                raise ValueError(f"{what} lists the name {entry!r}, and neither were names given "
                                 f"nor does the {pack.manifest.get('dataset', 'unknown')} pack "
                                 "have names to look it up")
            selected = sidecar.select(entry)
            if not selected:
                unknown.append(entry)
            found.extend((int(pack.neuron_ids[i]), i) for i in selected)
            continue
        if isinstance(entry, str):
            if by_name is None:
                by_name = {}
                for fid, name in names.items():
                    by_name.setdefault(name, []).append(int(fid))
            ids = sorted(by_name.get(entry, []))
            if len(ids) > 1:
                raise ValueError(f"{what}: {entry!r} is the name of {len(ids)} neurons: {ids}")
            if not ids:
                unknown.append(entry)
                continue
            entry = ids[0]
        elif isinstance(entry, bool) or not isinstance(entry, (int, np.integer)):
            raise TypeError(f"{what} must list neuron IDs or names, got {entry!r}")
        i = index.get(int(entry))
        if i is None:
            unknown.append(entry)
        else:
            found.append((int(entry), i))
    if unknown:
        dataset = pack.manifest.get("dataset", "unknown")
        raise ValueError(f"{what}: not in the {dataset} pack: {unknown}")
    return found


def _pack_sha256(pack: core.Pack) -> str:
    arrays = pack.manifest.get("arrays", {})
    text = "".join(f"{name} {arrays[name]['sha256']}\n" for name in sorted(arrays))
    return hashlib.sha256(text.encode()).hexdigest()


def _version() -> str:
    try:
        return importlib.metadata.version("mlx-lif-engine")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

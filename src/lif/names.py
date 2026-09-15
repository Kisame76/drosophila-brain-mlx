"""Cell-type names for a pack: a sidecar table with one row per model index.

MaleCNS v1.0 annotates every body with a type ("MN9"), an instance ("MN9_R"), a
class and a flywireType cross-reference. compile_pack_malecns writes those next to
the pack as names.parquet and records a hash of the table's content in the
manifest, so the hash does not depend on how a pyarrow version encodes the file.
FlyWire v630, as the published model uses it, ships no names, so that pack has no
sidecar and run_exp still takes names there from a flyid2name dict.

A name selects every neuron whose instance is that name and, if there is none,
every neuron whose type is: "MN9_R" is one neuron, "MN9" both.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lif.compile_pack import sha256_array

if TYPE_CHECKING:
    from lif import core

COLUMNS = ("type", "instance", "class", "flywireType")
FILE = "names.parquet"


def names_table(annotations: pa.Table, neuron_ids: np.ndarray) -> pa.Table:
    """bodyId and the name columns of annotations, one row per neuron in pack order."""
    missing = [c for c in ("bodyId", *COLUMNS) if c not in annotations.column_names]
    if missing:
        raise ValueError(f"annotations lack the columns {missing}")
    body = annotations["bodyId"].to_numpy().astype(np.int64)
    values, times = np.unique(body, return_counts=True)
    if (times > 1).any():
        raise ValueError(f"annotations list bodies more than once: {values[times > 1].tolist()}")
    ids = np.asarray(neuron_ids, dtype=np.int64)
    order = np.argsort(body, kind="stable")
    at = np.searchsorted(body[order], ids)
    found = at < body.size
    found[found] = body[order][at[found]] == ids[found]
    if not found.all():
        raise ValueError(f"neurons without an annotation row: {ids[~found].tolist()}")
    rows = pa.array(order[at])
    return pa.table({"bodyId": pa.array(ids, pa.int64()),
                     **{c: annotations[c].take(rows).cast(pa.string()) for c in COLUMNS}})


def content_sha256(table: pa.Table) -> str:
    """Hash of the column names and values, whatever the file encoding."""
    h = hashlib.sha256()
    for name in table.column_names:
        h.update(json.dumps([name, table[name].to_pylist()], ensure_ascii=False,
                            separators=(",", ":")).encode())
    return h.hexdigest()


def write_names(pack_dir: Path, table: pa.Table, source: str) -> dict:
    """Write table as the pack's names sidecar and record it in the manifest.

    source is the key under the manifest's sources of the file the names come from."""
    pack_dir = Path(pack_dir)
    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ids = np.load(pack_dir / "neuron_ids.npy", allow_pickle=False)
    if sha256_array(ids) != manifest["arrays"]["neuron_ids"]["sha256"]:
        raise ValueError(f"{pack_dir / 'neuron_ids.npy'} does not match the sha256 in its manifest")
    if (table.column_names != ["bodyId", *COLUMNS]
            or not np.array_equal(table["bodyId"].to_numpy(), ids)):
        raise ValueError(f"the names table does not list the pack's {ids.size} neurons in pack "
                         f"order with the columns bodyId, {', '.join(COLUMNS)}")
    partial = pack_dir / f"{FILE}.partial"
    pq.write_table(table, partial, compression="zstd")
    os.replace(partial, pack_dir / FILE)
    entry = {"file": FILE, "rows": table.num_rows, "columns": table.column_names,
             "sha256": content_sha256(table), "source": source}
    manifest.setdefault("sidecars", {})["names"] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return entry


class Names:
    """Model indices by name, and a label per neuron, from a names table."""

    def __init__(self, table: pa.Table):
        self.ids = table["bodyId"].to_numpy()
        self._instance = table["instance"].to_pylist()
        self._type = table["type"].to_pylist()
        self._by_instance: dict[str, list[int]] = defaultdict(list)
        self._by_type: dict[str, list[int]] = defaultdict(list)
        for i, (instance, type_) in enumerate(zip(self._instance, self._type, strict=True)):
            if instance is not None:
                self._by_instance[instance].append(i)
            if type_ is not None:
                self._by_type[type_].append(i)

    def select(self, name: str) -> list[int]:
        """Model indices of the neurons whose instance is name or, if none, whose type is."""
        return list(self._by_instance.get(name) or self._by_type.get(name) or [])

    def labels(self) -> dict[int, str]:
        """Instance, else type, per neuron ID, as rates(names=...) takes it."""
        return {int(b): instance or type_
                for b, instance, type_ in zip(self.ids, self._instance, self._type, strict=True)
                if instance or type_}


def load(pack: core.Pack) -> Names | None:
    """The pack's names, or None when it has no names sidecar."""
    entry = pack.manifest.get("sidecars", {}).get("names")
    if entry is None:
        return None
    if pack.path is None:
        raise ValueError("the pack records a names sidecar but was not loaded from a directory")
    path = pack.path / entry["file"]
    stat = path.stat()
    names = _read(str(path), entry["sha256"], stat.st_mtime_ns, stat.st_size)
    if not np.array_equal(names.ids, pack.neuron_ids):
        raise ValueError(f"{path} does not list the pack's neurons in pack order")
    return names


# Keyed on the file's modification time and size too, so a rewritten file is read
# and checked again rather than served from the cache.
@functools.cache
def _read(path: str, sha256: str, mtime_ns: int, size: int) -> Names:
    table = pq.read_table(path)
    if content_sha256(table) != sha256:
        raise ValueError(f"{path} does not match the sha256 in its manifest")
    return Names(table)

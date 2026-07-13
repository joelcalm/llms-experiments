"""Create the checked-in-contract but generated local Parquet smoke input."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


if __name__ == "__main__":
    path = Path(__file__).with_name("smoke.parquet")
    rows = [{"id": f"smoke-{index:03d}", "text": f"A short generic smoke-test statement number {index}."} for index in range(128)]
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    print(path)

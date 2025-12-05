import pyarrow as pa
from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.arrow")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    batch = []

    with open(file_path, "rb") as f:
        reader = pa.ipc.RecordBatchStreamReader(f)
        for record_batch in reader:
            for line in record_batch.to_pylist():
                text = line["input"] + " " + line["output"]
                batch.append({"text": text})
                if len(batch) == batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

import json
from pathlib import Path
from typing import Any, Iterable


def get_files(path: str) -> list[str]:
    return [str(file) for file in Path(path).rglob("*.jsonl")]


def get_text(file_path: str, batch_size: int = 2000) -> Iterable[list[dict[str, Any]]]:
    batch = []

    with open(file_path, mode="r", encoding="utf-8") as file:
        for line in file:
            obj = json.loads(line)
            text = " ".join(conv["content"] for conv in obj["conversations"])
            batch.append({"text": text})
            if len(batch) == batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

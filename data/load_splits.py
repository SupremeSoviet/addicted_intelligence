import json

from data.dataset import (
    DatasetItem,
    splits_paths,
)


def load_splits() -> dict[str, list[DatasetItem]]:
    items_by_split: dict[str, list[DatasetItem]] = {}
    for split_name, split_path in splits_paths.items():
        with split_path.open(encoding="utf-8") as split_file:
            items_by_split[split_name] = [DatasetItem(**json.loads(line)) for line in split_file]
    return items_by_split

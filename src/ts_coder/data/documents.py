from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    record_id: str
    repository_id: str
    relative_path: str
    text: str
    split: str


def accepted_documents(
    records: list[dict], contents: dict[str, str], split: str | None = None
) -> list[Document]:
    return [
        Document(
            r["record_id"],
            r["repository_id"],
            r["relative_path"],
            contents[r["record_id"]],
            r["split"],
        )
        for r in sorted(records, key=lambda x: x["record_id"])
        if r["included"] and (split is None or r["split"] == split)
    ]

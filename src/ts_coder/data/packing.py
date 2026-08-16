from .documents import Document


def pack_documents(
    documents: list[Document], tokenizer, context_length: int, eos_id: int
) -> list[dict]:
    packed = []
    for repo in sorted({x.repository_id for x in documents}):
        ids = []
        record_ids = []
        for doc in sorted(
            (x for x in documents if x.repository_id == repo), key=lambda x: x.relative_path
        ):
            values = tokenizer.encode(doc.text).ids + [eos_id]
            if ids and len(ids) + len(values) > context_length:
                packed.append({"repository_id": repo, "record_ids": record_ids, "token_ids": ids})
                ids = []
                record_ids = []
            while len(values) > context_length:
                packed.append(
                    {
                        "repository_id": repo,
                        "record_ids": [doc.record_id],
                        "token_ids": values[:context_length],
                    }
                )
                values = values[context_length:]
            ids.extend(values)
            record_ids.append(doc.record_id)
        if ids:
            packed.append({"repository_id": repo, "record_ids": record_ids, "token_ids": ids})
    return packed

"""Repository-context serialization with explicit boundaries."""

from xml.sax.saxutils import quoteattr  # nosec B406 - escaping output attributes, not parsing XML


def pack_repository(files: list[tuple[str, str]]) -> str:
    parts = ["<repo>"]
    for path, content in sorted(files):
        safe = path.replace("\\", "/").lstrip("/")
        if ".." in safe.split("/"):
            raise ValueError("repository path traversal")
        parts.extend([f"<file path={quoteattr(safe)}>", content, "</file>"])
    parts.append("</repo>")
    return "\n".join(parts)

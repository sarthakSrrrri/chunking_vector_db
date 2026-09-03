


from docling_core.types.doc import TableItem


def structured_chunking(
    document,
    max_chars: int = 2000,
) -> list[dict]:
    chunks = []
    current_text = []
    current_length = 0

    for item, _ in document.iterate_items():
        if isinstance(item, TableItem):
            text = item.export_to_markdown(document)
            element_type = "table"
        else:
            text = getattr(item, "text", "").strip()
            element_type = type(item).__name__

        if not text:
            continue

        # Keep tables as complete elements.
        if element_type == "table":
            if current_text:
                chunks.append({
                    "text": "\n\n".join(current_text),
                    "element_type": "text",
                })
                current_text = []
                current_length = 0

            chunks.append({
                "text": text,
                "element_type": "table",
            })
            continue

        if current_length + len(text) > max_chars and current_text:
            chunks.append({
                "text": "\n\n".join(current_text),
                "element_type": "text",
            })
            current_text = []
            current_length = 0

        current_text.append(text)
        current_length += len(text)

    if current_text:
        chunks.append({
            "text": "\n\n".join(current_text),
            "element_type": "text",
        })

    return chunks
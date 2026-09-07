from docling_core.types.doc import TableItem

from src.chunking.base import ChunkingStrategy


class StructuredChunking(ChunkingStrategy):

    @property
    def name(self) -> str:
        return "structured"

    def chunk(
        self,
        document,
        max_chars: int = 2000,
    ) -> list[str]:

        chunks = []
        current_text = []
        current_length = 0

        for item, _ in document.iterate_items():

            if isinstance(item, TableItem):
                text = item.export_to_markdown(document)

                if current_text:
                    chunks.append(
                        "\n\n".join(current_text)
                    )

                    current_text = []
                    current_length = 0

                chunks.append(text)
                continue

            text = getattr(
                item,
                "text",
                "",
            ).strip()

            if not text:
                continue

            if (
                current_length + len(text) > max_chars
                and current_text
            ):
                chunks.append(
                    "\n\n".join(current_text)
                )

                current_text = []
                current_length = 0

            current_text.append(text)
            current_length += len(text)

        if current_text:
            chunks.append(
                "\n\n".join(current_text)
            )

        return chunks
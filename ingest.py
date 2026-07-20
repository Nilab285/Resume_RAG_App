from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from typing import Optional, List, Dict

def _extract_table_text(item) -> Optional[str]:
    """
    Converts ANY table (simple, complex, nested, sparse)
    into flat key-value pairs.

    Strategy:
    - Row 0 is treated as the header (keys)
    - Each subsequent row maps header → cell value
    - Empty cells are skipped cleanly
    - Multi-column rows: each (header, cell) becomes its own pair

    Output example:
        Category: Languages
        Highlights: Java (8, 11), Kotlin, SQL

        Category: Frameworks
        Highlights: Spring Boot, Spring MVC, Hibernate
    """
    try:
        grid = item.data.grid
        if not grid or len(grid) < 1:
            return None

        # ── Extract header row ───────────────────────────────────────────
        headers = []
        for cell in grid[0]:
            if cell and hasattr(cell, "text") and cell.text.strip():
                headers.append(cell.text.strip())
            else:
                headers.append(None)  # placeholder for missing headers

        # ── Process data rows ────────────────────────────────────────────
        kv_blocks = []

        for row in grid[1:]:  # skip header row
            pairs = []

            for col_idx, cell in enumerate(row):
                # get cell value
                cell_val = None
                if cell and hasattr(cell, "text") and cell.text.strip():
                    cell_val = cell.text.strip()

                if not cell_val:
                    continue  # skip empty cells cleanly

                # get matching header key
                key = (
                    headers[col_idx]
                    if col_idx < len(headers) and headers[col_idx]
                    else f"Column_{col_idx + 1}"  # fallback for missing headers
                )

                pairs.append(f"{key}: {cell_val}")

            if pairs:
                kv_blocks.append("\n".join(pairs))

        if not kv_blocks:
            return None

        # Each row becomes its own block separated by blank line
        return "\n\n".join(kv_blocks)

    except Exception:
        return None



def extract_document_structure(pdf_path: str) -> List[Dict]:
    """
    Extract ordered chunks from PDF using Docling.

    Chunk hierarchy:
    ┌─────────────────────────────────────────────────────┐
    │  SectionHeaderItem  →  chunk_section = None         │
    │                        updates current_section       │
    ├─────────────────────────────────────────────────────┤
    │  TextItem           →  chunk_section = current_sec  │
    │                        chunk_type    = TextItem      │
    ├─────────────────────────────────────────────────────┤
    │  TableItem          →  chunk_section = current_sec  │
    │                        chunk_type    = TableItem     │
    │                        content       = markdown rows │
    └─────────────────────────────────────────────────────┘
    """
    # ── Disable OCR for text-based PDFs (faster, no OCR model needed) ──
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True   # ← enables table parsing

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document

    structured_content = []
    current_section: Optional[str] = None
    chunk_order = 1

    for item, level in doc.iterate_items():
        item_type = item.__class__.__name__

        # ── SECTION HEADER ──────────────────────────────────────────────
        if item_type == "SectionHeaderItem":
            if not hasattr(item, "text") or not item.text:
                continue
            text = item.text.strip()
            if not text:
                continue

            structured_content.append({
                "chunk_content": text,
                "chunk_type":    "SectionHeaderItem",
                "chunk_section": None,          # header owns no section
                "chunk_order":   chunk_order
            })
            current_section = text              # all next chunks belong here
            chunk_order += 1

        # ── TABLE ────────────────────────────────────────────────────────
        elif item_type == "TableItem":
            table_text = _extract_table_text(item)
            if not table_text:
                continue

            structured_content.append({
                "chunk_content": table_text,
                "chunk_type":    "TableItem",
                "chunk_section": current_section,   # belongs to last header
                "chunk_order":   chunk_order
            })
            chunk_order += 1

        # ── TEXT / LIST / OTHER ──────────────────────────────────────────
        else:
            if not hasattr(item, "text") or not item.text:
                continue
            text = item.text.strip()
            if not text:
                continue

            structured_content.append({
                "chunk_content": text,
                "chunk_type":    item_type,
                "chunk_section": current_section,
                "chunk_order":   chunk_order
            })
            chunk_order += 1

    return structured_content

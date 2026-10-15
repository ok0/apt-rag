from __future__ import annotations

from pathlib import Path

import pymupdf

PAGE_W = 595
PAGE_H = 842


def make_pdf(
    path: Path,
    pages: list[list[str]],
    *,
    page_numbers: bool = True,
    tables: dict[int, list[list[str]]] | None = None,
    image_pages: set[int] = frozenset(),
) -> Path:
    path = Path(path)
    tables = tables or {}
    doc = pymupdf.open()
    font = pymupdf.Font("cjk")
    for i, lines in enumerate(pages):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        tw = pymupdf.TextWriter(page.rect)
        if page_numbers:
            tw.append((72, 40), f"-{i + 1}-", font=font, fontsize=10)
        y = 72
        for s in lines:
            if s:
                tw.append((72, y), s, font=font, fontsize=10)
            y += 18
        if i in tables:
            rows = tables[i]
            ncols = max(len(r) for r in rows)
            for r, row in enumerate(rows):
                for c in range(ncols):
                    x0 = 72 + c * 150
                    y0 = 400 + r * 20
                    rect = pymupdf.Rect(x0, y0, x0 + 150, y0 + 20)
                    page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                    if c < len(row) and row[c]:
                        tw.append((x0 + 4, y0 + 20 - 6), row[c], font=font, fontsize=10)
        tw.write_text(page)
        if i in image_pages:
            pix = page.get_pixmap(dpi=100)
            doc.delete_page(i)
            new_page = doc.new_page(pno=i, width=PAGE_W, height=PAGE_H)
            new_page.insert_image(new_page.rect, pixmap=pix)
    doc.save(str(path))
    doc.close()
    return path

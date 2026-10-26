"""Load a bill from disk and convert it to what a provider can read.

Three kinds of input:

- PDF: sent as is to providers that read PDFs natively (Claude, OpenAI, Gemini);
  for the others (most local models) every page is rendered to a PNG image.
- PNG, JPEG, WEBP: sent as is. Every vision model reads them.
- Other image formats (HEIC/HEIF from iPhones, TIFF from scanners, BMP, GIF):
  converted to PNG first, because most APIs do not accept them. Multi-page
  TIFF files become one image per page.

Converted images are rotated according to their EXIF orientation (phone photos
are often stored sideways) and scaled down to at most MAX_SIDE_PX on the long
side, which keeps requests small without making printed text unreadable.

A file that cannot be decoded raises UnreadableDocument, never the decoding
library's own exception: a folder of real bills always ends up containing one
truncated download or one file that is not what its extension claims, and a
single bad file must not be able to kill a labeling or evaluation run.
"""

from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass
from pathlib import Path

PDF = "application/pdf"
# Sent as they are.
NATIVE_IMAGES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
# Converted to PNG before sending.
CONVERTED_IMAGES = {".heic": "image/heic", ".heif": "image/heif", ".tif": "image/tiff", ".tiff": "image/tiff",
                    ".bmp": "image/bmp", ".gif": "image/gif"}
MIME_BY_SUFFIX = {".pdf": PDF, **NATIVE_IMAGES, **CONVERTED_IMAGES}
SUPPORTED_SUFFIXES = tuple(MIME_BY_SUFFIX)
MAX_SIDE_PX = 2000
# Pages sent per document. Real bills put the POD, the contracted power or the
# residency flag on a later page, after the cover and the marketing pages, so the
# first few pages are not enough; beyond this every extra page is cost without data.
MAX_PAGES = 8
# pdfium is not thread-safe, and the eval renders pages from several threads at once:
# without this lock the process aborts with a malloc error. Only the rendering is
# serialized, so the provider calls stay parallel.
_PDFIUM_LOCK = threading.Lock()


class UnreadableDocument(ValueError):
    """This file cannot become pages: unsupported, unreadable, corrupt or encrypted."""


@dataclass(frozen=True)
class Part:
    """One binary input for the model: a whole PDF or a single page image."""

    data: bytes
    mime: str

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


@dataclass(frozen=True)
class Document:
    path: Path
    data: bytes
    mime: str

    @property
    def is_pdf(self) -> bool:
        return self.mime == PDF

    @property
    def needs_conversion(self) -> bool:
        return self.path.suffix.lower() in CONVERTED_IMAGES

    def to_parts(self, pdf_native: bool, max_pages: int = MAX_PAGES, dpi: int = 150) -> list[Part]:
        """What to send to a provider: the file itself, or PNG images made from it."""
        if self.is_pdf:
            if pdf_native:
                return [Part(self.data, self.mime)]
            return self._decode(render_pdf_pages, self.data, max_pages=max_pages, dpi=dpi)
        if self.needs_conversion:
            return self._decode(convert_image, self.data, max_pages=max_pages)
        return [Part(self.data, self.mime)]

    def _decode(self, decoder, *args, **kwargs) -> list[Part]:
        """Run a decoder, reporting any failure as UnreadableDocument.

        pdfium and Pillow each raise their own exception type for a broken file;
        callers should only ever have to handle one, and it has to name the file,
        because the point of the message is to tell you which one to go and fix.
        """
        try:
            return decoder(*args, **kwargs)
        except Exception as exc:
            raise UnreadableDocument(f"{self.path.name} is not a readable {self.mime}: {exc}") from exc


def is_supported(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MIME_BY_SUFFIX


def load_document(path: str | Path) -> Document:
    path = Path(path)
    mime = MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        supported = ", ".join(SUPPORTED_SUFFIXES)
        raise UnreadableDocument(f"Unsupported file type '{path.suffix}'. Supported: {supported}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise UnreadableDocument(f"{path.name} cannot be read: {exc}") from exc
    return Document(path=path, data=data, mime=mime)


def render_pdf_pages(pdf_bytes: bytes, max_pages: int = MAX_PAGES, dpi: int = 150) -> list[Part]:
    import pypdfium2 as pdfium

    parts: list[Part] = []
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(pdf_bytes)
        try:
            for index in range(min(len(pdf), max_pages)):
                image = pdf[index].render(scale=dpi / 72).to_pil()
                parts.append(_png_part(image))
        finally:
            pdf.close()
    return parts


def convert_image(data: bytes, max_pages: int = MAX_PAGES) -> list[Part]:
    """HEIC, TIFF, BMP, GIF -> one PNG per page (frame), upright and resized."""
    from PIL import Image, ImageOps, ImageSequence

    _register_heif()
    parts: list[Part] = []
    with Image.open(io.BytesIO(data)) as image:
        for index, frame in enumerate(ImageSequence.Iterator(image)):
            if index >= max_pages:
                break
            page = ImageOps.exif_transpose(frame.copy())
            page.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))  # only shrinks, keeps proportions
            parts.append(_png_part(page))
    return parts


def _png_part(image) -> Part:
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Part(buffer.getvalue(), "image/png")


def _register_heif() -> None:
    """Teach Pillow to open HEIC/HEIF (iPhone photos). No-op if the plugin is missing."""
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        return
    register_heif_opener()

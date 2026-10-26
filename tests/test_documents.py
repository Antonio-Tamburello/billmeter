"""Every supported file format ends up as something a provider can read."""

import io
from pathlib import Path

import pytest
from PIL import Image

from billmeter.documents import MAX_SIDE_PX, UnreadableDocument, is_supported, load_document


def png_size(part):
    return Image.open(io.BytesIO(part.data)).size


def test_supported_formats():
    for name in ("a.pdf", "a.PNG", "a.jpg", "a.jpeg", "a.webp", "a.heic", "a.HEIF",
                 "a.tif", "a.tiff", "a.bmp", "a.gif"):
        assert is_supported(name), name
    assert not is_supported("a.docx")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document("bill.docx")


def test_jpeg_is_sent_as_is(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (40, 20), "white").save(path)
    parts = load_document(path).to_parts(pdf_native=False)
    assert len(parts) == 1 and parts[0].mime == "image/jpeg" and parts[0].data == path.read_bytes()


def test_multipage_tiff_becomes_one_png_per_page(tmp_path):
    path = tmp_path / "scan.tiff"
    pages = [Image.new("L", (100, 140), color) for color in (255, 200, 150)]
    pages[0].save(path, save_all=True, append_images=pages[1:])
    parts = load_document(path).to_parts(pdf_native=True)  # even PDF-native providers get images
    assert [p.mime for p in parts] == ["image/png"] * 3
    assert len(load_document(path).to_parts(pdf_native=True, max_pages=2)) == 2


def test_big_photos_are_rotated_upright_and_shrunk(tmp_path):
    path = tmp_path / "phone.bmp"
    Image.new("RGB", (4000, 3000), "white").save(path)
    (part,) = load_document(path).to_parts(pdf_native=False)
    assert max(png_size(part)) == MAX_SIDE_PX

    sideways = tmp_path / "sideways.tiff"
    exif = Image.Exif()
    exif[0x0112] = 6  # EXIF orientation: "rotate 90° to display"
    Image.new("RGB", (300, 100), "white").save(sideways, exif=exif)
    (part,) = load_document(sideways).to_parts(pdf_native=False)
    assert png_size(part) == (100, 300)


def test_heic_photos_are_converted(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    path = tmp_path / "iphone.heic"
    try:
        pillow_heif.from_pillow(Image.new("RGB", (64, 48), "white")).save(path, quality=90)
    except Exception as exc:  # some builds can read HEIC but not write it
        pytest.skip(f"cannot create a HEIC file here: {exc}")
    (part,) = load_document(path).to_parts(pdf_native=True)
    assert part.mime == "image/png" and png_size(part) == (64, 48)


def test_pdf_rendering_is_thread_safe():
    """The eval renders pages from several threads: pdfium must not be touched concurrently.

    Without the lock in documents.py this aborts the whole process (SIGABRT, malloc error),
    so a failure here shows up as a crashed test run rather than an assertion error.
    """
    from concurrent.futures import ThreadPoolExecutor

    bills = sorted((Path(__file__).resolve().parents[1] / "dataset/synthetic/bills").glob("*.pdf"))[:8]
    if len(bills) < 4:
        pytest.skip("needs the synthetic dataset")

    def render(path):
        return len(load_document(path).to_parts(pdf_native=False))

    with ThreadPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(render, bills * 3))
    assert all(c >= 1 for c in counts)


def test_a_broken_file_is_reported_as_unreadable_not_as_a_library_crash(tmp_path):
    """A folder of real bills always holds one truncated download or one mislabelled file.

    pdfium and Pillow each raise their own exception for those; callers must see one
    type, naming the file, so a single bad file cannot kill a whole run.
    """
    broken_pdf = tmp_path / "truncated.pdf"
    broken_pdf.write_bytes(b"%PDF-1.4 and then nothing that is actually a PDF")
    with pytest.raises(UnreadableDocument, match="truncated.pdf"):
        load_document(broken_pdf).to_parts(pdf_native=False)

    not_an_image = tmp_path / "renamed.heic"
    not_an_image.write_bytes(b"this is plain text with the wrong extension")
    with pytest.raises(UnreadableDocument, match="renamed.heic"):
        load_document(not_an_image).to_parts(pdf_native=True)

    # Unsupported types and files that cannot even be opened use the same type.
    assert issubclass(UnreadableDocument, ValueError)  # callers written before it still work
    with pytest.raises(UnreadableDocument, match="Unsupported"):
        load_document("bill.docx")
    with pytest.raises(UnreadableDocument, match="cannot be read"):
        load_document(tmp_path / "gone.pdf")

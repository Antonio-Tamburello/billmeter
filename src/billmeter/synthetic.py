"""Generate fake but realistic Italian utility bills, with their expected values.

Because we generate the data ourselves, the ground truth is known for free and
contains no real personal data, so the dataset can be published.

Every bill contains traps that real bills have too:
  - a mailing address different from the supply address
  - F1/F2/F3 bands next to the period total
  - numbers in tax tables (excise duty per kWh, VAT)
  - gas consumption in Smc next to electricity in kWh on dual bills, and the gas
    excise (also in Smc) in the tax table
  - on some dual bills, a gas billing period different from the electricity one
  - business holders with a Partita IVA instead of a codice fiscale
  - bills without any yearly consumption printed
Some bills are also exported as tilted, blurry JPEG "phone photos".

The generator is reproducible: the same seed produces the same files byte for
byte, in any process, so a dataset change is always a real change in a diff.

Requires the optional dependencies: uv sync --extra dataset
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from . import fiscal_code

SUPPLIERS = {
    "A": "Luce Chiara S.p.A.",
    "B": "Aurora Energia S.r.l.",
    "C": "Nordgas & Luce S.p.A.",
}
# Belfiore codes of a few big cities (public data), used for fake birthplaces.
BIRTHPLACES = ["H501", "F205", "D612", "G273", "F839", "L219", "A944", "D969"]
SCENARIOS = [
    "power_resident", "power_nonresident", "gas", "dual",
    "business_power", "power_no_annual", "power_resident", "dual",
]


def it_int(value: int) -> str:
    """2150 -> '2.150' (Italian thousands separator)."""
    return f"{value:,}".replace(",", ".")


def it_dec(value: float) -> str:
    """4.5 -> '4,5'."""
    text = f"{value:.1f}" if value != int(value) else f"{int(value)}"
    return text.replace(".", ",")


def it_date(value: date, sep: str = "/") -> str:
    return value.strftime(f"%d{sep}%m{sep}%Y")


@dataclass
class FakeBill:
    case_id: str
    layout: str
    scenario: str
    holder_name: str
    tax_code: Optional[str]
    vat_number: Optional[str]
    supply: dict[str, str]
    mailing: dict[str, str]
    pod: Optional[str]
    pdr: Optional[str]
    is_resident: Optional[bool]
    power_kw: Optional[float]
    annual_kwh: Optional[int]
    period_start: date
    period_end: date
    period_kwh: Optional[int]
    bands: Optional[tuple[int, int, int]]
    gas_smc: Optional[int]
    gas_annual_smc: Optional[int]
    gas_period: Optional[tuple[date, date]]  # only when gas has its own period
    invoice_no: str

    @property
    def bill_type(self) -> str:
        return "dual" if self.pod and self.pdr else "power" if self.pod else "gas"

    def address_line(self, a: dict[str, str]) -> str:
        return f"{a['street']} {a['number']}, {a['cap']} {a['city']} {a['prov']}"

    def expected(self) -> dict[str, Any]:
        """The graded answer. The yearly estimates are left out on purpose: they are
        computed by normalize.py from fields already graded here, so grading them too
        would count the same mistake twice and tell you nothing about the model."""
        period_kwh = self.period_kwh if self.pod else None
        gas_start, gas_end = (d.isoformat() for d in self.gas_period) if self.gas_period else (None, None)
        return {
            "supplier_name": SUPPLIERS[self.layout],
            "bill_type": self.bill_type,
            "pod": self.pod,
            "pdr": self.pdr,
            "holder_name": self.holder_name,
            "holder_tax_code": self.tax_code,
            "supply_address": self.address_line(self.supply),
            "is_resident": self.is_resident,
            "contracted_power_kw": self.power_kw,
            "annual_consumption_kwh": self.annual_kwh,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "period_consumption_kwh": period_kwh,
            "gas_period_start": gas_start,
            "gas_period_end": gas_end,
            "gas_period_consumption_smc": self.gas_smc,
            "gas_annual_consumption_smc": self.gas_annual_smc,
        }


@lru_cache(maxsize=1)
def _cities() -> tuple[str, ...]:
    """Faker's Italian city names, in a fixed order.

    Faker builds that list with a set comprehension, so its order depends on
    PYTHONHASHSEED: `fake.city()` would return different cities in different
    processes even with the same seed, and the dataset would stop being
    reproducible. Sorting the list and drawing from it with our own seeded rng
    makes the choice depend on the seed only. Every other Faker field used here
    (street names, province codes, first and last names, company names) comes
    from an ordered tuple and is already stable.
    """
    from faker.providers.address.it_IT import Provider

    return tuple(sorted(Provider.cities))


def _address(fake, rng: random.Random) -> dict[str, str]:
    return {
        "street": fake.street_name(),
        "number": str(rng.randint(1, 180)),
        "cap": f"{rng.randint(10, 98)}{rng.randint(0, 999):03d}",
        "city": rng.choice(_cities()),
        "prov": fake.state_abbr(),
    }


def make_bill(index: int, rng: random.Random, fake) -> FakeBill:
    scenario = SCENARIOS[index % len(SCENARIOS)]
    has_power = scenario != "gas"
    has_gas = scenario in ("gas", "dual")
    business = scenario == "business_power"
    layout = "C" if has_gas else rng.choice(["A", "B"])

    if business:
        holder, tax_code = fake.company(), None
        vat = f"{rng.randint(0, 99999999999):011d}"
    else:
        female = rng.random() < 0.5
        first = fake.first_name_female() if female else fake.first_name_male()
        last = fake.last_name()
        birth = date(1950, 1, 1) + timedelta(days=rng.randint(0, 365 * 50))
        holder = f"{first} {last}"
        tax_code = fiscal_code.generate(first, last, birth, female, rng.choice(BIRTHPLACES))
        vat = None

    start = date(2026, rng.choice([1, 3, 5, 7]), 1)
    end = (start.replace(month=start.month + 2) - timedelta(days=1))
    period_kwh = rng.randint(150, 900) if has_power else None
    bands = None
    if period_kwh:
        f1 = round(period_kwh * rng.uniform(0.3, 0.45))
        f2 = round(period_kwh * rng.uniform(0.2, 0.3))
        bands = (f1, f2, period_kwh - f1 - f2)

    if business or scenario == "gas":
        resident = None
    elif scenario == "power_nonresident":
        resident = False
    else:
        resident = True

    show_annual = has_power and scenario != "power_no_annual" and rng.random() < 0.85
    # Dual bills: half of them bill gas on a different period (one month later), a common real case.
    gas_period = None
    if scenario == "dual" and rng.random() < 0.5:
        gas_start = date(start.year, start.month + 1, 1)
        gas_period = (gas_start, gas_start.replace(month=gas_start.month + 2) - timedelta(days=1))
    show_gas_annual = has_gas and rng.random() < 0.8
    return FakeBill(
        case_id=f"syn-{index + 1:03d}",
        layout=layout,
        scenario=scenario,
        holder_name=holder,
        tax_code=tax_code,
        vat_number=vat,
        supply=_address(fake, rng),
        mailing=_address(fake, rng),
        pod=f"IT{rng.choice(['001', '012', '020'])}E{rng.randint(0, 99999999):08d}" if has_power else None,
        pdr=f"{rng.randint(0, 10**14 - 1):014d}" if has_gas else None,
        is_resident=resident,
        power_kw=rng.choice([3.0, 3.0, 4.5, 6.0]) if has_power else None,
        # yearly figure coherent with the period, but not identical to the 365-day estimate
        annual_kwh=(round(period_kwh * 365 / ((end - start).days + 1) * rng.uniform(0.8, 1.2))
                    if show_annual and period_kwh else None),
        period_start=start,
        period_end=end,
        period_kwh=period_kwh,
        bands=bands,
        gas_smc=rng.randint(30, 260) if has_gas else None,
        gas_annual_smc=rng.randint(400, 1400) if show_gas_annual else None,
        gas_period=gas_period,
        invoice_no=f"{rng.randint(100000, 999999)}/{start.year}",
    )


# ------------------------------------------------------------------ rendering

class _Page:
    """Tiny helper over reportlab: write lines top-down."""

    def __init__(self, canvas, y: float = 800):
        self.c, self.y = canvas, y

    def text(self, x: float, value: str, size: float = 9, bold: bool = False, dy: float = 13) -> None:
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawString(x, self.y, value)
        self.y -= dy

    def gap(self, dy: float = 8) -> None:
        self.y -= dy

    def rule(self) -> None:
        self.c.line(40, self.y + 6, 555, self.y + 6)
        self.y -= 6


def _holder_lines(b: FakeBill) -> list[str]:
    if b.tax_code:
        return [b.holder_name, f"Codice fiscale: {b.tax_code}"]
    return [b.holder_name, f"Partita IVA: {b.vat_number}"]


def _usage_label(b: FakeBill) -> str:
    if b.scenario == "business_power":
        return "Altri usi"
    if b.is_resident is True:
        return "Uso domestico residente"
    if b.is_resident is False:
        return "Uso domestico non residente"
    return "Uso domestico"


def _power_section(p: _Page, b: FakeBill, labels: dict[str, str]) -> None:
    assert b.bands and b.power_kw and b.period_kwh is not None  # only called on power and dual bills
    f1, f2, f3 = b.bands
    p.text(40, labels["title"], 11, bold=True)
    p.text(40, f"{labels['pod']}: {b.pod}")
    p.text(40, f"{labels['power']}: {it_dec(b.power_kw)} kW")
    p.text(40, f"Tipologia: {_usage_label(b)}")
    p.text(40, f"Consumi per fascia: F1 {it_int(f1)} kWh - F2 {it_int(f2)} kWh - F3 {it_int(f3)} kWh")
    p.text(40, f"{labels['total']}: {it_int(b.period_kwh)} kWh", bold=True)
    if b.annual_kwh:
        p.text(40, f"{labels['annual']}: {it_int(b.annual_kwh)} kWh")
    p.gap()


def _gas_section(p: _Page, b: FakeBill) -> None:
    assert b.gas_smc is not None  # only called on gas and dual bills
    p.text(40, "FORNITURA GAS NATURALE", 11, bold=True)
    p.text(40, f"PDR - Punto di Riconsegna: {b.pdr}")
    if b.gas_period:
        p.text(40, f"Periodo di fatturazione gas: dal {it_date(b.gas_period[0])} al {it_date(b.gas_period[1])}")
    p.text(40, f"Consumo gas nel periodo: {it_int(b.gas_smc)} Smc")
    if b.gas_annual_smc:
        p.text(40, f"Consumo annuo gas: {it_int(b.gas_annual_smc)} Smc")
    p.text(40, "Coefficiente C: 1,02")
    p.gap()


def _taxes(p: _Page, b: FakeBill) -> None:
    p.text(40, "Dettaglio imposte", 10, bold=True)
    if b.period_kwh:
        p.text(40, f"Accisa energia elettrica  0,0227 €/kWh x {it_int(b.period_kwh)} kWh   "
                   f"€ {it_dec(round(0.0227 * b.period_kwh, 1))}")
    if b.gas_smc:
        p.text(40, f"Accisa gas  0,044 €/Smc x {it_int(b.gas_smc)} Smc")
    p.text(40, "Oneri di sistema  571,890000 €/POD/anno")
    p.text(40, f"IVA {'22' if b.scenario == 'business_power' else '10'}%")


def render_pdf(b: FakeBill) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    # invariant=1 drops the creation date and the random document id reportlab writes by
    # default, so the same bill always renders to the same bytes.
    c = canvas.Canvas(buffer, pagesize=A4, invariant=1)
    p = _Page(c)
    supply_line = b.address_line(b.supply)
    mailing_line = b.address_line(b.mailing)

    if b.layout == "A":
        p.text(40, SUPPLIERS["A"], 18, bold=True, dy=22)
        p.text(40, f"Bolletta n. {b.invoice_no} - emessa il {it_date(b.period_end + timedelta(days=12))}")
        p.rule()
        p.text(40, "Intestatario", 10, bold=True)
        for line in _holder_lines(b):
            p.text(40, line)
        p.text(40, f"Indirizzo di recapito: {mailing_line}")
        p.gap()
        p.text(40, f"Periodo di fatturazione: dal {it_date(b.period_start)} al {it_date(b.period_end)}", bold=True)
        p.text(40, f"Indirizzo di fornitura: {supply_line}")
        p.gap()
        _power_section(p, b, {"title": "DATI DELLA FORNITURA", "pod": "Codice POD", "power": "Potenza impegnata",
                              "total": "Totale consumo fatturato", "annual": "Consumo annuo"})
    elif b.layout == "B":
        c.setFont("Helvetica-Bold", 16)
        c.drawRightString(555, 800, SUPPLIERS["B"])
        p.y = 770
        p.text(40, f"Spett.le {b.holder_name}", bold=True)
        p.text(40, f"{b.mailing['street']} {b.mailing['number']}")
        p.text(40, f"{b.mailing['cap']} {b.mailing['city']} ({b.mailing['prov']})")
        p.gap(14)
        p.text(40, f"Bolletta del bimestre {it_date(b.period_start, '.')} - {it_date(b.period_end, '.')}",
               11, bold=True)
        p.text(40, f"Fattura {b.invoice_no}")
        p.rule()
        p.text(40, "Cliente", 10, bold=True)
        for line in _holder_lines(b):
            p.text(40, line)
        p.text(40, f"Luogo di fornitura: {supply_line}")
        p.gap()
        _power_section(p, b, {"title": "LA TUA FORNITURA DI ENERGIA ELETTRICA", "pod": "Punto di prelievo (POD)",
                              "power": "Potenza contrattualmente impegnata", "total": "Energia consumata nel periodo",
                              "annual": "Consumo degli ultimi 12 mesi"})
    else:
        p.text(40, SUPPLIERS["C"], 18, bold=True, dy=22)
        p.text(40, f"Documento n. {b.invoice_no}")
        p.text(40, f"Periodo: {it_date(b.period_start)} - {it_date(b.period_end)}")
        p.rule()
        p.text(40, "Titolare del contratto", 10, bold=True)
        for line in _holder_lines(b):
            p.text(40, line)
        p.text(40, f"Indirizzo fornitura: {supply_line}")
        p.text(40, f"Recapito: {mailing_line}")
        p.gap()
        if b.pdr:
            _gas_section(p, b)
        if b.pod:
            _power_section(p, b, {"title": "FORNITURA ENERGIA ELETTRICA", "pod": "POD", "power": "Potenza disponibile",
                                  "total": "Consumo totale del periodo", "annual": "Consumo annuo energia elettrica"})

    p.gap()
    _taxes(p, b)
    c.setFont("Helvetica", 7)
    c.drawString(40, 40, "Documento sintetico generato per test - dati inventati - billmeter")
    c.showPage()
    c.save()
    return buffer.getvalue()


def to_phone_photo(pdf_bytes: bytes, rng: random.Random) -> bytes:
    """Render the first page as a slightly tilted, blurry JPEG, like a phone picture."""
    from PIL import Image, ImageFilter

    from .documents import render_pdf_pages

    png = render_pdf_pages(pdf_bytes, max_pages=1, dpi=110)[0].data
    image = Image.open(io.BytesIO(png)).convert("RGB")
    image = image.rotate(rng.uniform(-3, 3), expand=True, fillcolor=(235, 232, 225))
    image = image.filter(ImageFilter.GaussianBlur(0.7))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=55)
    return buffer.getvalue()


def generate_dataset(out_dir: str | Path, n: int = 24, photos: int = 6, seed: int = 42) -> Path:
    from faker import Faker

    out_dir = Path(out_dir)
    (out_dir / "bills").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    Faker.seed(seed)
    fake = Faker("it_IT")

    cases = []
    for index in range(n):
        bill = make_bill(index, rng, fake)
        pdf = render_pdf(bill)
        (out_dir / "bills" / f"{bill.case_id}.pdf").write_bytes(pdf)
        tags = [bill.bill_type, bill.scenario, f"layout_{bill.layout}", "pdf"]
        cases.append({"id": bill.case_id, "file": f"bills/{bill.case_id}.pdf", "tags": tags,
                      "expected": bill.expected()})
        if index < photos:
            photo_id = f"{bill.case_id}-photo"
            (out_dir / "bills" / f"{photo_id}.jpg").write_bytes(to_phone_photo(pdf, rng))
            cases.append({"id": photo_id, "file": f"bills/{photo_id}.jpg",
                          "tags": [t for t in tags if t != "pdf"] + ["photo"], "expected": bill.expected()})

    manifest = {
        "name": "synthetic",
        "description": f"{n} synthetic bills + {photos} phone-photo variants, seed={seed}. All data is fake.",
        "contains_real_data": False,
        "cases": cases,
    }
    path = out_dir / "expected.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

"""Units sheet: plain explanations of measurement units that recur in the corpus, with a
concrete picture. Given to the writer when the text uses the unit (after a number).
Written by hand: these are standard definitions.
"""
import re

# (name, regex matched after a number, explanation)
UNITS = [
    ("µL (microlitre)", r"[µμu]L\b", "a millionth of a litre; a small raindrop is about 20 µL"),
    ("mL (millilitre)", r"mL\b", "a thousandth of a litre; a teaspoon holds about 5 mL"),
    ("µg (microgram)", r"[µμu]g\b", "a millionth of a gram; a grain of salt weighs about 60 µg"),
    ("mg (milligram)", r"mg\b", "a thousandth of a gram; a grain of rice weighs about 20 mg"),
    ("ng (nanogram)", r"ng\b", "a billionth of a gram"),
    ("mg/L, mg/kg", r"mg\s?(?:/|per\s)(?:L|kg)\b", "milligrams in each litre of water, or in each kilogram of soil or food; 1 mg/L is about one drop in 50 litres"),
    ("ppm, ppb", r"pp[mb]\b", "parts per million / per billion: 1 ppm is one part in a million, about one drop in a 50-litre tank"),
    ("mM, µM (millimolar, micromolar)", r"[mµμ]M\b", "how much of a substance is dissolved, counted in molecules rather than weight; µM is a thousand times weaker than mM"),
    ("kDa (kilodalton)", r"kDa\b", "a unit for the weight of very large molecules such as proteins; a typical protein is 20-100 kDa"),
    ("bp, kb (base pairs)", r"(?:bp|kb)\b", "the number of DNA letters; kb is a thousand letters"),
    ("µm (micrometre)", r"[µμu]m\b", "a thousandth of a millimetre; a human hair is about 70 µm thick"),
    ("nm (nanometre)", r"nm\b", "a millionth of a millimetre; the colours we see have wavelengths of about 400-700 nm"),
    ("ha, hm² (hectare)", r"(?:ha|hm²|hm2)\b", "a square of land 100 metres on each side, about one and a half football pitches"),
    ("km²", r"km²|km2\b", "a square one kilometre on each side; about 140 football pitches"),
    ("°C", r"°\s?C\b", "degrees Celsius; water freezes at 0 °C and boils at 100 °C"),
    ("rpm", r"rpm\b", "turns per minute of a spinning machine"),
    ("× g (centrifuge force)", r"[×x]\s?g\b", "how hard a spinning machine (centrifuge) pushes on a sample, as a multiple of Earth's gravity"),
    ("eV (electronvolt)", r"eV\b", "a tiny unit of energy used for single atoms and electrons"),
    ("m/z", r"m/z", "in a mass spectrometer, a molecule's weight divided by its electric charge; the machine sorts particles by it"),
    ("kPa, MPa (pressure)", r"[kM]Pa\b", "units of pressure; air pressure at sea level is about 100 kPa"),
    ("psu, ‰ (salinity)", r"(?:psu|‰)", "how salty water is; sea water is about 35 psu, fresh water close to 0"),
    ("t/ha, kg/ha", r"(?:t|kg)\s?(?:/|per\s)ha\b", "tonnes or kilograms harvested or added per hectare (a square of land 100 metres on each side)"),
]
_RX = [(n, re.compile(r"\d\s?(?:" + rx + ")"), e) for n, rx, e in UNITS]


def units_lines(scope: str) -> str:
    return "\n".join(f"- {n} [units sheet]: {e}" for n, rx, e in _RX if rx.search(scope or ""))

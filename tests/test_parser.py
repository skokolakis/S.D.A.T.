"""Parser regression tests."""
import io

import pytest

import SDAT

FREQS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def oe_psip_file() -> str:
    lines = [
        "OE PSIP Measurement",
        "PSIP_Version,2.8.0,",
        "Current Resistor[Ohms],100",
        "***End_Of_Header***",
        ",,Chan-1,Chan-1,",
        "Loop,Frequency[Hz],Magnitude[ratio],Phase_Shift[rad],",
    ]
    lines += [f"1,{f},1.5,-0.005," for f in FREQS]
    return "\n".join(lines)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_header_lines_do_not_leak_into_data(newline):
    raw = oe_psip_file().replace("\n", newline).encode()
    df, ref = SDAT.parse_sip_file(io.BytesIO(raw))[:2]
    assert ref == 100
    assert len(df) == len(FREQS)
    assert df["Frequency[Hz]"].tolist() == FREQS
    assert df["Loop"].astype(str).tolist() == ["1"] * len(FREQS)

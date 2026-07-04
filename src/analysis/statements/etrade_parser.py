"""E-Trade / Morgan Stanley CLIENT STATEMENT and legacy holdings PDF parser.

Modern E*TRADE statements (post Morgan Stanley acquisition) use the
CLIENT STATEMENT layout:

  Page 1–5: cover, disclosures, account summary (no per-security rows)
  HOLDINGS: stocks / ETFs / options / cash by asset class
  ACTIVITY: cash-flow activity by date

Per-security equity/ETF rows look like::

  MIND MEDICINE MINDMED INC NEW (MNMD) 2,000.000 $3.660 $7,447.69 $7,320.00 $(127.69)

Ticker is in parentheses after the security name — not a leading column.
pdfplumber table extraction is unreliable on these multi-column pages, so
we extract full-document text and match position rows with regex.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedPosition:
    symbol: str
    name: str = ""
    quantity: float = 0.0
    price: float = 0.0
    cost_basis: float = 0.0
    market_value: float = 0.0
    asset_type: str = ""


@dataclass
class ParsedActivity:
    date: str
    action: str
    symbol: str = ""
    description: str = ""
    quantity: float = 0.0
    price: float = 0.0
    amount: float = 0.0
    fees: float = 0.0


@dataclass
class ParsedStatement:
    variant: str = "unknown"
    period_start: str = ""
    period_end: str = ""
    ending_value: float | None = None
    positions: list[ParsedPosition] = field(default_factory=list)
    activities: list[ParsedActivity] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Equity / ETF / fund row: NAME (TICKER) QTY $PRICE $COST $MKT ...
# Ticker is 1–5 letters, optional class suffix (.A / -A). OCC option roots
# with spaces (e.g. "MNMD 240119C00004000") are intentionally excluded.
_PAREN_TICKER_POSITION = re.compile(
    r"(?P<name>[A-Z][A-Z0-9 &.'/\-]{1,80}?)\s+"
    r"\((?P<symbol>[A-Z]{1,5}(?:[.\-][A-Z])?)\)\s+"
    r"(?P<qty>-?[\d,]+\.\d+)\s+"
    r"\$?(?P<price>[\d,]+\.\d+)\s+"
    r"\$?(?P<cost>\(?-?[\d,]+\.?\d*\)?)\s+"
    r"\$?(?P<mkt>\(?-?[\d,]+\.?\d*\)?)",
)

# Legacy E*TRADE / simple layouts: TICKER ... QTY PRICE COST MKT
_LEADING_TICKER_POSITION = re.compile(
    r"^(?P<symbol>[A-Z]{1,5}(?:[.\-][A-Z])?)\s+"
    r"(?P<name>[A-Za-z][A-Za-z0-9 &.'/\-]{0,60}?)\s+"
    r"(?P<qty>-?[\d,]+\.\d+)\s+"
    r"\$?(?P<price>[\d,]+\.\d+)\s+"
    r"\$?(?P<cost>\(?-?[\d,]+\.?\d*\)?)\s+"
    r"\$?(?P<mkt>\(?-?[\d,]+\.?\d*\)?)",
)

_ACTIVITY_ROW = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}(?:/\d{2,4})?)\s+"
    r"(?:\d{1,2}/\d{1,2}(?:/\d{2,4})?\s+)?"
    r"(?P<action>Bought|Sold|Buy|Sell|Dividend|Interest|Transfer|Deposit|Withdrawal)\b"
    r"(?P<body>.*)$",
    re.IGNORECASE,
)

_TICKER_IN_TEXT = re.compile(r"\b([A-Z]{1,5}(?:[.\-][A-Z])?)\b")

# Summary / total / header noise that must never become a position.
_SKIP_NAME_TOKENS = {
    "STOCKS",
    "TOTAL",
    "TOTAL VALUE",
    "TOTAL STOCKS",
    "PERCENTAGE",
    "PERCENTAGE OF HOLDINGS",
    "CASH",
    "CASH, BDP, AND MMFS",
    "CASH BDP AND MMFS",
    "SECURITY DESCRIPTION",
    "EXCHANGE-TRADED",
    "EXCHANGE TRADED",
    "MUTUAL FUNDS",
    "FIXED INCOME",
    "OPTIONS",
    "BONDS",
    "ALLOCATION",
    "ASSET ALLOCATION",
    "MARKET VALUE",
    "BEGINNING",
    "ENDING",
    "NET UNSETTLED",
    "PROJECTED SETTLED",
    "COMMON STOCKS",
    "PREFERRED",
    "ALTERNATIVES",
    "ANNUITIES",
}

_HOLDINGS_START = re.compile(
    r"^\s*HOLDINGS\s*$|"
    r"^\s*ACCOUNT HOLDINGS\s*$|"
    r"^\s*POSITIONS\s*$",
    re.IGNORECASE,
)
_HOLDINGS_END = re.compile(
    r"^\s*ACTIVITY\s*$|"
    r"^\s*ACCOUNT ACTIVITY\s*$|"
    r"^\s*CASH FLOW ACTIVITY\b|"
    r"^\s*TRANSACTION HISTORY\s*$|"
    r"^\s*ALLOCATION OF ASSETS\s*$",
    re.IGNORECASE,
)


def _parse_money(text: str) -> float:
    cleaned = (text or "").strip()
    if not cleaned or cleaned in ("—", "-", "–", "N/A", "NA"):
        return 0.0
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").replace("$", "").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return 0.0
    return -value if negative else value


def _parse_qty(text: str) -> float:
    cleaned = (text or "").strip().replace(",", "")
    if not cleaned:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _detect_variant(full_text: str) -> str:
    upper = full_text.upper()
    if "CLIENT STATEMENT" in upper or "MORGAN STANLEY" in upper:
        return "client_statement"
    if "ACCOUNT HOLDINGS" in upper or "E*TRADE" in upper or "ETRADE" in upper:
        return "legacy_holdings"
    return "unknown"


def _extract_period(full_text: str) -> tuple[str, str]:
    patterns = [
        r"Beginning Total Value\s*\(as of\s*(\d{1,2}/\d{1,2}/\d{2,4})\).*?"
        r"Ending Total Value\s*\(as of\s*(\d{1,2}/\d{1,2}/\d{2,4})\)",
        r"For the Period\s+([A-Za-z]+)\s+(\d{1,2})-(\d{1,2}),\s*(\d{4})",
        r"Statement Period[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:to|–|-)\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\d{1,2}/\d{1,2}/\d{2,4})\s*(?:to|through|–|-)\s*(\d{1,2}/\d{1,2}/\d{2,4})",
    ]
    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        if "For the Period" in pat:
            month = month_map.get(m.group(1).lower(), 0)
            if not month:
                continue
            year = int(m.group(4))
            start_day = int(m.group(2))
            end_day = int(m.group(3))
            return (
                f"{month}/{start_day}/{year % 100:02d}",
                f"{month}/{end_day}/{year % 100:02d}",
            )
        return m.group(1), m.group(2)
    return "", ""


def _extract_ending_value(full_text: str) -> float | None:
    patterns = [
        r"Ending Total Value\s*\(as of[^)]*\)\s*\$?([\d,]+\.?\d*)",
        r"TOTAL ENDING VALUE\s*\$?([\d,]+\.?\d*)",
        r"Ending (?:Account )?Value[:\s]+\$?([\d,]+\.?\d*)",
        r"Total Account Value[:\s]+\$?([\d,]+\.?\d*)",
    ]
    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if m:
            return _parse_money(m.group(1))
    return None


def _is_noise_name(name: str) -> bool:
    upper = re.sub(r"\s+", " ", (name or "").strip().upper())
    if not upper:
        return True
    if upper in _SKIP_NAME_TOKENS:
        return True
    if any(upper.startswith(tok) for tok in _SKIP_NAME_TOKENS):
        return True
    if "PERCENTAGE" in upper or "TOTAL" in upper:
        return True
    return False


def _slice_holdings_lines(lines: list[str]) -> list[str]:
    """Return lines inside the HOLDINGS section only (all pages).

    Summary pages mention the word "holdings" in prose; we only enter the
    section on a standalone HOLDINGS / ACCOUNT HOLDINGS / POSITIONS header,
    and leave on ACTIVITY / ALLOCATION OF ASSETS.
    """
    in_holdings = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_holdings:
            if _HOLDINGS_START.match(stripped):
                in_holdings = True
            continue
        if _HOLDINGS_END.match(stripped):
            break
        out.append(stripped)
    return out


def _slice_activity_lines(lines: list[str]) -> list[str]:
    in_activity = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if not in_activity:
            if re.match(r"^(ACTIVITY|ACCOUNT ACTIVITY|CASH FLOW ACTIVITY)\b", upper):
                in_activity = True
            continue
        if re.match(r"^(GAIN/\(LOSS\)|INCOME AND DISTRIBUTION|EXPANDED DISCLOSURES)\b", upper):
            break
        out.append(stripped)
    return out


def _position_from_match(m: re.Match[str], *, asset_type: str = "") -> ParsedPosition | None:
    name = (m.group("name") or "").strip()
    symbol = (m.group("symbol") or "").strip().upper()
    if _is_noise_name(name) or not symbol:
        return None
    qty = _parse_qty(m.group("qty"))
    price = _parse_money(m.group("price"))
    cost = _parse_money(m.group("cost"))
    mkt = _parse_money(m.group("mkt"))
    if qty == 0 and mkt == 0:
        return None
    # Guard against summary rows that happen to match loosely.
    if abs(qty) > 0 and abs(qty) < 0.0001:
        return None
    return ParsedPosition(
        symbol=symbol,
        name=name,
        quantity=qty,
        price=price,
        cost_basis=cost if cost else qty * price,
        market_value=mkt if mkt else qty * price,
        asset_type=asset_type,
    )


def _infer_asset_type(context_upper: str) -> str:
    if "EXCHANGE-TRADED" in context_upper or "EXCHANGE TRADED" in context_upper or " ETF" in context_upper:
        return "etf"
    if "MUTUAL FUND" in context_upper:
        return "mutual_fund"
    if "OPTION" in context_upper:
        return "option"
    if "BOND" in context_upper or "FIXED INCOME" in context_upper:
        return "bond"
    if "STOCK" in context_upper or "COMMON STOCK" in context_upper or "EQUIT" in context_upper:
        return "stock"
    return ""


def _parse_positions_from_lines(lines: list[str]) -> list[ParsedPosition]:
    holdings_lines = _slice_holdings_lines(lines)
    search_lines = holdings_lines if holdings_lines else lines

    positions: list[ParsedPosition] = []
    seen: set[str] = set()
    section_ctx = ""

    for line in search_lines:
        upper = line.upper()
        if re.match(
            r"^(STOCKS|COMMON STOCKS|EXCHANGE-TRADED|EXCHANGE TRADED|"
            r"MUTUAL FUNDS|OPTIONS|BONDS|FIXED INCOME|PREFERRED|"
            r"CASH,?\s*BDP|ALTERNATIVES)\b",
            upper,
        ):
            section_ctx = upper
            continue

        asset_type = _infer_asset_type(section_ctx)
        # Prefer ticker-in-parens (Morgan Stanley / modern E*TRADE).
        matched = False
        for m in _PAREN_TICKER_POSITION.finditer(line):
            pos = _position_from_match(m, asset_type=asset_type or "stock")
            if pos is None:
                continue
            key = pos.symbol
            if key in seen:
                # Keep the first detailed row; later totals may re-mention symbols.
                continue
            seen.add(key)
            positions.append(pos)
            matched = True
        if matched:
            continue

        # Legacy leading-ticker rows (only inside a holdings slice).
        if holdings_lines:
            m = _LEADING_TICKER_POSITION.match(line)
            if m:
                pos = _position_from_match(m, asset_type=asset_type or "stock")
                if pos is not None and pos.symbol not in seen:
                    seen.add(pos.symbol)
                    positions.append(pos)

    return positions


def _parse_activities_from_lines(lines: list[str]) -> list[ParsedActivity]:
    activities: list[ParsedActivity] = []
    for line in _slice_activity_lines(lines):
        m = _ACTIVITY_ROW.match(line)
        if not m:
            continue
        date_str = m.group("date")
        action = m.group("action").lower()
        body = (m.group("body") or "").strip()

        # Trailing qty / price / amount tokens.
        nums = re.findall(r"\(?\$?-?[\d,]+\.\d+\)?", body)
        qty = _parse_qty(nums[-3]) if len(nums) >= 3 else 0.0
        price = _parse_money(nums[-2]) if len(nums) >= 2 else 0.0
        amount = _parse_money(nums[-1]) if nums else 0.0

        # Prefer a bare ticker token; fall back to first all-caps word.
        sym = ""
        for token in _TICKER_IN_TEXT.findall(body.upper()):
            if token in {
                "ACTED",
                "AGENT",
                "CALL",
                "PUT",
                "OPENING",
                "CLOSING",
                "INDEX",
                "OPTION",
                "TRADE",
                "UNSOLICITED",
                "BOUGHT",
                "SOLD",
            }:
                continue
            if len(token) <= 5:
                sym = token
                break

        activities.append(
            ParsedActivity(
                date=date_str,
                action=action,
                symbol=sym,
                description=body[:120],
                quantity=qty,
                price=price,
                amount=amount,
            )
        )
    return activities


def _extract_pdf_lines(path: Path) -> tuple[str, list[str], list[str]]:
    """Return (full_text, all_lines, warnings)."""
    try:
        import pdfplumber
    except ImportError:
        return "", [], ["pdfplumber not installed — run pip install pdfplumber"]

    full_text_parts: list[str] = []
    all_lines: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return "", [], ["PDF has no pages"]
            for page in pdf.pages:
                # layout=True preserves column alignment better on MS statements.
                try:
                    text = page.extract_text(layout=True) or ""
                except TypeError:
                    text = page.extract_text() or ""
                if not text.strip():
                    text = page.extract_text() or ""
                full_text_parts.append(text)
                all_lines.extend(text.splitlines())
    except Exception as ex:
        return "", [], [f"PDF read error: {ex}"]

    return "\n".join(full_text_parts), all_lines, []


def parse_etrade_statement(pdf_path: str | Path) -> ParsedStatement:
    """Parse an E-Trade / Morgan Stanley monthly statement PDF."""
    path = Path(pdf_path)
    if not path.is_file():
        return ParsedStatement(warnings=[f"File not found: {path}"])

    full_text, all_lines, warnings = _extract_pdf_lines(path)
    if warnings and not full_text:
        return ParsedStatement(warnings=warnings)

    variant = _detect_variant(full_text)
    period_start, period_end = _extract_period(full_text)
    ending_value = _extract_ending_value(full_text)
    positions = _parse_positions_from_lines(all_lines)
    activities = _parse_activities_from_lines(all_lines)

    if variant == "unknown":
        warnings.append("Could not detect E-Trade statement variant — review parsed rows carefully.")
    if not positions:
        warnings.append("No positions extracted — you may need to enter holdings manually.")
    else:
        holdings_lines = _slice_holdings_lines(all_lines)
        if not holdings_lines:
            warnings.append(
                "Holdings section header not found; positions were matched from full document text."
            )

    return ParsedStatement(
        variant=variant,
        period_start=period_start,
        period_end=period_end,
        ending_value=ending_value,
        positions=positions,
        activities=activities,
        warnings=warnings,
    )

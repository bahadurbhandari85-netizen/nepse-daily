#!/usr/bin/env python3
"""
fetch_nepse_daily.py
----------------------
Builds ONE combined CSV per trading day, matching the reference layout:

    [stock price table -- all listed securities, 24 columns]
    [blank row]
    [blank row]
    Sub Index,Open,High,Low,Close,Point,% Change,Turnover
    Banking SubIndex,...
    ... (all 13 sub-indices) ...
    NEPSE Index,...          <- appended last, same 8-column layout

Sources:
    Stock table   <- sharesansar.com/today-share-price
    Sub-indices   <- sharesansar.com/market, table 4 ("Sub Indices")
    NEPSE Index   <- sharesansar.com/market, table 1 ("Indices"),
                      just the "NEPSE Index" row, reshaped to the same
                      8-column layout as the sub-index rows

Output file is named just "<date>.csv" (e.g. 2026-07-21.csv) -- matching
the reference file, no site-name prefix.

Checks the SITE's own reported date before deciding whether to skip --
never assumes calendar-today matches the date the site is showing
(weekends/holidays lag), same discipline as the earlier scripts this
one replaces.

Usage:
    python fetch_nepse_daily.py                  # default: ./data_input
    python fetch_nepse_daily.py --outdir data_input
    python fetch_nepse_daily.py --force
"""

import argparse
import csv
import datetime
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

STOCK_URL = "https://www.sharesansar.com/today-share-price"
MARKET_URL = "https://www.sharesansar.com/market"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}
TIMEOUT_SECS = 20

NEPSE_TABLE_INDEX = 0      # "table 1" on /market, 1-indexed
SUBINDEX_TABLE_INDEX = 3   # "table 4" on /market, 1-indexed


def locate_subindex_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Finds the real 'Sub Index' header by CONTENT, not by a guessed row
    count. Handles two possible shapes:
      (a) pandas already parsed the header correctly (column 0 is
          literally named 'Sub Index') -- use the table as-is.
      (b) a caption/'As of' row sits above the real header inside the
          same <table>, so the real header text ends up as a data row
          within the first few rows -- find it by content and promote it.
    This replaces an earlier hardcoded "skip 2 rows" assumption that
    was built without being able to see the page's raw HTML from this
    sandbox, and turned out to be wrong: it silently deleted the first
    two real sub-indices (Banking, Development Bank) instead of two
    junk rows, since the real page's caption apparently isn't shaped
    the way that assumption guessed.
    """
    if str(raw.columns[0]).strip() == "Sub Index":
        return clean_numeric(raw.copy())

    for i in range(min(5, len(raw))):
        first_cell = str(raw.iloc[i, 0]).strip()
        if first_cell == "Sub Index":
            new_header = raw.iloc[i].tolist()
            data = raw.iloc[i + 1:].reset_index(drop=True)
            data.columns = new_header
            return clean_numeric(data.copy())

    raise ValueError(
        "Could not locate the 'Sub Index' header row in table 4 by "
        "content search -- site layout may have changed more than "
        "expected. First few rows for inspection:\n" + raw.head(5).to_string()
    )

SUBINDEX_HEADER = ["Sub Index", "Open", "High", "Low", "Close",
                    "Point", "% Change", "Turnover"]


def log(tag, msg):
    print(f"[{tag}] {msg}")


def fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECS)
    resp.raise_for_status()
    return resp.text


def extract_as_of_date(html: str, section_hint: str = None) -> str:
    """
    Strips HTML tags first, then searches the resulting PLAIN TEXT for
    the 'As of' date -- not the raw HTML string.

    The earlier version searched raw HTML directly with a plain regex
    (`As of\\s*:?\\s*(\\d{4}-\\d{2}-\\d{2})`), which only tolerates
    whitespace between "As of" and the date. If the real page wraps
    the date in its own tag -- e.g. `As of : <span>2026-07-21</span>`
    -- that gap is HTML markup, not whitespace, so the match silently
    failed and fell through to today's date every time. This couldn't
    be caught from the sandbox that originally wrote this function,
    since it only had a markdown-converted view of the page, which
    already had tags stripped -- the raw-HTML case was never actually
    exercised.

    Stripping tags first (via BeautifulSoup) sidesteps the problem
    entirely: it doesn't matter what tag structure wraps the date,
    because by the time the regex runs, there are no tags left to
    break the match on.
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    if section_hint:
        idx = text.find(section_hint)
        if idx != -1:
            window = text[idx:idx + 500]
            m = re.search(r"As of\s*:?\s*(\d{4}-\d{2}-\d{2})", window)
            if m:
                return m.group(1)

    m = re.search(r"As of\s*:?\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)

    log("WARN", "No 'As of' date found in page text; falling back to "
                "today's date. Verify this is correct.")
    return datetime.date.today().isoformat()


def is_numeric_col(s: pd.Series) -> bool:
    """Robust across pandas versions -- pandas >=3.0 uses a dedicated
    native `str` dtype for text instead of the legacy `object` dtype,
    so a strict `dtype == object` check silently misses it there."""
    return pd.api.types.is_numeric_dtype(s)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not is_numeric_col(df[col]):
            cleaned = df[col].astype(str).str.replace(",", "", regex=False)
            numeric = pd.to_numeric(cleaned, errors="coerce")
            if numeric.notna().mean() > 0.9:
                df[col] = numeric
    return df


def parse_stock_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise ValueError("No tables found on the stock-price page -- "
                          "site layout may have changed.")
    table = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    return clean_numeric(table.copy())


def parse_market_tables(html: str):
    """Returns (nepse_row, subindex_rows) as lists of lists, both already
    matching SUBINDEX_HEADER's column order and cleaned of comma
    formatting -- ready to write directly."""
    tables = pd.read_html(io.StringIO(html))
    if len(tables) <= max(NEPSE_TABLE_INDEX, SUBINDEX_TABLE_INDEX):
        raise ValueError(
            f"Expected at least {SUBINDEX_TABLE_INDEX + 1} tables on the "
            f"market page, found {len(tables)}. Site layout may have "
            f"changed -- inspect the raw tables."
        )

    # --- NEPSE Index: just its one row, reshaped to the 8-col layout ---
    nepse_table = clean_numeric(tables[NEPSE_TABLE_INDEX].copy())
    name_col = nepse_table.columns[0]
    nepse_matches = nepse_table[nepse_table[name_col] == "NEPSE Index"]
    if nepse_matches.empty:
        raise ValueError("Could not find a 'NEPSE Index' row in table 1 "
                          "on the market page.")
    r = nepse_matches.iloc[0]
    # Indices table columns: Index, Open, High, Low, Close, Point Change,
    # % Change, Turnover -- map positionally to the sub-index layout.
    nepse_row = ["NEPSE Index", r.iloc[1], r.iloc[2], r.iloc[3], r.iloc[4],
                 r.iloc[5], r.iloc[6], r.iloc[7]]

    # --- Sub Indices: locate the real header by content, not row count ---
    sub_df = locate_subindex_data(tables[SUBINDEX_TABLE_INDEX])
    subindex_rows = sub_df.values.tolist()

    return nepse_row, subindex_rows


def build_output_path(outdir: Path, date_str: str) -> Path:
    return outdir / f"{date_str}.csv"


def write_combined_csv(path: Path, stock_df: pd.DataFrame,
                        subindex_rows, nepse_row):
    width = len(stock_df.columns)

    def pad(row):
        row = list(row)
        return row + [""] * (width - len(row))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(stock_df.columns))
        for row in stock_df.values.tolist():
            writer.writerow(row)
        writer.writerow([""] * width)
        writer.writerow([""] * width)
        writer.writerow(pad(SUBINDEX_HEADER))
        for row in subindex_rows:
            writer.writerow(pad(row))
        writer.writerow(pad(nepse_row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data_input")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("INFO", f"Checking current data date at {STOCK_URL} ...")
    try:
        stock_html = fetch(STOCK_URL)
    except requests.RequestException as e:
        log("ERROR", f"Network request failed (stock page): {e}")
        sys.exit(1)
    as_of_date = extract_as_of_date(stock_html)
    log("INFO", f"Site reports data as of: {as_of_date}")

    out_path = build_output_path(outdir, as_of_date)
    if not args.force and out_path.exists():
        log("SKIP", f"{out_path} already exists. Use --force to override.")
        return

    try:
        market_html = fetch(MARKET_URL)
    except requests.RequestException as e:
        log("ERROR", f"Network request failed (market page): {e}")
        sys.exit(1)

    market_date = extract_as_of_date(market_html, section_hint="Indices")
    if market_date != as_of_date:
        log("WARN", f"Stock page date ({as_of_date}) and market page date "
                     f"({market_date}) don't match -- proceeding with "
                     f"{as_of_date} since that's the file name, but verify "
                     f"this wasn't a transient mismatch between the two "
                     f"requests.")

    try:
        stock_df = parse_stock_table(stock_html)
        nepse_row, subindex_rows = parse_market_tables(market_html)
    except ValueError as e:
        log("ERROR", str(e))
        sys.exit(1)

    write_combined_csv(out_path, stock_df, subindex_rows, nepse_row)
    log("OK", f"Saved {stock_df.shape[0]} stock rows + "
              f"{len(subindex_rows)} sub-indices + NEPSE Index to {out_path}")


if __name__ == "__main__":
    main()

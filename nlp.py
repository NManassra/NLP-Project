"""
updated_collector.py
---------------------
Batch processor that reads a list of institutions from Data.txt,
fetches their contextual information using DataCollector,
and saves the output into a CSV file with structured columns.

Run:
    pip install aiohttp beautifulsoup4
    python updated_collector.py
"""
import asyncio
import csv
import re
from pathlib import Path

from data_collector import DataCollector

INPUT_FILE = "Data.txt"
OUTPUT_FILE = "institutions_output.csv"
COLUMNS = ["Name", "Type", "Country", "YearFounded", "Website", "ShortDescription", "SourceText"]
BATCH_SIZE = 25

def read_institutions(file_path: str) -> list[str]:
    with open(file_path, encoding="utf-8") as f:
        return list(dict.fromkeys([line.strip() for line in f if line.strip()]))  # remove duplicates

def classify_type(index: int) -> str:
    if index < 32826:
        return "Bank"
    elif index < 32826 + 9363:
        return "University"
    else:
        return "Hospital"

def extract_year(text: str) -> str:
    match = re.search(r"\b(18|19|20)\d{2}\b", text)
    return match.group(0) if match else ""

def extract_short_description(text: str) -> str:
    lines = text.strip().split("\n")
    desc = lines[0] if lines else ""
    return desc[:300] + "…" if len(desc) > 300 else desc

def extract_country(text: str) -> str:
    match = re.search(r"\b(in|from|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    return match.group(2) if match else ""

def extract_homepage(text: str) -> str:
    match = re.search(r"https?://[\w./-]+", text)
    return match.group(0) if match else ""

async def fetch_single(collector, inst: str, index: int) -> dict:
    try:
        text = await collector._collect(inst)
        return {
            "Name": inst,
            "Type": classify_type(index),
            "Country": extract_country(text),
            "YearFounded": extract_year(text),
            "Website": extract_homepage(text),
            "ShortDescription": extract_short_description(text),
            "SourceText": text[:1000] + "…" if len(text) > 1000 else text,
        }
    except Exception as e:
        return {
            "Name": inst,
            "Type": classify_type(index),
            "Country": "",
            "YearFounded": "",
            "Website": "",
            "ShortDescription": "",
            "SourceText": f"ERROR: {e}"
        }

async def fetch_all(institutions: list[str], sources: list[str]) -> list[dict]:
    collector = DataCollector(sources=sources)
    results = []

    for i in range(0, len(institutions), BATCH_SIZE):
        batch = institutions[i:i + BATCH_SIZE]
        print(f"Processing batch {i} to {i+len(batch)-1}")
        coros = [fetch_single(collector, inst, i + j) for j, inst in enumerate(batch)]
        batch_results = await asyncio.gather(*coros)
        results.extend(batch_results)

    return results

def write_csv(rows: list[dict], out_path: str):
    with open(out_path, mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def main():
    institutions = read_institutions(INPUT_FILE)
    sources = ["wikipedia", "homepage", "news"]  # You can customize this list
    data = asyncio.run(fetch_all(institutions, sources))
    write_csv(data, OUTPUT_FILE)
    print(f"Done. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()

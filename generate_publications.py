"""
Parse publications.qmd and generate one .qmd file per paper in publications/.

Each input line looks something like:
  - Jones, A. L., & Kramer, R. (2021). Facial first impressions form two...
    *Cognitive Psychology*, 126, 101387. [doi:10.xxx](https://doi.org/10.xxx)
    [[📄]](assets/papers/2021/Jones%20&%20Kramer%202021.pdf)

We extract the fields, standardise your name, and write a .qmd stub.
"""

import re
from pathlib import Path
from urllib.parse import unquote

# Paths
SRC = Path("publications.qmd")
DST = Path("publications")
DST.mkdir(exist_ok=True)

# Read the source file
text = SRC.read_text()

# We want the bulleted paper entries only - skip the header prose
# Each entry starts with "- " at the beginning of a line
# Match each bulleted entry (may span multiple lines)
lines = text.split("\n")

# First, figure out the year boundaries - they're ## headings
current_year = None
entries = []
buffer = []

def flush_buffer():
    global buffer
    if buffer and current_year:
        entries.append((current_year, " ".join(buffer).strip()))
    buffer = []

for line in lines:
    # Year heading, e.g. "## 2025"
    year_match = re.match(r"^##\s+(\d{4})\s*$", line)
    if year_match:
        flush_buffer()
        current_year = int(year_match.group(1))
        continue

    # New paper bullet
    if line.startswith("- "):
        flush_buffer()
        buffer = [line[2:].strip()]
        continue

    # Continuation line (part of the previous paper)
    if buffer and line.strip():
        buffer.append(line.strip())

flush_buffer()

print(f"Found {len(entries)} paper entries.")

# Now parse each entry to extract fields
def standardise_author_list(s):
    """Turn 'Jones, A.' or '**Jones, A.**' into 'Jones, A. L.'"""
    # Remove markdown bolding
    s = s.replace("**", "")
    # Replace 'Jones, A.' (not followed by L.) with 'Jones, A. L.'
    # Be careful: we don't want to match "B. Jones, A." etc - only the bare form
    # Simple approach: replace "Jones, A." with "Jones, A. L." where the next char is not L
    s = re.sub(r"\bJones,\s*A\.(?!\s*L\.)", "Jones, A. L.", s)
    return s

def slugify(text, max_words=4):
    """Turn a title into a lowercase hyphenated slug, max N words."""
    text = re.sub(r"[^\w\s-]", "", text.lower())
    words = text.split()
    # Drop very short filler words
    skip = {"a", "an", "the", "of", "in", "on", "and", "or", "for", "to", "is", "with"}
    words = [w for w in words if w not in skip]
    return "-".join(words[:max_words])

def parse_entry(text):
    """
    Parse one entry like:
    'Jones, A. L., & Kramer, R. (2021). Facial first impressions... *Cognitive Psychology*, 126, 101387. [doi:10.xxx](https://doi.org/10.xxx) [[📄]](assets/papers/...pdf)'
    """
    # Extract PDF path
    pdf_match = re.search(r"\[\[📄\]\]\(([^)]+)\)", text)
    pdf_path = unquote(pdf_match.group(1)) if pdf_match else ""

    # Extract DOI
    doi_match = re.search(r"\[doi:([^\]]+)\]", text)
    doi = doi_match.group(1) if doi_match else ""

    # Strip the DOI and PDF markdown off the end for easier parsing
    body = text
    if doi_match:
        body = body[:doi_match.start()].strip()
    if "[[📄]]" in body:
        body = body.split("[[📄]]")[0].strip()
    # Remove trailing dot, bracket, etc.
    body = body.rstrip(" .")

    # Match authors up to "(YEAR)."
    m = re.match(r"^(.+?)\s*\((\d{4})\)\.\s*(.+)$", body, re.DOTALL)
    if not m:
        return None
    authors = m.group(1).strip()
    year = int(m.group(2))
    after_year = m.group(3).strip()

    # Now split title from journal. The title ends before the first *italic* section (the journal)
    # e.g. "Title of paper. *Journal Name*, volume, pages."
    journal_match = re.search(r"\*([^*]+)\*", after_year)
    if journal_match:
        title = after_year[:journal_match.start()].strip().rstrip(".")
        journal = journal_match.group(1).strip()
        rest = after_year[journal_match.end():].strip()
        # rest is typically ", 126, 101387." - split on commas
        rest = rest.lstrip(", ").rstrip(" .")
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        volume = parts[0] if len(parts) >= 1 else ""
        pages = parts[1] if len(parts) >= 2 else ""
    else:
        title = after_year.rstrip(".")
        journal = ""
        volume = ""
        pages = ""

    return {
        "authors": standardise_author_list(authors),
        "year": year,
        "title": title,
        "journal": journal,
        "volume": volume,
        "pages": pages,
        "doi": doi,
        "pdf": pdf_path,
    }

# Parse all entries and write files
generated = 0
skipped = 0
seen_slugs = {}

for year, entry_text in entries:
    parsed = parse_entry(entry_text)
    if not parsed:
        print(f"  SKIPPED (parse failed): {entry_text[:80]}...")
        skipped += 1
        continue

    # Build filename: YYYY-firstauthor-titleslug.qmd
    first_author = parsed["authors"].split(",")[0].strip().lower()
    first_author = re.sub(r"[^a-z]", "", first_author)
    title_slug = slugify(parsed["title"])
    base_slug = f"{parsed['year']}-{first_author}-{title_slug}"

    # Disambiguate if collision
    slug = base_slug
    n = 2
    while slug in seen_slugs:
        slug = f"{base_slug}-{n}"
        n += 1
    seen_slugs[slug] = True

    # Build the file content
    lines = ["---"]
    # Escape double quotes in title
    esc_title = parsed["title"].replace('"', '\\"')
    lines.append(f'title: "{esc_title}"')
    lines.append(f'authors: "{parsed["authors"]}"')
    lines.append(f'year: {parsed["year"]}')
    lines.append(f'date: {parsed["year"]}-01-01')
    if parsed["journal"]:
        esc_journal = parsed["journal"].replace('"', '\\"')
        lines.append(f'journal: "{esc_journal}"')
    if parsed["volume"]:
        lines.append(f'volume: "{parsed["volume"]}"')
    if parsed["pages"]:
        lines.append(f'pages: "{parsed["pages"]}"')
    if parsed["doi"]:
        lines.append(f'doi: "{parsed["doi"]}"')
    if parsed["pdf"]:
        lines.append(f'pdf: "{parsed["pdf"]}"')
    lines.append("categories: []")  # empty for now; we fill in Step 3
    lines.append("---")

    content = "\n".join(lines) + "\n"
    out_path = DST / f"{slug}.qmd"
    out_path.write_text(content)
    generated += 1

print(f"\nGenerated {generated} files in {DST}/")
print(f"Skipped {skipped}")
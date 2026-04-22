"""
Update the `categories: []` line in each publications/*.qmd file
with the first-pass category assignments.

Safe to re-run: just rewrites categories to the values in ASSIGNMENTS.
"""

import re
from pathlib import Path

PUB_DIR = Path("publications")

ASSIGNMENTS = {
    "2012-jones-signals-personality":                ["first-impressions", "morphometrics"],
    "2012-kramer-lack-sexual-dimorphism":            ["morphometrics"],
    "2013-kramer-sequential-effects":                ["attractiveness", "methodology"],
    "2013-scott-facial-cues":                        ["first-impressions"],
    "2014-jones-miscalibrations":                    ["cosmetics", "attractiveness"],
    "2015-jones-cosmetics-alter":                    ["cosmetics", "morphometrics"],
    "2015-jones-facial-cosmetics-have":              ["cosmetics", "attractiveness"],
    "2015-kramer-do-peoples-first":                  ["face-perception"],
    "2015-scott-facial-dimorphism":                  ["morphometrics"],
    "2016-diersch-timing":                           ["other"],
    "2016-jones-coloration":                         ["first-impressions"],
    "2016-jones-facial-cosmetics-attractiveness":    ["cosmetics", "attractiveness"],
    "2016-mileva-sex-differences":                   ["cosmetics", "first-impressions"],
    "2016-russell-facial-contrast-cue":              ["first-impressions"],
    "2017-jones-makeup-changes":                     ["cosmetics", "morphometrics"],
    "2017-jones-positive-facial":                    ["first-impressions"],
    "2017-mason-influence-opposite":                 ["other"],
    "2017-russell-facial-contrast-declines":         ["morphometrics"],
    "2017-sumanapala-have-i-grooved":                ["other"],
    "2018-jones-influence-shape":                    ["first-impressions", "morphometrics"],
    "2019-blagrove-testing-empathy":                 ["other"],
    "2019-callow-action-dual-tasks":                 ["other"],
    "2019-jones-biological-bases":                   ["attractiveness", "morphometrics"],
    "2019-jones-personality-faces":                  ["first-impressions"],
    "2019-russell-role-contrast-gain":               ["face-perception"],
    "2020-crawford-digital-cognitive":               ["other"],
    "2020-kramer-face-familiarity-image":            ["face-perception"],
    "2020-kramer-sequential-effects-facial":         ["attractiveness", "methodology", "bayesian"],
    "2021-childs-do-individual":                     ["face-perception"],
    "2021-holzleitner-do-3d-face":                   ["morphometrics"],
    "2021-jones-facial-first-impressions":           ["first-impressions", "methodology", "bayesian"],
    "2021-jones-facial-metrics-generated":           ["morphometrics", "methodology"],
    "2021-jones-height-shows-no":                    ["methodology", "bayesian"],
    "2021-jones-no-credible-evidence":               ["methodology", "bayesian"],
    "2021-kramer-individual-differences":            ["face-perception"],
    "2021-kramer-wanting-having":                    ["other"],
    "2021-moore-multidisciplinary":                  ["other"],
    "2022-batres-makeup-works":                      ["cosmetics", "attractiveness"],
    "2022-childs-perceptions-individuals":           ["other"],
    "2022-jaeger-which-facial":                      ["first-impressions"],
    "2022-kramer-incomplete-faces":                  ["face-perception"],
    "2022-ong-micro-longitudinal":                   ["other"],
    "2023-bobak-data-driven":                        ["methodology"],
    "2023-embling-associations-between":             ["other"],
    "2023-hadden-pre-frontal":                       ["other"],
    "2023-kramer-relationship-between-facial":       ["first-impressions", "attractiveness"],
    "2023-kramer-wisdom-inner-crowd":                ["face-perception"],
    "2023-satchell-beyond-reliability":              ["first-impressions", "methodology"],
    "2023-turner-neurocognitive":                    ["other"],
    "2024-jaeger-testing-perceivers":                ["first-impressions"],
    "2024-jones-decoding-language":                  ["first-impressions", "methodology"],
    "2024-kramer-no-influence-face":                 ["attractiveness"],
    "2024-kramer-psychometrics-rating":              ["attractiveness", "methodology"],
    "2025-james-predictors-pelvic":                  ["other"],
    "2025-jones-updating-evidence":                  ["morphometrics", "bayesian"],
    "2025-kannan-do-young":                          ["first-impressions"],
    "2025-kramer-ai-generated":                      ["face-perception"],
    "2025-kramer-face-familiarity-similarity":       ["face-perception"],
    "2025-rayner-patterns-recreational":             ["other"],
    "2025-satchell-do-we-look":                      ["first-impressions"],
    "2025-tree-exploring-insight":                   ["face-perception"],
    "2025-tree-how-prevalent":                       ["face-perception"],
    "2025-tree-upright-inverted":                    ["face-perception"],
    "2026-jaeger-who-likes":                         ["first-impressions"],
}

def format_categories(cats):
    """Turn ['a', 'b'] into '[a, b]' for YAML."""
    return "[" + ", ".join(cats) + "]"

def substitute_categories(text, new_line):
    """Replace the line starting with 'categories:' in YAML front matter."""
    return re.subn(r"^categories:.*$", new_line, text, count=1, flags=re.MULTILINE)

def main():
    all_files = sorted(PUB_DIR.glob("*.qmd"))
    
    updated = 0
    no_line = []
    missing = []
    ambiguous = []
    matched_files = set()
    
    for prefix, cats in ASSIGNMENTS.items():
        matches = [f for f in all_files if f.stem.startswith(prefix)]
        if len(matches) == 0:
            missing.append(prefix)
            continue
        if len(matches) > 1:
            ambiguous.append((prefix, [f.name for f in matches]))
            continue
        
        path = matches[0]
        matched_files.add(path.name)
        text = path.read_text()
        new_line = f"categories: {format_categories(cats)}"
        new_text, n = substitute_categories(text, new_line)
        if n == 0:
            no_line.append(path.name)
            continue
        path.write_text(new_text)
        updated += 1
    
    uncovered = [f.name for f in all_files if f.name not in matched_files]
    
    print(f"\nUpdated {updated} files\n")
    if missing:
        print(f"Assignment prefixes matching NO file ({len(missing)}):")
        for p in missing: print(f"  {p}")
        print()
    if ambiguous:
        print(f"Prefixes matching MULTIPLE files ({len(ambiguous)}):")
        for p, fs in ambiguous:
            print(f"  {p} -> {fs}")
        print()
    if no_line:
        print(f"Files with no 'categories:' line ({len(no_line)}):")
        for n in no_line: print(f"  {n}")
        print()
    if uncovered:
        print(f"Files with no assignment ({len(uncovered)}):")
        for f in uncovered: print(f"  {f}")
        print()

if __name__ == "__main__":
    main()
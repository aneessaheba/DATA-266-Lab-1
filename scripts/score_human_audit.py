"""
Score the 30 sample human audit and compute inter rater agreement.

The brief asks for a blinded audit of 30 fixed samples on style, content and
artifacts, rated by two people, with inter rater agreement reported. This
script does not produce ratings. It reads the ratings two people entered and
computes the agreement statistics from them.

Run from the repository root:

    python scripts/score_human_audit.py

It reads both members' human_audit_30_samples.csv, reports per criterion means
and agreement, and writes the summary back into each member's metrics folder.

Agreement is reported three ways because they answer different questions:

  exact agreement   the fraction of samples where both raters gave the same
                    score. Easy to read but harsh on a 1 to 5 scale, where
                    being one point apart is near agreement.
  within one        the fraction within one point. The usual measure for an
                    ordinal scale like this.
  quadratic kappa   Cohen's kappa with quadratic weights, which corrects for
                    the agreement expected by chance and penalises a two point
                    disagreement four times as much as a one point one. This
                    is the number to quote.
"""

import csv
import json
import sys
from pathlib import Path

CRITERIA = ["style", "content", "artifacts"]
MEMBERS = ["anees_saheba", "yashashree_shinde"]
SCALE = [1, 2, 3, 4, 5]


def quadratic_kappa(a, b):
    """Cohen's kappa with quadratic weights, for ordinal ratings."""
    n = len(a)
    if n == 0:
        return None
    k = len(SCALE)
    index = {s: i for i, s in enumerate(SCALE)}
    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight[i][j] * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    if den == 0:
        return 1.0
    return 1 - num / den


def read_ratings(path):
    """Return {criterion: (rater1 scores, rater2 scores)} for completed rows."""
    rows = list(csv.DictReader(path.open()))
    out = {}
    for c in CRITERIA:
        pairs = []
        for r in rows:
            one, two = r.get(f"rater1_{c}", ""), r.get(f"rater2_{c}", "")
            if one.strip() and two.strip():
                try:
                    pairs.append((int(one), int(two)))
                except ValueError:
                    print(f"  skipping a non numeric rating in {path.name}, {c}")
        out[c] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return out, len(rows)


def main():
    any_done = False
    for member in MEMBERS:
        path = Path(f"task3_gan/{member}/outputs/metrics/human_audit_30_samples.csv")
        if not path.is_file():
            print(f"{member}: no audit file at {path}")
            continue

        ratings, total_rows = read_ratings(path)
        done = len(ratings["style"][0])
        print(f"\n{member}: {done} of {total_rows} samples rated by both raters")
        if done == 0:
            print("  nothing to score yet. Both raters need to fill in a score "
                  "from 1 to 5 for style, content and artifacts.")
            continue

        any_done = True
        summary = {"samples_rated_by_both": done, "criteria": {}}
        for c in CRITERIA:
            one, two = ratings[c]
            exact = sum(1 for x, y in zip(one, two) if x == y) / len(one)
            within = sum(1 for x, y in zip(one, two) if abs(x - y) <= 1) / len(one)
            kappa = quadratic_kappa(one, two)
            m1, m2 = sum(one) / len(one), sum(two) / len(two)
            summary["criteria"][c] = {
                "rater1_mean": round(m1, 3),
                "rater2_mean": round(m2, 3),
                "combined_mean": round((m1 + m2) / 2, 3),
                "exact_agreement": round(exact, 3),
                "within_one_agreement": round(within, 3),
                "quadratic_weighted_kappa": round(kappa, 3) if kappa is not None else None,
            }
            print(f"  {c:<10} rater1 mean {m1:.2f}  rater2 mean {m2:.2f}  "
                  f"exact {exact:.0%}  within one {within:.0%}  kappa {kappa:.3f}")

        overall = sum(summary["criteria"][c]["combined_mean"] for c in CRITERIA) / 3
        kappas = [summary["criteria"][c]["quadratic_weighted_kappa"] for c in CRITERIA]
        summary["overall_mean_score"] = round(overall, 3)
        summary["mean_quadratic_kappa"] = round(sum(kappas) / len(kappas), 3)
        print(f"  overall mean score {overall:.2f}, mean kappa "
              f"{summary['mean_quadratic_kappa']:.3f}")

        out = path.parent / "human_audit_summary.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"  written {out}")
        print(f"  put {overall:.2f} in metrics_report.csv as human_audit_score")
        print(f"  put {summary['mean_quadratic_kappa']:.3f} as inter_rater_agreement")

    if not any_done:
        print("\nHow to fill the audit in:")
        print("  1. Open task3_gan/audit_sheets/ to see all 30 samples at once.")
        print("  2. Each rater scores every sample 1 to 5 on three criteria:")
        print("       style     does it look like a real Monet")
        print("       content   is the original scene still recognisable")
        print("       artifacts 5 means clean, 1 means obvious artifacts")
        print("  3. Rater 1 fills the rater1 columns, rater 2 the rater2 columns.")
        print("     Rate independently, without looking at the other's scores.")
        print("  4. Run this script again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

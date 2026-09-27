"""
data/print_session5_context.py

Reads data/processed/dataset_summary.json (produced by prepare_dataset.py)
and prints a ready-to-paste paragraph for the Session 5 Colab-notebook
prompt, with the real current numbers filled in -- so you never have to
hand-copy counts out of a JSON file into a prompt again.

Usage:
    python data/print_session5_context.py
"""

import json
from pathlib import Path

SUMMARY_PATH = Path("data/processed/dataset_summary.json")


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found -- run prepare_dataset.py first."
        )
    summary = json.loads(SUMMARY_PATH.read_text())

    overall = summary["overall"]
    train = summary["train"]
    val = summary["val"]
    test = summary["test"]

    by_label = train["by_label"]
    label_str = ", ".join(f"{k} {v}" for k, v in sorted(by_label.items(), key=lambda kv: -kv[1]))

    no_lesion_total = overall["by_label"].get("no_lesion", 0)
    positive_total = overall["n_images"] - no_lesion_total
    no_lesion_train = train["by_label"].get("no_lesion", 0)
    positive_train = train["n_images"] - no_lesion_train

    unknown_pct = 100 * overall["by_ita_bin"].get("unknown", 0) / overall["n_images"]

    zero_val_test_strata = []
    by_strata = overall["by_label_and_ita_bin"]
    val_strata = set(val["by_label_and_ita_bin"].keys())
    test_strata = set(test["by_label_and_ita_bin"].keys())
    for stratum in by_strata:
        if stratum not in val_strata and stratum not in test_strata:
            zero_val_test_strata.append(stratum)

    paragraph = f"""Data pipeline reality check: data/processed/train_manifest.csv has {train['n_images']} rows (val {val['n_images']}, test {test['n_images']}) -- size the throughput-probe cell's subset to the actual train manifest, not a fixed 15-20k. Label distribution in train: {label_str}. Stage 1's binary split in train is {positive_train} lesion-positive vs {no_lesion_train} no_lesion ({positive_train // max(no_lesion_train,1)}:1 imbalance, only {no_lesion_train} unique negative images) -- expect an early, possibly overfit-looking Stage 1 plateau; don't treat its first checkpoint as production-ready. ita_bin is 'unknown' for {unknown_pct:.0f}% of all rows. These (label, ita_bin) strata have ZERO val/test examples at all: {', '.join(zero_val_test_strata) if zero_val_test_strata else '(none)'} -- the fairness-report cells must not assume every class has skin-tone-stratified held-out data to evaluate."""

    print("\n" + "=" * 78)
    print("COPY EVERYTHING BELOW THIS LINE INTO THE SESSION 5 PROMPT:")
    print("=" * 78 + "\n")
    print(paragraph)
    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()

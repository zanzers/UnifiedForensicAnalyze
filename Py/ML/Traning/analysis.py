import pandas as pd
from collections import Counter

def analyze_features(excel_file):
    df = pd.read_excel(excel_file)

    # assume: df columns → [label, PRNU, ELA, Noise, ...]
    label_col = df.columns[0]

    classes = df[label_col].unique()

    for cls in sorted(classes):
        cls_df = df[df[label_col] == cls]

        print("\n============================")
        print(f" Class {cls} Summary")
        print("============================")

        # Count
        print(f"Count: {len(cls_df)} samples")

        # Raw labels inside class
        class_labels = [int(x) for x in cls_df[label_col].values]  # convert np.int64 → int
        print("Labels:", class_labels)


        # Feature analysis
        features = df.columns[1:]  # all columns except label

        for feat in features:
            min_val = cls_df[feat].min()
            max_val = cls_df[feat].max()
            mean_val = cls_df[feat].mean()

            print(f"\n{feat}: min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}")

            # Overlapping classes inside the same range
            overlapping = df[
                (df[feat] >= min_val) & 
                (df[feat] <= max_val)
            ][label_col].values

            # Convert numpy int64 → normal int
            overlapping = [int(x) for x in overlapping]

            counts = Counter(overlapping)

            print(f" → Labels in this range: {list(counts.keys())}")
            print(f" → Count per label: {dict(counts)}")
            print(f" → Total overlap: {len(overlapping)} samples")


if __name__ == "__main__":
    analyze_features("features_dataset.xlsx")

"""Measure cascade performance at every threshold from 0.00 to 1.00."""

from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluate_proskin_cascade import (
    REPORT_DIR, TEST_DIR, TestImages, find_test_labels, run_predictions, save_plot, scores,
)

OUTPUT = REPORT_DIR.with_name("proskin-cascade-threshold-sweep")
THRESHOLDS = np.arange(101) / 100
METRICS = {
    "macro_f1": "Macro F1 по семи классам",
    "weighted_f1": "Weighted F1",
    "accuracy": "Точность, %",
    "errors": "Число ошибок",
}


def sweep(frame):
    baseline = scores(frame.true_class, frame.base_pred)
    rows = []
    for threshold in THRESHOLDS:
        routed = frame.route.ne("ProSkin") & frame.base_entropy.ge(threshold)
        predicted = frame.cascade_pred.where(routed, frame.base_pred)
        rows.append({
            "threshold": threshold,
            "routed": int(routed.sum()),
            **scores(frame.true_class, predicted),
        })
    return pd.DataFrame(rows), baseline


def plot_metrics(curves, baseline):
    for key, label in METRICS.items():
        fig, ax = plt.subplots(figsize=(8.4, 4.5), layout="constrained")
        scale = 100 if key == "accuracy" else 1
        ax.plot(curves.threshold, curves[key] * scale, color="#29778b", lw=2.1,
                label="Каскад")
        ax.axhline(baseline[key] * scale, color="#485465", linestyle="--", lw=1.5,
                   label="Базовая ProSkin")
        reference = curves.loc[curves.threshold.isin([.3, .4])]
        ax.scatter(reference.threshold, reference[key] * scale, color="#cb5c36",
                   label="Пороги 0,30 и 0,40", zorder=4)
        ax.set(title=label + " на тесте ISIC 2018",
               xlabel="Порог нормированной энтропии базовой модели",
               ylabel=label, xlim=(-.01, 1.01), xticks=np.linspace(0, 1, 11))
        ax.grid(alpha=.22)
        ax.legend(loc="best")
        save_plot(fig, OUTPUT, f"{key}_vs_threshold.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels = find_test_labels(TEST_DIR)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Устройство: {device}. Снимков: {len(labels)}.", flush=True)
    with ZipFile(TEST_DIR / "ISIC2018_Task3_Test_Input.zip") as archive:
        # The specialist is determined by the base class, independently of the threshold.
        # A single pass at 0 computes both answers for every possible specialist route.
        frame = run_predictions(TestImages(archive, labels.image_id), labels, device, threshold=0.0)
    curves, baseline = sweep(frame)
    for key, value in baseline.items():
        curves[f"baseline_{key}"] = value
    curves.to_csv(OUTPUT / "threshold_metrics.csv", index=False)
    plot_metrics(curves, baseline)
    print(curves.loc[curves.threshold.isin([0, .3, .4, 1])].to_string(index=False))


if __name__ == "__main__":
    main()

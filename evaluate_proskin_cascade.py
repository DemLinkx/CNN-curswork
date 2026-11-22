"""Compare ProSkin and its entropy cascade on the official ISIC 2018 Task 3 test."""

import argparse
import ast
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from train_proskin_specialists import ALL_CLASSES, ProSkinCNN

ROOT = Path(__file__).resolve().parent
TEST_DIR = ROOT / "data" / "isic2018-test"
REPORT_DIR = ROOT / "graphics" / "proskin-cascade"
WEIGHTS = ROOT / "my-models"
BASELINE = WEIGHTS / "ProSkinCNN_20261022_231200_476285.pth"
ROUTES = dict(nv="ProSkinA", mel="ProSkinB", bkl="ProSkinC",
              bcc="ProSkinD", akiec="ProSkinE", df="ProSkinF")
CLASSES = np.array(ALL_CLASSES)
DEFAULT_THRESHOLD = 0.3
ERROR_THRESHOLD = 0.4
N_COLORS = 48


def find_test_labels(directory):
    with ZipFile(directory / "ISIC2018_Task3_Test_GroundTruth.zip") as archive:
        member, = [name for name in archive.namelist() if name.endswith("_GroundTruth.csv")]
        labels = pd.read_csv(archive.open(member))
    labels.columns = labels.columns.str.lower()
    encoded = labels[ALL_CLASSES]
    if labels.image.duplicated().any() or not (encoded.isin([0, 1]).all().all()
                                              and encoded.sum(axis=1).eq(1).all()):
        raise ValueError("Expected one unique image and one of the seven labels per row")
    ham_ids = set(pd.read_csv(ROOT / "data/HAM10000_metadata.csv", usecols=["image_id"]).image_id)
    if ham_ids.intersection(labels.image):
        raise ValueError("Test IDs overlap HAM10000")
    return pd.DataFrame({"image_id": labels.image, "true_class": encoded.idxmax(axis=1)})


class TestImages(Dataset):
    def __init__(self, archive, ids):
        self.archive = archive
        self.ids = ids.tolist()
        members = [name for name in archive.namelist() if Path(name).suffix.lower() in {".jpg", ".jpeg"}]
        self.members = {Path(name).stem: name for name in members}
        if set(self.members) != set(self.ids) or len(self.members) != len(members):
            raise ValueError("Image archive does not match the test labels")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def image(self, index):
        with Image.open(BytesIO(self.archive.read(self.members[self.ids[index]]))) as source:
            return source.convert("RGB")

    def __getitem__(self, index):
        return self.transform(self.image(index))


def load_model(path, classes, device):
    model = ProSkinCNN(len(classes)).to(device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return model.eval()


def output_entropy(probs):
    return -(probs * probs.clamp_min(1e-12).log2()).sum(dim=1).cpu().numpy()


@torch.inference_mode()
def predict(model, images, classes):
    probs = model(images).softmax(dim=1)
    return classes[probs.argmax(dim=1).cpu().numpy()], output_entropy(probs)


@torch.inference_mode()
def run_predictions(images, labels, device, threshold):
    base = load_model(BASELINE, CLASSES, device)
    specialists = {}
    for predicted_class, name in ROUTES.items():
        directory = WEIGHTS / "specialists"
        config = json.loads((directory / f"{name}_config.json").read_text())
        classes = np.asarray(config["classes"])
        if predicted_class not in classes or not set(classes).issubset(CLASSES):
            raise ValueError(f"Unexpected output classes in {name}: {classes}")
        specialists[predicted_class] = (load_model(directory / f"{name}_best.pth", classes, device), classes)
    results = []
    for batch, x in enumerate(DataLoader(images, batch_size=32), start=1):
        x = x.to(device)
        base_pred, base_bits = predict(base, x, CLASSES)
        base_entropy = base_bits / np.log2(len(CLASSES))
        final_pred, final_bits = base_pred.copy(), base_bits.copy()
        route = np.full(len(x), "ProSkin", dtype="<U8")
        final_k = np.full(len(x), len(CLASSES))
        for base_class, (model, classes) in specialists.items():
            mask = (base_pred == base_class) & (base_entropy >= threshold)
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            selected = x.index_select(0, torch.as_tensor(indices, device=device))
            final_pred[mask], final_bits[mask] = predict(model, selected, classes)
            route[mask] = ROUTES[base_class]
            final_k[mask] = len(classes)
        results.append(pd.DataFrame(dict(
            base_pred=base_pred, cascade_pred=final_pred, route=route,
            base_bits=base_bits, base_entropy=base_entropy,
            final_classes=final_k, cascade_bits=final_bits,
            cascade_entropy=final_bits / np.log2(final_k),
            cascade_entropy_log7=final_bits / np.log2(len(CLASSES)),
        )))
        if batch % 10 == 0:
            print(f"Обработано снимков: {sum(map(len, results))}/{len(images)}", flush=True)
    results = pd.concat([labels, pd.concat(results, ignore_index=True)], axis=1)
    for model in ("base", "cascade"):
        results[f"{model}_correct"] = results.true_class == results[f"{model}_pred"]
    return results


def scores(truth, pred):
    return dict(
        macro_f1=f1_score(truth, pred, labels=ALL_CLASSES, average="macro", zero_division=0),
        weighted_f1=f1_score(truth, pred, labels=ALL_CLASSES, average="weighted", zero_division=0),
        accuracy=accuracy_score(truth, pred),
        errors=int((truth != pred).sum()),
    )


def outcomes(frame):
    return dict(
        images=len(frame),
        changed=int((frame.base_pred != frame.cascade_pred).sum()),
        fixed=int((~frame.base_correct & frame.cascade_correct).sum()),
        new_errors=int((frame.base_correct & ~frame.cascade_correct).sum()),
        wrong_both=int((~frame.base_correct & ~frame.cascade_correct).sum()),
        base_errors=int((~frame.base_correct).sum()),
        cascade_errors=int((~frame.cascade_correct).sum()),
    )


def write_tables(frame, output, threshold):
    truth = frame.true_class
    report = {"images": len(frame), "threshold": threshold, "base": scores(truth, frame.base_pred),
              "cascade": scores(truth, frame.cascade_pred), "outcomes": outcomes(frame),
              "routed": int(frame.route.ne("ProSkin").sum())}
    report["macro_f1_delta"] = report["cascade"]["macro_f1"] - report["base"]["macro_f1"]
    report["accuracy_delta"] = report["cascade"]["accuracy"] - report["base"]["accuracy"]
    assert report["cascade"]["errors"] == report["base"]["errors"] - report["outcomes"]["fixed"] + report["outcomes"]["new_errors"]
    expected_route = frame.base_pred.map(ROUTES).where(frame.base_entropy.ge(threshold)).fillna("ProSkin")
    assert frame.route.eq(expected_route).all()
    assert (frame.route.ne("ProSkin") | frame.base_pred.eq(frame.cascade_pred)).all()
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    confusions = {}
    for model in ("base", "cascade"):
        pred = frame[f"{model}_pred"]
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, pred, labels=ALL_CLASSES, zero_division=0
        )
        rows.append(pd.DataFrame(dict(model=model, true_class=ALL_CLASSES, support=support,
                                      precision=precision, recall=recall, f1=f1)))
        matrix = confusion_matrix(truth, pred, labels=ALL_CLASSES)
        confusions[model] = pd.DataFrame(matrix, columns=ALL_CLASSES, index=ALL_CLASSES)
        confusions[model].to_csv(output / f"confusion_{model}.csv", index_label="true_class")
    class_scores = pd.concat(rows, ignore_index=True)
    class_scores.to_csv(output / "per_class_metrics.csv", index=False)
    routing = pd.DataFrame([{"route": route, **outcomes(group)} for route, group in frame.groupby("route")])
    routing.to_csv(output / "routing.csv", index=False)

    entropy_groups = []
    entropy_pairs = []
    for model in ("base", "cascade"):
        pred = f"{model}_pred"
        wrong = ~frame[f"{model}_correct"]
        entropy = frame[f"{model}_entropy"]
        bands = np.select([entropy.lt(0.3), entropy.gt(ERROR_THRESHOLD)],
                          ["< 0.3", "> 0.4"], default="0.3–0.4")
        for group_by in ("true_class", pred):
            group = frame.assign(band=bands, error=wrong).groupby([group_by, "band"], observed=True)
            summary = group.error.agg(images="size", errors="sum").reset_index()
            summary["error_rate"] = summary.errors / summary.images
            summary = summary.rename(columns={group_by: "class"})
            entropy_groups.append(summary.assign(model=model, group_by="true" if group_by == "true_class" else "predicted"))
        pairs = frame[wrong].assign(high=entropy[wrong].gt(ERROR_THRESHOLD)).groupby(["true_class", pred])
        pairs = pairs.high.agg(errors="size", high_entropy_errors="sum").reset_index()
        entropy_pairs.append(pairs.rename(columns={pred: "predicted_class"}).assign(model=model))
    pd.concat(entropy_groups).to_csv(output / "entropy_error_rates.csv", index=False)
    pairs = pd.concat(entropy_pairs)
    pairs.to_csv(output / "error_directions.csv", index=False)
    return report, class_scores, confusions, routing, pairs


def save_plot(fig, output, filename):
    fig.savefig(output / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plots(frame, report, class_scores, confusions, routing, pairs, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    x = np.arange(len(ALL_CLASSES))
    for i, (model, name, color) in enumerate((("base", "ProSkin", "#306d8f"), ("cascade", "Каскад", "#c36737"))):
        f1 = class_scores[class_scores.model == model].set_index("true_class").loc[ALL_CLASSES, "f1"]
        axes[0].bar(x + (i - .5) * .37, f1, .37, label=f"{name}, macro F1 = {report[model]['macro_f1']:.3f}", color=color)
    axes[0].set(xticks=x, xticklabels=ALL_CLASSES, ylim=(0, 1), ylabel="F1", title="F1 по классам на тесте")
    axes[0].legend()
    route_data = routing[routing.route.ne("ProSkin")]
    axes[1].bar(route_data.route, route_data.fixed, label="Исправил", color="#3c9471")
    axes[1].bar(route_data.route, -route_data.new_errors, label="Добавил ошибку", color="#c75250")
    axes[1].axhline(0, color="gray", lw=.8)
    axes[1].set(ylabel="Снимки", title="Что изменил каждый специалист")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=.24)
    fig.suptitle(f"Маршрутизация при H ≥ {report['threshold']:g}")
    save_plot(fig, output, "f1_and_routing.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    scale = max(matrix.to_numpy().max() for matrix in confusions.values())
    for ax, (model, matrix) in zip(axes, confusions.items()):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=scale)
        for (i, j), n in np.ndenumerate(matrix.to_numpy()):
            ax.text(j, i, n, ha="center", va="center", color="white" if n > scale / 2 else "#172e42")
        ax.set(xticks=x, yticks=x, xticklabels=ALL_CLASSES, yticklabels=ALL_CLASSES,
               xlabel="Предсказанный класс", ylabel="Истинный класс",
               title="Матрица ProSkin" if model == "base" else "Матрица каскада")
    save_plot(fig, output, "confusion_matrices.png")

    directions = pairs[pairs.model.eq("cascade")]
    order = directions.groupby("true_class").errors.sum().reindex(ALL_CLASSES, fill_value=0).sort_values(ascending=False).index
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for ax, true_class in zip(axes.flat, order):
        group = directions[directions.true_class.eq(true_class)].set_index("predicted_class")[
            ["errors", "high_entropy_errors"]
        ]
        group = group.reindex([cls for cls in ALL_CLASSES if cls != true_class], fill_value=0)
        group = group.sort_values("errors", ascending=False)
        high = group.high_entropy_errors
        ax.bar(group.index, high, color="#c75250", label="H > 0,4")
        ax.bar(group.index, group.errors - high, bottom=high, color="#306d8f", label="H ≤ 0,4")
        for i, count in enumerate(group.errors):
            if count:
                ax.annotate(str(count), (i, count), xytext=(0, 2), textcoords="offset points", ha="center")
        ax.set(ylim=(0, max(1, group.errors.max() * 1.2)), title=f"{true_class} → другой класс ({group.errors.sum()})",
               ylabel="Ошибок")
        ax.grid(axis="y", alpha=.24)
    axes.flat[0].legend()
    axes.flat[-1].axis("off")
    fig.suptitle("Направления ошибок каскада на тесте")
    save_plot(fig, output, "error_directions.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    bins = np.linspace(0, 1, 11)
    for ax, model, title in zip(axes, ("base", "cascade"), ("ProSkin", "Каскад")):
        correct = frame[f"{model}_correct"]
        h_column = "cascade_entropy_log7" if model == "cascade" else "base_entropy"
        ax.hist([frame.loc[correct, h_column], frame.loc[~correct, h_column]], bins=bins, stacked=True,
                color=["#306d8f", "#c75250"], label=["Верно", "Ошибка"])
        ax.set(xlim=(0, 1), xlabel="Энтропия ответа / log₂ 7", ylabel="Снимки", title=title)
        ax.legend()
        ax.grid(axis="y", alpha=.24)
    save_plot(fig, output, "output_entropy.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), layout="constrained")
    for i, (model, name, color) in enumerate((("base", "ProSkin", "#306d8f"), ("cascade", "Каскад", "#c36737"))):
        high = frame[frame[f"{model}_entropy"].gt(ERROR_THRESHOLD)]
        wrong = ~high[f"{model}_correct"]
        counts = high.groupby("true_class").size().reindex(ALL_CLASSES, fill_value=0)
        errors = high[wrong].groupby("true_class").size().reindex(ALL_CLASSES, fill_value=0)
        positions = x + (i - .5) * .37
        axes[0].bar(positions, errors / counts.where(counts.gt(0)), .37, label=name, color=color)
        bars = axes[1].bar(positions, errors, .37, label=name, color=color)
        axes[1].bar_label(bars, labels=[str(n) if n else "" for n in errors], fontsize=8, padding=2)
    axes[0].set(ylabel="Ошибки / снимки класса", ylim=(0, 1.03), title="Доля ошибок внутри истинного класса")
    axes[1].set(ylabel="Снимки", title="Число ошибок")
    axes[1].margins(y=.15)
    for ax in axes:
        ax.set(xticks=x, xticklabels=ALL_CLASSES, xlabel="Истинный класс")
        ax.grid(axis="y", alpha=.24)
        ax.legend()
    fig.suptitle("Ошибки при энтропии ответа H / log₂ K > 0,4")
    save_plot(fig, output, "high_entropy_class_errors.png")


def image_entropy_function():
    # Reuse the tested implementation in the notebook without executing its training cells.
    notebook = json.loads((ROOT / "main.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "def color_transition_entropy_n(" in source:
            node = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "color_transition_entropy_n")
            namespace = {"np": np, "Image": Image}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "main.ipynb", "exec"), namespace)
            return namespace[node.name]
    raise ValueError("color_transition_entropy_n is missing in main.ipynb")


def plot_image_entropy(frame, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True, layout="constrained")
    for axs, model, name in zip(axes, ("base", "cascade"), ("ProSkin", "Каскад")):
        h_column = "cascade_entropy_log7" if model == "cascade" else "base_entropy"
        correct = frame[f"{model}_correct"]
        groups = [(frame[~correct], "#c75250", "Ошибки"), (frame[correct], "#306d8f", "Верные ответы")]
        for (group, color, label), ax in zip(groups, axs[1:]):
            for target in (axs[0], ax):
                target.scatter(group.input_entropy, group[h_column], s=10, alpha=.38, color=color, label=label)
            ax.set_title(f"{label} (n={len(group)})")
        axs[0].set(title=name + " · Все снимки", ylabel="Энтропия ответа / log₂ 7")
        axs[0].legend()
        for ax in axs:
            ax.set_ylim(0, 1)
            ax.grid(alpha=.24)
    for ax in axes[-1]:
        ax.set_xlabel("Энтропия цвета / (3 log₂ 48)")
    save_plot(fig, output, "image_and_output_entropy.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    output = args.output
    if output is None:
        output = REPORT_DIR if args.threshold == DEFAULT_THRESHOLD else REPORT_DIR.with_name(
            f"{REPORT_DIR.name}-{args.threshold}"
        )
    output.mkdir(parents=True, exist_ok=True)
    labels = find_test_labels(args.test_dir)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Устройство: {device}. Снимков: {len(labels)}.", flush=True)
    with ZipFile(args.test_dir / "ISIC2018_Task3_Test_Input.zip") as archive:
        images = TestImages(archive, labels.image_id)
        frame = run_predictions(images, labels, device, args.threshold)
        summary = write_tables(frame, output, args.threshold)
        frame.to_csv(output / "predictions.csv", index=False)
        print(json.dumps(summary[0], indent=2), flush=True)
        plots(frame, *summary, output)
        entropy = image_entropy_function()
        input_bits = [entropy(N_COLORS, images.image(i))["mean_bits"] for i in range(len(images))]
    frame["input_bits"] = input_bits
    frame["input_entropy"] = frame.input_bits / (3 * np.log2(N_COLORS))
    frame.to_csv(output / "predictions.csv", index=False)
    plot_image_entropy(frame, output)


if __name__ == "__main__":
    main()

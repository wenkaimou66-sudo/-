"""Train/evaluate ASD versus HC classifier and create final submission.

Run with Python 3.7: python run_experiment.py --zip ..\excise.zip
All image features are computed without looking at labels. Every learned
classification step is re-fit inside each cross-validation training fold.
"""

import argparse
import csv
import html
import json
import os
import re
import time
import zipfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from mri_pipeline import PIPELINE_VERSION, preprocess, read_image_from_zip, save_qc_png, save_result


SEED = 20260927


class CancellationError(Exception):
    pass


def check_cancelled(event):
    if event is not None and event.is_set():
        raise CancellationError("User cancelled processing")


def load_manifest(zip_path):
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = set(archive.namelist())
        labels_text = archive.read("excise/train_labels.csv").decode("utf-8-sig")
        template_text = archive.read("excise/submission_example.csv").decode("utf-8-sig")
    train_rows = list(csv.DictReader(labels_text.splitlines()))
    template_rows = list(csv.DictReader(template_text.splitlines()))
    train = []
    test = []
    seen = set()
    for row in train_rows:
        sid = row["subject_id"].strip()
        label = int(row["label"])
        if not re.fullmatch(r"sub-\d{3}", sid) or sid in seen or label not in (1, 2):
            raise ValueError("Invalid or duplicate training row")
        if row.get("diagnosis", "").strip() != {1: "ASD", 2: "HC"}[label]:
            raise ValueError("Diagnosis/label mismatch for " + sid)
        member = "excise/train/" + sid + "_T1w.nii.gz"
        if member not in names:
            raise ValueError("Missing training image " + member)
        seen.add(sid)
        train.append((sid, label, member))
    for row in template_rows:
        sid = row["subject_id"].strip()
        if not re.fullmatch(r"sub-\d{3}", sid) or sid in seen:
            raise ValueError("Invalid or duplicate test row")
        member = "excise/test/" + sid + "_T1w.nii.gz"
        if member not in names:
            raise ValueError("Missing test image " + member)
        seen.add(sid)
        test.append((sid, member))
    actual_train = {x for x in names if re.fullmatch(r"excise/train/sub-\d{3}_T1w.nii.gz", x)}
    actual_test = {x for x in names if re.fullmatch(r"excise/test/sub-\d{3}_T1w.nii.gz", x)}
    if len(train) != 136 or len(test) != 10 or len(actual_train) != 136 or len(actual_test) != 10:
        raise ValueError("Unexpected train/test counts")
    return train, test


def extract_all(zip_path, output_dir, train, test, check_first=0,
                save_images=False, cancel_event=None):
    feature_dir = os.path.join(output_dir, "features")
    qc_dir = os.path.join(output_dir, "qc")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)
    records = [(sid, member) for sid, _, member in train] + list(test)
    if check_first:
        records = records[:check_first]
    vectors, feature_names, qc_rows = {}, None, []
    started = time.time()
    for number, (sid, member) in enumerate(records, 1):
        check_cancelled(cancel_event)
        cache = os.path.join(feature_dir, sid + "_features.npz")
        qc_path = os.path.join(feature_dir, sid + "_qc.json")
        valid_cache = False
        if os.path.exists(cache) and os.path.exists(qc_path):
            with open(qc_path, "r", encoding="utf-8") as source:
                qc = json.load(source)
            valid_cache = qc.get("pipeline_version") == PIPELINE_VERSION
            if number <= 6 or check_first:
                valid_cache = valid_cache and os.path.exists(os.path.join(qc_dir, sid + ".png"))
        if valid_cache:
            with np.load(cache, allow_pickle=False) as cached:
                vec = cached["features"].astype(np.float64)
                names = cached["feature_names"].tolist()
        else:
            image, affine = read_image_from_zip(zip_path, member)
            result = preprocess(image, affine)
            vec, names, qc = result["features"], result["feature_names"], result["qc"]
            save_result(feature_dir, sid, result, save_images=save_images)
            if number <= 6 or check_first:
                save_qc_png(os.path.join(qc_dir, sid + ".png"), result, sid)
        if not np.all(np.isfinite(vec)):
            raise ValueError("Non-finite features for " + sid)
        if feature_names is not None and names != feature_names:
            raise ValueError("Feature schema changed at " + sid)
        feature_names = names
        vectors[sid] = vec
        qc_rows.append({"subject_id": sid, **qc})
        elapsed = int(time.time() - started)
        print("[%d/%d] %s ready (%ds)" % (number, len(records), sid, elapsed), flush=True)
    with open(os.path.join(output_dir, "quality_control.json"), "w", encoding="utf-8") as out:
        json.dump(qc_rows, out, ensure_ascii=False, indent=2)
    return vectors, feature_names


def stratified_folds(y, splits, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(splits)]
    for cls in (0, 1):
        indices = np.where(y == cls)[0]
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, splits)):
            folds[fold].extend(int(i) for i in part)
    all_indices = np.arange(len(y))
    for fold in folds:
        test_idx = np.asarray(sorted(fold), dtype=int)
        train_idx = np.setdiff1d(all_indices, test_idx)
        yield train_idx, test_idx


class LogisticModel(object):
    def __init__(self, feature_count, c):
        self.feature_count = int(feature_count)
        self.c = float(c)

    def fit(self, x, y):
        mean = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd < 1e-10] = 1.0
        standard = (x - mean) / sd
        pos, neg = y == 1, y == 0
        fscore = (standard[pos].mean(axis=0) - standard[neg].mean(axis=0)) ** 2
        fscore /= (standard[pos].var(axis=0) / max(int(pos.sum()), 1) +
                   standard[neg].var(axis=0) / max(int(neg.sum()), 1) + 1e-8)
        selected = np.argsort(-fscore, kind="mergesort")[:self.feature_count]
        selected_x = standard[:, selected]
        weight = np.where(pos, len(y) / (2.0 * max(int(pos.sum()), 1)),
                          len(y) / (2.0 * max(int(neg.sum()), 1)))

        def objective(parameters):
            linear = selected_x.dot(parameters[:-1]) + parameters[-1]
            loss = (weight * (np.logaddexp(0.0, linear) - y * linear)).mean()
            loss += 0.5 * np.dot(parameters[:-1], parameters[:-1]) / (self.c * len(y))
            gradient_linear = weight * (expit(linear) - y) / len(y)
            gradient = np.empty_like(parameters)
            gradient[:-1] = selected_x.T.dot(gradient_linear) + parameters[:-1] / (self.c * len(y))
            gradient[-1] = gradient_linear.sum()
            return loss, gradient

        result = minimize(objective, np.zeros(len(selected) + 1), jac=True,
                          method="L-BFGS-B", options={"maxiter": 300})
        if not result.success and np.linalg.norm(result.jac) > 1e-3:
            raise RuntimeError("Logistic regression did not converge: " + result.message)
        self.mean_ = mean
        self.sd_ = sd
        self.selected_ = selected
        self.coef_ = result.x
        return self

    def predict_probability(self, x):
        z = ((x - self.mean_) / self.sd_)[:, self.selected_]
        return expit(z.dot(self.coef_[:-1]) + self.coef_[-1])


def metrics(y_true, probability, threshold=0.5):
    prediction = probability >= threshold
    pos = y_true == 1
    neg = ~pos
    tp = int(np.sum(prediction & pos))
    tn = int(np.sum(~prediction & neg))
    fp = int(np.sum(prediction & neg))
    fn = int(np.sum(~prediction & pos))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ranks = rankdata(probability)
    npos, nneg = int(pos.sum()), int(neg.sum())
    auc = float((ranks[pos].sum() - npos * (npos + 1) / 2) / max(npos * nneg, 1))
    return {"accuracy": float((tp + tn) / len(y_true)),
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def tune(x, y, seed, cancel_event=None):
    candidates = [(k, c) for k in (8, 16, 32, 64)
                  for c in (0.1, 1.0, 10.0, 100.0) if k <= x.shape[1]]
    folds = list(stratified_folds(y, 3, seed))
    best = None
    for k, c in candidates:
        check_cancelled(cancel_event)
        scores = []
        for train_idx, valid_idx in folds:
            model = LogisticModel(k, c).fit(x[train_idx], y[train_idx])
            p = model.predict_probability(x[valid_idx])
            scores.append(metrics(y[valid_idx], p)["balanced_accuracy"])
        score = float(np.mean(scores))
        priority = (score, -k, -c)
        if best is None or priority > best[0]:
            best = (priority, k, c)
    return best[1], best[2]


def nested_validation(x, y, cancel_event=None):
    probability = np.full(len(y), np.nan, dtype=np.float64)
    folds = []
    for outer, (train_idx, valid_idx) in enumerate(stratified_folds(y, 5, SEED), 1):
        check_cancelled(cancel_event)
        k, c = tune(x[train_idx], y[train_idx], SEED + outer, cancel_event)
        model = LogisticModel(k, c).fit(x[train_idx], y[train_idx])
        probability[valid_idx] = model.predict_probability(x[valid_idx])
        fold_metrics = metrics(y[valid_idx], probability[valid_idx])
        fold_metrics.update({"fold": outer, "features": k, "c": c,
                             "n_train": len(train_idx), "n_valid": len(valid_idx)})
        folds.append(fold_metrics)
        print("Outer fold %d: ACC %.3f, SEN %.3f, SPE %.3f (k=%d, C=%g)" %
              (outer, fold_metrics["accuracy"], fold_metrics["sensitivity"],
               fold_metrics["specificity"], k, c), flush=True)
    if not np.all(np.isfinite(probability)):
        raise RuntimeError("Missing cross-validation predictions")
    return probability, folds, metrics(y, probability)


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir, train, test, feature_names, cv, folds, best, predictions):
    def pc(value):
        return "%.1f%%" % (100.0 * value)

    fold_html = "".join(
        "<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%g</td></tr>" %
        (f["fold"], pc(f["accuracy"]), pc(f["sensitivity"]),
         pc(f["specificity"]), f["features"], f["c"]) for f in folds)
    prediction_html = "".join(
        "<tr><td>%s</td><td>%.3f</td><td>%s</td></tr>" %
        (html.escape(row["subject_id"]), row["p_asd"],
         "ASD (1)" if row["label"] == 1 else "HC (2)") for row in predictions)
    selected = ", ".join(html.escape(feature_names[i]) for i in best[2])
    report = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>ASD / HC T1 MRI 分类方法与结果</title>
<style>body{font:15px/1.6 'Microsoft YaHei',Arial,sans-serif;color:#15202b;max-width:900px;margin:30px auto}
h1,h2{color:#193657}table{border-collapse:collapse;width:100%%;font-size:13px}th,td{border:1px solid #bfcbd4;padding:6px;text-align:left}
th{background:#eaf2f7}.page{page-break-before:always}p{margin:8px 0}li{margin:4px 0}
@media print{body{margin:10mm;max-width:none;font-size:11pt}.page{page-break-before:always}}</style></head><body>
<h1>基于 T1 MRI 的 ASD / HC 二分类</h1>
<p>方法与结果报告 · Python 3.7 · 数据：ABIDE NYU 匿名子集</p>
<h2>1. 数据与任务</h2>
<p>训练集 136 例（ASD 62、HC 74），独立盲测集 10 例。ASD 编码为 1，HC 编码为 2。
测试集真值未提供。本报告中的测试结果是模型预测，不能称为测试准确率。</p>
<h2>2. 预处理与质量控制</h2>
<ol><li>读取 NIfTI-1 头和 sform 仿射，按物理坐标重采样到 3 mm 各向同性的 56×72×72 公共网格；检查数据类型、维度、仿射可逆性。</li>
<li>在固定解剖椭球范围内估计脑区掩膜，结合平滑、形态学和最大连通域；以空间低频场进行近似强度偏置校正。</li>
<li>按掩膜质心和尺度进行仿射对齐，建立本数据集的公共空间。该空间不是 MNI，不能将 ROI 解读为标准脑图谱分区。</li>
<li>用三成分高斯混合模型估计 CSF/GM/WM 样强度概率。该结果是近似组织类，需看 QC 图，不能等同经验证的临床分割。</li>
<li>导出脑区体积、全局概率比例和 2×3×3 空间格子的占有率及组织概率；共 %d 个特征。</li></ol>
<p>预处理逐人进行，完全不使用诊断标签；缓存版本 %s。质量控制记录和示例叠加图见输出目录。</p>
<div class="page"></div><h2>3. 建模与防止信息泄漏</h2>
<p>外层分层 5 折交叉验证估计泛化性能，内层分层 3 折选择特征数量 k 和正则化强度 C。
每个训练折单独计算均值、标准差和单变量类别区分分数，选出前 k 个特征，再拟合带类别平衡权重的 L2 逻辑回归。
外层验证样本只用于该折的最终预测。固定随机种子 %d，ASD 为阳性类；0.5 为判定阈值。</p>
<table><tr><th>外层折</th><th>准确率</th><th>灵敏度</th><th>特异度</th><th>k</th><th>C</th></tr>%s</table>
<p><b>外层留出预测汇总：</b>准确率 %s；灵敏度 %s；特异度 %s；平衡准确率 %s；ROC-AUC %.3f。
混淆矩阵：TP=%d，FN=%d，TN=%d，FP=%d。</p>
<p>最终模型在全部训练集上重新进行内层参数选择：k=%d，C=%g。选中的特征：%s。</p>
<div class="page"></div><h2>4. 盲测预测</h2>
<p>下表 p(ASD) 是模型输出概率。题目说明盲测集恰有 5 例 ASD 和 5 例 HC；提交标签按概率从高到低选 5 例 ASD。
这一已知组成约束仅用于盲测提交，不用于交叉验证指标计算。</p>
<table><tr><th>匿名编号</th><th>p(ASD)</th><th>提交标签</th></tr>%s</table>
<h2>5. 分析与局限</h2>
<p>样本量较小，且训练集类别不平衡；因此同时报告灵敏度、特异度和平衡准确率。
10 例测试集每错 1 例，准确率就变化 10 个百分点。扫描和人群混杂仍可能影响泛化。
公共空间是基于图像矩的仿射近似，掩膜和强度分割也不是专业脑提取与标准脑图谱分割；
必须逐例检查 QC，不能将模型用于临床诊断。真实测试准确率只能由掌握盲测标签的教师计算。</p>
<p>可复现文件：run_experiment.py、mri_pipeline.py、metrics.json、features.csv、predictions.csv、submission.csv。</p>
</body></html>""" % (len(feature_names), PIPELINE_VERSION, SEED, fold_html,
                   pc(cv["accuracy"]), pc(cv["sensitivity"]), pc(cv["specificity"]),
                   pc(cv["balanced_accuracy"]), cv["auc"], cv["tp"], cv["fn"], cv["tn"], cv["fp"],
                   best[0], best[1], selected, prediction_html)
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as out:
        out.write(report)


def run(zip_path, output_dir, check_first=0, save_images=False, cancel_event=None):
    os.makedirs(output_dir, exist_ok=True)
    train, test = load_manifest(zip_path)
    print("Manifest: %d train (%d ASD, %d HC), %d test" %
          (len(train), sum(label == 1 for _, label, _ in train),
           sum(label == 2 for _, label, _ in train), len(test)), flush=True)
    vectors, names = extract_all(zip_path, output_dir, train, test,
                                 check_first=check_first, save_images=save_images,
                                 cancel_event=cancel_event)
    if check_first:
        print("Smoke check complete; no model or submission created.", flush=True)
        return
    x = np.stack([vectors[sid] for sid, _, _ in train])
    y = np.asarray([int(label == 1) for _, label, _ in train], dtype=int)
    test_x = np.stack([vectors[sid] for sid, _ in test])
    write_csv(os.path.join(output_dir, "features.csv"), ["subject_id"] + names,
              [dict(zip(["subject_id"] + names, [sid] + vectors[sid].tolist()))
               for sid, _, _ in train])
    oof_probability, folds, cv = nested_validation(x, y, cancel_event)
    k, c = tune(x, y, SEED + 99, cancel_event)
    final = LogisticModel(k, c).fit(x, y)
    np.savez_compressed(os.path.join(output_dir, "model.npz"),
                        mean=final.mean_, sd=final.sd_, selected=final.selected_,
                        coef=final.coef_, feature_names=np.asarray(names),
                        pipeline_version=np.asarray(PIPELINE_VERSION),
                        k=np.asarray(k), c=np.asarray(c))
    probabilities = final.predict_probability(test_x)
    asd_indices = set(np.argsort(probabilities, kind="mergesort")[-5:].tolist())
    predictions = []
    for i, (sid, _) in enumerate(test):
        predictions.append({"subject_id": sid, "p_asd": float(probabilities[i]),
                            "unconstrained_label": 1 if probabilities[i] >= 0.5 else 2,
                            "label": 1 if i in asd_indices else 2})
    write_csv(os.path.join(output_dir, "submission.csv"), ["subject_id", "label"], predictions)
    write_csv(os.path.join(output_dir, "predictions.csv"),
              ["subject_id", "p_asd", "unconstrained_label", "label"], predictions)
    write_csv(os.path.join(output_dir, "out_of_fold_predictions.csv"),
              ["subject_id", "true_label", "p_asd", "predicted_label"],
              [{"subject_id": sid, "true_label": label,
                "p_asd": float(oof_probability[i]),
                "predicted_label": 1 if oof_probability[i] >= 0.5 else 2}
               for i, (sid, label, _) in enumerate(train)])
    info = {"python_target": "3.7", "pipeline_version": PIPELINE_VERSION,
            "seed": SEED, "training_count": len(train), "test_count": len(test),
            "feature_count": len(names), "nested_cv": cv, "outer_folds": folds,
            "final_model": {"k": k, "c": c,
                            "selected_features": [names[i] for i in final.selected_]},
            "test_truth_available": False,
            "test_assignment": "top five p(ASD) given the provided 5/5 test composition"}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as out:
        json.dump(info, out, ensure_ascii=False, indent=2)
    write_report(output_dir, train, test, names, cv, folds,
                 (k, c, final.selected_), predictions)
    print("Finished. CV metrics: " + json.dumps(cv), flush=True)
    print("Submission: " + os.path.join(output_dir, "submission.csv"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default=os.path.join("..", "excise.zip"))
    parser.add_argument("--output", default="output")
    parser.add_argument("--check-first", type=int, default=0,
                        help="Preprocess this many subjects only, without training")
    parser.add_argument("--save-images", action="store_true",
                        help="Save all corrected/mask/tissue NIfTI derivatives")
    args = parser.parse_args()
    run(os.path.abspath(args.zip_path), os.path.abspath(args.output),
        check_first=args.check_first, save_images=args.save_images)

"""Reproduce the MNI152/N4/atlas-guided ASD classification experiment.

Run from this project directory with Python 3.7:
    python run_mni_experiment.py
"""

import argparse
import csv
import html
import json
import os
import time

import numpy as np

from mni_pipeline import (VERSION, Reference, preprocess, read_zip_image,
                          save_derivatives, save_qc_png)
from run_experiment import (SEED, LogisticModel, check_cancelled, load_manifest,
                            nested_validation, tune, write_csv)


def extract(zip_path, output_dir, train, test, check_first=0,
            save_images=False, cancel_event=None):
    reference = Reference()
    feature_dir = os.path.join(output_dir, "features")
    qc_dir = os.path.join(output_dir, "qc")
    tmp_dir = os.path.join(output_dir, "tmp")
    for path in (feature_dir, qc_dir, tmp_dir):
        os.makedirs(path, exist_ok=True)
    subjects = [(sid, member) for sid, _, member in train] + list(test)
    if check_first:
        subjects = subjects[:check_first]
    vectors, feature_names, qc_rows = {}, None, []
    started = time.time()
    for index, (sid, member) in enumerate(subjects, 1):
        check_cancelled(cancel_event)
        feature_path = os.path.join(feature_dir, sid + "_features.npz")
        quality_path = os.path.join(feature_dir, sid + "_qc.json")
        qc_image = os.path.join(qc_dir, sid + ".png")
        cached = os.path.exists(feature_path) and os.path.exists(quality_path)
        if cached:
            with open(quality_path, "r", encoding="utf-8") as source:
                quality = json.load(source)
            cached = quality.get("pipeline_version") == VERSION
            if index <= 10 or check_first:
                cached = cached and os.path.exists(qc_image)
        if cached:
            with np.load(feature_path, allow_pickle=False) as saved:
                vector = saved["features"].astype(np.float64)
                names = saved["feature_names"].tolist()
        else:
            image = read_zip_image(zip_path, member, tmp_dir)
            result = preprocess(image, reference)
            save_derivatives(result, feature_dir, sid, reference,
                             save_images=save_images)
            vector, names, quality = (result["features"],
                                      result["feature_names"], result["qc"])
            with open(quality_path, "w", encoding="utf-8") as output:
                json.dump(quality, output, ensure_ascii=False, indent=2)
            if index <= 10 or check_first:
                save_qc_png(result, reference, qc_image, sid)
        if not np.all(np.isfinite(vector)) or feature_names not in (None, names):
            raise ValueError("Bad feature vector or schema at " + sid)
        feature_names = names
        vectors[sid] = vector
        qc_rows.append({"subject_id": sid, **quality})
        print("[%d/%d] %s, corr=%.3f, coverage=%.3f, elapsed=%ds" %
              (index, len(subjects), sid,
               quality["template_intensity_correlation"],
               quality["brain_coverage"], int(time.time() - started)), flush=True)
    with open(os.path.join(output_dir, "quality_control.json"), "w", encoding="utf-8") as output:
        json.dump(qc_rows, output, ensure_ascii=False, indent=2)
    return vectors, feature_names, qc_rows


def report_html(output_dir, train, test, names, folds, cv, final, predictions, qc_rows):
    def pct(number):
        return "%.1f%%" % (100.0 * number)

    table_folds = "".join(
        "<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%d</td><td>%g</td></tr>" %
        (fold["fold"], pct(fold["accuracy"]), pct(fold["sensitivity"]),
         pct(fold["specificity"]), fold["features"], fold["c"]) for fold in folds)
    table_predictions = "".join(
        "<tr><td>%s</td><td>%.3f</td><td>%d</td></tr>" %
        (html.escape(row["subject_id"]), row["p_asd"], row["label"])
        for row in predictions)
    corr = np.asarray([row["template_intensity_correlation"] for row in qc_rows])
    coverage = np.asarray([row["brain_coverage"] for row in qc_rows])
    selected = ", ".join(html.escape(names[i]) for i in final.selected_)
    content = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>ASD / HC 分类方法与结果</title><style>
@page{size:A4;margin:18mm}body{font-family:'Microsoft YaHei',Arial,sans-serif;font-size:11pt;line-height:1.5;color:#142635;max-width:850px;margin:24px auto}
h1{font-size:21pt;color:#173c60}h2{font-size:15pt;color:#173c60;margin-top:17px}
table{width:100%%;border-collapse:collapse;font-size:9pt}th,td{padding:5px;border:1px solid #8da5b5;text-align:left}th{background:#eaf2f7}
.page{page-break-before:always}p{margin:8px 0}li{margin:3px 0}
@media print{body{max-width:none;margin:0}.page{page-break-before:always}}
</style></head><body>
<h1>基于 T1 结构 MRI 的 ASD / HC 二分类</h1>
<p>Python 3.7 实验报告 · 匿名 ABIDE NYU 子集 · %s</p>
<h2>1. 数据和处理方法</h2>
<p>训练集 136 例，其中 ASD（标签 1）62 例、健康对照（标签 2）74 例；盲测集 10 例，真值不公开。
文件名中的匿名编号不用于任何特征或预测。</p>
<ol><li>读取 NIfTI sform 和 T1 原始影像，将每例缩小到约 2–3 mm 体素后用 N4 算法做偏置场校正。</li>
<li>使用公开 MNI152 2 mm T1 模板和脑掩膜，在脑区内以互信息为准则进行多分辨率仿射配准；保存每例变换矩阵。
配准结果均重采样至同一 MNI 网格。</li>
<li>在 MNI 脑掩膜内，结合模板的 CSF/GM/WM 概率先验与本例 T1 强度，使用三类高斯模型估计组织概率。
这些是近似组织概率，需结合质量控制图判断，不能当作临床分割。</li>
<li>提取全脑组织体积、组织比例、仿射体积比例，以及固定 MNI 网格 18 个区域内的 CSF/GM/WM 概率，共 %d 维。
体积按仿射雅可比校正。</li></ol>
<p>模板来源：<a href="https://github.com/neuroconductor/MNITemplate">neuroconductor/MNITemplate</a>；
其组织先验源于 FSL FAST 模板分割。软件版本、文件哈希见 resources/provenance.json。</p>
<p>质量控制：模板与配准后脑区的强度相关系数中位数 %.3f（范围 %.3f–%.3f）；
脑掩膜覆盖率中位数 %.3f。示例及标记复核病例的叠加图见 qc/。</p>
<div class="page"></div><h2>2. 特征选择、分类与交叉验证</h2>
<p>全部影像的预处理独立于诊断标签。外层分层 5 折交叉验证用于评估，内层分层 3 折以平衡准确率选择
特征数 k∈{8,16,32,64}（不超过特征总数）和 L2 逻辑回归正则强度 C∈{0.1,1,10,100}。
每折只在该折的训练数据上计算标准化参数与单变量特征排序；验证样本不参与拟合。
ASD 为阳性类，预测阈值 0.5，类别权重按训练折人数平衡，固定随机种子 %d。</p>
<table><tr><th>外层折</th><th>准确率</th><th>灵敏度</th><th>特异度</th><th>k</th><th>C</th></tr>%s</table>
<p><b>136 例外层留出预测汇总：</b>准确率 %s；灵敏度 %s；特异度 %s；
平衡准确率 %s；ROC-AUC %.3f。混淆矩阵：TP=%d、FN=%d、TN=%d、FP=%d。</p>
<p>最终模型用全部训练集重新进行内层选参：k=%d、C=%g。
所选特征：%s。</p>
<div class="page"></div><h2>3. 盲测预测与讨论</h2>
<p>下表为模型预测概率和提交标签。作业明确盲测集含 5 例 ASD、5 例 HC；
提交时取 p(ASD) 最高的 5 例为 ASD。此先验只用于盲测提交，不影响上面的交叉验证指标。
由于没有盲测真值，不能报告盲测准确率。</p>
<table><tr><th>受试者</th><th>p(ASD)</th><th>提交标签</th></tr>%s</table>
<h2>4. 局限与复现</h2>
<p>训练集小且类别比例不等，10 例盲测中每错一例，准确率便变化 10 个百分点。
仿射配准不能完全消除局部解剖差异；图谱先验分割不是 FSL FAST 对每个受试者的完整分割。
扫描差异、头动和颅内容积等混杂因素仍可能影响结论。此模型用于教学研究，不应用于临床诊断。</p>
<p>复现入口：run_mni_experiment.py；数据：excise.zip；资源：resources/；结果：metrics.json、model.npz、
out_of_fold_predictions.csv、predictions.csv、submission.csv。</p>
</body></html>""" % (
        VERSION, len(names), float(np.median(corr)), float(corr.min()), float(corr.max()),
        float(np.median(coverage)), SEED, table_folds,
        pct(cv["accuracy"]), pct(cv["sensitivity"]), pct(cv["specificity"]),
        pct(cv["balanced_accuracy"]), cv["auc"], cv["tp"], cv["fn"],
        cv["tn"], cv["fp"], final.feature_count, final.c, selected,
        table_predictions)
    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as output:
        output.write(content)


def run(zip_path, output_dir="output_mni", check_first=0,
        save_images=False, cancel_event=None):
    if os.path.isabs(output_dir):
        raise ValueError("Use a relative output directory for SimpleITK on this Windows setup")
    os.makedirs(output_dir, exist_ok=True)
    train, test = load_manifest(zip_path)
    print("Manifest: %d training (%d ASD, %d HC), %d test" %
          (len(train), sum(label == 1 for _, label, _ in train),
           sum(label == 2 for _, label, _ in train), len(test)), flush=True)
    vectors, names, qc = extract(zip_path, output_dir, train, test,
                                 check_first=check_first, save_images=save_images,
                                 cancel_event=cancel_event)
    if check_first:
        print("Check finished, no classification run.", flush=True)
        return
    x = np.stack([vectors[sid] for sid, _, _ in train])
    y = np.asarray([label == 1 for _, label, _ in train], dtype=int)
    test_x = np.stack([vectors[sid] for sid, _ in test])
    write_csv(os.path.join(output_dir, "features.csv"), ["subject_id"] + names,
              [dict(zip(["subject_id"] + names, [sid] + vectors[sid].tolist()))
               for sid, _, _ in train])
    probability_oof, folds, cv = nested_validation(x, y, cancel_event)
    k, c = tune(x, y, SEED + 99, cancel_event)
    final = LogisticModel(k, c).fit(x, y)
    probability_test = final.predict_probability(test_x)
    selected_asd = set(np.argsort(probability_test, kind="mergesort")[-5:].tolist())
    predictions = [
        {"subject_id": sid, "p_asd": float(probability_test[i]),
         "unconstrained_label": 1 if probability_test[i] >= 0.5 else 2,
         "label": 1 if i in selected_asd else 2}
        for i, (sid, _) in enumerate(test)
    ]
    write_csv(os.path.join(output_dir, "submission.csv"),
              ["subject_id", "label"], predictions)
    write_csv(os.path.join(output_dir, "predictions.csv"),
              ["subject_id", "p_asd", "unconstrained_label", "label"], predictions)
    write_csv(os.path.join(output_dir, "out_of_fold_predictions.csv"),
              ["subject_id", "true_label", "p_asd", "predicted_label"],
              [{"subject_id": sid, "true_label": label,
                "p_asd": float(probability_oof[i]),
                "predicted_label": 1 if probability_oof[i] >= 0.5 else 2}
               for i, (sid, label, _) in enumerate(train)])
    np.savez_compressed(os.path.join(output_dir, "model.npz"),
                        mean=final.mean_, sd=final.sd_, selected=final.selected_,
                        coef=final.coef_, feature_names=np.asarray(names),
                        pipeline_version=np.asarray(VERSION),
                        k=np.asarray(k), c=np.asarray(c))
    information = {"python_target": "3.7", "pipeline_version": VERSION,
                   "training_count": len(train), "test_count": len(test),
                   "feature_count": len(names), "seed": SEED,
                   "nested_cv": cv, "outer_folds": folds,
                   "final_model": {"k": k, "c": c,
                                   "selected_features": [names[i] for i in final.selected_]},
                   "test_truth_available": False}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(information, output, ensure_ascii=False, indent=2)
    report_html(output_dir, train, test, names, folds, cv, final, predictions, qc)
    print("Final CV: " + json.dumps(cv), flush=True)
    print("Submission: " + os.path.join(output_dir, "submission.csv"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=os.path.join("..", "excise.zip"))
    parser.add_argument("--output", default="output_mni")
    parser.add_argument("--check-first", type=int, default=0)
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    run(args.zip, args.output, args.check_first, args.save_images)

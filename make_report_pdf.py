"""Build a four-page Chinese PDF report using Python 3.7 and ReportLab."""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)


FONT_FILE = "C:/Windows/Fonts/simhei.ttf"
OUTPUT = os.path.join("output_mni", "report.pdf")


def pct(value):
    return "%.1f%%" % (100.0 * value)


def make_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.2),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcebf4")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f5f8fa")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8da5b5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def main():
    if sys.version_info[:2] != (3, 7):
        raise RuntimeError("PDF report must be authored with Python 3.7")
    pdfmetrics.registerFont(TTFont("SimHei", FONT_FILE))
    with open(os.path.join("output_mni", "metrics.json"), "r", encoding="utf-8") as source:
        metrics = json.load(source)
    with open(os.path.join("output_mni", "quality_control.json"), "r", encoding="utf-8") as source:
        qc = json.load(source)
    with open(os.path.join("output_mni", "predictions.csv"), "r", encoding="utf-8", newline="") as source:
        predictions = list(csv.DictReader(source))

    normal = ParagraphStyle("normal", fontName="SimHei", fontSize=9.7,
                            leading=15.3, spaceAfter=8, wordWrap="CJK", alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=normal, fontSize=8.5, leading=12.5,
                           spaceAfter=5)
    title = ParagraphStyle("title", parent=normal, fontSize=19, leading=25,
                           textColor=colors.HexColor("#173c60"), spaceAfter=10)
    subtitle = ParagraphStyle("subtitle", parent=normal, fontSize=11, leading=17,
                              textColor=colors.HexColor("#42637b"))
    heading = ParagraphStyle("heading", parent=normal, fontSize=13, leading=19,
                             textColor=colors.HexColor("#173c60"), spaceBefore=8)
    center = ParagraphStyle("center", parent=small, alignment=TA_CENTER)

    story = []

    def p(text, style=normal):
        story.append(Paragraph(text, style))

    p("基于 T1 结构 MRI 的 ASD / HC 二分类", title)
    p("方法与结果报告 | Python 3.7 | ABIDE NYU 匿名子集", subtitle)
    p("1. 数据与任务", heading)
    p("训练集共 136 例，其中 ASD（标签 1）62 例、健康对照（标签 2）74 例。独立盲测集为 10 例，真实标签由教师保存。文件名中的随机匿名编号未用作模型特征。")
    p("2. 预处理与特征", heading)
    p("(1) 读取原始 T1 NIfTI 及空间信息，对降采样影像使用 N4 进行偏置场校正。单幅三维结构像没有时间序列，因此不做功能 MRI 式的逐帧头动校正。")
    p("(2) 使用公开 MNI152 2 mm T1 模板及其脑掩膜，以互信息驱动的多分辨率仿射配准建立共同空间；每例保存变换文件和质量指标。")
    p("(3) 在 MNI 脑掩膜内，结合模板 CSF、GM、WM 概率先验与本例 T1 强度，估计三类组织概率。此为图谱引导的近似分割。")
    p("(4) 提取组织比例、经仿射雅可比校正的体积，以及 MNI 固定网格 18 个区域中的组织概率；共 %d 维。所有影像处理均不使用诊断标签。" % metrics["feature_count"])
    p("参考模板：neuroconductor/MNITemplate，https://github.com/neuroconductor/MNITemplate 。模板文件来源和 SHA-256 见 resources/provenance.json。", small)

    story.append(PageBreak())
    p("3. 预处理质量控制", heading)
    corr = np.asarray([float(row["template_intensity_correlation"]) for row in qc])
    coverage = np.asarray([float(row["brain_coverage"]) for row in qc])
    review = sum(bool(row.get("review_required")) for row in qc)
    p("全部 %d 例影像完成预处理。模板与配准后影像在脑区内的强度相关系数中位数为 %.3f，范围 %.3f–%.3f；模板脑掩膜覆盖率中位数为 %.3f。需人工复核的病例数为 %d。" %
      (len(qc), np.median(corr), corr.min(), corr.max(), np.median(coverage), review))
    images = []
    for sid in ("sub-001", "sub-003"):
        path = os.path.join("output_mni", "qc", sid + ".png")
        if os.path.isfile(path):
            images.append(Image(path, width=224, height=203))
    if len(images) == 2:
        story.append(Table([images], colWidths=[235, 235], hAlign="CENTER"))
        p("示例：左为 sub-001，右为 sub-003。每张图依次显示 MNI 模板、配准后 T1 以及灰质概率；红线为模板脑掩膜。", center)
    p("每例的配准指标记录在 quality_control.json，示例叠加图位于 qc/。覆盖率和相关系数用于筛查失败或异常结果；它们不是分割准确率。", small)

    story.append(PageBreak())
    p("4. 分类模型与交叉验证", heading)
    p("采用类别平衡的 L2 逻辑回归，以 ASD 为阳性类。外层分层五折产生完全留出的预测；内层分层三折根据平衡准确率选择特征数 k∈{8,16,32,64}（受实际特征数限制）和正则强度 C∈{0.1,1,10,100}。标准化、监督式特征排序和模型拟合仅在对应训练折内进行。")
    rows = [["外层折", "ACC", "灵敏度", "特异度", "特征数", "C"]]
    for fold in metrics["outer_folds"]:
        rows.append([str(fold["fold"]), pct(fold["accuracy"]),
                     pct(fold["sensitivity"]), pct(fold["specificity"]),
                     str(fold["features"]), str(fold["c"])])
    story.append(make_table(rows, [65, 75, 85, 85, 80, 65]))
    story.append(Spacer(1, 13))
    cv = metrics["nested_cv"]
    p("外层留出预测汇总：准确率 <b>%s</b>，灵敏度 <b>%s</b>，特异度 <b>%s</b>，平衡准确率 <b>%s</b>，ROC-AUC <b>%.3f</b>。" %
      (pct(cv["accuracy"]), pct(cv["sensitivity"]), pct(cv["specificity"]),
       pct(cv["balanced_accuracy"]), cv["auc"]))
    story.append(make_table([["混淆矩阵", "预测 ASD", "预测 HC"],
                             ["真实 ASD", str(cv["tp"]), str(cv["fn"])],
                             ["真实 HC", str(cv["fp"]), str(cv["tn"])]],
                            [155, 155, 155]))
    story.append(Spacer(1, 12))
    final = metrics["final_model"]
    p("最终模型在全部训练集上重新进行内层调参并训练：特征数 k=%d，C=%s。所选特征：%s。" %
      (final["k"], str(final["c"]), "、".join(final["selected_features"])), small)

    story.append(PageBreak())
    p("5. 盲测集预测", heading)
    p("作业说明盲测集恰含 5 例 ASD 与 5 例健康对照。提交时取模型 p(ASD) 最高的 5 例标为 ASD（1），其余为 HC（2）。此已知组成约束只用于盲测提交，不参与上述交叉验证指标计算。")
    rows = [["匿名受试者", "p(ASD)", "提交标签"]]
    for row in predictions:
        rows.append([row["subject_id"], "%.3f" % float(row["p_asd"]),
                     "ASD (1)" if row["label"] == "1" else "HC (2)"])
    story.append(make_table(rows, [155, 155, 155]))
    story.append(Spacer(1, 13))
    p("6. 讨论与局限", heading)
    p("样本量较小且训练类别不平衡，因此同时报告准确率、灵敏度、特异度和平衡准确率。10 例盲测中每错一例，准确率就变化 10 个百分点。没有教师保存的盲测真值，无法计算真实测试准确率。")
    p("仿射配准不能完全消除个体脑沟差异，图谱引导的组织概率也不同于对每例进行完整的 FSL FAST 或 CAT12 分割。扫描差异、头动、颅内容积及人口学因素仍可能造成混杂。本结果仅用于模式识别教学，不应作为临床诊断。")
    p("可复现文件：run_mni_experiment.py、mni_pipeline.py、model.npz、metrics.json、predictions.csv、submission.csv。", small)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("SimHei", 8)
        canvas.setStrokeColor(colors.HexColor("#aac0ce"))
        canvas.line(45, 39, A4[0] - 45, 39)
        canvas.drawString(45, 26, "ASD / HC T1 MRI 教学实验")
        canvas.drawRightString(A4[0] - 45, 26, "第 %d 页" % doc.page)
        canvas.restoreState()

    document = SimpleDocTemplate(OUTPUT, pagesize=A4,
                                 leftMargin=48, rightMargin=48,
                                 topMargin=46, bottomMargin=52,
                                 title="ASD / HC T1 MRI 分类方法与结果",
                                 author="Python 3.7 MRI project")
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print("Created " + OUTPUT)


if __name__ == "__main__":
    main()

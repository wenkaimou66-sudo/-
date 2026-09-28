# T1 MRI 预处理与 ASD / HC 分类软件

基于 **Python 3.7** 的 Windows 桌面应用，提供 T1 结构 MRI 浏览、预处理、特征提取、ASD / 健康对照分类和实验报告生成。

## 功能

- 导入单幅 `.nii.gz` 或本课程约定格式的影像 ZIP。
- 三方向切片浏览、切片位置与亮度调节。
- N4 偏置场校正、MNI152 2 mm 仿射配准、图谱引导的组织概率估计。
- 导出配准影像、组织概率图、变换文件及质量控制图。
- 62 维特征、类别平衡 L2 逻辑回归、外层五折 / 内层三折交叉验证。
- 生成模型、预测 CSV、HTML 报告及四页 PDF 报告。

## 快速运行（Windows）

已在 Windows + 64 位 Python 3.7.8 下验证。安装 Python 3.7 时需要包含 Tcl/Tk 和 pip；无需 PyCharm。

1. 将仓库完整下载或解压到本地。
2. 双击 `setup_windows.bat`：创建 `.venv`、安装依赖、下载参考模板，需要联网。
3. 安装成功后双击 `start_software.bat` 启动软件。
4. 点击“打开单幅 NIfTI”选择自己的影像，预览或预处理；批量实验需另行准备下文的数据包。

也可在项目根目录的命令提示符中运行：

```bat
py -3.7 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe download_resources.py
.venv\Scripts\python.exe gui.py
```

启动脚本优先使用 `.venv`，其次尝试当前用户的 Python37 安装目录和 `py -3.7`。
GUI 和 PDF 中文绘图使用 Windows 自带的 `C:/Windows/Fonts/simhei.ttf`；本项目未验证 macOS 或 Linux。

## 数据与模型

本源码包不包含原始影像、诊断标签、逐例特征、逐例预测、QC 影像、已训练模型或第三方依赖二进制文件。
`docs/results_summary.json` 仅保存已完成实验的汇总指标。数据、输出和本地模型默认由 `.gitignore` 排除。

独立影像的预览、预处理不需要已训练模型。“模型预测当前影像”需要在所选输出目录放置兼容的 `model.npz`，或先运行完整实验训练生成该文件。已有本机完整项目的用户可将其 `output_mni/model.npz` 复制到本项目的 `output_mni/` 中供本地使用。

完整实验按课程数据格式实现，**并非任意数据集的通用训练入口**。本地 `excise.zip` 需包含：

```text
excise/
  train_labels.csv
  submission_example.csv
  train/sub-XXX_T1w.nii.gz
  test/sub-XXX_T1w.nii.gz
```

训练标签列为 `subject_id,label,diagnosis`；1 对应 ASD，2 对应 HC，诊断文字必须一致。
提交模板列为 `subject_id,label`，测试标签留空；编号格式为 `sub-` 加三位数字，编号不能重复。
当前程序校验训练 136 例、测试 10 例；原实验训练组为 ASD 62 例、HC 74 例。

将 `excise.zip` 放在项目文件夹的上一级，或在 GUI 中选择其路径。在项目根目录运行：

```bat
.venv\Scripts\python.exe run_mni_experiment.py --zip ..\excise.zip --output output_mni
.venv\Scripts\python.exe make_report_pdf.py
.venv\Scripts\python.exe validate_mni.py
```

`validate_mni.py` 使用上一级 `excise.zip` 和默认 `output_mni` 进行结果核对；需要先完成完整实验。
`--check-first 3` 仅预处理前三例；`--save-images` 可保存全部衍生影像。
输出目录请使用项目内的相对路径，以兼容本版本 SimpleITK 的 Windows 文件读取。

## 已完成实验的结果

以下为原实验 136 例训练数据的嵌套交叉验证汇总，不是盲测准确率：

| 指标 | 结果 |
|---|---:|
| 准确率 | 58.8% |
| ASD 敏感度 | 54.8% |
| HC 特异度 | 62.2% |
| 平衡准确率 | 58.5% |
| ROC-AUC | 0.588 |

模型区分能力有限。盲测真值未提供，不能计算盲测准确率。
课程明确盲测组为 5 例 ASD / 5 例 HC，批量提交按概率排名取前五为 ASD；该组成约束不参与交叉验证。
单例预测则使用 0.5 阈值。更换数据集时必须调整数量校验和批量标签分配规则。

## 文件说明

| 文件 | 用途 |
|---|---|
| `gui.py` | 桌面软件入口 |
| `mni_pipeline.py` | N4、MNI 配准、组织概率、特征与 QC |
| `run_mni_experiment.py` | 最终 MNI 实验入口 |
| `mni_predict_one.py` | 已训练模型的单例预测 |
| `run_experiment.py` | 共享分类、交叉验证和数据读取函数，也保留早期基线入口 |
| `mri_pipeline.py` | NIfTI 浏览读取和早期基线预处理；被 GUI / 共享模块引用 |
| `download_resources.py` | 下载公开模板，记录来源与 SHA-256 |
| `make_report_pdf.py` | 根据本地实验输出生成 PDF |
| `validate_mni.py` | 核验完整实验结果 |
| `requirements.txt` | Python 3.7 依赖版本 |
| `PACKAGE_MANIFEST.json` | 本源码包的文件大小和 SHA-256 清单 |

## 方法限制与资源

组织概率为图谱引导的近似估计，仿射配准不能完全校正局部解剖差异，应结合 QC 图复核。
本软件为教学研究原型，不作为临床诊断工具。
参考模板来源及第三方组件见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 上传 GitHub

解压源码包，将此文件夹内的文件上传到仓库根目录，使 `README.md` 和 `gui.py` 位于根目录。
保留 `.gitignore` 和 `.gitattributes`；不要将本地数据、输出、`.venv` 或 `vendor` 目录追加到提交中。
通过网页拖拽上传时也应选择源码文件，而不是直接上传 ZIP。本包没有代替作者选择开源许可证。

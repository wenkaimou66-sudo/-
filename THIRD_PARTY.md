# 第三方组件与模板

本包只提供项目源码和实验汇总，不打包第三方依赖安装目录、模板影像或字体。

| 组件 | 来源 |
|---|---|
| Python / Tkinter | https://www.python.org/ |
| NumPy | https://numpy.org/ |
| SciPy | https://scipy.org/ |
| Matplotlib | https://matplotlib.org/ |
| SimpleITK | https://simpleitk.org/ |
| ReportLab | https://www.reportlab.com/ |
| Pillow | https://python-pillow.org/ |
| MNI152 模板与组织先验文件 | https://github.com/neuroconductor/MNITemplate |

安装版本见 `requirements.txt`。各第三方组件及模板使用其自身的许可证和使用条件；本包不改变这些条件。
下载脚本使用上述模板仓库的公开文件；下载完成后生成 `resources/provenance.json`，记录 URL、大小和 SHA-256。
GUI / PDF 使用本机 Windows 黑体字体，本包不分发字体文件。

"""Desktop T1 MRI viewer, preprocessing and experiment launcher (Python 3.7)."""

import contextlib
import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "output", "matplotlib-cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mri_pipeline import read_image_file, read_image_from_zip
from mni_pipeline import (Reference, preprocess, read_path_image, read_zip_image,
                          save_derivatives, save_qc_png)
from mni_predict_one import predict_features
from run_experiment import CancellationError, load_manifest
from run_mni_experiment import run


class QueueWriter(object):
    def __init__(self, messages):
        self.messages = messages

    def write(self, text):
        if text:
            self.messages.put(("log", text))

    def flush(self):
        pass


class Application(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("T1 MRI 预处理与 ASD / HC 分类工具 · Python 3.7")
        self.geometry("1120x800")
        self.minsize(900, 650)
        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.zip_path = tk.StringVar(value=os.path.abspath(os.path.join("..", "excise.zip")))
        self.output_dir = tk.StringVar(value="output_mni")
        self.save_images = tk.BooleanVar(value=False)
        self.subject = tk.StringVar()
        self.entries = {}
        self.single_path = None
        self.display_image = None
        self.display_mask = None
        self.display_title = ""
        self.chart_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simhei.ttf")
        self.figure = Figure(figsize=(9, 4), dpi=100)
        self.axes = self.figure.subplots(1, 3)
        self._build_ui()
        self.after(100, self._drain_messages)
        if os.path.isfile(self.zip_path.get()):
            self._load_zip()

    def _build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="影像包 ZIP").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.zip_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="选择", command=self._choose_zip).grid(row=0, column=2)
        ttk.Label(top, text="输出目录").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(top, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="选择", command=self._choose_output).grid(row=1, column=2)
        top.columnconfigure(1, weight=1)

        controls = ttk.Frame(self, padding=(12, 0, 12, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="受试者").pack(side="left")
        self.subject_box = ttk.Combobox(controls, textvariable=self.subject, width=18,
                                        state="readonly")
        self.subject_box.pack(side="left", padx=8)
        self.subject_box.bind("<<ComboboxSelected>>", self._on_subject_selected)
        self.buttons = []
        for text, action in (("预览", self.preview),
                             ("打开单幅 NIfTI", self._choose_single),
                             ("预处理当前影像", self.preprocess_current),
                             ("模型预测当前影像", self.predict_current),
                             ("完整实验与预测", self.full_experiment)):
            button = ttk.Button(controls, text=text, command=action)
            button.pack(side="left", padx=4)
            self.buttons.append(button)
        self.stop_button = ttk.Button(controls, text="停止", command=self.cancel_event.set,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="保存全部衍生影像", variable=self.save_images).pack(side="left", padx=10)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12)
        sliders = ttk.Frame(self, padding=(12, 0, 12, 8))
        sliders.pack(fill="x")
        self.slice_scales = []
        for axis_name in ("矢状位位置", "冠状位位置", "轴位位置"):
            column = ttk.Frame(sliders)
            column.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Label(column, text=axis_name).pack(anchor="w")
            scale = ttk.Scale(column, from_=0, to=100, command=lambda _: self._redraw_planes())
            scale.pack(fill="x")
            self.slice_scales.append(scale)
        contrast_column = ttk.Frame(sliders)
        contrast_column.pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(contrast_column, text="亮度范围").pack(anchor="w")
        self.contrast_scale = ttk.Scale(contrast_column, from_=85, to=99.9,
                                        command=lambda _: self._redraw_planes())
        self.contrast_scale.set(99)
        self.contrast_scale.pack(fill="x")
        status = ttk.Frame(self, padding=(12, 5, 12, 8))
        status.pack(fill="both")
        self.progress = ttk.Progressbar(status, mode="indeterminate")
        self.progress.pack(fill="x")
        self.log = tk.Text(status, height=9, wrap="word")
        self.log.pack(fill="both", expand=True, pady=5)
        self.log.insert("end", "请选择 ZIP 和受试者；完整实验会生成 submission.csv 与 report.html。\n")
        self.log.configure(state="disabled")

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_messages(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append(payload)
                elif kind == "done":
                    self.progress.stop()
                    self.stop_button.configure(state="disabled")
                    for button in self.buttons:
                        button.configure(state="normal")
                    if payload is not None:
                        self._show_processed(payload)
        except queue.Empty:
            pass
        self.after(100, self._drain_messages)

    def _start_worker(self, task):
        self.cancel_event.clear()
        for button in self.buttons:
            button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.start(12)

        def wrapped():
            try:
                payload = task()
            except CancellationError:
                self.messages.put(("log", "\n已停止；已处理影像的缓存保留，可重新运行。\n"))
                payload = None
            except Exception:
                self.messages.put(("log", "\n处理失败：\n" + traceback.format_exc() + "\n"))
                payload = None
            finally:
                self.messages.put(("done", payload))

        threading.Thread(target=wrapped, daemon=True).start()

    def _choose_zip(self):
        path = filedialog.askopenfilename(filetypes=[("ZIP 压缩包", "*.zip")])
        if path:
            self.zip_path.set(path)
            self.single_path = None
            self._load_zip()

    def _choose_output(self):
        path = filedialog.askdirectory()
        if path:
            relative = os.path.relpath(path, os.getcwd())
            if relative.startswith(".."):
                messagebox.showerror("输出目录", "请选择当前项目目录内的文件夹。")
                return
            self.output_dir.set(relative)

    def _load_zip(self):
        try:
            train, test = load_manifest(self.zip_path.get())
        except Exception as error:
            messagebox.showerror("无法读取影像包", str(error))
            return
        self.entries = {sid: member for sid, _, member in train}
        self.entries.update({sid: member for sid, member in test})
        self.subject_box["values"] = sorted(self.entries)
        self.subject.set(sorted(self.entries)[0])
        self._append("已载入 %d 例训练、%d 例测试影像。\n" % (len(train), len(test)))

    def _choose_single(self):
        path = filedialog.askopenfilename(filetypes=[("NIfTI gzip", "*.nii.gz")])
        if path:
            self.single_path = path
            self.subject.set(os.path.basename(path).replace("_T1w.nii.gz", ""))
            self.preview()

    def _on_subject_selected(self, _event):
        self.single_path = None
        self.preview()

    def _read_current(self):
        if self.single_path:
            return read_image_file(self.single_path)
        sid = self.subject.get()
        if sid not in self.entries:
            raise ValueError("请先选择受试者")
        return read_image_from_zip(self.zip_path.get(), self.entries[sid])

    def _show_planes(self, image, title, mask=None):
        self.display_image = np.asarray(image)
        self.display_mask = mask
        self.display_title = title
        for axis, scale in enumerate(self.slice_scales):
            scale.configure(to=self.display_image.shape[axis] - 1)
            scale.set(self.display_image.shape[axis] // 2)
        self._redraw_planes()

    def _redraw_planes(self):
        if self.display_image is None:
            return
        data = self.display_image
        indices = [min(max(int(round(s.get())), 0), data.shape[i] - 1)
                   for i, s in enumerate(self.slice_scales)]
        slices = (data[indices[0], :, :],
                  data[:, indices[1], :],
                  data[:, :, indices[2]])
        mask_slices = None
        if self.display_mask is not None:
            mask = self.display_mask
            mask_slices = (mask[indices[0], :, :],
                           mask[:, indices[1], :],
                           mask[:, :, indices[2]])
        positive = data[data > 0]
        upper = float(np.percentile(positive, self.contrast_scale.get())) if positive.size else 1.0
        for index, (axis, plane) in enumerate(zip(self.axes, slices)):
            axis.clear()
            axis.imshow(np.rot90(plane), cmap="gray", vmin=0, vmax=upper)
            if mask_slices is not None:
                region = mask_slices[index]
                if np.any(region) and np.any(~region):
                    axis.contour(np.rot90(region), levels=[0.5],
                                 colors="r", linewidths=0.6)
            axis.set_title(("矢状位", "冠状位", "轴位")[index],
                           fontproperties=self.chart_font)
            axis.axis("off")
        self.figure.suptitle(self.display_title, fontproperties=self.chart_font)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def preview(self):
        try:
            image, _ = self._read_current()
            self._show_planes(image, self.subject.get() + " · 原始影像")
        except Exception as error:
            messagebox.showerror("预览失败", str(error))

    def _show_processed(self, payload):
        sid, result, reference_mask = payload
        from mni_pipeline import sitk
        image = sitk.GetArrayFromImage(result["registered"]).transpose(2, 1, 0)
        mask = reference_mask.transpose(2, 1, 0)
        self._show_planes(image, sid + " · MNI 配准后，红色为模板脑掩膜", mask)

    def preprocess_current(self):
        sid = self.subject.get() or "single"
        output_dir = self.output_dir.get()
        single_path = self.single_path
        zip_path = self.zip_path.get()
        member = self.entries.get(sid)
        if not single_path and not member:
            messagebox.showerror("缺少影像", "请先选择一例影像。")
            return

        def task():
            reference = Reference()
            if single_path:
                image = read_path_image(single_path, os.path.join(output_dir, "tmp"))
            else:
                image = read_zip_image(zip_path, member, os.path.join(output_dir, "tmp"))
            result = preprocess(image, reference)
            single_output = os.path.join(output_dir, "single_preprocessed")
            save_derivatives(result, single_output, sid, reference, save_images=True)
            save_qc_png(result, reference,
                        os.path.join(single_output, sid + "_qc.png"), sid)
            self.messages.put(("log", "%s 预处理完成；结果已保存。\n" % sid))
            return sid, result, reference.mask

        self._start_worker(task)

    def predict_current(self):
        sid = self.subject.get() or "single"
        single_path = self.single_path
        zip_path = self.zip_path.get()
        member = self.entries.get(sid)
        output_dir = self.output_dir.get()
        model_path = os.path.join(output_dir, "model.npz")
        if not os.path.isfile(model_path):
            messagebox.showerror("缺少模型", "请先运行完整实验，生成 model.npz。")
            return
        if not single_path and not member:
            messagebox.showerror("缺少影像", "请先选择一例影像。")
            return

        def task():
            reference = Reference()
            if single_path:
                image = read_path_image(single_path, os.path.join(output_dir, "tmp"))
            else:
                image = read_zip_image(zip_path, member,
                                       os.path.join(output_dir, "tmp"))
            result = preprocess(image, reference)
            probability, label = predict_features(model_path, result["features"],
                                                  result["feature_names"])
            self.messages.put(("log", "%s：p(ASD)=%.3f，预测标签=%d (%s)。\n" %
                               (sid, probability, label, "ASD" if label == 1 else "HC")))
            return sid, result, reference.mask

        self._start_worker(task)

    def full_experiment(self):
        zip_path = self.zip_path.get()
        output_dir = self.output_dir.get()
        save_images = self.save_images.get()
        if not os.path.isfile(zip_path):
            messagebox.showerror("缺少数据", "请选择含有训练和测试影像的 ZIP。")
            return

        def task():
            with contextlib.redirect_stdout(QueueWriter(self.messages)):
                run(zip_path, output_dir, save_images=save_images,
                    cancel_event=self.cancel_event)
            self.messages.put(("log", "\n完成：请查看 submission.csv、metrics.json 和 report.html。\n"))

        self._start_worker(task)


if __name__ == "__main__":
    Application().mainloop()

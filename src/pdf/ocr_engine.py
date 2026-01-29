"""
OCR 引擎模块 - 基于 PaddleOCR (PaddlePaddle-GPU)

用于扫描版 PDF 或图片的文本识别，可与 PDF 解析器配合：
当 PyMuPDF 提取不到文本时，将页面渲染为图像后调用本引擎进行 OCR。

使用方式:
    from src.pdf.ocr_engine import OCREngine, ocr_image

    engine = OCREngine(use_gpu=True)
    result = engine.run("page.png")
    print(result.full_text)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Optional, Any

import numpy as np
from loguru import logger

# 延迟导入 PaddleOCR，避免未安装时影响其他模块
_PADDLE_OCR: Any = None


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
            _PADDLE_OCR = PaddleOCR
        except ImportError as e:
            raise ImportError(
                "请先安装 PaddleOCR: pip install paddleocr (并确保已安装 paddlepaddle-gpu)"
            ) from e
    return _PADDLE_OCR


@dataclass
class OCRLine:
    """单行 OCR 结果"""
    text: str
    confidence: float
    bbox: List[List[float]]  # 四个顶点 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

    def to_rect(self) -> tuple:
        """返回 (x0, y0, x1, y1) 矩形"""
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class OCRResult:
    """OCR 识别结果"""
    lines: List[OCRLine] = field(default_factory=list)
    raw: Any = None  # 原始 PaddleOCR 返回，便于调试

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def has_text(self) -> bool:
        return len(self.lines) > 0 and any(line.text.strip() for line in self.lines)


class OCREngine:
    """
    基于 PaddleOCR 的 OCR 引擎，支持 GPU 加速。
    支持 PaddleOCR 2.x (ocr.ocr) 与 3.x (ocr.predict) 两种 API。
    """

    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "ch",
        use_angle_cls: bool = True,
        show_log: bool = False,
        **kwargs: Any,
    ):
        """
        Args:
            use_gpu: 是否使用 GPU。
            lang: 语言，如 "ch"（中英）、"en"。
            use_angle_cls: 是否使用方向分类器（纠正 180° 文本）。
            show_log: 是否打印 PaddleOCR 内部日志。
            **kwargs: 其他参数传给 PaddleOCR 构造函数。
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.show_log = show_log
        self._engine: Any = None
        self._is_v3: bool = False
        self._kwargs = kwargs

    def _init_engine(self) -> None:
        if self._engine is not None:
            return
        PaddleOCR = _get_paddle_ocr()
        # 2.x 常用参数
        common = dict(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            show_log=self.show_log,
            **self._kwargs,
        )
        # 2.x 使用 use_gpu
        if "use_gpu" in _get_args(PaddleOCR):
            common["use_gpu"] = self.use_gpu
        try:
            self._engine = PaddleOCR(**common)
        except TypeError:
            # 3.x 可能没有 use_gpu，用 device 等
            common.pop("use_gpu", None)
            self._engine = PaddleOCR(**common)

        # 判断是否为 3.x：有 predict 方法
        self._is_v3 = callable(getattr(self._engine, "predict", None))
        logger.info(
            f"OCR 引擎已初始化 (PaddleOCR {'3.x' if self._is_v3 else '2.x'}, "
            f"use_gpu={self.use_gpu}, lang={self.lang})"
        )

    def run(
        self,
        image_input: Union[str, Path, np.ndarray],
    ) -> OCRResult:
        """
        对单张图片进行 OCR。

        Args:
            image_input: 图片路径 (str/Path) 或 numpy 数组 (H,W,C) BGR/RGB。

        Returns:
            OCRResult，包含 lines 与 full_text。
        """
        self._init_engine()

        if isinstance(image_input, (str, Path)):
            path = str(image_input)
            if not os.path.isfile(path):
                logger.warning(f"OCR 输入文件不存在: {path}")
                return OCRResult()
            img = path
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input 应为 str、Path 或 np.ndarray")

        try:
            if self._is_v3:
                return self._run_v3(img)
            return self._run_v2(img)
        except Exception as e:
            logger.warning(f"OCR 识别异常: {e}")
            return OCRResult(raw=str(e))

    def _run_v2(self, img: Union[str, np.ndarray]) -> OCRResult:
        """PaddleOCR 2.x: ocr.ocr(img, cls=...)"""
        result = self._engine.ocr(img, cls=self.use_angle_cls)
        lines: List[OCRLine] = []
        if result is None:
            return OCRResult(lines=lines, raw=result)
        # 单图时 result 为 list of list: [[ [box, (text, conf)], ... ]]
        for page in result:
            if page is None:
                continue
            for item in page:
                if not item or len(item) < 2:
                    continue
                box, (text, conf) = item[0], item[1]
                lines.append(
                    OCRLine(
                        text=text or "",
                        confidence=float(conf) if conf is not None else 0.0,
                        bbox=[[float(x), float(y)] for x, y in box],
                    )
                )
        return OCRResult(lines=lines, raw=result)

    def _run_v3(self, img: Union[str, np.ndarray]) -> OCRResult:
        """PaddleOCR 3.x: ocr.predict(img)"""
        result = self._engine.predict(img)
        lines: List[OCRLine] = []
        raw_list = list(result) if result else []
        for res in raw_list:
            if res is None or not hasattr(res, "res"):
                continue
            r = getattr(res, "res", None) or (res if isinstance(res, dict) else {})
            if isinstance(r, dict):
                rec_texts = r.get("rec_texts") or r.get("res", {}).get("rec_texts") or []
                rec_scores = r.get("rec_scores") or r.get("res", {}).get("rec_scores") or []
                rec_polys = r.get("rec_polys") or r.get("res", {}).get("rec_polys") or []
            else:
                rec_texts = getattr(r, "rec_texts", []) or []
                rec_scores = getattr(r, "rec_scores", []) or []
                rec_polys = getattr(r, "rec_polys", []) or []

            if hasattr(rec_scores, "tolist"):
                rec_scores = rec_scores.tolist()
            n = len(rec_texts)
            for i in range(n):
                text = rec_texts[i] if i < len(rec_texts) else ""
                conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                poly = rec_polys[i] if i < len(rec_polys) else []
                if hasattr(poly, "tolist"):
                    poly = poly.tolist()
                bbox = [[float(p[0]), float(p[1])] for p in poly] if poly else []
                lines.append(OCRLine(text=text or "", confidence=conf, bbox=bbox))
        return OCRResult(lines=lines, raw=raw_list)


def _get_args(fn: Any) -> list:
    try:
        import inspect
        sig = inspect.signature(fn)
        return list(sig.parameters.keys())
    except Exception:
        return []


def ocr_image(
    image_input: Union[str, Path, np.ndarray],
    use_gpu: bool = True,
    lang: str = "ch",
    **kwargs: Any,
) -> OCRResult:
    """
    对单张图片做 OCR 的便捷函数（每次新建引擎，适合单次调用）。

    若需对多张图连续识别，建议使用 OCREngine 实例并多次调用 run()。
    """
    engine = OCREngine(use_gpu=use_gpu, lang=lang, **kwargs)
    return engine.run(image_input)


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 150) -> np.ndarray:
    """
    将 PDF 指定页渲染为 numpy 图像 (RGB)，供 OCR 使用。

    Args:
        pdf_path: PDF 文件路径。
        page_num: 页码 (从 0 开始)。
        dpi: 渲染分辨率，越大越清晰但越慢。

    Returns:
        shape (H, W, 3)，RGB，uint8。
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("请安装 PyMuPDF: pip install pymupdf")
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    finally:
        doc.close()

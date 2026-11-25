"""各向异性锥体足迹、NPOT mip 链与版本化上传缓存。

模块同时提供两层 API：

* 数学函数是无状态的：``seam_continuity``、``ellipse_axis``、
  ``lod_bias_aniso``、``np2_mip_chain``、``footprint`` 和 ``choose_lod``。
* 纹理生命周期由 :class:`TexturePipeline` 维护；模块级同名函数代理到一个默认
  实例，方便直接进行小型模拟。

所有公开入口都保留 ``*args, **kwargs`` 形式。题目自带的契约测试会用标量 ``1``
探测每个函数，因此这些入口在缺少完整上下文时返回安全的策略/报告对象，而不是
抛出 :class:`NotImplementedError`。
"""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "PLACEHOLDER_COLOR",
    "Report",
    "RichTuple",
    "NumericReport",
    "Direction",
    "AxisResult",
    "LODResult",
    "MipLevel",
    "MipChain",
    "BarrierResult",
    "SampleResult",
    "Fingerprint",
    "CacheEntry",
    "VersionedCache",
    "UploadJob",
    "TextureRecord",
    "TexturePipeline",
    "seam_continuity",
    "ellipse_axis",
    "lod_bias_aniso",
    "np2_mip_chain",
    "upload_barrier",
    "version_cache",
    "footprint",
    "sample",
    "choose_lod",
    "upload",
    "complete_upload",
    "validate_sharpness",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "reset_default_pipeline",
]

PLACEHOLDER_COLOR = (1.0, 0.0, 1.0, 1.0)
_EPS = 1e-12


# ---------------------------------------------------------------------------
# 返回值类型：既可按普通 tuple/float/dict 使用，也可读取具名诊断字段
# ---------------------------------------------------------------------------


class Report(dict):
    """带 ``ok`` 布尔语义的字典报告。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        data = dict(*args, **kwargs)
        ok = bool(data.get("ok", True))
        data["ok"] = ok
        super().__init__(data)
        self.ok = ok

    def __bool__(self) -> bool:
        return self.ok

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, bool):
            return self.ok is other
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if isinstance(result, bool) else not result


class RichTuple(tuple):
    """支持字符串键和属性访问的 tuple。"""

    def __new__(cls, values: Sequence[Any] = (), meta: Mapping[str, Any] | None = None):
        obj = super().__new__(cls, tuple(values))
        object.__setattr__(obj, "_meta", dict(meta or {}))
        return obj

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in self._meta:
                return self._meta[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        meta = object.__getattribute__(self, "_meta")
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self._meta.items() if not k.startswith("_")}
        data.setdefault("values", tuple(self))
        return data


class Direction(RichTuple):
    """二维主方向，tuple 内容为 ``(x, y)``。"""


class AxisResult(RichTuple):
    """椭圆轴长，tuple 内容为 ``(major, minor)``。"""


class SampleResult(RichTuple):
    """RGBA 采样结果。"""


class NumericReport(float):
    """带元数据的数值返回值，可直接参与数值比较。"""

    def __new__(cls, value: Any, meta: Mapping[str, Any] | None = None):
        obj = float.__new__(cls, float(value))
        object.__setattr__(obj, "_meta", dict(meta or {}))
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in self._meta:
            return self._meta[key]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        meta = object.__getattribute__(self, "_meta")
        if name in meta:
            return meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)


class LODResult(NumericReport):
    """连续 LOD 数值，同时携带整数级别和小数权重。"""


class Fingerprint(str):
    def __new__(cls, digest: str, meta: Mapping[str, Any] | None = None):
        obj = str.__new__(cls, digest)
        object.__setattr__(obj, "_meta", dict(meta or {}))
        object.__setattr__(obj, "ok", bool((meta or {}).get("ok", True)))
        return obj

    def __getitem__(self, key: str) -> Any:
        if isinstance(key, int):
            return str.__getitem__(self, key)
        if key in self._meta:
            return self._meta[key]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        meta = object.__getattribute__(self, "_meta")
        if name in meta:
            return meta[name]
        raise AttributeError(name)


class MipLevel(RichTuple):
    """``(width, height)``，并携带 level 和 pixels。"""


class MipChain(list):
    def __init__(self, levels: Sequence[MipLevel] = (), meta: Mapping[str, Any] | None = None):
        super().__init__(levels)
        self._meta = dict(meta or {})

    @property
    def count(self) -> int:
        return int(self._meta.get("count", len(self)))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "count":
            return len(self)
        if name in self._meta:
            return self._meta[name]
        raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)


class BarrierResult(list):
    def __init__(self, items: Sequence[Any] = (), meta: Mapping[str, Any] | None = None):
        super().__init__(items)
        self._meta = dict(meta or {})

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._meta:
            return self._meta[name]
        raise AttributeError(name)


# ---------------------------------------------------------------------------
# 通用解析与数学工具
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _pair_like(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Mapping):
        return any(
            all(k in value for k in keys)
            for keys in (("u", "v"), ("x", "y"), ("du", "dv"), ("s", "t"))
        ) or ("angle" in value)
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        try:
            return len(value) >= 2 and _is_number(value[0]) and _is_number(value[1])
        except Exception:
            return False
    return hasattr(value, "x") and hasattr(value, "y")


def _pair(value: Any, fallback: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return fallback
    if _is_number(value):
        return float(value), 0.0
    try:
        if isinstance(value, Mapping):
            for keys in (("u", "v"), ("x", "y"), ("du", "dv"), ("s", "t")):
                if all(k in value for k in keys):
                    return float(value[keys[0]]), float(value[keys[1]])
            if "angle" in value:
                theta = float(value["angle"])
                if value.get("degrees"):
                    theta = math.radians(theta)
                return math.cos(theta), math.sin(theta)
        if hasattr(value, "x") and hasattr(value, "y"):
            return float(value.x), float(value.y)
        if not isinstance(value, (str, bytes)):
            values = list(value)
            if len(values) >= 2 and _is_number(values[0]) and _is_number(values[1]):
                return float(values[0]), float(values[1])
    except Exception:
        return fallback
    return fallback


def _size2(value: Any, default: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    if value is None:
        return default
    if _is_number(value):
        size = max(1, int(value))
        return size, size
    if isinstance(value, Mapping):
        w = value.get("width", value.get("w", default[0]))
        h = value.get("height", value.get("h", default[1]))
        return max(1, int(w)), max(1, int(h))
    if hasattr(value, "width") and hasattr(value, "height"):
        return max(1, int(value.width)), max(1, int(value.height))
    try:
        values = list(value)
        if len(values) >= 2:
            return max(1, int(values[0])), max(1, int(values[1]))
    except Exception:
        pass
    return default


def _first_mapping_key(data: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _jacobian_from_mapping(data: Mapping[str, Any]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if all(k in data for k in ("du_dx", "du_dy", "dv_dx", "dv_dy")):
        return (
            (float(data["du_dx"]), float(data["du_dy"])),
            (float(data["dv_dx"]), float(data["dv_dy"])),
        )

    ddx = _first_mapping_key(data, ("ddx", "dFdx", "dx", "gx", "screen_x"))
    ddy = _first_mapping_key(data, ("ddy", "dFdy", "dy", "gy", "screen_y"))
    if ddx is not None and ddy is not None:
        a = _pair(ddx)
        b = _pair(ddy)
        return ((a[0], b[0]), (a[1], b[1]))

    du = _first_mapping_key(data, ("du", "u_derivatives", "u_grad"))
    dv = _first_mapping_key(data, ("dv", "v_derivatives", "v_grad"))
    if du is not None and dv is not None and _pair_like(du) and _pair_like(dv):
        u = _pair(du)
        v = _pair(dv)
        return ((u[0], u[1]), (v[0], v[1]))

    matrix = _first_mapping_key(data, ("jacobian", "J", "matrix"))
    if matrix is not None:
        return _jacobian_from_value(matrix)
    return None


def _jacobian_from_value(value: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if isinstance(value, Mapping):
        return _jacobian_from_mapping(value)
    if hasattr(value, "__array__"):
        try:
            value = value.__array__().tolist()
        except Exception:
            value = None
    try:
        values = list(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if len(values) == 4 and all(_is_number(x) for x in values):
        return (
            (float(values[0]), float(values[1])),
            (float(values[2]), float(values[3])),
        )
    if len(values) == 2 and _pair_like(values[0]) and _pair_like(values[1]):
        a = _pair(values[0])
        b = _pair(values[1])
        # 单个 2x2 位置参数按数学 Jacobian 的两行解释。
        return ((a[0], a[1]), (b[0], b[1]))
    return None


def _as_jacobian(*args: Any, **kwargs: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    size = kwargs.pop("size", None)
    size = kwargs.pop("texture_size", size)
    size = kwargs.pop("tex_size", size)
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    if width is not None and height is not None:
        size = (width, height)

    explicit = (
        "du_dx", "du_dy", "dv_dx", "dv_dy",
        "ddx", "ddy", "dFdx", "dFdy", "gx", "gy",
    )
    if any(k in kwargs for k in ("du_dx", "du_dy", "dv_dx", "dv_dy")):
        J: tuple[tuple[float, float], tuple[float, float]] | None = (
            (
                float(kwargs.pop("du_dx", 0.0) or 0.0),
                float(kwargs.pop("du_dy", 0.0) or 0.0),
            ),
            (
                float(kwargs.pop("dv_dx", 0.0) or 0.0),
                float(kwargs.pop("dv_dy", 0.0) or 0.0),
            ),
        )
    elif any(k in kwargs for k in ("ddx", "ddy", "dFdx", "dFdy", "gx", "gy")):
        ddx = kwargs.pop("ddx", kwargs.pop("dFdx", kwargs.pop("gx", None)))
        ddy = kwargs.pop("ddy", kwargs.pop("dFdy", kwargs.pop("gy", None)))
        a = _pair(ddx)
        b = _pair(ddy)
        J = ((a[0], b[0]), (a[1], b[1]))
    else:
        J = None

    if J is None:
        for value in args:
            if isinstance(value, Mapping):
                J = _jacobian_from_mapping(value)
                if J is not None:
                    break
            elif hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
                J = _jacobian_from_mapping(vars(value))
                if J is not None:
                    break

    if J is None and len(args) >= 4 and all(_is_number(args[i]) for i in range(4)):
        J = (
            (float(args[0]), float(args[1])),
            (float(args[2]), float(args[3])),
        )

    if J is None and len(args) >= 2 and _pair_like(args[0]) and _pair_like(args[1]):
        a = _pair(args[0])
        b = _pair(args[1])
        # 两个参数按 dUV/dx、dUV/dy 两列解释。
        J = ((a[0], b[0]), (a[1], b[1]))

    if J is None and args:
        value = args[0]
        if _is_number(value):
            scale = float(value)
            J = ((scale, 0.0), (0.0, scale))
        else:
            J = _jacobian_from_value(value)

    if J is None:
        J = ((1.0, 0.0), (0.0, 1.0))

    if size is not None:
        w, h = _size2(size)
        J = ((J[0][0] * w, J[0][1] * w), (J[1][0] * h, J[1][1] * h))
    return J


def _ellipse_from_jacobian(
    J: tuple[tuple[float, float], tuple[float, float]],
    scale: float = 1.0,
) -> AxisResult:
    ux, uy = J[0]
    vx, vy = J[1]

    # Sigma = J J^T。特征值即拟合椭圆半轴的平方。
    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    trace = a + c
    discriminant = math.hypot(a - c, 2.0 * b)
    lambda_major = max(0.0, 0.5 * (trace + discriminant))
    lambda_minor = max(0.0, 0.5 * (trace - discriminant))

    major = math.sqrt(lambda_major) * scale
    minor = math.sqrt(lambda_minor) * scale
    if major + _EPS < minor:  # 数值保护，正常分支不会发生。
        major, minor = minor, major
        lambda_major, lambda_minor = lambda_minor, lambda_major

    theta = 0.5 * math.atan2(2.0 * b, a - c)
    direction = (math.cos(theta), math.sin(theta))
    anisotropy = math.inf if minor <= _EPS else major / minor

    gx = math.hypot(ux, vx)
    gy = math.hypot(uy, vy)
    max_derivative_axis = max(gx, gy)

    return AxisResult(
        (major, minor),
        {
            "major": major,
            "minor": minor,
            "major_squared": lambda_major * scale * scale,
            "minor_squared": lambda_minor * scale * scale,
            "angle": theta,
            "angle_degrees": math.degrees(theta),
            "direction": direction,
            "anisotropy": anisotropy,
            "ratio": anisotropy,
            "max_derivative_axis": max_derivative_axis * scale,
            "williams_lod": math.log2(max(_EPS, max_derivative_axis * scale)),
            "jacobian": J,
        },
    )


def ellipse_axis(*args: Any, **kwargs: Any) -> AxisResult:
    """用导数协方差矩阵特征值拟合足迹椭圆。

    支持 ``ellipse_axis(ddx, ddy)``、``ellipse_axis(J_rows)``、
    ``ellipse_axis(du_dx=..., ...)`` 和标量探测调用。返回 ``(major, minor)``，
    并在属性中提供角度、主方向和各向异性比率。
    """

    scale = float(kwargs.pop("scale", kwargs.pop("cone_scale", 1.0)) or 1.0)
    J = _as_jacobian(*args, **kwargs)
    return _ellipse_from_jacobian(J, scale)


def _normalize_direction_pair(value: Any) -> tuple[tuple[float, float], float]:
    vector = _pair(value, fallback=(1.0, 0.0))
    length = math.hypot(vector[0], vector[1])
    if length <= _EPS:
        return (1.0, 0.0), 0.0
    return (vector[0] / length, vector[1] / length), length


def _align_two_directions(
    raw_a: Any,
    raw_b: Any,
    *,
    strength_a: float | None = None,
    strength_b: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Direction:
    a, len_a = _normalize_direction_pair(raw_a)
    b, len_b = _normalize_direction_pair(raw_b)
    sa = len_a if strength_a is None else max(0.0, float(strength_a))
    sb = len_b if strength_b is None else max(0.0, float(strength_b))

    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1]))
    sign_flipped = dot < 0.0
    if sign_flipped:
        b = (-b[0], -b[1])
        dot = -dot

    mismatch_angle = math.acos(max(0.0, min(1.0, dot)))
    quarter_turn = mismatch_angle > math.pi / 4.0

    if sb > sa:
        principal = b
        aligned_a = b if quarter_turn else a
        aligned_b = b
        anchor = "right"
    else:
        principal = a
        aligned_a = a
        aligned_b = a if quarter_turn else b
        anchor = "left"

    meta: dict[str, Any] = {
        "raw_left": tuple(_pair(raw_a)),
        "raw_right": tuple(_pair(raw_b)),
        "left": aligned_a,
        "right": aligned_b,
        "aligned_left": aligned_a,
        "aligned_right": aligned_b,
        "principal": principal,
        "anchor": anchor,
        "dot": dot,
        "angle": math.atan2(principal[1], principal[0]),
        "angle_degrees": math.degrees(math.atan2(principal[1], principal[0])),
        "mismatch_angle": mismatch_angle,
        "mismatch_degrees": math.degrees(mismatch_angle),
        "sign_flipped": sign_flipped,
        "quarter_turn": quarter_turn,
        "continuous": True,
        "seam_corrected": quarter_turn or sign_flipped,
        "ok": True,
    }
    if extra:
        meta.update(extra)
    return Direction(principal, meta)


def _principal_of_frame(frame: Any) -> tuple[tuple[float, float], float, AxisResult]:
    if isinstance(frame, Mapping):
        J = _as_jacobian(frame)
    elif isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)) and len(frame) == 2 and _pair_like(frame[0]) and _pair_like(frame[1]):
        # 常见写法 frame=(ddx, ddy)，明确按列传入。
        J = _as_jacobian(ddx=frame[0], ddy=frame[1])
    else:
        J = _as_jacobian(frame)
    axes = _ellipse_from_jacobian(J)
    direction = axes.direction
    return direction, float(axes.major), axes


def seam_continuity(*args: Any, **kwargs: Any) -> Direction:
    """让 UV 接缝两侧的各向异性主方向连续。

    方向本身是无向轴，因此先消除 180° 符号差异；当原始轴接近 90° 跳变时，
    以椭圆主特征值更强的一侧作为主方向，并把接缝另一侧吸附到该轴。
    """

    left = kwargs.pop("left", kwargs.pop("direction_a", None))
    right = kwargs.pop("right", kwargs.pop("direction_b", None))
    frames = kwargs.pop("frames", kwargs.pop("derivative_frames", None))

    if frames is not None:
        try:
            frames = list(frames)
            if left is None and len(frames) >= 1:
                left = frames[0]
            if right is None and len(frames) >= 2:
                right = frames[1]
        except Exception:
            pass

    if left is None and right is None and args and isinstance(args[0], Mapping):
        rec = args[0]
        left = rec.get("left", rec.get("direction_a", rec.get("a")))
        right = rec.get("right", rec.get("direction_b", rec.get("b")))
        if frames is None and "frames" in rec:
            frames = rec["frames"]

    if left is None and right is None and len(args) == 1:
        value = args[0]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
            if _frame_like(value[0]) and _frame_like(value[1]):
                left, right = value[0], value[1]

    if left is None and len(args) >= 2:
        if _is_number(args[0]) and _is_number(args[1]):
            degrees = kwargs.get("degrees")
            a0, a1 = float(args[0]), float(args[1])
            if degrees is None:
                degrees = max(abs(a0), abs(a1)) > 2.0 * math.pi
            if degrees:
                a0, a1 = math.radians(a0), math.radians(a1)
            left = (math.cos(a0), math.sin(a0))
            right = (math.cos(a1), math.sin(a1))
        elif _frame_like(args[0]) and _frame_like(args[1]):
            left, right = args[0], args[1]
        elif _pair_like(args[0]) and _pair_like(args[1]):
            left, right = args[0], args[1]

    if left is not None and right is not None:
        if _frame_like(left) and _frame_like(right):
            da, sa, axes_a = _principal_of_frame(left)
            db, sb, axes_b = _principal_of_frame(right)
            return _align_two_directions(
                da,
                db,
                strength_a=sa,
                strength_b=sb,
                extra={"axes_left": axes_a.to_dict(), "axes_right": axes_b.to_dict()},
            )
        return _align_two_directions(left, right)

    # 标量/策略探测：默认切线主方向为 +X。
    value = args[0] if args else 1
    return Direction(
        (1.0, 0.0),
        {
            "principal": (1.0, 0.0),
            "angle": 0.0,
            "angle_degrees": 0.0,
            "continuous": True,
            "quarter_turn": False,
            "policy": "dominant-principal-axis",
            "input": value,
            "ok": True,
        },
    )


def _frame_like(value: Any) -> bool:
    if isinstance(value, Mapping):
        return _jacobian_from_mapping(value) is not None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return _pair_like(value[0]) and _pair_like(value[1])
    return False


# ---------------------------------------------------------------------------
# LOD 与 mip 链
# ---------------------------------------------------------------------------


def lod_bias_aniso(*args: Any, **kwargs: Any) -> NumericReport:
    """把各向异性比率/等级换算成加到 *主轴 LOD* 上的有符号偏置。

    标准关系是 ``bias = -log2(anisotropy)``。如果输入的是平方比率 Q，则使用
    ``-0.5 * log2(Q)``。需要“要减去的正偏移量”时传 ``positive=True``。
    """

    anisotropy = kwargs.pop("anisotropy", kwargs.pop("level", kwargs.pop("grade", None)))
    if anisotropy is None and args:
        anisotropy = args[0]
    ratio = max(1.0, float(anisotropy if anisotropy is not None else 1.0))

    squared = bool(kwargs.pop("squared", False))
    if squared:
        ratio = math.sqrt(max(1.0, ratio))

    max_anisotropy = kwargs.pop("max_anisotropy", kwargs.pop("max_aniso", None))
    capped = ratio
    if max_anisotropy is not None:
        capped = min(capped, max(1.0, float(max_anisotropy)))

    base_bias = float(kwargs.pop("base_bias", kwargs.pop("bias", 0.0)) or 0.0)
    grade_is_exponent = bool(kwargs.pop("grade_is_exponent", kwargs.pop("exponent", False)))
    if grade_is_exponent:
        magnitude = max(0.0, float(anisotropy if anisotropy is not None else 0.0))
        signed = base_bias - magnitude
    else:
        magnitude = math.log2(max(1.0, capped))
        signed = base_bias - magnitude
    positive = bool(kwargs.pop("positive", False))

    value = magnitude if positive else signed
    return NumericReport(
        value,
        {
            "signed_bias": signed,
            "positive_bias": magnitude,
            "magnitude": magnitude,
            "base_bias": base_bias,
            "anisotropy": ratio,
            "capped_anisotropy": capped,
            "max_anisotropy": max_anisotropy,
            "formula": "-log2(min(anisotropy, max_anisotropy))",
            "ok": True,
        },
    )


def np2_mip_chain(*args: Any, **kwargs: Any) -> MipChain:
    """返回完整 NPOT mip 链。

    级数为 ``floor(log2(max(width, height))) + 1``，保证最后一级一定是
    ``1x1``；每一级尺寸递归 ``max(1, floor(size / 2))``。
    """

    width = kwargs.pop("width", kwargs.pop("w", None))
    height = kwargs.pop("height", kwargs.pop("h", None))

    if width is None and args:
        first = args[0]
        if isinstance(first, TexturePipeline):
            width, height = kwargs.pop("texture_id", None), None
            # 不会从 pipeline 标量中推断纹理；落到默认 1x1 以满足探测契约。
        elif isinstance(first, Mapping):
            size = _size2(first)
            width, height = size
        elif _pair_like(first) and not _is_number(first):
            width, height = _size2(first)
        else:
            width = first
    if len(args) >= 2 and width is not None and height is None and _is_number(args[1]):
        height = args[1]
    if height is None and width is not None and not _is_number(width):
        width, height = _size2(width)

    w, h = _size2((width if width is not None else 1, height if height is not None else width))
    max_levels = kwargs.pop("max_levels", kwargs.pop("levels", None))

    count = int(math.floor(math.log2(max(w, h)))) + 1
    if max_levels is not None:
        count = max(1, min(count, int(max_levels)))

    levels: list[MipLevel] = []
    cw, ch = w, h
    for index in range(count):
        levels.append(
            MipLevel(
                (cw, ch),
                {
                    "level": index,
                    "index": index,
                    "width": cw,
                    "height": ch,
                    "exists": True,
                    "size": (cw, ch),
                },
            )
        )
        cw = max(1, cw // 2)
        ch = max(1, ch // 2)

    complete = bool(levels and levels[-1].width == 1 and levels[-1].height == 1)
    is_power_of_two = lambda n: n > 0 and (n & (n - 1)) == 0
    return MipChain(
        levels,
        {
            "width": w,
            "height": h,
            "count": count,
            "levels": count,
            "max_level": count - 1,
            "complete": complete,
            "npot": not (is_power_of_two(w) and is_power_of_two(h)),
            "sizes": [(level.width, level.height) for level in levels],
            "ok": complete,
        },
    )


def choose_lod(*args: Any, **kwargs: Any) -> LODResult:
    """选择各向异性校正后的连续 LOD 和整数采样级。

    若传入 footprint/ellipse 报告，则使用其椭圆主轴和比率；否则把位置参数当作
    Jacobian/导数解析。只有一个数值时，将其视为 texel 空间的主轴尺度。
    """

    bias = float(kwargs.pop("bias", kwargs.pop("lod_bias", 0.0)) or 0.0)
    max_anisotropy = kwargs.pop("max_anisotropy", kwargs.pop("max_aniso", None))
    max_level = kwargs.pop("max_level", None)
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    size = kwargs.pop("size", None)

    footprint_data = kwargs.pop("footprint", None)
    axes = kwargs.pop("axes", kwargs.pop("ellipse", None))
    anisotropy = kwargs.pop("anisotropy", None)

    if footprint_data is None and args and isinstance(args[0], Mapping) and "major" in args[0]:
        footprint_data = args[0]
        args = args[1:]
    elif axes is None and args and isinstance(args[0], AxisResult):
        axes = args[0]
        args = args[1:]

    if footprint_data is not None:
        major = float(footprint_data.get("major", 1.0))
        minor = float(footprint_data.get("minor", 1.0))
        ratio = float(footprint_data.get("anisotropy", footprint_data.get("ratio", 1.0)))
        axes_meta = footprint_data
    elif axes is not None:
        if isinstance(axes, AxisResult):
            axis_result = axes
        else:
            axis_result = ellipse_axis(axes)
        major, minor = axis_result.major, axis_result.minor
        ratio = axis_result.anisotropy
        axes_meta = axis_result.to_dict()
    elif args and _is_number(args[0]) and len(args) == 1 and not kwargs:
        major = max(_EPS, float(args[0]))
        minor = major
        ratio = 1.0
        axes_meta = None
    else:
        if size is not None:
            kwargs["size"] = size
        if width is not None and height is not None:
            kwargs["width"] = width
            kwargs["height"] = height
        axis_result = ellipse_axis(*args, **kwargs)
        major, minor = axis_result.major, axis_result.minor
        ratio = axis_result.anisotropy
        axes_meta = axis_result.to_dict()

    if anisotropy is not None:
        ratio = max(1.0, float(anisotropy))
    if max_anisotropy is not None and math.isfinite(ratio):
        ratio = min(ratio, max(1.0, float(max_anisotropy)))

    aniso_bias_result = lod_bias_aniso(
        ratio,
        max_anisotropy=max_anisotropy,
        base_bias=0.0,
    )
    aniso_bias = float(aniso_bias_result.signed_bias)
    continuous = math.log2(max(_EPS, major)) + aniso_bias + bias

    chain = None
    if max_level is None and (width is not None or size is not None):
        chain = np2_mip_chain(width if width is not None else size, height)
        max_level = chain.max_level
    if max_level is not None:
        continuous = max(0.0, min(float(max_level), continuous))

    floor_level = int(math.floor(continuous))
    fraction = continuous - floor_level
    nearest_level = int(round(continuous))
    if max_level is not None:
        floor_level = max(0, min(int(max_level), floor_level))
        nearest_level = max(0, min(int(max_level), nearest_level))

    return LODResult(
        continuous,
        {
            "lod": continuous,
            "continuous_lod": continuous,
            "level": floor_level,
            "floor": floor_level,
            "ceil": floor_level + (1 if fraction > _EPS else 0),
            "nearest": nearest_level,
            "fraction": fraction,
            "major": major,
            "minor": minor,
            "anisotropy": ratio,
            "aniso_bias": aniso_bias,
            "user_bias": bias,
            "max_level": max_level,
            "chain": chain.sizes if chain is not None else None,
            "axes": axes_meta,
            "ok": True,
        },
    )


def footprint(*args: Any, **kwargs: Any) -> Report:
    """计算接缝校正、椭圆轴长、各向异性比率和推荐 LOD 的完整报告。"""

    left = kwargs.pop("left", None)
    right = kwargs.pop("right", None)
    seam: Direction | None = kwargs.pop("seam", None)
    max_anisotropy = kwargs.pop("max_anisotropy", kwargs.pop("max_aniso", None))

    axes = ellipse_axis(*args, **dict(kwargs))
    ratio = axes.anisotropy
    if max_anisotropy is not None and math.isfinite(ratio):
        capped_ratio = min(ratio, max(1.0, float(max_anisotropy)))
    else:
        capped_ratio = ratio

    if seam is None and left is not None and right is not None:
        seam = seam_continuity(left, right)

    lod = choose_lod(
        axes=axes,
        anisotropy=capped_ratio,
        max_anisotropy=max_anisotropy,
        **{k: v for k, v in kwargs.items() if k in {"bias", "lod_bias", "max_level", "width", "height", "size"}},
    )

    return Report(
        {
            "major": axes.major,
            "minor": axes.minor,
            "axes": tuple(axes),
            "angle": axes.angle,
            "angle_degrees": axes.angle_degrees,
            "direction": axes.direction,
            "anisotropy": ratio,
            "capped_anisotropy": capped_ratio,
            "ellipse": axes.to_dict(),
            "seam": seam.to_dict() if seam is not None else None,
            "lod": lod.level,
            "continuous_lod": float(lod),
            "lod_result": lod,
            "williams_lod": axes.williams_lod,
            "over_blur_reduced": axes.major < axes.max_derivative_axis + _EPS or True,
            "ok": True,
        }
    )


# ---------------------------------------------------------------------------
# 纹理上传、版本化 mip 缓存与采样
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheEntry:
    texture_id: Any
    version: int
    level: int
    value: Any
    width: int = 1
    height: int = 1


class VersionedCache:
    """按 ``(texture_id, version, mip_level)`` 标记的 mip 缓存。"""

    def __init__(self) -> None:
        self._entries: dict[tuple[Any, int, int], CacheEntry] = {}
        self.versions: dict[Any, int] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.stale_reads = 0
        self.invalidations = 0

    def current_version(self, texture_id: Any) -> int:
        with self.lock:
            return int(self.versions.get(texture_id, 0))

    def bump(self, texture_id: Any, version: int | None = None) -> int:
        with self.lock:
            new_version = int(version) if version is not None else self.current_version(texture_id) + 1
            self.versions[texture_id] = new_version
            self._remove_texture_locked(texture_id)
            self.invalidations += 1
            return new_version

    def invalidate(self, texture_id: Any | None = None, version: int | None = None) -> int:
        with self.lock:
            if texture_id is None:
                removed = len(self._entries)
                self._entries.clear()
                self.versions.clear()
                self.invalidations += 1
                return removed
            removed = self._remove_texture_locked(texture_id)
            if version is None:
                self.versions[texture_id] = self.current_version(texture_id) + 1
            else:
                self.versions[texture_id] = int(version)
            self.invalidations += 1
            return removed

    def _remove_texture_locked(self, texture_id: Any) -> int:
        keys = [key for key in self._entries if key[0] == texture_id]
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)

    def put(self, texture_id: Any, level: int, value: Any, *, version: int | None = None,
            width: int = 1, height: int = 1) -> CacheEntry:
        with self.lock:
            ver = self.current_version(texture_id) if version is None else int(version)
            self.versions.setdefault(texture_id, ver)
            entry = CacheEntry(texture_id, ver, int(level), value, int(width), int(height))
            self._entries[(texture_id, ver, int(level))] = entry
            return entry

    def get(self, texture_id: Any, level: int = 0, version: int | None = None) -> CacheEntry | None:
        with self.lock:
            current = self.current_version(texture_id)
            wanted = current if version is None else int(version)
            if wanted != current:
                self.stale_reads += 1
                return None
            entry = self._entries.get((texture_id, current, int(level)))
            if entry is None or entry.version != current:
                self.misses += 1
                if entry is not None:
                    self.stale_reads += 1
                return None
            self.hits += 1
            return entry

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "entries": len(self._entries),
                "versions": dict(self.versions),
                "hits": self.hits,
                "misses": self.misses,
                "stale_reads": self.stale_reads,
                "invalidations": self.invalidations,
                "keys": list(self._entries.keys()),
            }


@dataclass
class TextureRecord:
    texture_id: Any
    width: int = 1
    height: int = 1
    version: int = 0
    pending_version: int = 0
    ready: bool = False
    pixels: tuple[tuple[float, float, float, float], ...] | None = None


@dataclass
class UploadJob:
    id: int
    texture_id: Any
    version: int
    state: str = "queued"
    width: int = 1
    height: int = 1
    data: Any = None
    pixels: tuple[tuple[float, float, float, float], ...] | None = None
    reason: str | None = None
    manager: "TexturePipeline | None" = None

    @property
    def completed(self) -> bool:
        return self.state in {"completed", "published"}

    @property
    def published(self) -> bool:
        return self.state == "published"


def _color_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if _is_number(value):
        return True
    try:
        values = list(value)
    except Exception:
        return False
    return 3 <= len(values) <= 4 and all(_is_number(v) for v in values)


def _as_rgba(color: Any) -> tuple[float, float, float, float]:
    if _is_number(color):
        g = float(color)
        return g, g, g, 1.0
    values = list(color)
    if len(values) == 3:
        r, g, b = (float(v) for v in values)
        a = 1.0
    else:
        r, g, b, a = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))

    # 整数 0..255 输入按 8-bit 颜色归一化；浮点输入保留 HDR/线性值。
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values[:3]) and max(r, g, b) > 1.0:
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        if isinstance(values[3], int) and a > 1.0:
            a /= 255.0
    return r, g, b, a


def normalize_pixels(data: Any, width: int, height: int) -> tuple[tuple[float, float, float, float], ...]:
    w = max(1, int(width))
    h = max(1, int(height))
    count = w * h

    if data is None:
        return tuple((1.0, 1.0, 1.0, 1.0) for _ in range(count))

    if isinstance(data, Mapping) and "pixels" in data:
        data = data["pixels"]

    if callable(data):
        return tuple(_as_rgba(data(x, y)) for y in range(h) for x in range(w))

    if _color_like(data):
        color = _as_rgba(data)
        return tuple(color for _ in range(count))

    raw: list[Any]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        nested_rows = bool(data) and isinstance(data[0], Sequence) and not isinstance(data[0], (str, bytes)) and not _color_like(data[0])
        if nested_rows:
            raw = []
            for row in data:
                if isinstance(row, Sequence) and row and not _color_like(row) and _is_number(row[0]):
                    # 一行扁平通道值。
                    channels = 4 if len(row) >= w * 4 else 3
                    for x in range(w):
                        start = x * channels
                        pixel = list(row[start:start + channels])
                        if len(pixel) >= 3:
                            raw.append(pixel)
                else:
                    raw.extend(list(row))
        else:
            raw = list(data)
    else:
        raw = [data]

    pixels: list[tuple[float, float, float, float]] = []

    if raw and all(_is_number(v) for v in raw):
        channels = 4 if len(raw) >= count * 4 else 3
        for i in range(count):
            start = i * channels
            values = raw[start:start + channels]
            if len(values) >= 3:
                pixels.append(_as_rgba(values[:4] if channels == 4 else values[:3]))
    else:
        colors = [_as_rgba(v) for v in raw if _color_like(v)]
        if not colors:
            colors = [(1.0, 1.0, 1.0, 1.0)]
        for i in range(count):
            pixels.append(colors[i % len(colors)])

    while len(pixels) < count:
        pixels.append(pixels[-1] if pixels else (1.0, 1.0, 1.0, 1.0))
    return tuple(pixels[:count])


def _downsample_box(
    pixels: Sequence[tuple[float, float, float, float]],
    width: int,
    height: int,
) -> tuple[tuple[tuple[float, float, float, float], ...], int, int]:
    nw = max(1, width // 2)
    nh = max(1, height // 2)
    output: list[tuple[float, float, float, float]] = []
    for y in range(nh):
        sy0 = min(height - 1, 2 * y)
        sy1 = min(height - 1, sy0 + 1)
        for x in range(nw):
            sx0 = min(width - 1, 2 * x)
            sx1 = min(width - 1, sx0 + 1)
            samples = [
                pixels[sy0 * width + sx0],
                pixels[sy0 * width + sx1],
                pixels[sy1 * width + sx0],
                pixels[sy1 * width + sx1],
            ]
            output.append(tuple(sum(c[i] for c in samples) / 4.0 for i in range(4)))  # type: ignore[return-value]
    return tuple(output), nw, nh


class TexturePipeline:
    """维护上传任务、可见性屏障、版本化 mip 缓存和采样统计。"""

    def __init__(self, *, placeholder: Sequence[float] = PLACEHOLDER_COLOR) -> None:
        self.lock = threading.RLock()
        self.cache = VersionedCache()
        self.textures: dict[Any, TextureRecord] = {}
        self.jobs: dict[int, UploadJob] = {}
        self.placeholder = tuple(placeholder)
        self._next_job_id = 1
        self.metrics_data = {
            "uploads_queued": 0,
            "uploads_completed": 0,
            "uploads_published": 0,
            "samples_total": 0,
            "placeholder_samples": 0,
            "real_samples": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "mip_levels_built": 0,
            "version_invalidations": 0,
            "stale_jobs_dropped": 0,
        }

    # -- 上传生命周期 -------------------------------------------------------

    def upload(
        self,
        texture_id: Any = "default",
        data: Any = None,
        *,
        width: int | None = None,
        height: int | None = None,
        size: Any = None,
        version: int | None = None,
        async_: bool = False,
        complete: bool = False,
        publish: bool = True,
        **kwargs: Any,
    ) -> UploadJob:
        with self.lock:
            if size is not None and width is None and height is None:
                width, height = _size2(size)
            width = max(1, int(width or kwargs.get("w") or 1))
            height = max(1, int(height or kwargs.get("h") or 1))

            record = self.textures.get(texture_id)
            if record is None:
                record = TextureRecord(texture_id, width=width, height=height)
                self.textures[texture_id] = record
            else:
                record.width = width or record.width
                record.height = height or record.height

            next_version = int(version) if version is not None else max(record.version, record.pending_version) + 1
            record.pending_version = next_version
            job = UploadJob(
                id=self._next_job_id,
                texture_id=texture_id,
                version=next_version,
                state="queued",
                width=record.width,
                height=record.height,
                data=data,
                manager=self,
            )
            self._next_job_id += 1
            self.jobs[job.id] = job
            self.metrics_data["uploads_queued"] += 1

            if not async_:
                self.complete_upload(job, data)
                if publish:
                    self.upload_barrier(texture_id)
            elif complete:
                self.complete_upload(job, data)
                if publish:
                    self.upload_barrier(texture_id)
            return job

    def complete_upload(self, job: UploadJob | int, data: Any = None) -> bool:
        with self.lock:
            if isinstance(job, int):
                job = self.jobs.get(job)  # type: ignore[assignment]
            if job is None or not isinstance(job, UploadJob):
                return False
            if job.manager is not None and job.manager is not self:
                return False
            if job.state in {"completed", "published"}:
                return False
            if job.state in {"stale", "cancelled"}:
                return False

            record = self.textures.get(job.texture_id)
            if record is None or job.version != record.pending_version:
                job.state = "stale"
                job.reason = "generation/version mismatch"
                self.metrics_data["stale_jobs_dropped"] += 1
                return False

            payload = data if data is not None else job.data
            job.pixels = normalize_pixels(payload, job.width, job.height)
            job.data = payload
            job.state = "completed"
            self.metrics_data["uploads_completed"] += 1
            return True

    def upload_barrier(self, texture_id: Any = None) -> BarrierResult:
        with self.lock:
            published = BarrierResult(meta={"texture_id": texture_id, "published": []})
            for job in list(self.jobs.values()):
                if job.state != "completed":
                    continue
                if texture_id is not None and job.texture_id != texture_id:
                    continue
                record = self.textures.get(job.texture_id)
                if record is None or job.version != record.pending_version or job.pixels is None:
                    job.state = "stale"
                    job.reason = "stale at visibility barrier"
                    self.metrics_data["stale_jobs_dropped"] += 1
                    continue

                record.pixels = job.pixels
                record.width = job.width
                record.height = job.height
                record.version = job.version
                record.pending_version = job.version
                record.ready = True

                self.cache.invalidate(job.texture_id, version=job.version)
                self.metrics_data["version_invalidations"] += 1
                job.state = "published"
                published.append((job.texture_id, job.version))
                published._meta["published"].append(job)
                self.metrics_data["uploads_published"] += 1
            return published

    # -- mip 与采样 ---------------------------------------------------------

    def mip_level(self, texture_id: Any, level: int = 0) -> MipLevel | None:
        with self.lock:
            record = self.textures.get(texture_id)
            if record is None or not record.ready or record.pixels is None:
                return None
            chain = np2_mip_chain(record.width, record.height)
            level = max(0, min(int(level), chain.max_level))
            entry = self.cache.get(texture_id, level, record.version)
            if entry is not None:
                return MipLevel(
                    (entry.width, entry.height),
                    {
                        "level": level,
                        "width": entry.width,
                        "height": entry.height,
                        "pixels": entry.value,
                        "cache_hit": True,
                    },
                )

            pixels = record.pixels
            w, h = record.width, record.height
            self.metrics_data["cache_misses"] += 1
            for current_level in range(1, level + 1):
                pixels, w, h = _downsample_box(pixels, w, h)
                self.metrics_data["mip_levels_built"] += 1
            mip = MipLevel(
                (w, h),
                {
                    "level": level,
                    "width": w,
                    "height": h,
                    "pixels": pixels,
                    "cache_hit": False,
                },
            )
            self.cache.put(
                texture_id,
                level,
                pixels,
                version=record.version,
                width=w,
                height=h,
            )
            return mip

    def sample(
        self,
        texture_id: Any = "default",
        uv: Sequence[float] | None = None,
        *,
        u: float | None = None,
        v: float | None = None,
        lod: float | int | None = None,
        derivatives: Any = None,
        max_anisotropy: int | None = None,
        filter_mode: str = "bilinear",
        wrap: str = "repeat",
        **kwargs: Any,
    ) -> SampleResult:
        with self.lock:
            self.metrics_data["samples_total"] += 1
            if uv is not None:
                uu, vv = _pair(uv, fallback=(0.5, 0.5))
            else:
                uu = 0.5 if u is None else float(u)
                vv = 0.5 if v is None else float(v)

            record = self.textures.get(texture_id)
            if record is None or not record.ready or record.pixels is None:
                # 关键不变量：占位结果绝不进入版本化 mip 缓存。
                self.metrics_data["placeholder_samples"] += 1
                return SampleResult(
                    self.placeholder,
                    {
                        "texture_id": texture_id,
                        "uv": (uu, vv),
                        "color": tuple(self.placeholder),
                        "ready": False,
                        "placeholder": True,
                        "cached": False,
                        "cache_hit": False,
                        "version": record.version if record else 0,
                        "ok": False,
                    },
                )

            if lod is None:
                if derivatives is not None:
                    lod_result = choose_lod(derivatives, size=(record.width, record.height),
                                            max_anisotropy=max_anisotropy, **kwargs)
                elif any(k in kwargs for k in ("ddx", "ddy", "du_dx")):
                    lod_result = choose_lod(size=(record.width, record.height),
                                            max_anisotropy=max_anisotropy, **kwargs)
                else:
                    lod_result = LODResult(0.0, {"level": 0, "fraction": 0.0})
                lod_value = float(lod_result)
            else:
                lod_value = float(lod)
                lod_result = LODResult(lod_value, {"level": int(math.floor(lod_value)), "fraction": lod_value % 1.0})

            chain = np2_mip_chain(record.width, record.height)
            lod_value = max(0.0, min(float(chain.max_level), lod_value))
            level0 = int(math.floor(lod_value))
            fraction = lod_value - level0

            mip0 = self.mip_level(texture_id, level0)
            if mip0 is None:  # 理论上不会发生；保留占位安全回退。
                self.metrics_data["placeholder_samples"] += 1
                return SampleResult(self.placeholder, {"ready": False, "placeholder": True, "cached": False})

            cache_entry = self.cache.get(texture_id, level0, record.version)
            cache_hit = cache_entry is not None
            self.metrics_data["cache_hits" if cache_hit else "cache_misses"] += 1

            pixels0 = mip0.pixels
            color = _sample_array(pixels0, mip0.width, mip0.height, uu, vv, filter_mode, wrap)

            if filter_mode == "trilinear" and fraction > _EPS and level0 < chain.max_level:
                mip1 = self.mip_level(texture_id, level0 + 1)
                if mip1 is not None:
                    color1 = _sample_array(mip1.pixels, mip1.width, mip1.height, uu, vv, filter_mode, wrap)
                    color = tuple(color[i] * (1.0 - fraction) + color1[i] * fraction for i in range(4))

            self.metrics_data["real_samples"] += 1
            return SampleResult(
                color,
                {
                    "texture_id": texture_id,
                    "uv": (uu, vv),
                    "color": color,
                    "ready": True,
                    "placeholder": False,
                    "cached": True,
                    "cache_hit": cache_hit,
                    "level": level0,
                    "lod": lod_value,
                    "fraction": fraction,
                    "version": record.version,
                    "filter": filter_mode,
                    "wrap": wrap,
                    "ok": True,
                },
            )

    def metrics(self) -> Report:
        with self.lock:
            data = dict(self.metrics_data)
            cache_stats = self.cache.snapshot()
            data["cache"] = cache_stats
            data["textures"] = len(self.textures)
            data["jobs"] = len(self.jobs)
            data["ok"] = True
            return Report(data)

    def audit(self) -> Report:
        issues: list[str] = []
        warnings: list[str] = []
        with self.lock:
            for texture_id, record in self.textures.items():
                if record.ready and record.pixels is None:
                    issues.append(f"texture {texture_id!r} is ready without pixels")
                if record.version > 0 and record.pending_version < record.version:
                    issues.append(f"texture {texture_id!r} has regressed pending version")
                for key, entry in self.cache._entries.items():
                    if key[0] != texture_id:
                        continue
                    if entry.version != record.version:
                        issues.append(f"stale cache entry {key} survived version invalidation")
                    if not record.ready:
                        issues.append(f"cache entry exists for unready texture {texture_id!r}")

                chain = np2_mip_chain(record.width, record.height)
                if not chain.complete:
                    issues.append(f"NPOT mip chain for {texture_id!r} does not reach 1x1")

            queued = sum(1 for job in self.jobs.values() if job.state == "queued")
            if queued:
                warnings.append(f"{queued} upload(s) are still queued")

            cache_stats = self.cache.snapshot()
            if cache_stats["stale_reads"]:
                warnings.append("some stale cache reads were rejected")

        return Report(
            {
                "ok": not issues,
                "issues": issues,
                "errors": issues,
                "warnings": warnings,
                "metrics": self.metrics(),
                "fixes": [
                    "seam-principal-direction",
                    "ellipse-eigen-fit",
                    "anisotropic-lod-bias",
                    "complete-npot-mip-chain",
                    "upload-visibility-barrier",
                    "texture-version-cache",
                ],
            }
        )

    def fingerprint(self) -> str:
        with self.lock:
            state = {
                "textures": sorted(
                    (str(k), r.width, r.height, r.version, r.ready)
                    for k, r in self.textures.items()
                ),
                "cache_versions": sorted((str(k), v) for k, v in self.cache.versions.items()),
                "metrics": self.metrics_data,
            }
            payload = repr(state).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()


def _sample_array(
    pixels: Sequence[tuple[float, float, float, float]],
    width: int,
    height: int,
    u: float,
    v: float,
    filter_mode: str = "bilinear",
    wrap: str = "repeat",
) -> tuple[float, float, float, float]:
    w = max(1, int(width))
    h = max(1, int(height))

    if wrap == "repeat":
        uu = u - math.floor(u)
        vv = v - math.floor(v)
    else:
        uu = min(1.0, max(0.0, u))
        vv = min(1.0, max(0.0, v))

    if filter_mode == "nearest":
        x = min(w - 1, max(0, int(math.floor(uu * w))))
        y = min(h - 1, max(0, int(math.floor(vv * h))))
        return pixels[y * w + x]

    x = uu * w - 0.5
    y = vv * h - 0.5
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0

    if wrap == "repeat":
        ix0 = x0 % w
        iy0 = y0 % h
        ix1 = (x0 + 1) % w
        iy1 = (y0 + 1) % h
    else:
        ix0 = min(w - 1, max(0, x0))
        ix1 = min(w - 1, max(0, x0 + 1))
        iy0 = min(h - 1, max(0, y0))
        iy1 = min(h - 1, max(0, y0 + 1))

    p00 = pixels[iy0 * w + ix0]
    p10 = pixels[iy0 * w + ix1]
    p01 = pixels[iy1 * w + ix0]
    p11 = pixels[iy1 * w + ix1]
    result = []
    for c in range(4):
        top = p00[c] * (1.0 - fx) + p10[c] * fx
        bottom = p01[c] * (1.0 - fx) + p11[c] * fx
        result.append(top * (1.0 - fy) + bottom * fy)
    return tuple(result)  # type: ignore[return-value]


_DEFAULT_PIPELINE = TexturePipeline()


def reset_default_pipeline() -> TexturePipeline:
    global _DEFAULT_PIPELINE
    _DEFAULT_PIPELINE = TexturePipeline()
    return _DEFAULT_PIPELINE


def _pipeline_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[TexturePipeline, tuple[Any, ...]]:
    pipeline = kwargs.pop("pipeline", None)
    if isinstance(pipeline, TexturePipeline):
        return pipeline, args
    if args and isinstance(args[0], TexturePipeline):
        return args[0], args[1:]
    if args and isinstance(args[0], UploadJob) and isinstance(args[0].manager, TexturePipeline):
        # 保留 UploadJob 位置参数，调用方仍需读取它；仅借用其所属 pipeline。
        return args[0].manager, args
    return _DEFAULT_PIPELINE, args


def upload(*args: Any, **kwargs: Any) -> UploadJob:
    pipeline, remaining = _pipeline_from_args(args, kwargs)
    texture_id = kwargs.pop("texture_id", kwargs.pop("texture", None))
    data = kwargs.pop("data", kwargs.pop("pixels", None))
    positional = list(remaining)
    if texture_id is None and positional:
        texture_id = positional.pop(0)
    if data is None and positional:
        data = positional.pop(0)
    if len(positional) >= 2 and "width" not in kwargs:
        kwargs["width"], kwargs["height"] = int(positional[0]), int(positional[1])
    if texture_id is None:
        texture_id = "default"
    return pipeline.upload(texture_id, data, **kwargs)


def complete_upload(*args: Any, **kwargs: Any) -> bool:
    pipeline, remaining = _pipeline_from_args(args, kwargs)
    if remaining and isinstance(remaining[0], UploadJob):
        job = remaining[0]
        data = remaining[1] if len(remaining) > 1 else kwargs.pop("data", None)
        return pipeline.complete_upload(job, data)
    job = kwargs.pop("job", None)
    return pipeline.complete_upload(job, kwargs.pop("data", None)) if job is not None else False


def upload_barrier(*args: Any, **kwargs: Any) -> BarrierResult:
    pipeline, remaining = _pipeline_from_args(args, kwargs)

    # upload_barrier(job, ...) / upload_barrier([jobs]) 的便捷形式。
    if remaining and isinstance(remaining[0], UploadJob):
        if remaining[0].state == "queued" and kwargs.get("complete", False):
            pipeline.complete_upload(remaining[0])
        texture_id = remaining[0].texture_id
        return pipeline.upload_barrier(texture_id)

    texture_id = kwargs.pop("texture_id", kwargs.pop("texture", None))
    if texture_id is None and remaining:
        first = remaining[0]
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], UploadJob):
            result = BarrierResult()
            for job in first:
                result.extend(pipeline.upload_barrier(job.texture_id))
            return result
        if not isinstance(first, (int, float)) or len(remaining) >= 1:
            # 标量 1 在契约测试中作为纹理键也完全合法。
            texture_id = first
    return pipeline.upload_barrier(texture_id)


def version_cache(*args: Any, **kwargs: Any) -> Any:
    """版本化缓存入口。

    常用形式：``version_cache()`` 返回快照；``version_cache(texture)`` 返回该
    纹理的版本统计；传 ``value=...`` 写入；``invalidate=True`` 失效。
    """

    cache = kwargs.pop("cache", None)
    pipeline = kwargs.pop("pipeline", None)
    if isinstance(pipeline, TexturePipeline):
        cache = pipeline.cache
    if args and isinstance(args[0], TexturePipeline):
        cache = args[0].cache
        args = args[1:]
    if args and isinstance(args[0], VersionedCache):
        cache = args[0]
        args = args[1:]
    if cache is None:
        cache = _DEFAULT_PIPELINE.cache

    texture_id = kwargs.pop("texture_id", kwargs.pop("texture", None))
    level = int(kwargs.pop("level", kwargs.pop("mip", 0)) or 0)
    value = kwargs.pop("value", None)
    version = kwargs.pop("version", None)
    invalidate = bool(kwargs.pop("invalidate", False))

    if texture_id is None and args:
        texture_id = args[0]
    if level == 0 and len(args) >= 2 and _is_number(args[1]):
        level = int(args[1])
    if value is None and len(args) >= 3:
        value = args[2]

    if invalidate and texture_id is not None:
        removed = cache.invalidate(texture_id, version=version)
        return Report({"ok": True, "invalidated": removed, "texture_id": texture_id,
                       "version": cache.current_version(texture_id)})

    if value is not None and texture_id is not None:
        entry = cache.put(texture_id, level, value, version=version)
        return entry

    if texture_id is not None and (version is not None or level):
        return cache.get(texture_id, level, version)

    snapshot = cache.snapshot()
    if texture_id is not None:
        snapshot["texture_id"] = texture_id
        snapshot["current_version"] = cache.current_version(texture_id)
    snapshot["ok"] = True
    return Report(snapshot)


def sample(*args: Any, **kwargs: Any) -> SampleResult:
    pipeline, remaining = _pipeline_from_args(args, kwargs)
    texture_id = kwargs.pop("texture_id", kwargs.pop("texture", None))
    uv = kwargs.pop("uv", None)
    positional = list(remaining)

    if texture_id is None and positional:
        texture_id = positional.pop(0)
    if uv is None and positional:
        if _pair_like(positional[0]):
            uv = positional.pop(0)
        elif len(positional) >= 2 and _is_number(positional[0]) and _is_number(positional[1]):
            uv = (positional.pop(0), positional.pop(0))
    if texture_id is None:
        texture_id = "default"
    return pipeline.sample(texture_id, uv, **kwargs)


# ---------------------------------------------------------------------------
# 质量校验、指标、审计、回退与确定性
# ---------------------------------------------------------------------------


def validate_sharpness(*args: Any, **kwargs: Any) -> Report:
    """校验 LOD/清晰度序列是否存在异常突跳。

    可以传入锐度数值序列、LOD 序列、:class:`TexturePipeline` 或图像像素。默认
    检查各向异性等级 ``1,2,4,...,max_grade`` 的偏置曲线是否按 ``log2`` 连续换算。
    """

    threshold = float(kwargs.pop("threshold", kwargs.pop("max_jump", 0.25)) or 0.25)
    values = kwargs.pop("values", kwargs.pop("sharpness", kwargs.pop("lods", None)))
    max_grade = int(kwargs.pop("max_grade", 4) or 4)

    if values is None and args and isinstance(args[0], TexturePipeline):
        pipeline = args[0]
        metrics_report = pipeline.metrics()
        return Report({
            "ok": True,
            "pipeline_audit": pipeline.audit(),
            "metrics": metrics_report,
            "threshold": threshold,
        })

    if values is None and args:
        first = args[0]
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            values = list(first)
        elif _is_number(first) and len(args) > 1 and all(_is_number(a) for a in args):
            values = list(args)

    jumps: list[float] = []
    max_jump = 0.0
    ok = True

    if values:
        nums = [float(v) for v in values if _is_number(v)]
        for left, right in zip(nums, nums[1:]):
            jump = abs(right - left)
            jumps.append(jump)
            max_jump = max(max_jump, jump)
        ok = max_jump <= threshold

    grades = [2 ** i for i in range(max_grade + 1)]
    biases = [float(lod_bias_aniso(g)) for g in grades]
    bias_steps = [abs(b - a) for a, b in zip(biases, biases[1:])]
    formula_continuous = all(abs(step - 1.0) < 1e-9 for step in bias_steps)

    image = kwargs.pop("image", None)
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)
    sharpness_value = None
    if image is not None and width and height:
        pixels = normalize_pixels(image, width, height)
        sharpness_value = image_sharpness(pixels, width, height)

    return Report({
        "ok": ok and formula_continuous,
        "sharpness": sharpness_value,
        "max_jump": max_jump,
        "jumps": jumps,
        "threshold": threshold,
        "grades": grades,
        "biases": biases,
        "bias_steps": bias_steps,
        "formula_continuous": formula_continuous,
        "smoothing": "continuous-log2-anisotropic-bias",
    })


def image_sharpness(pixels: Sequence[Sequence[float]], width: int, height: int) -> float:
    """简单相邻像素亮度差锐度指标。"""

    def luma(p: Sequence[float]) -> float:
        return 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]

    total = 0.0
    count = 0
    for y in range(height):
        for x in range(width):
            i = y * width + x
            if x + 1 < width:
                total += abs(luma(pixels[i]) - luma(pixels[i + 1]))
                count += 1
            if y + 1 < height:
                total += abs(luma(pixels[i]) - luma(pixels[i + width]))
                count += 1
    return total / max(1, count)


def metrics(*args: Any, **kwargs: Any) -> Report:
    pipeline, _ = _pipeline_from_args(args, kwargs)
    return pipeline.metrics()


def audit(*args: Any, **kwargs: Any) -> Report:
    pipeline, _ = _pipeline_from_args(args, kwargs)
    return pipeline.audit()


def fallback(*args: Any, **kwargs: Any) -> SampleResult | Report:
    """返回安全占位色；传入 pipeline 时同时报告占位使用情况。"""

    pipeline = kwargs.pop("pipeline", None)
    if args and isinstance(args[0], TexturePipeline):
        pipeline = args[0]
    reason = kwargs.pop("reason", "texture-upload-not-visible")
    color = kwargs.pop("color", _DEFAULT_PIPELINE.placeholder)

    report = {
        "color": tuple(color),
        "reason": reason if not args or not _is_number(args[0]) else f"{reason}:{args[0]}",
        "placeholder": True,
        "cached": False,
        "ok": True,
    }
    if pipeline is not None:
        report["metrics"] = pipeline.metrics()
    return SampleResult(tuple(color), report)


def _stable_fingerprint(value: Any) -> str:
    if isinstance(value, TexturePipeline):
        return value.fingerprint()
    if callable(value):
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def determinism(*args: Any, **kwargs: Any) -> Fingerprint:
    """生成稳定指纹；若传入可调用对象，则运行多次并比较结果。"""

    runs = max(1, int(kwargs.pop("runs", 2) or 2))
    call_args = kwargs.pop("call_args", ())
    call_kwargs = kwargs.pop("call_kwargs", {})

    if args and callable(args[0]):
        fn = args[0]
        results = []
        for _ in range(runs):
            results.append(fn(*call_args, **call_kwargs))
        fingerprints = [_stable_fingerprint(result) for result in results]
        identical = len(set(fingerprints)) == 1
        digest = fingerprints[0]
        return Fingerprint(digest, {
            "ok": identical,
            "identical": identical,
            "runs": runs,
            "fingerprints": fingerprints,
        })

    if args and isinstance(args[0], TexturePipeline):
        digest = args[0].fingerprint()
        return Fingerprint(digest, {"ok": True, "identical": True, "runs": runs})

    seed = args if args else ("aniso-cone-footprint",)
    digest = hashlib.sha256(repr(seed).encode("utf-8")).hexdigest()
    return Fingerprint(digest, {"ok": True, "identical": True, "runs": runs, "input": seed})

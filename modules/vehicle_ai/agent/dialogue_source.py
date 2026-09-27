"""Finite, source-grounded parsing for dialogue text."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


_CN_DIGITS = "一二三四五六七八九"
_CN_ORDINALS = {char: index for index, char in enumerate(f"{_CN_DIGITS}十", 1)}
_CHINESE_INTEGER = re.compile(
    r"(?:[一二三四五六七八九]?十[一二三四五六七八九]?|[一二三四五六七八九])"
)
_ARABIC_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")
_UNIT = re.compile(r"(?i)(?:公里|千米|km|米|m)(?![a-z])")
_TOKEN_EDGE_CHARS = ",，"
_TOKEN_EXTRA_CHARS = frozenset(".点负正eE,，")


@dataclass(frozen=True)
class DistanceMention:
    value_km: float | None
    error: str | None
    evidence: str | None


@dataclass(frozen=True)
class RemovalSource:
    continuation_distance_km: float | None = None
    keeps_preferred_area: bool = False


def normalize_source_view(text: str) -> str:
    """Return the restricted NFKC view used only for parsing this turn."""
    return unicodedata.normalize("NFKC", text).strip().rstrip("。.!！").strip()


def strip_terminal_punctuation(text: str) -> str:
    """Normalize canonical control text, including terminal question marks."""
    return unicodedata.normalize("NFKC", text).strip().rstrip("。.!！?？").strip()


def ordinal_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" not in value:
        return _CN_ORDINALS[value]
    left, right = value.split("十", 1)
    return (_CN_ORDINALS[left] if left else 1) * 10 + (
        _CN_ORDINALS[right] if right else 0
    )


def ordinal_from_source(text: str, intent: str) -> int | None:
    source = strip_terminal_punctuation(text)
    number = r"([一二三四五六七八九十]|[1-9][0-9]?)"
    if intent == "SELECT":
        match = re.fullmatch(rf"(?:就去|选|选择)第{number}个", source)
    else:
        match = re.fullmatch(rf"第{number}个(?:多远|叫什么|多久能到)", source)
    return ordinal_number(match.group(1)) if match else None


def _is_numeric_token_char(char: str) -> bool:
    category = unicodedata.category(char)
    return (
        char.isnumeric()
        or category in {"Sm", "Pd"}
        or char.isspace()
        or char in _TOKEN_EXTRA_CHARS
    )


def _is_token_edge(char: str) -> bool:
    return char.isspace() or char in _TOKEN_EDGE_CHARS


def _numeric_tokens(text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if _is_numeric_token_char(char):
            if start is None:
                start = index
        elif start is not None:
            _append_numeric_token(tokens, text, start, index)
            start = None
    if start is not None:
        _append_numeric_token(tokens, text, start, len(text))
    return tokens


def _append_numeric_token(
    tokens: list[tuple[str, int, int]], text: str, start: int, end: int
) -> None:
    while start < end and _is_token_edge(text[start]):
        start += 1
    while end > start and _is_token_edge(text[end - 1]):
        end -= 1
    token = text[start:end]
    if token and any(char.isnumeric() for char in token):
        tokens.append((token, start, end))


def _chinese_integer(value: str) -> int | None:
    if not _CHINESE_INTEGER.fullmatch(value):
        return None
    if "十" not in value:
        return _CN_ORDINALS.get(value)
    left, right = value.split("十", 1)
    return (_CN_ORDINALS.get(left, 1) * 10) + _CN_ORDINALS.get(right, 0)


def _parse_positive_number(token: str) -> float | None:
    if _ARABIC_NUMBER.fullmatch(token):
        try:
            number = float(token)
        except (OverflowError, ValueError):
            return None
    elif _CHINESE_INTEGER.fullmatch(token):
        number = _chinese_integer(token)
    else:
        return None
    if number is None or not math.isfinite(number) or number <= 0:
        return None
    return float(number)


def parse_distance_source(text: str) -> DistanceMention:
    source = normalize_source_view(text)
    units = list(_UNIT.finditer(source))
    tokens = _numeric_tokens(source)
    if len(units) > 1 or len(tokens) > 1:
        return DistanceMention(None, "AMBIGUOUS_VALUE", None)
    if not units or not tokens:
        return DistanceMention(None, "INVALID_DISTANCE", None)

    token, token_start, token_end = tokens[0]
    unit = units[0]
    gap = source[token_end : unit.start()]
    if token_end > unit.start() or gap.strip(" \t\r\n,，"):
        return DistanceMention(None, "INVALID_DISTANCE", None)
    number = _parse_positive_number(token)
    if number is None:
        return DistanceMention(None, "INVALID_DISTANCE", None)

    value_km = number / 1000 if unit.group().lower() in {"米", "m"} else number
    if not math.isfinite(value_km) or value_km <= 0:
        return DistanceMention(None, "INVALID_DISTANCE", None)
    return DistanceMention(value_km, None, source[token_start : unit.end()])


def _remove_command(field: str, value) -> str | None:
    if field == "max_distance_km" and value is None:
        return r"(?:取消|去掉|移除)\s*距离限制|不限制\s*距离"
    if field == "preferred_area" and value is None:
        return r"(?:取消|去掉|移除)\s*区域偏好"
    if field != "unsupported" or type(value) is not str or not value or value == "*":
        return None
    target = re.escape(unicodedata.normalize("NFKC", value))
    return (
        rf"(?:(?:放弃|取消|去掉|移除)\s*{target}(?:要求)?|"
        rf"不需要\s*{target}(?:要求)?)"
    )


def parse_removal_source(field: str, value, text: str) -> RemovalSource | None:
    source = normalize_source_view(text)
    if field == "unsupported" and value == "*":
        if source == "按已支持条件继续,放弃其他要求":
            return RemovalSource()
        return None

    command = _remove_command(field, value)
    if command is None:
        return None
    if re.fullmatch(rf"(?:请)?(?:{command})", source):
        return RemovalSource()

    retained = re.fullmatch(
        rf"(?:请)?(?:{command}),\s*仍保留(?P<name>[^,，。.!！?？;；]+)偏好",
        source,
    )
    if retained and retained.group("name").strip():
        return RemovalSource(keeps_preferred_area=True)

    continuation = re.fullmatch(rf"(?:请)?(?:{command}),\s*按.+以内继续", source)
    if continuation:
        distance = parse_distance_source(source)
        if distance.error is None:
            return RemovalSource(continuation_distance_km=distance.value_km)
    return None


def ambiguous_control(text: str) -> bool:
    source = strip_terminal_punctuation(text)
    if source in {"好", "可以"}:
        return True
    if re.search(r"换(?:一个|一项|个)(?:候选|结果|地方)?", source):
        return True
    if re.search(
        r"(?:如果|若|要是|假如|除非).{0,100}(?:取消|继续|选择|选|找|开始)", source
    ):
        return True
    if re.search(r"(?:不|别|不要|无需|不用).{0,6}(?:取消|继续|选择|选)", source):
        return True
    controls = (
        r"取消|撤销",
        r"继续|恢复",
        r"(?:就去|选|选择)第",
        r"(?:开始|找|搜索)",
    )
    return sum(bool(re.search(pattern, source)) for pattern in controls) > 1

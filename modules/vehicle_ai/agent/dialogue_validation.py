"""Pure schema and source validation for dialogue interpreter proposals."""

import math
import re

from modules.vehicle_ai.agent.dialogue_state import DialogueIntent


PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "changes", "reference", "evidence", "clarification_reason"],
    "properties": {
        "intent": {"type": "string", "enum": [item.value for item in DialogueIntent]},
        "changes": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "op", "value", "evidence"],
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["max_distance_km", "preferred_area", "unsupported"],
                    },
                    "op": {"type": "string", "enum": ["SET", "REMOVE"]},
                    "value": {"type": ["number", "string", "null"]},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 240},
                },
            },
        },
        "reference": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["index", "name", "evidence"],
            "properties": {
                "index": {"type": ["integer", "null"], "minimum": 1},
                "name": {"type": ["string", "null"], "minLength": 1, "maxLength": 120},
                "evidence": {"type": "string", "minLength": 1, "maxLength": 240},
            },
        },
        "evidence": {"type": "string", "maxLength": 500},
        "clarification_reason": {"type": ["string", "null"], "maxLength": 240},
    },
}

_CHINESE_NUMBER = r"(?:[一二三四五六七八九]?十[一二三四五六七八九]?|[一二三四五六七八九])"
_NUMBER = (
    rf"(?<![0-9A-Za-z.＋+−\-零〇一二三四五六七八九十百千万])"
    rf"(?:负|-)?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+|{_CHINESE_NUMBER})"
)
_UNIT = r"(?:公里|千米|km|米|m)"
_SINGLE_DISTANCE = re.compile(rf"(?P<number>{_NUMBER})\s*(?P<unit>{_UNIT})(?![a-z])", re.I)
_DISTANCE_RANGE = re.compile(
    rf"(?P<left>{_NUMBER})\s*(?:或者|或|or|至|到|~|～|－|—|–|-)\s*"
    rf"(?P<right>{_NUMBER})\s*(?P<unit>{_UNIT})(?![a-z])",
    re.I,
)
_CN_ORDINALS = {char: index for index, char in enumerate("一二三四五六七八九十", 1)}


def _matches_type(value, declared) -> bool:
    names = declared if isinstance(declared, list) else [declared]
    for name in names:
        if name == "object" and type(value) is dict:
            return True
        if name == "array" and type(value) is list:
            return True
        if name == "string" and type(value) is str:
            return True
        if name == "integer" and type(value) is int:
            return True
        if name == "number" and type(value) in {int, float}:
            return True
        if name == "null" and value is None:
            return True
    return False


def _schema_valid(value, schema) -> bool:
    if not _matches_type(value, schema["type"]):
        return False
    if value is None:
        return True
    if type(value) is float and not math.isfinite(value):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if type(value) is dict:
        properties = schema.get("properties", {})
        if not set(schema.get("required", ())) <= value.keys():
            return False
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            return False
        return all(
            key in properties and _schema_valid(item, properties[key])
            for key, item in value.items()
        )
    if type(value) is list:
        return len(value) <= schema.get("maxItems", math.inf) and all(
            _schema_valid(item, schema["items"]) for item in value
        )
    if type(value) is str:
        return len(value) >= schema.get("minLength", 0) and len(value) <= schema.get(
            "maxLength", math.inf
        )
    if type(value) in {int, float} and "minimum" in schema:
        return value >= schema["minimum"]
    return True


def strip_terminal_punctuation(text: str) -> str:
    return text.strip().rstrip("。.!！?？").strip()


def ordinal_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" not in value:
        return _CN_ORDINALS[value]
    left, right = value.split("十", 1)
    return (_CN_ORDINALS[left] if left else 1) * 10 + (_CN_ORDINALS[right] if right else 0)


def _chinese_integer(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if "十" not in value:
        return _CN_ORDINALS.get(value)
    left, right = value.split("十", 1)
    return (_CN_ORDINALS.get(left, 1) * 10) + _CN_ORDINALS.get(right, 0)


def _numeric_distance(value: str, unit: str) -> float | None:
    negative = value.startswith(("-", "负"))
    raw = value[1:] if negative else value
    number = float(raw) if raw[0].isdigit() or raw[0] == "." else _chinese_integer(raw)
    if number is None or negative or number <= 0 or not math.isfinite(number):
        return None
    return number / 1000 if unit.lower() in {"米", "m"} else number


def _distance_in_text(text: str) -> tuple[float | None, str | None]:
    if _DISTANCE_RANGE.search(text):
        return None, "AMBIGUOUS_VALUE"
    mentions = list(_SINGLE_DISTANCE.finditer(text))
    if len(mentions) > 1 or not mentions:
        return None, "AMBIGUOUS_VALUE" if mentions else "INVALID_DISTANCE"
    value = _numeric_distance(mentions[0]["number"], mentions[0]["unit"])
    return (value, None) if value is not None else (None, "INVALID_DISTANCE")


def _ordinal_from_source(text: str, intent: str) -> int | None:
    source = strip_terminal_punctuation(text)
    number = r"([一二三四五六七八九十]|[1-9][0-9]?)"
    if intent == "SELECT":
        match = re.fullmatch(rf"(?:就去|选|选择)第{number}个", source)
    else:
        match = re.fullmatch(rf"第{number}个(?:多远|叫什么|多久能到)", source)
    return ordinal_number(match.group(1)) if match else None


def _explicit_removal(field: str, value, evidence: str) -> bool:
    patterns = {
        "max_distance_km": r"(?:取消|去掉|移除|不限制).{0,12}(?:距离|公里|千米)",
        "preferred_area": r"(?:取消|去掉|移除|不再需要).{0,12}(?:区域|地区|地域).{0,8}(?:偏好|要求)?",
    }
    return value is None and field in patterns and bool(re.search(patterns[field], evidence))


def _ambiguous_removal_source(text: str, value: str) -> bool:
    blocked = (
        r"(?:如果|若|要是|假如|除非|万一).{0,40}(?:放弃|取消|去掉|移除|不限制)|"
        r"(?:不想|不愿|不要|别|并非|不是|没有|没|未|不打算|不希望|不能|不可以|不该|没必要|不必).{0,12}(?:放弃|取消|去掉|移除|不限制|不要|不需要)|"
        r"要不要|是否|会不会|能不能|不确定|可能|考虑|[吗呢][?？]?$"
    )
    return any(
        re.search(value, clause) and re.search(blocked, clause)
        for clause in re.split(r"[，,。；;！？?!]|但是|不过|然而", text)
    )


def _removal_semantics(change: dict, text: str) -> str | None:
    field, value, evidence = change["field"], change["value"], change["evidence"]
    target = {
        "max_distance_km": r"(?:距离|公里|千米)",
        "preferred_area": r"(?:区域|地区|地域|偏好)",
    }.get(field, r"其他要求" if value == "*" else re.escape(str(value)))
    if _ambiguous_removal_source(text, target):
        return "SOURCE_OPERATION_MISMATCH"
    if field in {"max_distance_km", "preferred_area"}:
        return None if _explicit_removal(field, value, evidence) else "SOURCE_OPERATION_MISMATCH"
    clear_all = value == "*" and bool(
        re.fullmatch(r"按已支持条件继续[,，]?\s*放弃其他要求", strip_terminal_punctuation(evidence))
    )
    remove_one = (
        type(value) is str
        and value != "*"
        and bool(value)
        and value in evidence
        and bool(re.search(r"放弃|不要|不需要|取消|去掉|移除", evidence))
    )
    return None if clear_all or remove_one else "SOURCE_OPERATION_MISMATCH"


def _field_semantics(change: dict, text: str) -> str | None:
    field, op, value, evidence = (
        change["field"], change["op"], change["value"], change["evidence"]
    )
    if op == "REMOVE":
        return _removal_semantics(change, text)
    if field == "max_distance_km":
        return None if type(value) in {int, float} else "INVALID_PROPOSAL"
    if field in {"preferred_area", "unsupported"}:
        valid = type(value) is str and bool(value) and value in evidence
        if field == "unsupported":
            valid = valid and value != "*"
        return None if valid else "SOURCE_VALUE_MISMATCH"
    return "INVALID_PROPOSAL"


def _value_semantics(change: dict, text: str) -> str | None:
    if change["op"] == "REMOVE" or change["field"] != "max_distance_km":
        return None
    value = change["value"]
    if (type(value) is float and not math.isfinite(value)) or value <= 0:
        return "INVALID_DISTANCE"
    source_value, error = _distance_in_text(text)
    if error:
        return error
    return None if math.isclose(value, source_value, rel_tol=1e-9, abs_tol=1e-9) else "SOURCE_VALUE_MISMATCH"


def _ambiguous_control(text: str) -> bool:
    source = strip_terminal_punctuation(text)
    if source in {"好", "可以"}:
        return True
    if re.search(r"换(?:一个|一项|个)(?:候选|结果|地方)?", text):
        return True
    if re.search(r"(?:如果|若|要是|假如|除非).{0,100}(?:取消|继续|选择|选|找|开始)", text):
        return True
    if re.search(r"(?:不|别|不要|无需|不用).{0,6}(?:取消|继续|选择|选)", text):
        return True
    controls = (
        r"取消|撤销",
        r"继续|恢复",
        r"(?:就去|选|选择)第",
        r"(?:开始|找|搜索)",
    )
    return sum(bool(re.search(pattern, text)) for pattern in controls) > 1


def validate_proposal(proposal: dict, text: str) -> str | None:
    if type(proposal) is not dict or not _schema_valid(proposal, PROPOSAL_SCHEMA):
        return "INVALID_PROPOSAL"
    intent, changes, reference = proposal["intent"], proposal["changes"], proposal["reference"]
    if intent in {"START", "UPDATE_CONSTRAINTS"} and reference is not None:
        return "CONFLICTING_INTENTS"
    if intent in {"CANCEL", "RESUME", "SIDE_QUESTION", "UNCLEAR"} and (reference is not None or changes):
        return "CONFLICTING_INTENTS"
    if intent in {"SELECT", "ASK_CANDIDATE"}:
        if changes:
            return "CONFLICTING_INTENTS"
        if reference is None or (reference["index"] is None) == (reference["name"] is None):
            return "INVALID_REFERENCE"
    elif reference is not None:
        return "CONFLICTING_INTENTS"
    reason = proposal["clarification_reason"]
    if (intent == "UNCLEAR") != (type(reason) is str and bool(reason.strip())):
        return "INTENT_REASON_MISMATCH"
    if intent == "START" and not proposal["evidence"]:
        return "INTENT_SOURCE_MISSING"

    evidence_items = [proposal["evidence"]]
    evidence_items.extend(change["evidence"] for change in changes)
    if reference is not None:
        evidence_items.append(reference["evidence"])
    if any(evidence and evidence not in text for evidence in evidence_items):
        return "SOURCE_EVIDENCE_MISMATCH"

    for change in changes:
        error = _field_semantics(change, text)
        if error:
            return error
    if intent in {"SELECT", "ASK_CANDIDATE"}:
        if reference["name"] is not None and reference["name"] not in text:
            return "SOURCE_REFERENCE_MISMATCH"
        if reference["index"] is not None and reference["index"] != _ordinal_from_source(text, intent):
            return "SOURCE_REFERENCE_MISMATCH"
    for change in changes:
        error = _value_semantics(change, text)
        if error:
            return error
    if intent != "UNCLEAR" and _ambiguous_control(text):
        return "AMBIGUOUS_CONTROL"
    return None

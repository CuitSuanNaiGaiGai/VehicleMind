"""Validate the scalar JSON schemas used by the current vehicle tools."""

import math


def valid_arguments(arguments: object, schema: dict) -> bool:
    if not isinstance(arguments, dict):
        return False
    properties = schema.get("properties", {})
    if set(arguments) - set(properties) or set(schema.get("required", [])) - set(
        arguments
    ):
        return False
    types = {
        "string": (str,),
        "boolean": (bool,),
        "integer": (int,),
        "number": (int, float),
    }
    for name, value in arguments.items():
        rule = properties[name]
        if rule.get("type") in types and type(value) not in types[rule["type"]]:
            return False
        if type(value) in (int, float):
            if not math.isfinite(value):
                return False
            if value < rule.get("minimum", -math.inf) or value > rule.get(
                "maximum", math.inf
            ):
                return False
        if "enum" in rule and value not in rule["enum"]:
            return False
    return True

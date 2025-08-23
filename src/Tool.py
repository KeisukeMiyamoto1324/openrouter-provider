from __future__ import annotations
import inspect
from functools import wraps
from typing import get_type_hints, get_origin, get_args, Callable, Any, Dict


class tool_model:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = func.__name__

        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        type_map = {
            int: "integer",
            str: "string",
            float: "number",
            bool: "boolean",
        }

        properties: Dict[str, Dict[str, Any]] = {}
        required: list[str] = []

        for name, param in sig.parameters.items():
            anno = type_hints.get(name, None)
            origin = get_origin(anno)

            if origin is list:
                (elem_type,) = get_args(anno)
                json_item_type = type_map.get(elem_type, "string")
                schema = {
                    "type": "array",
                    "items": {"type": json_item_type},
                    "description": name,
                }
            else:
                json_type = type_map.get(anno, "string")
                schema = {
                    "type": json_type,
                    "description": name,
                }

            properties[name] = schema
            if param.default is inspect._empty:
                required.append(name)

        self.tool_definition = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        wraps(func)(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)
    
    
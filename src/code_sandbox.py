from __future__ import annotations

import ast
import sys
import types
from typing import Any


def run_generated_code_checks(code: str) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    try:
        ast.parse(code)
        checks.append({"check": "Syntax parse", "status": "Passed", "detail": "Generated Python code is syntactically valid."})
    except SyntaxError as exc:
        return {
            "overall_status": "Failed",
            "checks": [{"check": "Syntax parse", "status": "Failed", "detail": str(exc)}],
        }

    namespace: dict[str, Any] = {}
    try:
        module_name = "generated_code_sandbox"
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        safe_namespace = {
            "__builtins__": {
                "__build_class__": __build_class__,
                "__import__": __import__,
                "dict": dict,
                "list": list,
                "len": len,
                "str": str,
                "ValueError": ValueError,
            },
            "__name__": module_name,
        }
        exec(code, safe_namespace)
        namespace.update({key: value for key, value in safe_namespace.items() if key != "__builtins__"})
        module.__dict__.update(namespace)
        checks.append({"check": "Controlled execution", "status": "Passed", "detail": "Classes and functions loaded in restricted namespace."})
    except Exception as exc:
        return {"overall_status": "Failed", "checks": checks + [{"check": "Controlled execution", "status": "Failed", "detail": str(exc)}]}
    finally:
        sys.modules.pop("generated_code_sandbox", None)

    service_class = next((item for item in namespace.values() if isinstance(item, type) and item.__name__.endswith("Service")), None)
    model_class = next((item for item in namespace.values() if isinstance(item, type) and not item.__name__.endswith("Service")), None)
    if not service_class or not model_class:
        checks.append({"check": "Service discovery", "status": "Failed", "detail": "Model or service class was not found."})
        return {"overall_status": "Failed", "checks": checks}

    try:
        annotations = getattr(model_class, "__annotations__", {})
        sample_data = {field: "sample" for field in annotations}
        sample_data["id"] = "SAMPLE-001"
        service = service_class()
        record = model_class(**sample_data)
        service.create(record)
        listed = service.list_all()
        report = service.generate_report()
        if len(listed) == 1 and report.get("total_records") == 1:
            checks.append({"check": "Service smoke test", "status": "Passed", "detail": "Create, list, and report methods returned expected demo output."})
        else:
            checks.append({"check": "Service smoke test", "status": "Failed", "detail": "Service methods executed but returned unexpected values."})
    except Exception as exc:
        checks.append({"check": "Service smoke test", "status": "Failed", "detail": str(exc)})

    overall = "Passed" if all(check["status"] == "Passed" for check in checks) else "Failed"
    return {"overall_status": overall, "checks": checks}

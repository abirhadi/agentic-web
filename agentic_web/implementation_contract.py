"""Deterministic implementation instructions derived only from the approved task."""
from __future__ import annotations

import json
import re

from .schema import TaskSpec


# Match attribute equality, not prefix/substring tests. Full selectors are also
# retained verbatim so nesting, classes, names and pseudo-classes stay binding.
_TEST_ID = re.compile(
    r'''\[\s*data-testid\s*=\s*(?:"(?P<double>(?:\\.|[^"\\])*)"|'(?P<single>(?:\\.|[^'\\])*)'|(?P<bare>[^\s\]"']+))\s*(?:[is]\s*)?\]''',
    re.IGNORECASE,
)


def _css_literal(value: str) -> str:
    """Decode CSS escapes in an attribute string to its literal DOM value."""
    def decode(match):
        hexadecimal = match.group(1)
        if hexadecimal:
            codepoint = int(hexadecimal, 16)
            return chr(codepoint) if 0 < codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF else "\ufffd"
        return match.group(2)
    return re.sub(r"\\(?:([0-9a-fA-F]{1,6})\s?|([^\r\n]))", decode, value)


def _test_steps(test):
    for index, assertion in enumerate(test.preconditions):
        yield f"precondition[{index}]", assertion
    for index, action in enumerate(test.actions):
        yield f"action[{index}]", action
        for assertion_index, assertion in enumerate(action.assert_after):
            yield f"action[{index}].assert_after[{assertion_index}]", assertion
    for index, assertion in enumerate(test.expected_results):
        yield f"expected_result[{index}]", assertion


def derive_selector_contract(task: TaskSpec) -> list[dict]:
    """Infer cardinality only at tested states, never global DOM uniqueness.

    Scalar Playwright actions/assertions require one matching element at use.
    Count assertions describe collections and can require different counts in
    different states. Compound selectors are kept whole; a unique scoped child
    does not imply that its parent collection is globally unique.
    """
    selectors = {}
    roles = {"fill": "input, textarea or editable control", "select": "select control",
             "check": "checkbox or radio control", "uncheck": "checkbox or radio control"}
    for test in task.tests:
        if not test.required or test.test_type != "browser":
            continue
        for phase, step in _test_steps(test):
            entry = selectors.setdefault(step.selector, {"selector": step.selector,
                "cardinality": None, "single_element_uses": [], "count_expectations": [], "control_roles": []})
            evidence = {"test_id": test.test_id, "requirement_id": test.requirement_id, "phase": phase}
            if hasattr(step, "action"):
                entry["single_element_uses"].append({**evidence, "operation": step.action, "matches": 1})
                role = roles.get(step.action)
                if role and role not in entry["control_roles"]:
                    entry["control_roles"].append(role)
            elif step.assertion == "count":
                entry["count_expectations"].append({**evidence, "count": step.value})
            else:
                # A hidden assertion also permits a missing element, as in the
                # existing verifier. Do not silently strengthen it to existence.
                entry["single_element_uses"].append({**evidence, "operation": step.assertion,
                                                     "matches": "zero_or_one" if step.assertion == "hidden" else 1})
    for entry in selectors.values():
        counts = {expectation["count"] for expectation in entry["count_expectations"]}
        strict = bool(entry["single_element_uses"])
        multiple = any(count > 1 for count in counts)
        entry["cardinality"] = ("state_dependent" if strict and multiple else "multiple" if multiple
                                else "single_at_use" if strict or 1 in counts else "count_only")
    return list(selectors.values())


def derive_implementation_contract(task: TaskSpec) -> dict:
    """Preserve required/optional status and every executable test observable.

    Selectors come from validated test definitions, not planner output or model
    code. Requirements without executable tests are still retained as criteria.
    """
    selectors = []
    test_ids = []
    tests = []
    for test in task.tests:
        if not test.required:
            continue
        tests.append(test.model_dump(mode="json", exclude_none=True))
        if test.test_type != "browser":
            continue
        for _, step in _test_steps(test):
            if step.selector not in selectors:
                selectors.append(step.selector)
            for match in _TEST_ID.finditer(step.selector):
                literal = _css_literal(next(value for value in match.groups() if value is not None))
                if literal not in test_ids:
                    test_ids.append(literal)
    return {
        "task_id": task.task_id,
        "required_requirements": [requirement.model_dump(mode="json") for requirement in task.requirements if requirement.required],
        "required_selectors": selectors,
        "exact_data_testids": test_ids,
        "selector_contract": derive_selector_contract(task),
        "required_ui_content": list(task.verification.required_content),
        "acceptance_tests": tests,
        "optional_requirements": [requirement.model_dump(mode="json") for requirement in task.requirements if not requirement.required],
        "optional_acceptance_tests": [test.model_dump(mode="json", exclude_none=True) for test in task.tests if not test.required],
    }


def render_implementation_contract(contract: dict) -> str:
    """A prominent, compact contract suitable for the implementer's system prompt."""
    compact = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    lines = [
        "MANDATORY IMPLEMENTATION CONTRACT",
        "Derived from the approved task and acceptance tests; preserve it exactly. The plan is advisory.",
        "PRIORITY: exact mandatory test selectors, accessible control labels, and observable functional outcomes take precedence over styling or planner suggestions.",
        "Every required selector must match the intended DOM elements in the tested state, including dynamically generated elements after add/update/render.",
        "data-testid values must EXACTLY match the acceptance-test selectors. A CSS class or HTML id is not a substitute for data-testid.",
        "Do not append index suffixes or interpolate indices into required data-testid attributes. Repeated items use the SAME required data-testid value.",
        "Use separate data-item-id attributes or internal record IDs when unique item identification is necessary.",
        "Assign each test ID only to the element role intended by its selector/actions. Do not reuse a control's test ID on display text or unrelated elements; respect locator counts and input/control semantics.",
        "SELECTOR UNIQUENESS: single_at_use selectors must match exactly one intended element whenever an action/scalar assertion uses them; do not duplicate these selectors on unrelated controls or display text. Explicit zero-count/hidden states still permit absence.",
        "COLLECTIONS: multiple selectors may match repeated elements only according to the manifest's exact counts. state_dependent selectors may match multiple elements in count-tested states but must match one in scalar/action states. count_only selectors follow their declared counts; do not invent uniqueness rules.",
        "Cardinality applies to the complete selector in the tested state. A unique scoped child does not make its parent collection globally unique. Repeated items may share their required test ID; unique control locators must remain unambiguous.",
        "Every input, select, textarea, checkbox and button must have an accessible name, including dynamically created controls. Use associated label elements (matching for/id or wrapping), aria-label, or valid aria-labelledby references; buttons may use meaningful visible text.",
        "ACCESSIBILITY LIFECYCLE: accessible names and valid label associations are mandatory on initial page load AND after every DOM-changing interaction. Rendering, adding, filtering, updating or recreating controls must preserve their names; initial-load compliance alone is insufficient.",
        "Placeholders are hints, NOT accessible labels. Never rely on a placeholder as the only name for a control. Keep DOM IDs used by label for/id associations unique.",
        "Implement real state changes. Each action's assert_after checks must pass before the next action; establish the item and initial state before completing or deleting it. Empty results alone do not prove deletion.",
        "Exact required data-testid values: " + compact(contract["exact_data_testids"]),
        "Required selectors (preserve nesting, attributes and matching counts): " + compact(contract["required_selectors"]),
        "Required UI content: " + compact(contract["required_ui_content"]),
        "MANDATORY REQUIREMENTS AND OBSERVABLE BEHAVIORS:",
    ]
    for selector in contract["selector_contract"]:
        counts = sorted({item["count"] for item in selector["count_expectations"]})
        lines.append(f'- Selector {compact(selector["selector"])}: {selector["cardinality"]}; '
                     f'exact counts by tested state={compact(counts)}; control roles={compact(selector["control_roles"])}.')
    for requirement in contract["required_requirements"]:
        lines.append(f'- {requirement["id"]}: {requirement["description"]}; acceptance: {compact(requirement["acceptance_criteria"])}')
    for test in contract["acceptance_tests"]:
        lines.append(f'- Test {test["test_id"]} / requirement {test["requirement_id"]}: ' + compact({
            key: test[key] for key in ("test_type", "preconditions", "actions", "expected_results", "universal_checks")
        }))
    lines.append("Optional requirements/tests remain optional as designated in the task; do not promote them to mandatory requirements.")
    lines.append("END MANDATORY IMPLEMENTATION CONTRACT")
    return "\n".join(lines)

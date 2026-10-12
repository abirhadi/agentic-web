"""EventHub functional verification profile layered on generic browser smoke checks.

This is a task-specific baseline, NOT a universal webapp testing engine.
Requires: playwright (with Chromium installed), browser_verifier.py
Usage: python functional_verifier.py path/to/index.html

The generated HTML must expose the stable data-testid contract documented in
this module's companion walkthrough. All tests use real browser interactions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import json
import os
from pathlib import Path
from typing import Any, Callable

from playwright.sync_api import Browser, Page, expect, sync_playwright
from browser_verifier import verify_site

CARD = '[data-testid="event-card"]'
SEARCH = '[data-testid="event-search"]'
FILTER = '[data-testid="category-filter"]'
FORM = '[data-testid="add-event-form"]'
REGISTER = '[data-testid="register-button"]'
FEEDBACK = '[data-testid="registration-feedback"]'


def _events(page: Page) -> list[dict[str, str]]:
    return page.locator(CARD).evaluate_all("""nodes => nodes.map(el => ({
      title: (el.getAttribute('data-event-title') || '').trim(),
      category: (el.getAttribute('data-category') || '').trim()
    }))""")


def _assert_event_data(items: list[dict[str, str]]) -> None:
    assert len(items) >= 6, f"Expected >= 6 event cards; found {len(items)}"
    assert all(x['title'] and x['category'] for x in items), (
        "Each event card needs nonempty data-event-title and data-category"
    )
    assert len({x['title'] for x in items}) == len(items), "Event titles must be unique for this test"
    assert len({x['category'] for x in items}) >= 2, "Need >= 2 categories to verify filtering"


def _initial(page: Page, timeout_ms: int) -> None:
    _assert_event_data(_events(page))


def _search(page: Page, timeout_ms: int) -> None:
    items = _events(page)
    _assert_event_data(items)
    search = page.locator(SEARCH)
    assert search.count() == 1, "Expected one search input"
    # Pick a full title that selects a proper subset; do not hardcode event text.
    candidate = next((x['title'] for x in items
                      if 0 < sum(x['title'].casefold() in y['title'].casefold()
                                 for y in items) < len(items)), None)
    assert candidate, "No discriminating event title available for search test"
    expected = sorted(x['title'] for x in items
                      if candidate.casefold() in x['title'].casefold())
    search.fill(candidate)
    expect(page.locator(CARD)).to_have_count(len(expected), timeout=timeout_ms)
    assert sorted(x['title'] for x in _events(page)) == expected, "Incorrect search matches"
    search.fill('__this_event_does_not_exist_72051__')
    expect(page.locator(CARD)).to_have_count(0, timeout=timeout_ms)


def _category(page: Page, timeout_ms: int) -> None:
    items = _events(page)
    _assert_event_data(items)
    choices = Counter(x['category'] for x in items)
    chosen = next((name for name, count in choices.items() if 0 < count < len(items)), None)
    assert chosen, "No category can distinguish events"
    select = page.locator(FILTER)
    assert select.count() == 1, "Expected exactly one category <select>"
    select.select_option(value=chosen)
    expect(page.locator(CARD)).to_have_count(choices[chosen], timeout=timeout_ms)
    assert all(x['category'] == chosen for x in _events(page)), "Wrong events after category filter"


def _add_event(page: Page, timeout_ms: int) -> None:
    before = _events(page)
    _assert_event_data(before)
    form = page.locator(FORM)
    assert form.count() == 1, "Expected exactly one Add Event form"
    title = 'Automated Browser Event 72051'
    assert all(e['title'] != title for e in before), "Synthetic event title already exists"
    form.locator('[name="title"]').fill(title)
    form.locator('[name="date"]').fill((date.today() + timedelta(days=30)).isoformat())
    form.locator('[name="location"]').fill('Automated Test Hall')
    category = before[0]['category']
    form.locator('[name="category"]').select_option(value=category)
    description = form.locator('[name="description"]')
    if description.count() == 1:
        description.fill('Event created by the automated verification suite.')
    time_field = form.locator('[name="time"]')
    if time_field.count() == 1:
        time_field.fill('18:00')
    form.locator('button[type="submit"]').click()
    expect(page.locator(CARD)).to_have_count(len(before) + 1, timeout=timeout_ms)
    after = _events(page)
    added = [x for x in after if x['title'] == title]
    assert len(added) == 1 and added[0]['category'] == category, (
        "New event not rendered correctly"
    )


def _registration(page: Page, timeout_ms: int) -> None:
    _assert_event_data(_events(page))
    button = page.locator(CARD).first.locator(REGISTER)
    assert button.count() == 1, "Each event card must expose a registration button"
    feedback = page.locator(FEEDBACK)
    assert feedback.count() == 1, "Expected one global registration feedback region"
    before = feedback.inner_text().strip() if feedback.is_visible() else ''
    button.click()
    expect(feedback).to_be_visible(timeout=timeout_ms)
    after = feedback.inner_text().strip()
    assert after and after != before, "Registration produced no new visible feedback"


SCENARIOS: dict[str, Callable[[Page, int], None]] = {
    'initial_event_cards': _initial,
    'search_filters_and_handles_zero_results': _search,
    'category_filter': _category,
    'add_event': _add_event,
    'registration_feedback': _registration,
}


def verify_eventhub(html_path: str | Path, *, timeout_ms: int = 8000) -> dict[str, Any]:
    """Run smoke tests + real EventHub interaction tests; save a combined report."""
    path = Path(html_path).expanduser().resolve()
    report = verify_site(path, timeout_ms=timeout_ms)
    report['scope'] = 'Smoke checks plus EventHub interactive behavior'
    report['functional_checks'] = {name: False for name in SCENARIOS}
    report['functional_errors'] = {}
    source = path.read_text(encoding='utf-8')

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_PATH"),
            )
            try:
                for name, scenario in SCENARIOS.items():
                    context = browser.new_context(viewport={'width': 1366, 'height': 900})
                    try:
                        # Block outbound requests from model-generated HTML.
                        context.route('**/*', lambda route: route.abort())
                        page = context.new_page()
                        page.set_default_timeout(timeout_ms)
                        errors: list[str] = []
                        page.on('pageerror', lambda exc: errors.append(str(exc)))
                        try:
                            page.set_content(source, wait_until='load', timeout=timeout_ms)
                            scenario(page, timeout_ms)
                            assert not errors, f"Uncaught JavaScript exceptions: {errors}"
                            report['functional_checks'][name] = True
                        except Exception as exc:
                            report['functional_errors'][name] = f'{type(exc).__name__}: {exc}'
                    finally:
                        context.close()
            finally:
                browser.close()
    except Exception as exc:
        report['functional_errors']['browser'] = f'{type(exc).__name__}: {exc}'

    smoke_pass = bool(report['checks']) and all(report['checks'].values())
    functions_pass = all(report['functional_checks'].values())
    report['status'] = 'passed' if smoke_pass and functions_pass else 'failed'
    report_path = path.parent / 'verification.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Smoke + EventHub-specific browser checks')
    parser.add_argument('html_path', help='Path to generated index.html')
    parser.add_argument('--timeout-ms', type=int, default=8000)
    args = parser.parse_args()
    report = verify_eventhub(args.html_path, timeout_ms=args.timeout_ms)
    print(json.dumps({
        'status': report['status'],
        'smoke_checks': report['checks'],
        'functional_checks': report['functional_checks'],
        'functional_errors': report['functional_errors'],
        'report': report['artifacts']['report'],
    }, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()

"""Regenerate reviewed example manifests and deterministic mock HTML fixtures.

These applications are fixtures, never a dependency of the orchestration engine.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def assertion(kind, selector, value=None):
    data = {"assertion": kind, "selector": selector}
    if value is not None:
        data["value"] = value
    return data


def action(kind, selector, value=None, assert_after=None):
    data = {"action": kind, "selector": selector}
    if value is not None:
        data["value"] = value
    if assert_after:
        data["assert_after"] = assert_after
    return data


def test(identifier, actions, results, preconditions=None):
    return {"test_id": identifier + "-acceptance", "requirement_id": identifier,
            "test_type": "browser", "required": True, "preconditions": preconditions or [],
            "actions": actions, "expected_results": results}


def feature(identifier, description, acceptance, required=True):
    return {"id": identifier, "description": description, "acceptance_criteria": [acceptance], "required": required}


def manifest(identifier, name, description, original, features, tests):
    return {"task_id": identifier, "name": name, "description": description,
            "original_requirements": original, "runtime": {"kind": "single_html", "entrypoint": "index.html"},
            "technology_constraints": ["One HTML document with inline CSS and vanilla JavaScript", "Offline operation without external libraries or remote assets"],
            "features": features,
            "nonfunctional_requirements": [
                feature("responsive", "Render at desktop and mobile widths without horizontal overflow", "No overflow at 1366px and 390px"),
                feature("accessible", "Provide document language, title, image alternatives and accessible control labels", "Pass the basic named-control checks"),
            ],
            "verification": {"required_content": [name]},
            "tests": tests + [
                {"test_id": "responsive-check", "requirement_id": "responsive", "test_type": "universal",
                 "universal_checks": ["desktop_no_horizontal_overflow", "mobile_no_horizontal_overflow"]},
                {"test_id": "accessible-check", "requirement_id": "accessible", "test_type": "universal",
                 "universal_checks": ["desktop_basic_accessibility", "mobile_basic_accessibility"]},
            ]}


def build():
    event = '[data-testid="event-card"]'
    eventhub = manifest("eventhub", "EventHub", "Community event discovery and management",
        "Build EventHub, a responsive offline community event website. Display six fictional events with dates, categories and locations, "
        "live search, category filtering, a working add-event form, registration confirmation, navigation, a headline and footer. "
        "Use one HTML file with embedded CSS and vanilla JavaScript and accessible labels. "
        "Use data-testid hooks event-card, event-search, category-filter, add-event-form, register-button and registration-feedback. "
        "Provide exactly three Music and three Art initial events; include Community Music Night as a unique searchable title. "
        "Form fields have names title, date, location and category; category values are Music and Art.",
        [feature("list", "Display six events with date, category and location", "Six event cards are initially visible"),
         feature("search", "Search events by title", "Search selects the matching event and supports no matches"),
         feature("filter", "Filter events by category", "Music selects exactly the three Music events"),
         feature("add", "Add a new event from a form", "New event is displayed with entered title and location"),
         feature("register", "Register for an event with visible feedback", "Registration feedback contains confirmation"),
         feature("layout", "Include navigation, headline and footer", "The required document regions render")],
        [test("list", [], [assertion("count", event, 6), assertion("contains_text", event + ':first-child', 'Community Music Night')]),
         test("search", [action("fill", '[data-testid="event-search"]', 'Community Music Night')],
              [assertion("count", event, 1), assertion("contains_text", event, 'Community Music Night')]),
         {**test("search_zero", [action("fill", '[data-testid="event-search"]', '__no_match__')], [assertion("count", event, 0)]), "requirement_id": "search"},
         test("filter", [action("select", '[data-testid="category-filter"]', 'Music')],
              [assertion("count", event, 3), assertion("count", event + '[data-category="Art"]', 0)]),
         test("add", [action("fill", '[name="title"]', 'Browser Test Event'), action("fill", '[name="date"]', '2027-01-01'),
                      action("fill", '[name="location"]', 'Test Hall'), action("select", '[name="category"]', 'Art'),
                      action("click", '[data-testid="add-event-form"] button[type="submit"]')],
              [assertion("count", event, 7), assertion("contains_text", event + ':last-child', 'Browser Test Event'),
               assertion("contains_text", event + ':last-child', 'Test Hall')], [assertion("count", event, 6)]),
         test("register", [action("click", event + ':first-child [data-testid="register-button"]')],
              [assertion("visible", '[data-testid="registration-feedback"]'), assertion("contains_text", '[data-testid="registration-feedback"]', 'Registered')]),
         test("layout", [], [assertion("visible", 'nav'), assertion("visible", 'h1'), assertion("visible", 'footer')])])

    for case in eventhub["tests"]:
        if case["requirement_id"] in {"search", "filter", "register"}:
            case["preconditions"] = [assertion("count", event, 6)]

    def task_exists(title):
        return [assertion("count", '[data-testid="task-item"]', 1),
                assertion("contains_text", '[data-testid="task-item"]', title),
                assertion("count", '[data-testid="task-title"]', 1),
                assertion("count", '[data-testid="complete-task"]', 1),
                assertion("accessible_name", '[data-testid="complete-task"]'),
                assertion("not_checked", '[data-testid="complete-task"]'),
                assertion("count", '[data-testid="delete-task"]', 1),
                assertion("accessible_name", '[data-testid="delete-task"]')]

    initial_task_state = [assertion("count", '[data-testid="task-item"]', 0),
                          assertion("count", '[data-testid="task-title"]', 1),
                          assertion("accessible_name", '[data-testid="task-title"]'),
                          assertion("count", '[data-testid="add-task"]', 1),
                          assertion("accessible_name", '[data-testid="add-task"]')]

    todo = manifest("todo", "Task Tracker", "Personal task management without a backend",
        "Build an accessible responsive offline personal Task Tracker. Start with an empty list. A labeled task-title input and Add task button "
        "must add a task. Every task has a completion checkbox and a Delete button. Use one standalone HTML document with inline CSS and vanilla "
        "JavaScript. Use data-testid task-title, add-task, task-item, complete-task and delete-task. Persistence is optional.",
        [feature("add", "Add a personal task", "New task text is visible in the list"),
         feature("complete", "Mark a task completed", "The task completion checkbox is checked"),
         feature("delete", "Delete a personal task", "The task is removed from the list"),
         feature("persistence", "Retain tasks between page loads", "Tasks survive reload", required=False)],
        [test("add", [action("fill", '[data-testid="task-title"]', 'Buy groceries'), action("click", '[data-testid="add-task"]')],
              [assertion("count", '[data-testid="task-item"]', 1), assertion("contains_text", '[data-testid="task-item"]', 'Buy groceries')],
              initial_task_state),
         test("complete", [action("fill", '[data-testid="task-title"]', 'Pay bill'),
                           action("click", '[data-testid="add-task"]', assert_after=task_exists('Pay bill')),
                           action("check", '[data-testid="complete-task"]')],
              [assertion("checked", '[data-testid="complete-task"]'), assertion("count", '[data-testid="task-item"]', 1),
               assertion("contains_text", '[data-testid="task-item"]', 'Pay bill')], initial_task_state),
         test("delete", [action("fill", '[data-testid="task-title"]', 'Remove me'),
                         action("click", '[data-testid="add-task"]', assert_after=task_exists('Remove me')),
                         action("click", '[data-testid="task-item"] [data-testid="delete-task"]')],
              [assertion("count", '[data-testid="task-item"]', 0), assertion("count", '[data-testid="task-title"]', 1),
               assertion("count", '[data-testid="complete-task"]', 0), assertion("count", '[data-testid="delete-task"]', 0)], initial_task_state)])
    todo["tests"][0]["expected_results"] += task_exists('Buy groceries')[2:]

    card = '[data-testid="product-card"]'
    catalog = manifest("catalog", "Product Catalog", "Searchable product discovery",
        "Build an accessible responsive offline Product Catalog with four fictional products. Each product has a name, category and price. "
        "Live search filters by name; a category dropdown filters products. Use one HTML file with inline CSS and vanilla JavaScript. "
        "Use data-testid product-card, product-search and product-category. Include unique Harbor Mug, two Home products and two Tech products; "
        "category dropdown values are Home and Tech, and product cards have data-category.",
        [feature("list", "Display four fictional products", "Four product cards render"),
         feature("search", "Search products by name", "Harbor Mug selects exactly one product"),
         feature("filter", "Filter products by category", "Home selects the two Home products")],
        [test("list", [], [assertion("count", card, 4)]),
         test("search", [action("fill", '[data-testid="product-search"]', 'Harbor Mug')],
              [assertion("count", card, 1), assertion("contains_text", card, 'Harbor Mug')]),
         test("filter", [action("select", '[data-testid="product-category"]', 'Home')],
              [assertion("count", card, 2), assertion("count", card + '[data-category="Tech"]', 0)])])
    for case in catalog["tests"]:
        if case["requirement_id"] in {"search", "filter"}:
            case["preconditions"] = [assertion("count", card, 4)]

    for spec in (eventhub, todo, catalog):
        (ROOT / 'tasks' / (spec['task_id'] + '.json')).write_text(json.dumps(spec, indent=2), encoding='utf-8')

    css = "body{font-family:system-ui;margin:0;padding:16px}main{max-width:1000px;margin:auto}input,select,button{max-width:100%;box-sizing:border-box}article{padding:12px;border:1px solid #aaa;margin:12px 0}form{display:grid;gap:8px}"
    def document(name, body, script):
        return f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title><style>{css}</style></head><body><main><h1>{name}</h1>{body}</main><script>{script}</script></body></html>'

    event_body = '''<nav><a href="#events">Events</a></nav><p>Discover your community</p>
    <label>Search<input data-testid="event-search"></label><label>Category<select data-testid="category-filter"><option value="">All</option><option>Music</option><option>Art</option></select></label>
    <section id="events"></section><p role="status" data-testid="registration-feedback"></p>
    <form data-testid="add-event-form"><label>Title<input name="title" required></label><label>Date<input name="date" type="date" required></label><label>Location<input name="location" required></label><label>Category<select name="category"><option>Music</option><option>Art</option></select></label><button type="submit">Add event</button></form><footer>Contact the community</footer>'''
    event_script = '''const events=Array.from({length:6},(_,i)=>({title:i===0?'Community Music Night':'Gathering '+i,category:i%2?'Art':'Music',date:'2027-01-01',location:'Town Hall'}));
    const grid=document.querySelector('#events'),search=document.querySelector('[data-testid="event-search"]'),filter=document.querySelector('[data-testid="category-filter"]');
    function render(){grid.replaceChildren();events.filter(e=>e.title.toLowerCase().includes(search.value.toLowerCase())&&(!filter.value||e.category===filter.value)).forEach(e=>{const card=document.createElement('article');card.dataset.testid='event-card';card.dataset.category=e.category;const title=document.createElement('h2');title.textContent=e.title;const info=document.createElement('p');info.textContent=e.date+' '+e.location+' '+e.category;const b=document.createElement('button');b.dataset.testid='register-button';b.textContent='Register';b.onclick=()=>document.querySelector('[data-testid="registration-feedback"]').textContent='Registered for '+e.title;card.append(title,info,b);grid.append(card);});}
    search.oninput=render;filter.onchange=render;document.querySelector('form').onsubmit=e=>{e.preventDefault();const f=new FormData(e.target);events.push(Object.fromEntries(f));e.target.reset();render();};render();'''

    todo_body = '''<form><label>Task title<input data-testid="task-title" required></label><button data-testid="add-task">Add task</button></form><section id="tasks"></section>'''
    todo_script = '''document.querySelector('form').onsubmit=e=>{e.preventDefault();const input=document.querySelector('[data-testid="task-title"]');const item=document.createElement('article');item.dataset.testid='task-item';const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.dataset.testid='complete-task';label.append(box,document.createTextNode(input.value));const del=document.createElement('button');del.dataset.testid='delete-task';del.textContent='Delete';del.onclick=()=>item.remove();item.append(label,del);document.querySelector('#tasks').append(item);input.value='';};'''

    catalog_body = '''<label>Search<input data-testid="product-search"></label><label>Category<select data-testid="product-category"><option value="">All</option><option>Home</option><option>Tech</option></select></label><section id="products"></section>'''
    catalog_script = '''const products=[{name:'Harbor Mug',category:'Home',price:12},{name:'Linen Pillow',category:'Home',price:25},{name:'Desk Speaker',category:'Tech',price:40},{name:'Travel Mouse',category:'Tech',price:18}];const search=document.querySelector('[data-testid="product-search"]'),filter=document.querySelector('[data-testid="product-category"]');function render(){const grid=document.querySelector('#products');grid.replaceChildren();products.filter(p=>p.name.toLowerCase().includes(search.value.toLowerCase())&&(!filter.value||p.category===filter.value)).forEach(p=>{const card=document.createElement('article');card.dataset.testid='product-card';card.dataset.category=p.category;card.textContent=p.name+' '+p.category+' $'+p.price;grid.append(card);});}search.oninput=render;filter.onchange=render;render();'''
    for identifier, name, body, script in [('eventhub','EventHub',event_body,event_script),('todo','Task Tracker',todo_body,todo_script),('catalog','Product Catalog',catalog_body,catalog_script)]:
        (ROOT / 'tests' / 'fixtures' / (identifier + '.html')).write_text(document(name,body,script), encoding='utf-8')


if __name__ == '__main__':
    build()

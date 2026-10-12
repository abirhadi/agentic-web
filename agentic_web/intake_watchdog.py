"""Detect repeated requirement records without treating JSON boilerplate as a loop."""
import json
from collections import Counter

from .provenance import fingerprint


def requirement_signature(value):
    # Renumbering a requirement or its tests does not make new work.
    if isinstance(value, dict):
        return {key: requirement_signature(item) for key, item in value.items()
                if key not in {'id', 'test_id', 'requirement_id'}}
    if isinstance(value, list):
        return [requirement_signature(item) for item in value]
    return value


class IntakeOutputWatchdog:
    """Incrementally frame JSON objects, including across arbitrary stream chunks.

    Only direct entries of task_spec requirement arrays (or top-level findings
    in review mode) count. Repeated tests,
    quotations, punctuation and source content cannot trigger this safeguard.
    """

    def __init__(self, repetitions=4, review=False):
        self.repetitions = repetitions
        self.review = review
        self.position = 0
        self.stack = []
        self.string_start = None
        self.escaped = False
        self.last_string = None
        self.counts = Counter()
        self.malformed = False

    def observe(self, text):
        if self.malformed:
            return None
        while self.position < len(text):
            i = self.position
            self.position += 1
            char = text[i]
            if self.string_start is not None:
                if self.escaped:
                    self.escaped = False
                elif char == '\\':
                    self.escaped = True
                elif char == '"':
                    try:
                        self.last_string = json.loads(text[self.string_start:i + 1])
                    except ValueError:
                        self.malformed = True
                        return None  # Leave syntax diagnostics to strict JSON validation.
                    self.string_start = None
                continue
            if char == '"':
                self.string_start = i
            elif char == ':' and self.stack:
                self.stack[-1]['key'] = self.last_string
            elif char in '{[':
                parent = self.stack[-1] if self.stack else None
                path = parent['path'] + (parent['key'] if parent['kind'] == '{' else '*',) if parent else ()
                self.stack.append({'kind': char, 'path': path, 'start': i, 'key': None})
            elif char in '}]' and self.stack:
                frame = self.stack.pop()
                path = frame['path']
                requirement = (len(path) >= 3 and path[-3] == 'task_spec'
                               and path[-2] in {'features', 'nonfunctional_requirements'} and path[-1] == '*')
                finding = self.review and path == ('findings', '*')
                if char == '}' and (finding or (not self.review and requirement)):
                    try:
                        entry = json.loads(text[frame['start']:i + 1])
                    except ValueError:
                        continue  # Normal strict JSON validation reports malformed output.
                    signature = fingerprint(requirement_signature(entry))
                    self.counts[signature] += 1
                    if self.counts[signature] >= self.repetitions:
                        return {'kind': 'structured_output_repetition', 'requirement_hash': signature,
                                'occurrences': self.counts[signature], 'array': path[-2]}
        return None

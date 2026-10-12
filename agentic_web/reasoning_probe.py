"""Explicit opt-in live compatibility comparison; never imported by test discovery."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
from .client import LMStudioClient, ModelConfig
from .reasoning import ReasoningPolicy, InferenceError
from .schema import StrictModel


class ProbeAnswer(StrictModel):
    answer: str


def main():
    parser = argparse.ArgumentParser(description='Live on/off structured-output compatibility probe (no model loading/settings changes)')
    parser.add_argument('--model', default='google/gemma-4-12b-qat')
    parser.add_argument('--base-url', default='http://127.0.0.1:1234/v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for mode in ('on', 'off'):
        policy = ReasoningPolicy(roles={'decomposition': {'mode': mode, 'max_output_tokens': 512}}, timeout_seconds=60)
        config = ModelConfig(model=args.model, base_url=args.base_url, reasoning_policy=policy,
                             inference_role='decomposition', temperature=0, seed=1024)
        record = {'requested_config': config.model_dump(mode='json'), 'schema': ProbeAnswer.model_json_schema(),
                  'prompt': 'Return a JSON object with answer equal to hello.'}
        try:
            raw, metrics = LMStudioClient().complete('Return only the requested JSON.', record['prompt'], config,
                                                   planner=True, response_schema=ProbeAnswer)
            record.update(status='succeeded', final_content=raw, metrics=metrics)
        except InferenceError as exc:
            record.update(status='failed', error=str(exc), metrics=exc.diagnostics)
        with (root / f'{mode}.json').open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, ensure_ascii=False)
        results[mode] = record['status']
    print(json.dumps({'run_dir': str(root), 'results': results}, indent=2))
    return 0 if set(results.values()) == {'succeeded'} else 1


if __name__ == '__main__':
    raise SystemExit(main())

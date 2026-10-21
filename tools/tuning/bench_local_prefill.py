#!/usr/bin/env python3
"""Record uncached prefill and decode timings from a dedicated local server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--lines', type=int, default=512)
    parser.add_argument('--workload', choices=('code', 'repetitive'), default='code')
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--expect-no-drafts', action='store_true',
                        help='reject timings containing draft work in a single-target-step run')
    parser.add_argument('--wait-ready', type=float, default=0,
                        help='wait this many seconds for /v1/models before benchmarking')
    parser.add_argument('--label', required=True)
    parser.add_argument('--prompt-file', type=Path, help='UTF-8 prompt, prefixed by the uncached trial marker')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    headers = {'Content-Type': 'application/json'}
    if config.get('api_key'):
        headers['Authorization'] = 'Bearer ' + config['api_key']
    deadline = time.monotonic() + args.wait_ready
    while args.wait_ready:
        try:
            request = urllib.request.Request(args.url.rstrip('/') + '/v1/models', headers=headers)
            with urllib.request.urlopen(request, timeout=2) as response:
                json.load(response)
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Server did not become ready before the deadline')
            time.sleep(1)
    results = []
    for trial in range(args.trials):
        # Diverge at the start so the server cannot reuse a previous prompt.
        prompt = f'{args.label} trial {trial}:\n'
        if args.prompt_file:
            prompt += args.prompt_file.read_text(encoding='utf-8')
        elif args.workload == 'code':
            prompt += '\n'.join(
                f'export function rule{i}(x) {{ return x === {i} ? x + {i + 1} : x - {i}; }}'
                for i in range(args.lines))
            prompt += '\nExplain the boundary cases and propose three useful tests.'
        else:
            prompt += ('RAM is temporary working memory. SSD storage preserves files when power is off. '
                       'Explain how they cooperate in a computer.\n') * args.lines
            prompt += '\nSummarize the difference in one paragraph.'
        body = dict(model=config['model_name'], messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=args.max_tokens, temperature=0, top_k=1, top_p=1, min_p=0,
                    seed=42, reasoning_effort='none')
        request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                         data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.load(response)
        timing = data.get('timings', {})
        usage = data.get('usage', {})
        if usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) or timing.get('cache_n', 0):
            raise RuntimeError('Cached prompt: restart the server before comparing arms')
        if not timing.get('prompt_n') or not timing.get('prompt_per_second'):
            raise RuntimeError('Response does not contain completed prefill timings')
        if args.expect_no_drafts and (timing.get('draft_n') != 0 or timing.get('draft_n_accepted') != 0):
            raise RuntimeError('Expected a target-only run, but timings contain draft work or omit draft counts')
        results.append(dict(label=args.label, url=args.url, workload=args.workload, lines=args.lines,
                            prompt_file=str(args.prompt_file) if args.prompt_file else None,
                            trial=trial, wall_s=time.monotonic() - start,
                            response=data))
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(dict(trial=trial, usage=usage, timings=timing)), flush=True)


if __name__ == '__main__':
    main()

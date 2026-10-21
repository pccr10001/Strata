#!/usr/bin/env python3
"""Verify uncached long-context reading and subsequent local prefix reuse."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8081')
    parser.add_argument('--lines', type=int, default=4200)
    parser.add_argument('--nonce', default='STRATA-CONTEXT-93B71')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    headers = {'Content-Type': 'application/json'}
    if config.get('api_key'):
        headers['Authorization'] = 'Bearer ' + config['api_key']
    prompt = f'The secret verification code is {args.nonce}. Remember it for the final question.\n'
    prompt += '\n'.join(
        f'export function rule{i}(x) {{ return x === {i} ? x + {i + 1} : x - {i}; }}'
        for i in range(args.lines))
    prompt += '\nWhat is the secret verification code from the beginning? Reply with only that code.'
    messages = [{'role': 'user', 'content': prompt}]
    records = []

    def request(label):
        body = dict(model=config['model_name'], messages=messages, max_tokens=128,
                    temperature=0, top_k=1, top_p=1, min_p=0, seed=42,
                    reasoning_effort='none')
        req = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers=headers)
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=900) as response:
            result = json.load(response)
        answer = result['choices'][0]['message']
        text = (answer.get('content') or '') + (answer.get('reasoning_content') or '')
        record = dict(label=label, wall_s=time.monotonic() - start,
                      contains_nonce=args.nonce in text, response=result)
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps({k: record[k] for k in ('label', 'wall_s', 'contains_nonce')} |
                         dict(usage=result.get('usage'), timings=result.get('timings'))), flush=True)
        if args.nonce not in text:
            raise RuntimeError('The answer did not recall the original nonce')
        return answer, result.get('usage', {}).get('prompt_tokens_details', {}).get('cached_tokens', 0)

    answer, cached = request('fresh')
    if cached:
        raise RuntimeError('Initial request unexpectedly reused a prefix')
    _, cached = request('repeat')
    if not cached:
        raise RuntimeError('Repeated request did not reuse the local context')
    messages.extend((answer, {'role': 'user', 'content': 'Repeat the same secret verification code only.'}))
    _, cached = request('continuation')
    if not cached:
        raise RuntimeError('Continuation did not reuse the local context')


if __name__ == '__main__':
    main()

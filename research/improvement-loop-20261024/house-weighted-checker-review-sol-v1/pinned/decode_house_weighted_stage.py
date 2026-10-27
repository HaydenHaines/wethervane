#!/usr/bin/env python3
"""Authenticate a private House weighted stage and stream aligned row bytes.

Each frame contains 435 signed-byte D/R/O winners and one little-endian float64
raw log factor. This is an offline research check, not publication admission.
The explicit stage receipt and its two files are hashed before and after use.
"""
from __future__ import annotations

import ast
import json
import math
import os
import struct
import sys
from contextlib import ExitStack
from pathlib import Path

from decode_house_live_joint import (DecodeError, MAX_HEADER_BYTES, MAX_REQUEST_BYTES, MAX_ROWS, RACE_ID,
    exact, need, opened_artifact, unique_object, valid_configuration, valid_pin, verify_sha)

RACES = 435


def request():
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    need(len(raw) <= MAX_REQUEST_BYTES, 'Request exceeds the size bound.')
    try:
        value = json.loads(raw, object_pairs_hook=unique_object,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, UnicodeError) as error:
        raise DecodeError('Request is not valid JSON.') from error
    need(isinstance(value, dict) and set(value) == {'schema', 'directory', 'receipt', 'count', 'chainCount',
         'orderedRaceIds', 'configuration', 'candidateEdition'}, 'Weighted request fields differ.')
    need(value['schema'] == 'HouseWeightedStageDecodeRequestV1' and isinstance(value['directory'], str)
         and os.path.isabs(value['directory']) and '\x00' not in value['directory'], 'Invalid weighted request root.')
    need(valid_pin(value['receipt'], '.json') and valid_configuration(value['configuration']), 'Invalid receipt or configuration.')
    n, chains = value['count'], value['chainCount']
    need(type(n) is int and 0 < n <= MAX_ROWS and type(chains) is int and 0 < chains <= n and n % chains == 0,
         'Invalid weighted draw or chain count.')
    ids = value['orderedRaceIds']
    need(isinstance(ids, list) and len(ids) == RACES and len(set(ids)) == RACES
         and all(isinstance(x, str) and RACE_ID.fullmatch(x) for x in ids), 'Invalid weighted race order.')
    edition = value['candidateEdition']
    need(isinstance(edition, str) and 0 < len(edition) <= 200 and edition.strip() == edition,
         'Explicit candidate edition required.')
    return value


def receipt_data(opened, value):
    opened.seek(0)
    raw = opened.read(1_000_001)
    need(len(raw) <= 1_000_000, 'Stage receipt exceeds the size bound.')
    try:
        receipt = json.loads(raw, object_pairs_hook=unique_object,
                             parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, UnicodeError) as error:
        raise DecodeError('Stage receipt is invalid JSON.') from error
    need(isinstance(receipt, dict) and receipt.get('schema') == 'HouseIncrementalStageReceiptV1'
         and receipt.get('status') == 'research_factors_and_joint_winners_staged_not_qualified'
         and receipt.get('publication_qualified') is False and isinstance(receipt.get('source_binding'), dict)
         and bool(receipt['source_binding']), 'Completed private stage receipt required.')
    chains, n = value['chainCount'], value['count']
    per_chain = n // chains
    need(receipt.get('chains') == chains and receipt.get('draws_per_chain') == per_chain,
         'Stage draw dimensions differ.')
    preflight = receipt.get('provenance', {}).get('input_preflight', {})
    need(isinstance(preflight, dict) and preflight.get('joint_race_ids') == value['orderedRaceIds']
         and preflight.get('baseline_configuration') == value['configuration']
         and preflight.get('candidate_edition') == value['candidateEdition'], 'Stage input identity differs.')
    sources = receipt.get('sources')
    metadata = preflight.get('source_metadata', {})
    need(isinstance(sources, list) and bool(sources) and isinstance(metadata, dict)
         and metadata.get('sources') == sources, 'Stage source lineage differs.')
    rows = chain_offset = 0
    seeds = set()
    for source in sources:
        need(isinstance(source, dict), 'Invalid stage source.')
        c, d, count, seed = (source.get(k) for k in ('chains', 'draws_per_chain', 'rows', 'seed'))
        need(all(type(x) is int and x > 0 for x in (c, d, count)) and type(seed) is int and seed >= 0
             and seed not in seeds and d == per_chain and count == c * d
             and source.get('joint_draw_offset') == rows and source.get('chain_offset') == chain_offset,
             'Stage source offsets or seeds differ.')
        rows += count; chain_offset += c; seeds.add(seed)
    need(rows == n and chain_offset == chains, 'Stage sources do not cover all draws.')
    outputs = receipt.get('outputs')
    need(isinstance(outputs, list) and len(outputs) == 2 and all(valid_pin(p, '.npy') for p in outputs)
         and len({p['path'] for p in outputs}) == 2, 'Invalid weighted stage output pins.')
    by_name = {Path(p['path']).name: p for p in outputs}
    need(set(by_name) == {'winner-codes.npy', 'log-factors.npy'}
         and all(Path(p['path']).parent == Path(value['receipt']['path']).parent for p in outputs),
         'Stage outputs must belong to the pinned receipt directory.')
    return by_name


def header(stream, name, count, file_size):
    need(exact(stream, 6, name) == b'\x93NUMPY', 'Invalid stage NPY magic.')
    major, minor = exact(stream, 2, name)
    need(major in (1, 2, 3) and minor == 0, 'Unsupported stage NPY version.')
    width = 2 if major == 1 else 4
    length = struct.unpack('<H' if major == 1 else '<I', exact(stream, width, name))[0]
    need(0 < length <= MAX_HEADER_BYTES, 'Stage NPY header exceeds size bound.')
    try:
        source = exact(stream, length, name).decode('utf-8' if major == 3 else 'latin1').strip()
        node = ast.parse(source, mode='eval').body
        need(isinstance(node, ast.Dict) and len(node.keys) == 3 and all(isinstance(k, ast.Constant) for k in node.keys)
             and len({k.value for k in node.keys}) == 3, 'Invalid stage NPY header fields.')
        fields = ast.literal_eval(node)
    except (ValueError, SyntaxError, UnicodeError) as error:
        raise DecodeError('Invalid stage NPY header.') from error
    dtype, shape, item = ('|i1', (count, RACES), 1) if name == 'winner-codes.npy' else ('<f8', (count,), 8)
    need(isinstance(fields, dict) and set(fields) == {'descr', 'fortran_order', 'shape'}
         and fields['descr'] == dtype and fields['fortran_order'] is False and fields['shape'] == shape
         and all(type(x) is int for x in fields['shape']), 'Stage NPY shape, dtype or order differs.')
    need(file_size == 6 + 2 + width + length + math.prod(shape) * item, 'Stage NPY file size differs.')


def main():
    value = request()
    with ExitStack() as stack:
        receipt, receipt_path, receipt_initial = opened_artifact(value['directory'], value['receipt'], 'Stage receipt')
        stack.callback(receipt.close)
        verify_sha(receipt, value['receipt']['sha256'], receipt_initial, receipt_path, 'Stage receipt')
        pins = receipt_data(receipt, value)
        opened = {}
        for name in ('winner-codes.npy', 'log-factors.npy'):
            stream, path, initial = opened_artifact(value['directory'], pins[name], name)
            stack.callback(stream.close)
            verify_sha(stream, pins[name]['sha256'], initial, path, name)
            stream.seek(0)
            header(stream, name, value['count'], initial.st_size)
            opened[name] = (stream, path, initial)
        winners, factors = opened['winner-codes.npy'][0], opened['log-factors.npy'][0]
        for offset in range(0, value['count'], 64):
            rows = min(64, value['count'] - offset)
            winner_bytes = exact(winners, rows * RACES, 'winner codes')
            factor_bytes = exact(factors, rows * 8, 'log factors')
            for i in range(rows):
                row = winner_bytes[i * RACES:(i + 1) * RACES]
                factor = factor_bytes[i * 8:(i + 1) * 8]
                need(all(x <= 2 for x in row) and math.isfinite(struct.unpack('<d', factor)[0]),
                     'Invalid winner code or nonfinite log factor.')
                sys.stdout.buffer.write(row + factor)
        for name, (stream, path, initial) in opened.items():
            verify_sha(stream, pins[name]['sha256'], initial, path, name)
        verify_sha(receipt, value['receipt']['sha256'], receipt_initial, receipt_path, 'Stage receipt')


if __name__ == '__main__':
    try:
        main()
    except (DecodeError, OSError, ValueError, KeyError, TypeError) as error:
        print(f'Weighted stage decode failed: {error}', file=sys.stderr)
        sys.exit(1)

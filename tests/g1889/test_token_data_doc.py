"""token-data-1889.md, the token contract, against the code: the row map and
widths, each row type's slots (every slot from 1 to its width less one, once),
the tile block, the relation ids and static planes, and the sizes and edge
bound it states. test_extraction.py and test_extraction_relations.py check
what the slots and records hold."""

from pathlib import Path
import re

import pytest

from g1889.relations import RelationId, RelationSize, StaticRelationId, static_relation_planes
from g1889.token_data import TileBlock, TokenDataSize, TokenRow, TokenWidth, get_num_tokens, get_token_widths

from tests.g1889.test_extraction_relations import BOUNDS

DOC = (Path(__file__).resolve().parents[2] / 'token-data-1889.md').read_text()
ROW_TYPES = ('GLOBAL', 'PRICE', 'DIVIDEND', 'CORP', 'PRIVATE', 'TRAIN', 'HEX', 'CANDIDATE', 'PLAYER')


def _table(heading: str) -> list[list[str]]:
    """The body of the first table under ``heading``, as cells."""
    lines = DOC[DOC.index(f'\n{heading}\n'):].splitlines()[1:]
    start = next(i for i, line in enumerate(lines) if line.startswith('|'))
    rows = []
    for line in lines[start + 2:]:
        if not line.startswith('|'):
            return rows
        rows.append([cell.strip() for cell in line.strip('|').split('|')])
    return rows


def _slots(cell: str) -> list[int]:
    return [s for a, b in re.findall(r'(\d+)(?:-(\d+))?', cell) for s in range(int(a), int(b or a) + 1)]


def test_the_row_map_and_widths():
    rows = _table('## Rows')
    assert [name for _, name, *_ in rows] == list(ROW_TYPES)
    widths = get_token_widths(5, 1)
    for i, (first, name, count, width, _) in enumerate(rows):
        start = int(first)
        assert start == TokenRow[f'ROW_{name}'] and int(width) == TokenWidth[f'TW_{name}'], name
        end = int(rows[i + 1][0]) if name != 'PLAYER' else get_num_tokens(5)
        assert count == ('max_players' if name == 'PLAYER' else str(end - start)), name
        assert set(widths[start:end]) == {int(width)}, name
    assert '`TOKEN_DIM` = 108' in DOC and TokenDataSize.TOKEN_DIM == 108
    assert '`num_tokens(max_players) = 90 + max_players`: 93, 94, 95' in DOC
    assert [get_num_tokens(n) for n in (3, 4, 5)] == [93, 94, 95] and TokenRow.ROW_PLAYER == 90


@pytest.mark.parametrize('name', ROW_TYPES)
def test_each_row_type_lists_every_slot_once(name):
    width = int(TokenWidth[f'TW_{name}'])
    slots = [s for cell, *_ in _table(f'## {name} ({width})') for s in _slots(cell)]
    assert sorted(slots) == list(range(1, width)), name


def test_the_tile_block():
    offsets = [s for cell, *_ in _table(f'## The tile block ({int(TileBlock.TW_TILE_BLOCK)})') for s in _slots(cell)]
    assert sorted(offsets) == list(range(TileBlock.TW_TILE_BLOCK)) and TileBlock.TILE_BLOCK == 2
    assert _table('## HEX (105)')[1][0] == _table('## CANDIDATE (108)')[1][0] == '2-77'


def test_the_relations():
    rows = _table('### Dynamic relations (IPC)')
    assert [(int(i), name) for i, name, *_ in rows] == [(int(r), r.name[4:]) for r in RelationId
                                                        if r != RelationId.NUM_RELATIONS]
    assert f'`num_relations` = {int(RelationId.NUM_RELATIONS)}' in DOC
    static = _table('### Static relations (model buffers)')
    named = {int(i): name for i, name, _ in static[1:]}
    assert static[0][:2] == ['0-5', 'HEX_SIDE_0 to HEX_SIDE_5'] and StaticRelationId.SREL_HEX_SIDE == 0
    assert named == {int(r): r.name[5:] for r in StaticRelationId
                     if r not in (StaticRelationId.SREL_HEX_SIDE, StaticRelationId.NUM_STATIC_RELATIONS)}
    assert static_relation_planes(5).shape == (10, 95, 95) == (StaticRelationId.NUM_STATIC_RELATIONS, 95, 95)
    assert '`max_relation_edges` = **2,048**' in DOC and RelationSize.MAX_RELATION_EDGES == 2048
    assert 'bounded by **1,844**' in DOC and sum(BOUNDS.values()) == 1844

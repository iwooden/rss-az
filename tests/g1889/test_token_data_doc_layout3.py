"""token-data-1889.md's layout 3 against the code, as
``tests/g1889/test_token_data_doc.py`` holds layouts 1 and 2: the parts'
names, shapes, and offsets, the trunk rows and widths, each row type's slots
(every slot from 1 to its width less one, once), the pair blocks' channels,
the choice table's columns, and each phase block's slots (every slot from 0
to its width less one, once). ``tests/g1889/test_extraction_layout3.py``
checks what the slots hold."""

import math
import re

import pytest

from g1889.token_data import (
    ChoiceTable3,
    InputOffset3,
    PairChannels3,
    TokenWidth3,
    TrunkRow3,
    get_input_parts,
    get_input_size,
)

from tests.g1889.test_token_data_doc import DOC, _slots, _table

LAYOUT = 3
ROW_TYPES = ('GLOBAL', 'CORP', 'PRIVATE', 'CENTER', 'PLAYER')
PHASE_BLOCKS = re.findall(r'\n#### (.+) \((\d+)\)\n', DOC[DOC.index('\n## Layout 3\n'):])


def _count(cell, max_players):
    """A table's count or offset: digits with commas, and terms in P."""
    total = 0
    for term in cell.replace(',', '').split('+'):
        term = term.strip()
        total += int(term[:-1] or 1) * max_players if term.endswith('P') else int(term)
    return total


def test_the_shapes_table_is_the_parts():
    rows = _table('### Shapes, layout 3')
    for capacity in (3, 4, 5):
        parts = get_input_parts(capacity, layout_version=LAYOUT)
        assert len(rows) == len(parts) + 1
        for (_, name, shape, offset, values), part in zip(rows, parts):
            assert name == f'`{part.name}`'
            assert tuple(_count(d, capacity) for d in shape.split('×')) == part.shape, part.name
            assert _count(offset, capacity) == part.offset, part.name
            if capacity == 5:
                assert int(values.replace(',', '')) == math.prod(part.shape), part.name
        total = rows[-1]
        assert total[0] == 'Total' and _count(total[2], capacity) == get_input_size(capacity, layout_version=LAYOUT)
    assert int(rows[-1][4].replace(',', '')) == get_input_size(5, layout_version=LAYOUT) == 5108
    assert '4,998, 5,053, and 5,108 values' in DOC


def test_the_rows_table_is_the_trunk():
    rows = _table('### Rows, layout 3')
    assert [name for _, name, *_ in rows] == list(ROW_TYPES)
    parts = {p.name: p for p in get_input_parts(5, layout_version=LAYOUT)}
    for first, name, count, width, _ in rows:
        assert int(first) == TrunkRow3[f'TRUNK3_{name}'] == parts[name].rows[0][0], name
        assert int(width) == TokenWidth3[f'TW3_{name}'] == parts[name].shape[1], name
        assert count == ('max_players' if name == 'PLAYER' else str(parts[name].rows[0][1])), name


@pytest.mark.parametrize('name', ROW_TYPES)
def test_each_row_type_lists_every_layout_3_slot_once(name):
    width = int(TokenWidth3[f'TW3_{name}'])
    slots = [s for cell, *_ in _table(f'### {name}, layout 3 ({width})') for s in _slots(cell)]
    assert sorted(slots) == list(range(1, width)), name


def test_the_pair_blocks_table_has_each_blocks_channels():
    channels, block = {}, None
    for name, shape, channel, *_ in _table('### Pair blocks, layout 3'):
        if name:
            block = name.replace(' × ', '_')
            channels[block] = (shape, [])
        channels[block][1].append(int(channel))
    parts = {p.name: p for p in get_input_parts(5, layout_version=LAYOUT)}
    for block, (shape, listed) in channels.items():
        assert listed == list(range(len(listed))), block
        assert len(listed) == int(PairChannels3[f'CH3_{block}']) == parts[block].shape[-1], block
        assert tuple(_count(d, 5) for d in shape.split('×')) == parts[block].shape, block
    assert set(channels) == {'PLAYER_CORP', 'PLAYER_PRIVATE', 'CORP_PRIVATE', 'CORP_CENTER', 'CENTER_CENTER'}


def test_the_choice_tables_columns_and_phase_blocks():
    columns = {cell: content for cell, content in _table('#### Columns')}
    assert list(columns) == ['0', '1, 2', '3-9', '10-46']
    assert _slots('0') == [ChoiceTable3.CHOICE3_HEX]
    assert _slots('1, 2') == [ChoiceTable3.CHOICE3_TILE, ChoiceTable3.CHOICE3_ROTATION]
    assert _slots('3-9') == list(range(ChoiceTable3.CHOICE3_SHARED, ChoiceTable3.CHOICE3_PHASE))
    assert _slots('10-46') == list(range(ChoiceTable3.CHOICE3_PHASE, ChoiceTable3.CHOICE3_WIDTH))
    shared = _table('#### The shared block')
    assert [(int(column), slot) for column, slot, *_ in shared] == [(3 + i, f'S{i}') for i in range(7)]
    widths = {}
    for heading, width in PHASE_BLOCKS:
        widths[heading] = int(width)
        if int(width):
            slots = [s for cell, *_ in _table(f'#### {heading} ({width})') for s in _slots(cell)]
            assert sorted(slots) == list(range(int(width))), heading
    assert max(widths.values()) == ChoiceTable3.CHOICE3_PHASE_WIDTH == widths['TILE and ER_BONUS'] == 37
    assert len(widths) == 11 and widths['DISCARD and PORT'] == 0
    assert int(InputOffset3.IN3_CHOICES) + ChoiceTable3.CHOICE3_ROWS * ChoiceTable3.CHOICE3_WIDTH \
        == int(InputOffset3.IN3_PLAYERS)

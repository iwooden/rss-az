"""Attention-relation planes for Graphormer-style token attention bias.

Each relation plane is a directed ``(query_token, key_token)`` uint8
matrix over the token capacity used by the model. For example,
``CORP_OWNS_COMPANY`` marks ``[corp_token, company_token]`` so a corp query
can be biased toward reading from its owned company key.

Binary planes contain 0/1, while share-count planes contain raw counts (0..7).
Sparse eval IPC carries uint8 (relation, query, key, value) records. Only the
model normalizes quantities; worker extraction preserves raw integer counts.
"""

from __future__ import annotations

from enum import IntEnum

from core.data import PY_SHARE_DIVISOR
from core.relations import (
    AttentionRelationIndex,
    get_attention_relation_coord_width,
    get_max_attention_relation_edges,
    get_num_attention_relations,
)


class AttentionRelation(IntEnum):
    """Directed relation planes over the model token list."""

    CORP_OWNS_COMPANY = int(AttentionRelationIndex.REL_CORP_OWNS_COMPANY)
    COMPANY_OWNED_BY_CORP = int(AttentionRelationIndex.REL_COMPANY_OWNED_BY_CORP)
    PLAYER_OWNS_COMPANY = int(AttentionRelationIndex.REL_PLAYER_OWNS_COMPANY)
    COMPANY_OWNED_BY_PLAYER = int(AttentionRelationIndex.REL_COMPANY_OWNED_BY_PLAYER)
    FI_OWNS_COMPANY = int(AttentionRelationIndex.REL_FI_OWNS_COMPANY)
    COMPANY_OWNED_BY_FI = int(AttentionRelationIndex.REL_COMPANY_OWNED_BY_FI)
    PLAYER_OWNS_CORP_SHARES = int(AttentionRelationIndex.REL_PLAYER_OWNS_CORP_SHARES)
    CORP_HAS_PLAYER_SHAREHOLDER = int(
        AttentionRelationIndex.REL_CORP_HAS_PLAYER_SHAREHOLDER
    )
    PLAYER_PRESIDENT_OF_CORP = int(AttentionRelationIndex.REL_PLAYER_PRESIDENT_OF_CORP)
    CORP_PRESIDENT_PLAYER = int(AttentionRelationIndex.REL_CORP_PRESIDENT_PLAYER)
    PLAYER_CORP_SHARE_COUNT = int(AttentionRelationIndex.REL_PLAYER_CORP_SHARE_COUNT)
    CORP_PLAYER_SHARE_COUNT = int(AttentionRelationIndex.REL_CORP_PLAYER_SHARE_COUNT)


NUM_ATTENTION_RELATIONS = get_num_attention_relations()
MAX_ATTENTION_RELATION_EDGES = get_max_attention_relation_edges()
ATTENTION_RELATION_COORD_WIDTH = get_attention_relation_coord_width()

# Applied once per forward by models, identically for dense and sparse inputs.
ATTENTION_RELATION_SCALES = tuple(
    1.0 / float(PY_SHARE_DIVISOR)
    if relation in (
        AttentionRelation.PLAYER_CORP_SHARE_COUNT,
        AttentionRelation.CORP_PLAYER_SHARE_COUNT,
    ) else 1.0
    for relation in AttentionRelation
)

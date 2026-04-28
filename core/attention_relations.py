"""Attention-relation planes for Graphormer-style token attention bias.

Each relation plane is a directed ``(query_token, key_token)`` boolean
matrix. For example, ``CORP_OWNS_COMPANY`` marks ``[corp_token, company_token]``
so a corp query can be biased toward reading from its owned company key.

The shared eval IPC path stores these planes as uint8, not torch.bool, because
PyTorch/Numpy boolean tensors are still byte-addressed and uint8 matches the
existing legal-mask wire dtype.
"""

from __future__ import annotations

from enum import IntEnum

from core.relations import AttentionRelationIndex, get_num_attention_relations


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


NUM_ATTENTION_RELATIONS = get_num_attention_relations()

#!/usr/bin/env python3
"""Check committed_action_ids accuracy across all games.

Simulates both the old and new Ruby extractor undo tracking logic
on raw action streams to find games where the fix changes committed_ids.

Usage:
    python tests/18xx_games/debug/check_committed_ids.py [game_id...]
"""

import json
import glob
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

SKIP_TYPES = {'program_share_pass', 'program_close_pass', 'program_disable', 'message'}


def simulate_committed_ids(actions, use_engine_stack=False):
    """Simulate the Ruby extractor's undo/redo tracking.

    Args:
        actions: Raw action list from game JSON.
        use_engine_stack: If True, use the fixed logic (track engine_action_stack
            to avoid popping snapshots for skip_type undos). If False, use the
            old logic (always pop one snapshot per undo).

    Returns:
        Set of committed action IDs (action IDs that would remain in the
        snapshot stack at the end).
    """
    # snapshot_ids tracks action_ids of snapshots on the stack
    snapshot_ids = []
    undo_groups = []  # stack of groups

    # For the fixed version: track all processed action IDs+types
    engine_stack = []  # list of (action_id, action_type)

    for action in actions:
        action_id = action.get('id')
        action_type = action.get('type', '')

        if action_type == 'undo':
            if use_engine_stack:
                engine_group = []
                snap_group = []

                target_id = action.get('action_id')
                if target_id is not None:
                    while engine_stack and engine_stack[-1][0] > target_id:
                        engine_group.append(engine_stack.pop())
                    while snapshot_ids and snapshot_ids[-1] > target_id:
                        snap_group.append(snapshot_ids.pop())
                else:
                    if engine_stack:
                        undone = engine_stack.pop()
                        engine_group.append(undone)
                        if undone[1] not in SKIP_TYPES and snapshot_ids:
                            snap_group.append(snapshot_ids.pop())

                if engine_group:
                    undo_groups.append(('engine', engine_group, snap_group))
            else:
                # OLD logic: always pop one snapshot per undo
                group = []
                target_id = action.get('action_id')
                if target_id is not None:
                    while snapshot_ids and snapshot_ids[-1] > target_id:
                        group.append(snapshot_ids.pop())
                else:
                    if snapshot_ids:
                        group.append(snapshot_ids.pop())

                if group:
                    undo_groups.append(('old', group))
            continue

        if action_type == 'redo':
            if use_engine_stack:
                if undo_groups:
                    _, engine_group, snap_group = undo_groups.pop()
                    for item in reversed(engine_group):
                        engine_stack.append(item)
                    for sid in reversed(snap_group):
                        snapshot_ids.append(sid)
            else:
                if undo_groups:
                    _, group = undo_groups.pop()
                    for sid in reversed(group):
                        snapshot_ids.append(sid)
            continue

        # Regular action
        if use_engine_stack:
            engine_stack.append((action_id, action_type))

        if action_type not in SKIP_TYPES:
            snapshot_ids.append(action_id)

    return set(snapshot_ids)


def check_game(game_id, verbose=False):
    """Check a single game for committed_ids differences.

    Returns (old_set, new_set, added, removed) or None if no undo/redo.
    """
    json_path = os.path.join(DATA_DIR, f'{game_id}.json')
    if not os.path.exists(json_path):
        return None

    data = json.loads(open(json_path).read())
    actions = data.get('actions', [])

    has_undo = any(a.get('type') in ('undo', 'redo') for a in actions)
    if not has_undo:
        return None

    old_committed = simulate_committed_ids(actions, use_engine_stack=False)
    new_committed = simulate_committed_ids(actions, use_engine_stack=True)

    added = new_committed - old_committed
    removed = old_committed - new_committed

    if verbose and (added or removed):
        # Look up action types for changed IDs
        action_by_id = {a['id']: a for a in actions if 'id' in a}
        if added:
            print(f"  ADDED to committed ({len(added)}):")
            for aid in sorted(added):
                a = action_by_id.get(aid, {})
                print(f"    {aid}: {a.get('type', '?')} entity={a.get('entity', '?')}")
        if removed:
            print(f"  REMOVED from committed ({len(removed)}):")
            for aid in sorted(removed):
                a = action_by_id.get(aid, {})
                print(f"    {aid}: {a.get('type', '?')} entity={a.get('entity', '?')}")

    return old_committed, new_committed, added, removed


def main():
    game_ids = sys.argv[1:] if len(sys.argv) > 1 else None

    if game_ids is None:
        # Discover all games
        game_ids = sorted(
            os.path.basename(f).removesuffix('.json')
            for f in glob.glob(os.path.join(DATA_DIR, '*.json'))
            if not f.endswith('_extract.json')
        )

    affected = []
    for gid in game_ids:
        result = check_game(gid, verbose=True)
        if result is None:
            continue
        old_set, new_set, added, removed = result
        if added or removed:
            print(f"Game {gid}: +{len(added)} -{len(removed)} committed IDs changed")
            affected.append(gid)

    print(f"\n{'='*60}")
    print(f"Total games checked: {len(game_ids)}")
    print(f"Games affected by fix: {len(affected)}")
    if affected:
        print(f"Affected: {affected}")


if __name__ == '__main__':
    main()

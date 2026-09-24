# Non-standard AlphaZero choices

Where search, self-play, and training depart from textbook AlphaZero. Defaults
are from `train/config.py`. Active runs are `train_configs/bigger-multi.json`
(v2) and `train_configs/v3-09-20-26.json` (v3). Tuned values change, so check
the config. Self-play policy diagnostics are in `train/POLICY_METRICS.md`.

## Search (`mcts/search.py`, `mcts/mcts_core.pyx`, `mcts/node.py`)

- **Per-player values, no rotation.** Each model's `value_head` applies tanh
  to every player token. Output is padded to `effective_max_players`, and
  evaluators slice it to the actual count. PUCT reads
  `value_sums[a, active_player_id]` using canonical ids
  (`_select_child_impl`). Player tokens stay in canonical order, with no
  negamax sign flip.
- **A tree edge is one multi-choice decision.** `DRIVER.apply_action`
  (`_auto_chain`) runs automated phases and single-legal-action decisions.
  Consecutive nodes can have any active player.
- **Plain PUCT, parent-value FPU.** U is `c_puct * P * sqrt(N_parent) /
  (1 + N)` with no `c_base` term. `N_parent` is `node.visit_count`, which
  counts the node's own evaluation. Unvisited edges hold the node's own NN
  value (`expand_node_sparse`). The first real backup replaces that value
  instead of adding to it (`_backup_node`).
- **Leaf-locked batching.** `run_search` queues up to `search_batch_size`
  leaves per evaluator call. Each queued leaf's parent-edge Q becomes `-inf`.
  `propagate_lock` locks an ancestor edge once all its child edges are locked.
  Visits are counted at selection and values at backup, so in-flight ancestor
  Q shrinks toward 0. A batch ends early if all root edges are locked or a
  pending leaf is re-selected. Terminal leaves back up without an NN call.
- **Subtree reuse, lazy catch-up.** `prepare_reuse_root` compacts `StatePool`
  in place to the chosen child's subtree. `_reset_root_for_reuse` zeroes only
  the new root's edge counts, so fresh Dirichlet noise matters; deeper nodes
  keep their stats. When PUCT picks an edge below its child's old visit count
  (`DESCEND_VIRTUAL_BACKUP`), `virtual_backup` adds the child's mean value once.
  Each such backup uses one of `num_simulations`. Reused searches therefore
  make fewer NN calls, and targets include replayed visits.
- **Dynamic Dirichlet alpha.** By default (`dirichlet_dynamic`), alpha is
  `dirichlet_alpha_numerator / num_legal` (10/n). `dirichlet_alpha` applies
  only when dynamic mode is off. Every root gets noise, reused roots included.
- **ACQ price cap.** `max_acq_price_actions > 0` (v3 config: 8) keeps only the
  lowest and highest halves of legal `ACQ_SELECT_PRICE` offsets
  (`enumerate_policy_actions`, `_filter_acq_price_root_priors`). The cap covers
  search, sampling, replay masks, and targets. Middle prices stay engine-legal
  but are never played or trained.

## Self-play targets (`train/self_play.py` `play_game`)

- **Separate action and target temperatures.** Actions sample visits under
  `temp_*`. Replay policy targets use `policy_target_temp_*`
  (`scale_visit_counts_by_temperature`). Each schedule holds its initial value
  through move `start`, then reaches its final value linearly at `end`. Moves
  count every searched decision in the game. Both default to 1.0 -> 0.5 over
  moves 60-120, so targets are sharpened and play never becomes greedy. If
  scalar start and end are both 0, per-player-count `*_anneal_starts/_ends`
  lists apply. Both active configs use those lists.
- **A0GB value targets.** `get_greedy_leaf_value` follows max-visit children
  to a terminal node or to a node whose best child has 0 visits. That node's
  `value_sum / visit_count` is its single NN value or its terminal reward.
- **Outcome/A0GB blend by epoch.** `compute_epoch_config` gives
  `value_blend_alpha = 0` before zero-indexed epoch `value_blend_start_epoch`
  (10), rising linearly to 1 at `value_blend_end_epoch` (200). Targets are
  `alpha * A0GB + (1 - alpha) * terminal_values`, frozen per replay row.
  Without an `EpochConfig`, they are pure A0GB. The v2 config ramps over epochs
  100-300. The v3 config sets both to 999, making targets pure game outcomes.
- **Blended terminal reward.** `compute_terminal_values` (`mcts/evaluator.py`)
  returns `terminal_blend * rank + (1 - terminal_blend) * margin`, default
  0.75. Rank spaces +1..-1 evenly with averaged ties. Margin is
  `n/(n-1) * (nw - mean) / max_nw`. In-tree terminal nodes use the same reward.
  Evaluators default to 0.5, so pass `config.terminal_blend`.
- **Epoch schedules.** `c_puct` anneals linearly from 3.5 to 2.5 over
  `c_puct_anneal_epochs` (20). Paths without an `EpochConfig` use
  `c_puct_final`. If all four `mcts_sims_*`/`mcts_ramp_*` fields are set, the
  simulation count also ramps; both active configs set them.

## Replay and training (`train/replay_buffer.py`, `train/trainer.py`, `train/main.py`)

- **Mixed player counts, one model.** `num_players: 0` with `min_players` and
  `max_players` (3-5 in both active configs) pads states, tokens, and values to
  `effective_max_players`. `build_epoch_player_count_schedule` splits games
  evenly by count, giving the remainder to the lowest. Value MSE ignores padded
  player slots.
- **Replay stores raw state.** Rows hold compact int16 state, dense masks and
  targets, and padded values. Tokens and relation planes are rebuilt with the
  current extractors when sampled, so extractor changes also affect old rows.
  Sampling is uniform.
- **Synchronous generations.** Each epoch plays `games_per_epoch` games with
  frozen weights, trains, then syncs eval servers. Training waits for
  `min_buffer_size`, then runs `training_steps_per_epoch * len(buffer) /
  buffer_capacity` steps (`_scaled_training_steps`). Early epochs therefore
  train only briefly.
- **Step-based LR schedule.** Warmup plus cosine decay to `lr_min` spans
  `lr_decay_end_epoch * training_steps_per_epoch` optimizer steps. Because of
  step scaling, `lr_min` arrives after that epoch.
- **Optimizer and loss.** The default `"muon"` puts 2-D weights on
  `torch.optim.Muon` (`match_rms_adamw`) and other parameters on an auxiliary
  AdamW. Both active configs use `"adamw"`. Weight decay skips Embedding,
  LayerNorm, and RMSNorm modules; `bias`; `*embeds`; `relation_bias_mult`;
  `relation_gains`; and `phase_mod.weight`. Policy cross-entropy uses masked
  logits and sharpened targets, and value loss is masked MSE. The
  multi-process trainer runs fp32 without compilation.

## Inference IPC (`train/eval_server.py`)

- **Wire format.** Workers write fp16 tokens, uint8 `UNIFIED_LOGIT_DIM` legal
  masks, and uint8 relation records (`MAX_ATTENTION_RELATION_EDGES`, 4). Each
  record is `(relation_id, query, key, value)`. Workers set a lock-free bitmap
  bit. The server masks and softmaxes on GPU and returns dense fp32 priors plus
  canonical values. Workers gather legal slots through `build_action_lut` and
  run no torch ops on the hot path.
- **Two relation formats.** Model `forward` accepts dense planes (trainer,
  `NNEvaluator`) or sparse records (eval server). It scatters sparse records
  straight into the per-head attention bias; the server never builds dense
  planes. V2 reads its original 10 binary relations, while v3 reads all
  relations and scales edges by value. Keep both formats in parity.
- **Precision and weights.** Eval servers keep autocast (`eval_dtype`) open
  with `cache_enabled=False`, because same-device servers share trainer
  parameters through CUDA IPC. After training, every server reloads a CPU
  snapshot (`_sync_eval_servers`). `bucketed` mode pads launches to powers of 2.

## Operations (`train/main.py`)

- **Shutdown.** `q` + Enter (TTY only) drains in-flight games into replay.
  If the epoch quota and `min_buffer_size` are met, it trains first. It then
  saves a checkpoint and replay. Ctrl-C exits without saving; workers ignore
  SIGINT.
- **Resume.** Config is the checkpoint's, overridden by `--config` JSON, then
  CLI flags; RNG state is restored. Replay loads from the single, overwritten
  `<checkpoint_dir>/replay_buffer/`, even when resuming an older checkpoint.
  A capacity, player-range, or shape mismatch skips it with only a warning.
- **Model selection.** `model_path` selects the implementation (default
  `nn/transformer-v2.py`). Its `INPUT_LAYOUT_VERSION` sets token width for
  every consumer (`get_model_input_spec`). `phase_conditioning`, `d_proj`, and
  `price_slot_fourier_bands` are v2-only, and `relation_input_mixing` is
  v3-only. Engine flags `v3_behavior` and `acq_same_president` are independent
  of the model (see `README.md`).

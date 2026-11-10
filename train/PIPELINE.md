# Non-standard AlphaZero choices

Where search, self-play, and training depart from textbook AlphaZero. Defaults
are from `train/config.py`. Active runs are `train_configs/bigger-multi.json`
(v2) and `train_configs/v3-09-20-26.json` (v3). Tuned values change, so check
the config. Self-play policy diagnostics are in `train/POLICY_METRICS.md`.

Search, the evaluators, the eval server, replay, the trainer, and `play_game`
reach the game only through its `games.Game` adapter, which each process
builds with `get_game(config)`. Dimensions come from the adapter and its
`policy` layout (`unified_dim`, `max_action_size`, `build_action_lut()`).
Notes below name RSS's implementations where they matter
([`1889/pipeline-refactor.md`](../1889/pipeline-refactor.md)).

## Search (`mcts/search.py`, `mcts/mcts_core.pyx`, `mcts/node.py`)

- **Per-player values, no rotation.** Each model's `value_head` applies tanh
  to every player token. Output is padded to `effective_max_players`, and
  evaluators slice it to the actual count. PUCT reads
  `value_sums[a, active_player_id]` using canonical ids
  (`_select_child_impl`). Player tokens stay in canonical order, with no
  negamax sign flip. With v3 `zero_sum_values`, the value head subtracts the
  mean over real players after tanh, so values sum to zero and may exceed ±1.
- **A tree edge is one multi-choice decision.** `game.apply_action` (RSS:
  `DRIVER.apply_action`'s `_auto_chain`) runs automated phases and
  single-legal-action decisions.
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
- **Optional forced playouts.** With `forced_playouts_k > 0` (default 0;
  KataGo uses 2), self-play roots select any child with fewer than
  `sqrt(k * P * N)` visits first, using the noised priors and the root's
  `visit_count` as N. Locked edges are never forced. Only `play_game` sets
  `MCTSConfig.forced_playouts_k`; tournament, live play and analysis leave
  it at 0.
- **ACQ price cap.** `max_acq_price_actions > 0` (v3 config: 8) keeps only the
  lowest and highest halves of legal `ACQ_SELECT_PRICE` offsets. It is a
  runtime flag on `GameState`, outside the `int16` row: self-play states get it
  from the config, and search inherits it from the root, as it does every
  runtime flag: `StatePool` keeps the root state, and `game.scratch_state`
  builds expansion states that carry its flags.
  `enumerate_policy_actions` applies it, so root and leaf priors are a softmax
  over the capped set; it covers search, sampling, replay masks, and targets.
  Middle prices stay engine-legal but are never played or trained. Tournament
  seats and live play search under their checkpoint's cap; the live server's
  `--max-acq-price-actions` overrides it.

## Self-play targets (`train/self_play.py` `play_game`)

- **Separate action and target temperatures.** Actions sample visits under
  `temp_*`. Replay policy targets use `policy_target_temp_*`
  (`scale_visit_counts_by_temperature`). Each schedule holds its initial value
  through move `start`, then reaches its final value linearly at `end`. Moves
  count every searched decision in the game. Both default to 1.0 -> 0.5 over
  moves 60-120, so targets are sharpened and play never becomes greedy. If
  scalar start and end are both 0, per-player-count `*_anneal_starts/_ends`
  lists apply. Both active configs use those lists.
- **Optional action windows by game length.** With `temp_anneal_start_pcts`
  and `temp_anneal_end_pcts` (shares 0-1, one per player count; default
  unset), the action temperature's window is those shares of the last finished
  epoch's mean game length at that player count: 30% and 60% of 500 moves
  anneal over moves 150-300. The main loop turns them into move numbers each
  epoch and sends them in `EpochConfig`; `schedule/temp_anneal_*` logs them.
  Lengths count only games played from the opening, and a player count with
  none in an epoch keeps its last length. A count with no length yet, as every
  count in a run's first epoch, plays at `temp_initial` throughout. The
  config's move numbers then apply only to `play_game` calls whose
  `EpochConfig` carries no windows; the target temperature keeps its own.
- **Optional starts from supplied positions.** With `start_positions_path` and
  `human_start_fraction` f > 0 (default 0; 1889 only), the main process loads
  the game's start-position file once (`Game.load_start_positions`), which
  refuses one built for another state layout, player capacity, or set of
  engine options. Each epoch, `build_epoch_start_schedule` starts f of each
  player count's games, to the nearest game, from a position: the last of
  every 1/f of that count's games, so they are spread through the epoch, each
  drawn uniformly with replacement from that count's positions. A count with
  none plays from the opening. The draw is seeded by `seed`, the epoch, and
  the player count, apart from the run's generator, so game seeds are the same
  with starts or without. The worker's task carries the row and its decision
  index in its source game. `play_game` plays from a copy of the row
  (`start_row`) under the config's engine options, and both temperature
  schedules and the policy metrics' stages read that index plus the decisions
  played (`start_move`), so the game anneals as one from the opening would
  there. Its first `start_explore_moves` (20) decisions sample actions at
  `temp_initial` whatever the schedule says, so games from one position part
  ways; targets keep their schedule. `total_moves` and the per-move means
  count only the decisions played. A started game's rows carry replay's flag
  bit 0 and are trained on like any others. Its stats go to
  `self_play_started` and `self_play_started_<n>p`, with `games` and
  `start_move_mean`, and stay out of `self_play_aggregate`, `self_play_<n>p`,
  and the lengths behind the temperature windows; the console counts every
  game together.
- **A0GB value targets.** `get_greedy_leaf_value` follows max-visit children
  to a terminal node or to a node whose best child has 0 visits. That node's
  `value_sum / visit_count` is its single NN value or its terminal reward.
- **Policy-target pruning.** `policy_target_visits` (`mcts/search.py`)
  chooses the visits that targets are built from, per
  `policy_target_pruning`. `none` uses raw visits. Otherwise the most-visited
  child keeps all its visits, and every other child loses visits while its
  PUCT, with Q held fixed, stays below the best child's unpruned PUCT; a child
  left with one visit drops to zero. `forced` is KataGo's version: noised
  priors, removal capped at the child's forced allotment. `raw_prior` uses
  the unnoised prior with no cap, so it also strips visits that root noise or
  since-revised Q estimates earned. Forced playouts require one of these.
  Action sampling and A0GB use unpruned visits.
- **Outcome/A0GB blend by epoch.** `compute_epoch_config` gives
  `value_blend_alpha = 0` before zero-indexed epoch `value_blend_start_epoch`
  (10), rising linearly to `value_blend_final` (default 1) at
  `value_blend_end_epoch` (200). Targets are
  `alpha * A0GB + (1 - alpha) * terminal_values`, frozen per replay row.
  Without an `EpochConfig`, they are pure A0GB. The v2 config ramps over epochs
  100-300; the v3 config over 10-200.
- **Blended terminal reward.** `compute_terminal_values` (`mcts/evaluator.py`)
  returns `terminal_blend * rank + (1 - terminal_blend) * margin`, default
  0.75. Rank spaces +1..-1 evenly with averaged ties. Margin is
  `n/(n-1) * (nw - mean) / max_nw`. In-tree terminal nodes use the same reward.
  Evaluators default to 0.5, so pass `config.terminal_blend`.
- **Optional soft rank.** With `terminal_rank_tau` > 0 (default 0; no active
  config sets it), the rank term is each player's mean over opponents of
  `tanh((nw - opponent_nw) / (tau * mean_nw))`. It equals the hard rank once
  every gap is several `tau` wide and keeps its tie values, but a near-tie no
  longer swings the target by a whole place. It is for games whose final net
  worths bunch, as 1889's do ([`1889/v1-review.md`](../1889/v1-review.md) §3).
  Evaluators default to 0, so pass `config.terminal_rank_tau` beside the blend.
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
  targets, and padded values. Tokens and relation inputs are rebuilt with the
  current extractors when sampled, so extractor changes also affect old rows.
- **Rows know their game.** Each row carries its game's serial id, flags, and
  final net worths in canonical order (-1 past the game's players). One
  `add_stacked` call is one game and takes the buffer's next id, so ids never
  decrease in write order. Flag bit 0 marks a game started from a supplied
  position (`play_game`'s `start_row`). Rows saved before these columns load as
  unknown: id -1, flags 0, net worths -1.
- **Optional epoch window.** The main loop calls `begin_epoch` and the buffer
  records each epoch's first game id. With `buffer_epochs` N > 0, sampling
  draws only from the rows of the newest N epochs that still hold rows: the
  ring's newest run of rows. The `min_buffer_size` check and
  `buffer/priority_*` see that window too. `buffer_capacity` is then only the
  memory cap; a window that outgrows it loses its oldest rows, with a warning
  once an epoch. Rows older than the record, as in a buffer saved without it,
  count as one epoch. With N = 0 (default), sampling is over the whole buffer.
- **Optional prioritized replay.** Each row stores KL(policy target ||
  unnoised prior) from self-play (`kl_to_prior`). With
  `replay_priority_fraction` f > 0, `round(f * batch_size)` rows per batch
  are drawn with replacement in proportion to `KL ** replay_priority_exponent`
  and the rest uniformly; rows saved without a priority count as average. The
  policy gradient uses the batch as drawn. Value loss and all reported
  training metrics weight rows by `uniform / mixture` probability, so they
  estimate uniform values; `policy_kl_sampled` is the unweighted KL actually
  trained on. `buffer/priority_*` scalars track the sampling skew. With
  f = 0 (default), sampling is uniform without replacement.
- **Synchronous generations.** Each epoch plays `games_per_epoch` games with
  frozen weights, trains, then syncs eval servers. Training waits for
  `min_buffer_size`, then runs `training_steps_per_epoch * len(buffer) /
  buffer_capacity` steps (`_scaled_training_steps`). Early epochs therefore
  train only briefly. With `buffer_epochs`, the fraction is instead the
  window's epochs over `buffer_epochs`.
- **Optional steps from a replay ratio.** With `replay_ratio` r > 0 (default
  0; it needs `buffer_epochs`), `training_steps_per_epoch` is unused. An epoch
  trains `r * its rows / batch_size` steps, times the same window fraction
  (`_epoch_training_steps`), so a row is drawn about r times over the
  `buffer_epochs` epochs it is sampled from, in the first epochs as later. The
  epoch's rows are those of its games still in the buffer, with any a graceful
  shutdown left it before a resume.
- **LR schedule in steps, or in epochs.** Warmup plus cosine decay to `lr_min` spans
  `lr_decay_end_epoch * training_steps_per_epoch` optimizer steps. Because of
  step scaling, `lr_min` arrives after that epoch. With `replay_ratio`, steps
  vary by epoch and the schedule reads progress in epochs instead: the epoch's
  index plus the share of its steps done (`Trainer.begin_epoch`). It is the
  same curve, the step schedule's value had every epoch the current one's
  steps, so the two agree at constant steps. `lr_min` then arrives at
  `lr_decay_end_epoch`, and an epoch that trains nothing still counts. The
  capacity probe and `cloud/profile_1889.py` train without epochs, so they
  set `replay_ratio` to 0 and keep the step schedule.
- **Optimizer and loss.** The default `"muon"` puts 2-D hidden-layer weights
  on `torch.optim.Muon` (`match_rms_adamw`) and other parameters on an
  auxiliary AdamW. Layers listed by the model's `input_output_layers()`
  (raw-feature input projections and final head Linears) stay on AdamW with
  decay; policy-head hidden layers remain on Muon. Both active configs use `"adamw"`. Weight decay skips Embedding,
  LayerNorm, and RMSNorm modules; `bias`; `*embeds`; `relation_bias_mult`;
  `relation_gains`; and `phase_mod.weight`. Policy cross-entropy uses masked
  logits and sharpened targets, and value loss is masked MSE. The
  multi-process trainer runs fp32 without compilation.
- **Gradient norm.** `epoch/grad_norm_avg` and `_max` are the total norm
  before `grad_clip`, and `epoch/grad_clipped_share` the share of steps clipped.
- **Optional held-out loss.** With `heldout_rows` n > 0 (default 0; 1889-v1's
  design sets 32,768), each epoch, once self-play has added its games and
  before it trains, n of its rows (all, if fewer) are drawn uniformly without
  replacement: rows no step has trained on. The graceful shutdown's training
  draws them too; an epoch it leaves untrained draws them when replayed. The
  draw is seeded by `seed` and the epoch, apart from the run's generator, so
  training and games are the same with it or without. `Trainer.score_batches`
  scores the rows with the current weights by the training loss's own input
  path and functions, in eval mode under `no_grad`, with no update and no
  importance weights; the capacity probe scores its held-out rows the same
  way. `heldout/value_mse` and `heldout/policy_kl` (cross-entropy minus target
  entropy) cover the sample. The `_opening` suffix covers games played from
  the opening, `_q1` to `_q4` their rows by quarter of the game (a row's place
  among its game's rows in the buffer), and `_<phase>` by decision phase, also
  opening games only; `_started` covers games started from a position.
  `heldout/rows` and `heldout/rows_started` count rows, and
  `heldout/value_mse_over_training` divides by the epoch's mean training value
  loss.

## Inference IPC (`train/eval_server.py`)

- **Wire format.** Workers write fp16 tokens, uint8 legal masks over the
  policy's `unified_dim` slots, and uint8 relation records
  (`game.max_relation_edges`, 4). Each record is
  `(relation_id, query, key, value)`. Workers set a lock-free bitmap bit. The
  server masks and softmaxes on GPU and returns dense fp32 priors plus
  canonical values. Workers gather legal slots through the policy's
  `build_action_lut()` and run no torch ops on the hot path.
  `SharedEvalBuffers` carries the policy layout to the server, never the
  `Game`: each process builds its own.
- **Two relation formats, per game.** `Game.relation_format` sets what replay,
  the trainer, `NNEvaluator`, the eval server, and the compile warmups send. For
  RSS (`"dense"`), model `forward` accepts dense planes (trainer, `NNEvaluator`)
  or sparse records (eval server). It scatters sparse records straight into the
  per-head attention bias; the server never builds dense planes. V2 reads its
  original 10 binary relations, while v3 reads all relations and scales edges by
  value. For 1889 (`"flat"`), all of them send flat `(N, 5)` int16 records: the
  caller flattens the padded wire records on the CPU
  (`flatten_relation_records`). N is a dynamic dimension in both batch-shape
  modes. A dynamic launch ships its real records; a bucketed one rounds them up
  with value-0 rows to B × K, K a power of two from 256 to 2,048, the small
  set of shapes CUDA graphs need. 1889's model also accepts dense planes, for
  tests. Keep each model's formats in parity.
- **Precision and weights.** Eval servers keep autocast (`eval_dtype`) open
  with `cache_enabled=False`, because same-device servers share trainer
  parameters through CUDA IPC. Without autocast (no `eval_dtype`, or a CPU
  server) the server upcasts the fp16 wire tokens to fp32 on the device. After
  training, every server reloads a CPU snapshot (`_sync_eval_servers`).
  `bucketed` mode pads launches to powers of 2.
- **Launch and stage tags.** `eval_server/` gives each epoch's real and
  launched (padded) rows, launches by launched size, and the serve loop's time
  by stage (`profile_stats.STAGES`); GPU work it doesn't wait on is uncounted.

## Operations (`train/main.py`)

- **Shutdown.** `q` + Enter (TTY only) drains in-flight games into replay.
  If the epoch quota and `min_buffer_size` are met, it trains first. It then
  saves a checkpoint and replay. Ctrl-C exits without saving; workers ignore
  SIGINT.
- **Resume.** Config is the checkpoint's, overridden by `--config` JSON, then
  CLI flags; RNG state is restored. An override that changes the game is an
  error. Replay loads from the single, overwritten
  `<checkpoint_dir>/replay_buffer/`, even when resuming an older checkpoint.
  A buffer saved at another capacity keeps its newest rows that fit, in order,
  priorities and game columns included. The epoch record is saved with the
  buffer: resuming into a partly played epoch adds its new games to that
  epoch, and any other resume begins a new one. Another game's buffer, or a player-range or shape mismatch,
  skips it with only a warning. Checkpoints and buffers without a game are
  RSS's. A changed `training_steps_per_epoch` moves the LR schedule's span from
  the restored step: the scheduler keeps its step, and this run's lambda. With
  `replay_ratio` the rate depends only on the epoch and the share of its steps
  done, so a resume continues it exactly, and turning `replay_ratio` on at a
  resume moves the rate to that epoch's. The checkpoint stores the game
  lengths `temp_anneal_*_pcts` read, as of its epoch, so a resumed epoch, or
  one replayed after a shutdown, plays in the windows it would have. A
  checkpoint saved without them plays its next epoch at `temp_initial`. An
  epoch's start positions follow from `seed`, the epoch, and the file, not
  from the restored RNG state, so a resumed epoch draws the starts it would
  have. An epoch replayed after a shutdown draws them again for the same
  games, under new game seeds, beside any of its games the shutdown left in
  the buffer.
- **Model selection.** `model_path` selects the implementation; every config
  that builds a model names one (no default). A module that defines
  `build_model(config, spec)` builds its own model for any game; any other is
  an RSS transformer. Every module declares `INPUT_LAYOUT_VERSION`, which sets
  token width for every consumer (`get_model_input_spec`).
  `phase_conditioning`, `d_proj`, and `price_slot_fourier_bands` are v2-only,
  `relation_input_mixing` is v3-only, and `zero_sum_values` is v3's and 1889's.
  Engine flags `v3_behavior` and `acq_same_president` are independent of the model (see `README.md`).

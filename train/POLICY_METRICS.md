# Monitoring policy learning and overconfidence

New self-play scalars are written every epoch under
`self_play_aggregate/policy` and `self_play_{3p,4p,5p}/policy`.
For example, start with `self_play_aggregate/policy/all/prior_top1_mean`.
Collection is automatic and does not require strategy traces or a config change.

Each prefix contains `all`, `phase/<name>`, and `stage/<name>` groups. Phase names
match the existing training loss names (`invest`, `bid`, `ipo`, etc.). Stages are
relative to the **policy-target temperature schedule's decision index**:
`pre_anneal` through its start, `anneal` strictly between start and end, and
`post_anneal` at/after its end. If start and end coincide, the boundary belongs
to `pre_anneal`, matching the temperature function. These are not game turns or
equal thirds of a game. Each player count uses its own configured window.

## Denominators and distributions

- All averages weight **decisions**, not games. `decision_count` counts positions
  with at least two search-available actions. Forced positions are excluded from
  every mean and fraction, and reported separately as `forced_count`.
- `prior` is the network policy **before root Dirichlet noise**, normalized over
  search-available actions, including the acquisition-price action restriction.
  Reused subtrees supply their existing NN priors; no extra inference is run.
- `search` is normalized raw root visit counts, including subtree reuse backups.
- `target` is those visits after the configured policy-target temperature.
- Probabilities and fractions use **0–1**, not 0–100. Entropies and KL use nats.
- These metrics compare the same position's distributions at self-play time.
  They are distinct from `epoch/policy_kl_avg`, which measures the changing
  training network against previously stored replay targets.

## Useful plots

| Scalar within each group | Meaning |
| --- | --- |
| `prior_top1_mean` | Average largest unnoised NN prior; multiply by 100 for percentage. |
| `prior_above_95_fraction`, `prior_above_98_fraction` | Fraction of non-forced decisions whose largest prior exceeds 0.95 or 0.98. Corresponding `_count` scalars give population sizes. |
| `prior_entropy_mean` | Average NN policy entropy. Interpret alongside phase/action counts. |
| `search_top1_mean`, `search_entropy_mean` | Concentration of raw search visits. |
| `target_top1_mean`, `target_entropy_mean` | Concentration after target sharpening. |
| `search_top1_changed_fraction` | Fraction where none of the most-visited actions is a highest-prior action. Ties are treated as agreement (prior tolerance 1e-7). |
| `search_choice_prior_mean` | Prior probability of the most-visited action, favoring the highest prior among tied visit winners. |
| `search_changed_choice_prior_mean` | Same, conditional on search changing the preferred action. Its denominator is `search_changed_count`. |
| `search_changed_from_above_95_fraction`, `search_changed_from_above_98_fraction` | Change rate within the corresponding high-confidence prior population. |
| `search_unvisited_fraction_mean` | Average fraction of available actions receiving zero root visits. |
| `search_kl_to_prior_mean` | KL(raw search visits \|\| unnoised prior). |
| `target_kl_to_prior_mean` | KL(temperature-adjusted target \|\| unnoised prior). |

Conditional means/fractions are omitted when their denominator is zero;
they do not imply a zero measurement. Groups containing only forced moves
likewise emit counts but no averages. KL logarithms floor probabilities at
1e-12 to keep float32 softmax underflow finite; zero-weight terms contribute zero.

Three `target_kl_*_contribution` scalars partition the target KL:

- `different_top`: target and prior have disjoint sets of preferred actions.
- `same_top_prior_sharper`: they share a preferred action, but prior entropy is
  lower than target entropy by more than 1e-7 nats.
- `same_top_prior_not_sharper`: remaining agreement positions, including equal
  entropy.

Each contribution divides by **all non-forced decisions**, so the three sum
to `target_kl_to_prior_mean`. `same_top_prior_sharper_fraction` reports how often
the second category occurs. Entropy compares overall concentration, not whether
every individual action probability is over/underestimated.

Growing search disagreement can reflect useful corrections, exploration noise,
or unreliable search; it does not by itself establish stronger play. A falling
change rate among 98%-confident priors can mean either good confidence or search
starvation. Inspect examples or compare search budgets to distinguish them.
Root noise remains enabled according to the training config: only the *prior
measurement* is unnoised. No extra noise-free searches are performed.

## Training fit by phase

`epoch/policy_target_entropy_<phase>_avg` and `epoch/policy_kl_<phase>_avg`
accompany the existing `epoch/policy_loss_<phase>_avg`. Cross-entropy equals
target entropy plus KL, helping distinguish more difficult targets from worse
fitting. `epoch/policy_samples_<phase>` counts sampled training rows, including
repeated draws and forced decisions. These diagnostics use the existing batch
forward pass and scalar transfer; they do not affect gradients.

Phase loss, target entropy and KL are weighted by sampled rows across the epoch.
The existing phase loss curves therefore change from averaging nonempty batch
means to averaging sampled rows when this logging is installed. Global loss
aggregation and training objectives are unchanged.

The collection adds one sparse prior copy per root and small CPU reductions per
played decision; it adds no NN evaluations and stores no extra replay arrays.
Worker IPC carries only additive summaries. A running process must be restarted
from a checkpoint with this code to begin collecting the new metrics; old events
cannot supply the missing prior measurements retroactively.

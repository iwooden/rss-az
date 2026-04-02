Fix the sparse checkout issue that beads sometimes causes in git worktrees.

## Instructions

Run `git sparse-checkout disable` in the current worktree. This resets `core.sparseCheckout` to false in the worktree's `config.worktree` file.

Then verify the fix by running `git config core.sparseCheckout` and confirming it returns `false`.

## Background

Beads uses a dedicated sparse-checkout worktree at `.git/beads-worktrees/beads-sync` that only checks out `.beads/` files. Sometimes this sparse checkout config leaks into user worktrees, setting `core.sparseCheckout = true` with a pattern of `/.beads/` only. This can cause files to appear missing or git operations to behave unexpectedly.

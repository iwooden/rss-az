# Self-Play Analysis: seed=42, 6400 simulations/move
# Checkpoint: checkpoints/backups/tfinal-small/checkpoint_epoch_0325.pt
# Noise: epsilon=0.15, dynamic alpha=5.0/K | Terminal blend: 0.75

Phase: INVEST  |  Turn: 1  |  CoO Level: 1  |  Active Player: 0  |  End Card: no

**Players**
  P0: $30 (NW $30) order=0 income=$0
  P1: $30 (NW $30) order=1 income=$0
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [3]: BSE (fv=$2, 1★, inc=$1), KME (fv=$5, 1★, inc=$2), MHE (fv=$8, 1★, inc=$2)

**Deck**: 17 remaining


---

### Step 0: P0 [INVEST]

  NN Values: P0=+0.078, P1=-0.050, P2=-0.054
  NN Priors (top 4 of 4 legal):
     1.  48.7% ( -4.2pp) ███████████████████ AUCTION slot 1 (KME, face $5)
     2.  23.2% ( +2.0pp) █████████ AUCTION slot 0 (BSE, face $2)
     3.  21.6% ( -0.6pp) ████████ AUCTION slot 2 (MHE, face $8)
     4.   6.5% ( +2.7pp) ██ PASS (INVEST)

  MCTS Visits (top 4, 6400 total):
     1.  4394 (68.7%) Q=+0.022 ███████████████████████████ AUCTION slot 1 (KME, face $5)
     2.   965 (15.1%) Q=+0.006 ██████ AUCTION slot 2 (MHE, face $8)
     3.   807 (12.6%) Q=-0.007 █████ AUCTION slot 0 (BSE, face $2)
     4.   234 ( 3.7%) Q=-0.019 █ PASS (INVEST)
  A0GB Value: P0=-0.159, P1=+0.391, P2=-0.196 (depth: 42)

  **Action: AUCTION slot 1 (KME, face $5)**

Phase: BID_IN_AUCTION  |  Turn: 1  |  CoO Level: 1  |  Active Player: 0  |  End Card: no

**Players**
  P0: $30 (NW $30) order=0 income=$0
  P1: $30 (NW $30) order=1 income=$0
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [3]: BSE (fv=$2, 1★, inc=$1), KME (fv=$5, 1★, inc=$2), MHE (fv=$8, 1★, inc=$2)

**Deck**: 17 remaining

**Auction**: KME current bid=$0 high bidder=P-1 starter=P0

### Step 1: P0 [BID_IN_AUCTION]

  **Auction**: KME current bid=$0 high bidder=P-1 starter=P0

  NN Values: P0=+0.043, P1=-0.020, P2=-0.042
  NN Priors (top 10 of 15 legal):
     1.  85.0% (-11.0pp) ██████████████████████████████████ BID $9
     2.   3.6% ( +0.1pp) █ BID $6
     3.   3.3% ( -0.1pp) █ BID $8
     4.   3.2% ( -0.3pp) █ BID $7
     5.   2.3% ( -0.3pp)  BID $5
     6.   0.6% ( +7.8pp)  BID $11
     7.   0.6% ( +0.2pp)  BID $16
     8.   0.4% ( +1.2pp)  BID $12
     9.   0.4% ( +1.1pp)  BID $10
    10.   0.2% ( -0.0pp)  BID $13

  MCTS Visits (top 10, 6400 total):
     1.  6084 (95.1%) Q=+0.026 ██████████████████████████████████████ BID $9
     2.    60 ( 0.9%) Q=-0.042  BID $6
     3.    58 ( 0.9%) Q=-0.036  BID $8
     4.    57 ( 0.9%) Q=-0.158  BID $11
     5.    51 ( 0.8%) Q=-0.035  BID $7
     6.    40 ( 0.6%) Q=-0.033  BID $5
     7.    27 ( 0.4%) Q=-0.060  BID $12
     8.    16 ( 0.2%) Q=-0.084  BID $10
     9.     1 ( 0.0%) Q=-0.641  BID $13
    10.     1 ( 0.0%) Q=-0.730  BID $14
  A0GB Value: P0=+0.197, P1=+0.070, P2=-0.183 (depth: 41, vbackups: 4393)

  **Action: BID $9**

### Step 2: P1 [BID_IN_AUCTION]

  **Auction**: KME current bid=$9 high bidder=P0 starter=P0

  NN Values: P0=+0.015, P1=-0.005, P2=-0.056
  NN Priors (top 10 of 11 legal):
     1.  83.6% ( -9.0pp) █████████████████████████████████ PASS (BID_IN_AUCTION)
     2.  12.9% ( -1.7pp) █████ BID $10
     3.   1.3% ( +1.4pp)  BID $11
     4.   0.5% ( +2.5pp)  BID $12
     5.   0.4% ( +1.7pp)  BID $13
     6.   0.3% ( +0.9pp)  BID $17
     7.   0.3% ( +1.4pp)  BID $16
     8.   0.3% ( +0.0pp)  BID $15
     9.   0.2% ( +0.6pp)  BID $14
    10.   0.1% ( +1.3pp)  BID $18

  MCTS Visits (top 10, 6400 total):
     1.  6084 (95.1%) Q=+0.020 ██████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   188 ( 2.9%) Q=-0.044 █ BID $10
     3.    35 ( 0.5%) Q=-0.067  BID $11
     4.    31 ( 0.5%) Q=-0.117  BID $12
     5.    23 ( 0.4%) Q=-0.080  BID $13
     6.    21 ( 0.3%) Q=-0.111  BID $16
     7.    14 ( 0.2%) Q=-0.122  BID $18
     8.     1 ( 0.0%) Q=-0.738  BID $14
     9.     1 ( 0.0%) Q=-0.793  BID $15
    10.     1 ( 0.0%) Q=-0.848  BID $17
  A0GB Value: P0=+0.099, P1=+0.063, P2=-0.132 (depth: 39, vbackups: 6068)

  **Action: PASS (BID_IN_AUCTION)**

### Step 3: P2 [BID_IN_AUCTION]

  **Auction**: KME current bid=$9 high bidder=P0 starter=P0

  NN Values: P0=+0.001, P1=+0.009, P2=-0.050
  NN Priors (top 10 of 11 legal):
     1.  90.4% (-13.5pp) ████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   7.5% ( -1.0pp) ██ BID $10
     3.   0.7% ( +4.9pp)  BID $11
     4.   0.3% ( -0.0pp)  BID $17
     5.   0.3% ( -0.0pp)  BID $12
     6.   0.2% ( +0.3pp)  BID $13
     7.   0.2% ( +1.9pp)  BID $15
     8.   0.2% ( +1.2pp)  BID $14
     9.   0.1% ( +3.9pp)  BID $16
    10.   0.1% ( +2.4pp)  BID $18

  MCTS Visits (top 10, 6400 total):
     1.  6167 (96.4%) Q=-0.067 ██████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   122 ( 1.9%) Q=-0.122  BID $10
     3.    46 ( 0.7%) Q=-0.223  BID $11
     4.    17 ( 0.3%) Q=-0.551  BID $18
     5.    15 ( 0.2%) Q=-0.413  BID $16
     6.    15 ( 0.2%) Q=-0.313  BID $14
     7.    14 ( 0.2%) Q=-0.288  BID $15
     8.     1 ( 0.0%) Q=-0.535  BID $12
     9.     1 ( 0.0%) Q=-0.664  BID $13
    10.     1 ( 0.0%) Q=-0.844  BID $17
  A0GB Value: P0=+0.099, P1=+0.063, P2=-0.132 (depth: 38, vbackups: 6069)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 1  |  CoO Level: 2  |  Active Player: 1  |  End Card: no

**Players**
  P0: $21 (NW $26) order=0 income=$2  companies=[KME]
  P1: $30 (NW $30) order=1 income=$0
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [2]: BSE (fv=$2, 1★, inc=$1), MHE (fv=$8, 1★, inc=$2)

**Deck**: 16 remaining


### Step 4: P1 [INVEST]

  NN Values: P0=+0.031, P1=+0.022, P2=-0.063
  NN Priors (top 3 of 3 legal):
     1.  58.9% ( -1.6pp) ███████████████████████ AUCTION slot 0 (BSE, face $2)
     2.  29.6% ( -2.3pp) ███████████ AUCTION slot 1 (MHE, face $8)
     3.  11.5% ( +3.8pp) ████ PASS (INVEST)

  MCTS Visits (top 3, 6400 total):
     1.  5719 (89.4%) Q=+0.027 ███████████████████████████████████ AUCTION slot 0 (BSE, face $2)
     2.   491 ( 7.7%) Q=-0.036 ███ AUCTION slot 1 (MHE, face $8)
     3.   190 ( 3.0%) Q=-0.070 █ PASS (INVEST)
  A0GB Value: P0=+0.099, P1=+0.063, P2=-0.132 (depth: 37, vbackups: 6140)

  **Action: AUCTION slot 0 (BSE, face $2)**

Phase: BID_IN_AUCTION  |  Turn: 1  |  CoO Level: 2  |  Active Player: 1  |  End Card: no

**Players**
  P0: $21 (NW $26) order=0 income=$2  companies=[KME]
  P1: $30 (NW $30) order=1 income=$0
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [2]: BSE (fv=$2, 1★, inc=$1), MHE (fv=$8, 1★, inc=$2)

**Deck**: 16 remaining

**Auction**: BSE current bid=$0 high bidder=P-1 starter=P1

### Step 5: P1 [BID_IN_AUCTION]

  **Auction**: BSE current bid=$0 high bidder=P-1 starter=P1

  NN Values: P0=+0.041, P1=+0.014, P2=-0.082
  NN Priors (top 10 of 15 legal):
     1.  73.5% (-11.0pp) █████████████████████████████ BID $5
     2.   8.2% ( +5.7pp) ███ BID $4
     3.   8.1% ( -1.2pp) ███ BID $3
     4.   7.9% ( -0.9pp) ███ BID $2
     5.   0.7% ( +2.6pp)  BID $6
     6.   0.4% ( +2.4pp)  BID $12
     7.   0.4% ( +0.4pp)  BID $7
     8.   0.3% ( -0.0pp)  BID $8
     9.   0.2% ( +0.5pp)  BID $9
    10.   0.1% ( -0.0pp)  BID $16

  MCTS Visits (top 10, 6400 total):
     1.  5211 (81.4%) Q=+0.031 ████████████████████████████████ BID $5
     2.   476 ( 7.4%) Q=+0.007 ██ BID $4
     3.   319 ( 5.0%) Q=+0.016 █ BID $2
     4.   314 ( 4.9%) Q=+0.016 █ BID $3
     5.    25 ( 0.4%) Q=+0.002  BID $7
     6.    24 ( 0.4%) Q=-0.140  BID $6
     7.    23 ( 0.4%) Q=-0.144  BID $12
     8.     1 ( 0.0%) Q=-0.414  BID $8
     9.     1 ( 0.0%) Q=-0.539  BID $9
    10.     1 ( 0.0%) Q=-0.656  BID $10
  A0GB Value: P0=+0.067, P1=+0.188, P2=-0.252 (depth: 38, vbackups: 5705)

  **Action: BID $5**

### Step 6: P2 [BID_IN_AUCTION]

  **Auction**: BSE current bid=$5 high bidder=P1 starter=P1

  NN Values: P0=+0.022, P1=+0.002, P2=-0.049
  NN Priors (top 10 of 12 legal):
     1.  92.0% (-13.8pp) ████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   4.4% ( -0.5pp) █ BID $6
     3.   0.8% ( -0.1pp)  BID $7
     4.   0.7% ( +0.5pp)  BID $8
     5.   0.5% ( +1.3pp)  BID $10
     6.   0.4% ( +5.8pp)  BID $9
     7.   0.4% ( +0.3pp)  BID $11
     8.   0.3% ( +0.3pp)  BID $13
     9.   0.2% ( +4.4pp)  BID $12
    10.   0.2% ( +1.8pp)  BID $15

  MCTS Visits (top 10, 6400 total):
     1.  6199 (96.9%) Q=-0.079 ██████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.    47 ( 0.7%) Q=-0.262  BID $9
     3.    39 ( 0.6%) Q=-0.193  BID $6
     4.    30 ( 0.5%) Q=-0.297  BID $12
     5.    24 ( 0.4%) Q=-0.124  BID $8
     6.    23 ( 0.4%) Q=-0.321  BID $15
     7.    20 ( 0.3%) Q=-0.188  BID $10
     8.    14 ( 0.2%) Q=-0.125  BID $7
     9.     1 ( 0.0%) Q=-0.730  BID $11
    10.     1 ( 0.0%) Q=-0.816  BID $13
  A0GB Value: P0=-0.301, P1=+0.303, P2=+0.024 (depth: 48, vbackups: 5168)

  **Action: PASS (BID_IN_AUCTION)**

### Step 7: P0 [BID_IN_AUCTION]

  **Auction**: BSE current bid=$5 high bidder=P1 starter=P1

  NN Values: P0=+0.048, P1=-0.010, P2=-0.060
  NN Priors (top 10 of 12 legal):
     1.  95.2% (-14.2pp) ██████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   2.9% ( -0.2pp) █ BID $6
     3.   0.7% ( +3.7pp)  BID $7
     4.   0.3% ( +0.0pp)  BID $8
     5.   0.2% ( +0.4pp)  BID $12
     6.   0.2% ( +0.4pp)  BID $13
     7.   0.2% ( +0.2pp)  BID $9
     8.   0.1% ( +0.3pp)  BID $14
     9.   0.1% ( +1.8pp)  BID $10
    10.   0.1% ( +0.1pp)  BID $15

  MCTS Visits (top 10, 6400 total):
     1.  6294 (98.3%) Q=+0.026 ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.    37 ( 0.6%) Q=-0.054  BID $6
     3.    25 ( 0.4%) Q=-0.390  BID $11
     4.    23 ( 0.4%) Q=-0.252  BID $7
     5.    14 ( 0.2%) Q=-0.167  BID $10
     6.     1 ( 0.0%) Q=-0.357  BID $8
     7.     1 ( 0.0%) Q=-0.539  BID $9
     8.     1 ( 0.0%) Q=-0.770  BID $12
     9.     1 ( 0.0%) Q=-0.816  BID $13
    10.     1 ( 0.0%) Q=-0.844  BID $14
  A0GB Value: P0=-0.301, P1=+0.303, P2=+0.024 (depth: 47, vbackups: 6198)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 1  |  CoO Level: 2  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $26) order=0 income=$2  companies=[KME]
  P1: $25 (NW $27) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [1]: MHE (fv=$8, 1★, inc=$2)

**Deck**: 15 remaining


### Step 8: P2 [INVEST]

  NN Values: P0=-0.030, P1=-0.057, P2=-0.022
  NN Priors (top 2 of 2 legal):
     1.  57.9% ( -1.4pp) ███████████████████████ AUCTION slot 0 (MHE, face $8)
     2.  42.1% ( +1.4pp) ████████████████ PASS (INVEST)

  MCTS Visits (top 2, 6400 total):
     1.  4684 (73.2%) Q=-0.073 █████████████████████████████ PASS (INVEST)
     2.  1716 (26.8%) Q=-0.106 ██████████ AUCTION slot 0 (MHE, face $8)
  A0GB Value: P0=-0.301, P1=+0.303, P2=+0.024 (depth: 46, vbackups: 6293)

  **Action: PASS (INVEST)**

### Step 9: P0 [INVEST]

  NN Values: P0=+0.036, P1=-0.038, P2=-0.078
  NN Priors (top 2 of 2 legal):
     1.  96.3% ( -5.1pp) ██████████████████████████████████████ AUCTION slot 0 (MHE, face $8)
     2.   3.7% ( +5.1pp) █ PASS (INVEST)

  MCTS Visits (top 2, 6400 total):
     1.  5207 (81.4%) Q=+0.011 ████████████████████████████████ AUCTION slot 0 (MHE, face $8)
     2.  1193 (18.6%) Q=+0.025 ███████ PASS (INVEST)
  A0GB Value: P0=-0.016, P1=+0.009, P2=+0.036 (depth: 37, vbackups: 4683)

  **Action: AUCTION slot 0 (MHE, face $8)**

Phase: BID_IN_AUCTION  |  Turn: 1  |  CoO Level: 2  |  Active Player: 0  |  End Card: no

**Players**
  P0: $21 (NW $26) order=0 income=$2  companies=[KME]
  P1: $25 (NW $27) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=2 income=$0

**FI**: $4 income=$5

**Auction Row** [1]: MHE (fv=$8, 1★, inc=$2)

**Deck**: 15 remaining

**Auction**: MHE current bid=$0 high bidder=P-1 starter=P0

### Step 10: P0 [BID_IN_AUCTION]

  **Auction**: MHE current bid=$0 high bidder=P-1 starter=P0

  NN Values: P0=+0.063, P1=-0.064, P2=-0.070
  NN Priors (top 10 of 14 legal):
     1.  93.3% (-14.0pp) █████████████████████████████████████ BID $13
     2.   3.5% ( +0.4pp) █ BID $12
     3.   0.6% ( +1.6pp)  BID $8
     4.   0.6% ( -0.1pp)  BID $20
     5.   0.5% ( +6.9pp)  BID $9
     6.   0.5% ( +2.0pp)  BID $14
     7.   0.4% ( -0.1pp)  BID $10
     8.   0.3% ( -0.0pp)  BID $11
     9.   0.2% ( +0.5pp)  BID $15
    10.   0.0% ( -0.0pp)  BID $16

  MCTS Visits (top 10, 6400 total):
     1.  5998 (93.7%) Q=+0.017 █████████████████████████████████████ BID $13
     2.   159 ( 2.5%) Q=-0.028  BID $9
     3.   113 ( 1.8%) Q=-0.012  BID $12
     4.    42 ( 0.7%) Q=-0.036  BID $8
     5.    34 ( 0.5%) Q=-0.062  BID $14
     6.    21 ( 0.3%) Q=-0.226  BID $19
     7.    13 ( 0.2%) Q=+0.006  BID $11
     8.    13 ( 0.2%) Q=-0.003  BID $10
     9.     2 ( 0.0%) Q=-0.854  BID $21
    10.     1 ( 0.0%) Q=-0.412  BID $15
  A0GB Value: P0=-0.016, P1=+0.009, P2=+0.036 (depth: 36, vbackups: 5204)

  **Action: BID $13**

### Step 11: P1 [BID_IN_AUCTION]

  **Auction**: MHE current bid=$13 high bidder=P0 starter=P0

  NN Values: P0=+0.021, P1=+0.007, P2=-0.057
  NN Priors (top 10 of 10 legal):
     1.  97.1% (-14.6pp) ██████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   1.6% ( +5.4pp)  BID $14
     3.   0.3% ( +0.2pp)  BID $15
     4.   0.2% ( +1.2pp)  BID $16
     5.   0.2% ( +0.0pp)  BID $20
     6.   0.2% ( +0.2pp)  BID $17
     7.   0.2% ( -0.0pp)  BID $18
     8.   0.2% ( +1.1pp)  BID $19
     9.   0.1% ( +5.9pp)  BID $21
    10.   0.0% ( +0.6pp)  BID $22

  MCTS Visits (top 10, 6400 total):
     1.  6327 (98.9%) Q=+0.056 ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.    33 ( 0.5%) Q=-0.210  BID $14
     3.    23 ( 0.4%) Q=-0.105  BID $16
     4.    11 ( 0.2%) Q=-0.761  BID $21
     5.     1 ( 0.0%) Q=-0.711  BID $17
     6.     1 ( 0.0%) Q=-0.766  BID $18
     7.     1 ( 0.0%) Q=-0.805  BID $19
     8.     1 ( 0.0%) Q=-0.439  BID $15
     9.     1 ( 0.0%) Q=-0.836  BID $20
    10.     1 ( 0.0%) Q=-0.895  BID $22
  A0GB Value: P0=+0.142, P1=+0.299, P2=-0.375 (depth: 38, vbackups: 5997)

  **Action: PASS (BID_IN_AUCTION)**

### Step 12: P2 [BID_IN_AUCTION]

  **Auction**: MHE current bid=$13 high bidder=P0 starter=P0

  NN Values: P0=+0.011, P1=+0.018, P2=-0.044
  NN Priors (top 10 of 10 legal):
     1.  98.6% (-14.8pp) ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   0.7% ( +4.2pp)  BID $14
     3.   0.2% ( +0.2pp)  BID $15
     4.   0.1% ( +1.5pp)  BID $16
     5.   0.1% ( +0.5pp)  BID $18
     6.   0.1% ( +1.5pp)  BID $17
     7.   0.1% ( +1.8pp)  BID $19
     8.   0.1% ( +0.8pp)  BID $20
     9.   0.0% ( +4.0pp)  BID $22
    10.   0.0% ( +0.2pp)  BID $21

  MCTS Visits (top 10, 6400 total):
     1.  6303 (98.5%) Q=-0.094 ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.    43 ( 0.7%) Q=-0.262  BID $14
     3.    15 ( 0.2%) Q=-0.325  BID $16
     4.    14 ( 0.2%) Q=-0.418  BID $17
     5.    12 ( 0.2%) Q=-0.605  BID $19
     6.     9 ( 0.1%) Q=-0.879  BID $22
     7.     1 ( 0.0%) Q=-0.738  BID $18
     8.     1 ( 0.0%) Q=-0.480  BID $15
     9.     1 ( 0.0%) Q=-0.820  BID $20
    10.     1 ( 0.0%) Q=-0.852  BID $21
  A0GB Value: P0=+0.142, P1=+0.299, P2=-0.375 (depth: 37, vbackups: 6326)

  **Action: PASS (BID_IN_AUCTION)**

  ↳ auto: PASS (INVEST)
  ↳ auto: PASS (INVEST)
  ↳ auto: PASS (INVEST)
  ↳ auto: AUTO:WRAP_UP
  ↳ auto: AUTO:INCOME
  ↳ auto: AUTO:END_CARD

Phase: IPO  |  Turn: 1  |  CoO Level: 2  |  Active Player: 0  |  End Card: no

**Players**
  P0: $12 (NW $25) order=2 income=$4  companies=[KME, MHE]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=0 income=$0

**FI**: $9 income=$5

**Auction Row** [3]: AKE (fv=$6, 1★, inc=$2), BD (fv=$13, 2★, inc=$3), PR (fv=$19, 2★, inc=$3)

**Deck**: 14 remaining

**IPO**: MHE

### Step 13: P0 [IPO]

  **IPO**: MHE

  NN Values: P0=+0.059, P1=-0.053, P2=-0.073
  NN Priors (top 9 of 9 legal):
     1.  96.0% (-13.1pp) ██████████████████████████████████████ IPO MHE → float SI
     2.   2.5% ( -0.2pp)  PASS (IPO)
     3.   0.3% ( +0.8pp)  IPO MHE → float VM
     4.   0.3% ( +0.4pp)  IPO MHE → float OS
     5.   0.2% ( +0.1pp)  IPO MHE → float JS
     6.   0.2% (+10.0pp)  IPO MHE → float S
     7.   0.2% ( +0.1pp)  IPO MHE → float SM
     8.   0.2% ( +1.9pp)  IPO MHE → float DA
     9.   0.2% ( +0.0pp)  IPO MHE → float PR

  MCTS Visits (top 9, 6400 total):
     1.  6301 (98.5%) Q=+0.019 ███████████████████████████████████████ IPO MHE → float SI
     2.    40 ( 0.6%) Q=-0.038  PASS (IPO)
     3.    24 ( 0.4%) Q=-0.543  IPO MHE → float S
     4.    16 ( 0.2%) Q=-0.245  IPO MHE → float DA
     5.    10 ( 0.2%) Q=-0.468  IPO MHE → float VM
     6.     6 ( 0.1%) Q=-0.410  IPO MHE → float OS
     7.     1 ( 0.0%) Q=-0.143  IPO MHE → float PR
     8.     1 ( 0.0%) Q=-0.242  IPO MHE → float JS
     9.     1 ( 0.0%) Q=-0.205  IPO MHE → float SM
  A0GB Value: P0=+0.142, P1=+0.299, P2=-0.375 (depth: 36, vbackups: 6295)

  **Action: IPO MHE → float SI**

Phase: PAR  |  Turn: 1  |  CoO Level: 2  |  Active Player: 0  |  End Card: no

**Players**
  P0: $12 (NW $25) order=2 income=$4  companies=[KME, MHE]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=0 income=$0

**FI**: $9 income=$5

**Auction Row** [3]: AKE (fv=$6, 1★, inc=$2), BD (fv=$13, 2★, inc=$3), PR (fv=$19, 2★, inc=$3)

**Deck**: 14 remaining

**PAR**: MHE -> SI

### Step 14: P0 [PAR]

  **PAR**: MHE -> SI

  NN Values: P0=+0.069, P1=-0.076, P2=-0.058
  NN Priors (top 5 of 5 legal):
     1.  71.2% ( -9.3pp) ████████████████████████████ PAR SI @$14 (IPO MHE)
     2.  18.6% ( -1.7pp) ███████ PAR SI @$11 (IPO MHE)
     3.   5.3% ( +0.3pp) ██ PAR SI @$12 (IPO MHE)
     4.   3.5% ( +4.8pp) █ PAR SI @$13 (IPO MHE)
     5.   1.4% ( +5.9pp)  PAR SI @$10 (IPO MHE)

  MCTS Visits (top 5, 6400 total):
     1.  3186 (49.8%) Q=+0.038 ███████████████████ PAR SI @$13 (IPO MHE)
     2.  1597 (25.0%) Q=+0.027 █████████ PAR SI @$11 (IPO MHE)
     3.  1315 (20.5%) Q=-0.023 ████████ PAR SI @$14 (IPO MHE)
     4.   195 ( 3.0%) Q=+0.002 █ PAR SI @$12 (IPO MHE)
     5.   107 ( 1.7%) Q=-0.051  PAR SI @$10 (IPO MHE)
  A0GB Value: P0=-0.043, P1=+0.023, P2=-0.009 (depth: 46, vbackups: 5541)

  **Action: PAR SI @$13 (IPO MHE)**

Phase: IPO  |  Turn: 1  |  CoO Level: 2  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=0 income=$0

**FI**: $9 income=$5

**Auction Row** [3]: AKE (fv=$6, 1★, inc=$2), BD (fv=$13, 2★, inc=$3), PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 14 remaining

**IPO**: KME

### Step 15: P0 [IPO]

  **IPO**: KME

  NN Values: P0=+0.011, P1=-0.044, P2=-0.046
  NN Priors (top 8 of 8 legal):
     1.  98.8% (-14.3pp) ███████████████████████████████████████ PASS (IPO)
     2.   0.3% ( +1.3pp)  IPO KME → float DA
     3.   0.2% ( +1.7pp)  IPO KME → float SM
     4.   0.2% ( +2.9pp)  IPO KME → float OS
     5.   0.1% ( +5.4pp)  IPO KME → float PR
     6.   0.1% ( +1.7pp)  IPO KME → float S
     7.   0.1% ( +1.3pp)  IPO KME → float VM
     8.   0.1% ( -0.0pp)  IPO KME → float JS

  MCTS Visits (top 8, 6400 total):
     1.  6349 (99.2%) Q=+0.051 ███████████████████████████████████████ PASS (IPO)
     2.    17 ( 0.3%) Q=-0.431  IPO KME → float PR
     3.     9 ( 0.1%) Q=-0.374  IPO KME → float OS
     4.     8 ( 0.1%) Q=-0.512  IPO KME → float S
     5.     6 ( 0.1%) Q=-0.296  IPO KME → float DA
     6.     6 ( 0.1%) Q=-0.430  IPO KME → float SM
     7.     4 ( 0.1%) Q=-0.451  IPO KME → float VM
     8.     1 ( 0.0%) Q=-0.044  IPO KME → float JS
  A0GB Value: P0=+0.093, P1=+0.293, P2=-0.328 (depth: 35, vbackups: 3185)

  **Action: PASS (IPO)**

### Step 16: P1 [IPO]

  **IPO**: BSE

  NN Values: P0=+0.071, P1=-0.027, P2=-0.112
  NN Priors (top 8 of 8 legal):
     1.  77.5% ( -7.8pp) ███████████████████████████████ PASS (IPO)
     2.  20.4% ( -3.7pp) ████████ IPO BSE → float DA
     3.   1.1% ( +0.2pp)  IPO BSE → float PR
     4.   0.3% ( +1.5pp)  IPO BSE → float SM
     5.   0.2% ( +1.0pp)  IPO BSE → float VM
     6.   0.2% ( +1.0pp)  IPO BSE → float S
     7.   0.2% ( +0.5pp)  IPO BSE → float OS
     8.   0.2% ( +7.4pp)  IPO BSE → float JS

  MCTS Visits (top 8, 6400 total):
     1.  6100 (95.3%) Q=+0.077 ██████████████████████████████████████ PASS (IPO)
     2.   234 ( 3.7%) Q=-0.004 █ IPO BSE → float DA
     3.    27 ( 0.4%) Q=-0.287  IPO BSE → float JS
     4.    11 ( 0.2%) Q=-0.106  IPO BSE → float SM
     5.    11 ( 0.2%) Q=-0.059  IPO BSE → float PR
     6.     7 ( 0.1%) Q=-0.147  IPO BSE → float VM
     7.     6 ( 0.1%) Q=-0.288  IPO BSE → float S
     8.     4 ( 0.1%) Q=-0.205  IPO BSE → float OS
  A0GB Value: P0=+0.093, P1=+0.293, P2=-0.328 (depth: 34, vbackups: 6328)

  **Action: PASS (IPO)**

--- Turn 2 ---

Phase: INVEST  |  Turn: 2  |  CoO Level: 2  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=0 income=$0

**FI**: $9 income=$5

**Auction Row** [3]: AKE (fv=$6, 1★, inc=$2), BD (fv=$13, 2★, inc=$3), PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 14 remaining


### Step 17: P2 [INVEST]

  NN Values: P0=+0.075, P1=-0.059, P2=-0.120
  NN Priors (top 5 of 5 legal):
     1.  48.7% ( -2.3pp) ███████████████████ AUCTION slot 0 (AKE, face $6)
     2.  30.0% ( -3.1pp) ████████████ AUCTION slot 1 (BD, face $13)
     3.  11.9% ( -0.1pp) ████ PASS (INVEST)
     4.   7.0% ( +1.9pp) ██ BUY SI share
     5.   2.4% ( +3.7pp)  AUCTION slot 2 (PR, face $19)

  MCTS Visits (top 5, 6400 total):
     1.  2678 (41.8%) Q=-0.146 ████████████████ AUCTION slot 1 (BD, face $13)
     2.  1787 (27.9%) Q=-0.141 ███████████ PASS (INVEST)
     3.  1597 (25.0%) Q=-0.171 █████████ AUCTION slot 0 (AKE, face $6)
     4.   202 ( 3.2%) Q=-0.192 █ BUY SI share
     5.   136 ( 2.1%) Q=-0.195  AUCTION slot 2 (PR, face $19)
  A0GB Value: P0=+0.120, P1=+0.240, P2=-0.322 (depth: 34, vbackups: 6099)

  **Action: AUCTION slot 1 (BD, face $13)**

Phase: BID_IN_AUCTION  |  Turn: 2  |  CoO Level: 2  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $30 (NW $30) order=0 income=$0

**FI**: $9 income=$5

**Auction Row** [3]: AKE (fv=$6, 1★, inc=$2), BD (fv=$13, 2★, inc=$3), PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 14 remaining

**Auction**: BD current bid=$0 high bidder=P-1 starter=P2

### Step 18: P2 [BID_IN_AUCTION]

  **Auction**: BD current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=+0.089, P1=-0.005, P2=-0.121
  NN Priors (top 10 of 15 legal):
     1.  54.1% ( -7.2pp) █████████████████████ BID $15
     2.  36.6% ( -5.8pp) ██████████████ BID $14
     3.   6.7% ( -0.6pp) ██ BID $13
     4.   0.9% ( -0.1pp)  BID $16
     5.   0.4% ( +1.7pp)  BID $17
     6.   0.4% ( -0.0pp)  BID $22
     7.   0.2% ( +3.6pp)  BID $18
     8.   0.2% ( +5.1pp)  BID $19
     9.   0.2% ( +0.3pp)  BID $26
    10.   0.1% ( +1.1pp)  BID $21

  MCTS Visits (top 10, 6400 total):
     1.  5400 (84.4%) Q=-0.110 █████████████████████████████████ BID $15
     2.   682 (10.7%) Q=-0.187 ████ BID $14
     3.   169 ( 2.6%) Q=-0.177 █ BID $13
     4.    33 ( 0.5%) Q=-0.328  BID $18
     5.    33 ( 0.5%) Q=-0.394  BID $19
     6.    30 ( 0.5%) Q=-0.230  BID $17
     7.    27 ( 0.4%) Q=-0.167  BID $16
     8.    15 ( 0.2%) Q=-0.331  BID $21
     9.     5 ( 0.1%) Q=-0.912  BID $27
    10.     1 ( 0.0%) Q=-0.582  BID $20
  A0GB Value: P0=-0.367, P1=+0.005, P2=+0.369 (depth: 53, vbackups: 2666)

  **Action: BID $15**

### Step 19: P1 [BID_IN_AUCTION]

  **Auction**: BD current bid=$15 high bidder=P2 starter=P2

  NN Values: P0=+0.118, P1=+0.043, P2=-0.167
  NN Priors (top 10 of 12 legal):
     1.  93.6% (-13.0pp) █████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   4.7% ( -0.8pp) █ BID $16
     3.   0.6% ( -0.1pp)  BID $17
     4.   0.2% ( +0.3pp)  BID $18
     5.   0.2% ( +4.4pp)  BID $19
     6.   0.2% ( +0.5pp)  BID $26
     7.   0.2% ( +0.2pp)  BID $21
     8.   0.1% ( +0.0pp)  BID $22
     9.   0.1% ( +6.4pp)  BID $25
    10.   0.1% ( +0.4pp)  BID $23

  MCTS Visits (top 10, 6400 total):
     1.  6303 (98.5%) Q=+0.039 ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.    28 ( 0.4%) Q=-0.128  BID $16
     3.    23 ( 0.4%) Q=-0.245  BID $19
     4.    15 ( 0.2%) Q=-0.691  BID $25
     5.    13 ( 0.2%) Q=-0.156  BID $24
     6.    11 ( 0.2%) Q=-0.041  BID $18
     7.     2 ( 0.0%) Q=-0.185  BID $17
     8.     1 ( 0.0%) Q=-0.598  BID $20
     9.     1 ( 0.0%) Q=-0.762  BID $22
    10.     1 ( 0.0%) Q=-0.691  BID $21
  A0GB Value: P0=-0.367, P1=+0.005, P2=+0.369 (depth: 52, vbackups: 5399)

  **Action: PASS (BID_IN_AUCTION)**

  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 2  |  CoO Level: 2  |  Active Player: 1  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $15 (NW $28) order=0 income=$3  companies=[BD]

**FI**: $9 income=$5

**Auction Row** [2]: AKE (fv=$6, 1★, inc=$2), PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 13 remaining


### Step 20: P1 [INVEST]

  NN Values: P0=+0.150, P1=+0.024, P2=-0.216
  NN Priors (top 4 of 4 legal):
     1.  68.5% ( -8.8pp) ███████████████████████████ BUY SI share
     2.  29.9% ( -3.0pp) ███████████ AUCTION slot 0 (AKE, face $6)
     3.   1.1% ( +4.0pp)  PASS (INVEST)
     4.   0.5% ( +7.8pp)  AUCTION slot 1 (PR, face $19)

  MCTS Visits (top 4, 6400 total):
     1.  4320 (67.5%) Q=+0.049 ███████████████████████████ AUCTION slot 0 (AKE, face $6)
     2.  1937 (30.3%) Q=+0.016 ████████████ BUY SI share
     3.    84 ( 1.3%) Q=-0.077  AUCTION slot 1 (PR, face $19)
     4.    59 ( 0.9%) Q=-0.062  PASS (INVEST)
  A0GB Value: P0=-0.367, P1=+0.005, P2=+0.369 (depth: 51, vbackups: 6136)

  **Action: AUCTION slot 0 (AKE, face $6)**

Phase: BID_IN_AUCTION  |  Turn: 2  |  CoO Level: 2  |  Active Player: 1  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $26 (NW $28) order=1 income=$1  companies=[BSE]
  P2: $15 (NW $28) order=0 income=$3  companies=[BD]

**FI**: $9 income=$5

**Auction Row** [2]: AKE (fv=$6, 1★, inc=$2), PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 13 remaining

**Auction**: AKE current bid=$0 high bidder=P-1 starter=P1

### Step 21: P1 [BID_IN_AUCTION]

  **Auction**: AKE current bid=$0 high bidder=P-1 starter=P1

  NN Values: P0=+0.105, P1=+0.058, P2=-0.209
  NN Priors (top 10 of 15 legal):
     1.  84.0% (-12.1pp) █████████████████████████████████ BID $7
     2.   7.6% ( +1.3pp) ███ BID $8
     3.   7.5% ( -1.1pp) ███ BID $6
     4.   0.3% ( -0.1pp)  BID $9
     5.   0.2% ( -0.0pp)  BID $14
     6.   0.1% ( +2.2pp)  BID $10
     7.   0.1% ( +0.0pp)  BID $11
     8.   0.1% ( +1.0pp)  BID $12
     9.   0.0% ( -0.0pp)  BID $15
    10.   0.0% ( +0.9pp)  BID $19

  MCTS Visits (top 10, 6400 total):
     1.  6248 (97.6%) Q=+0.028 ███████████████████████████████████████ BID $7
     2.    58 ( 0.9%) Q=-0.114  BID $6
     3.    42 ( 0.7%) Q=-0.241  BID $8
     4.    15 ( 0.2%) Q=-0.190  BID $10
     5.    13 ( 0.2%) Q=-0.384  BID $12
     6.    12 ( 0.2%) Q=-0.478  BID $13
     7.     3 ( 0.0%) Q=-0.832  BID $17
     8.     2 ( 0.0%) Q=-0.902  BID $20
     9.     1 ( 0.0%) Q=-0.346  BID $11
    10.     1 ( 0.0%) Q=-0.132  BID $9
  A0GB Value: P0=-0.270, P1=-0.193, P2=+0.373 (depth: 58, vbackups: 4318)

  **Action: BID $7**

  ↳ auto: PASS (BID_IN_AUCTION)

### Step 22: P2 [BID_IN_AUCTION]

  **Auction**: AKE current bid=$7 high bidder=P1 starter=P1

  NN Values: P0=+0.083, P1=+0.048, P2=-0.166
  NN Priors (top 9 of 9 legal):
     1.  50.4% ( -7.7pp) ████████████████████ PASS (BID_IN_AUCTION)
     2.  42.7% ( -4.8pp) █████████████████ BID $8
     3.   3.2% ( +1.2pp) █ BID $9
     4.   1.0% ( -0.1pp)  BID $10
     5.   1.0% ( +1.5pp)  BID $14
     6.   0.8% ( +0.1pp)  BID $11
     7.   0.5% ( +2.4pp)  BID $12
     8.   0.3% ( +5.9pp)  BID $15
     9.   0.1% ( +1.6pp)  BID $13

  MCTS Visits (top 9, 6400 total):
     1.  5867 (91.7%) Q=-0.033 ████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   367 ( 5.7%) Q=-0.164 ██ BID $8
     3.    39 ( 0.6%) Q=-0.348  BID $15
     4.    36 ( 0.6%) Q=-0.188  BID $9
     5.    32 ( 0.5%) Q=-0.188  BID $14
     6.    27 ( 0.4%) Q=-0.235  BID $12
     7.    18 ( 0.3%) Q=-0.212  BID $13
     8.     8 ( 0.1%) Q=-0.177  BID $10
     9.     6 ( 0.1%) Q=-0.204  BID $11
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 57, vbackups: 5556)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 2  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $25) order=2 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $19 (NW $27) order=1 income=$3  companies=[BSE, AKE]
  P2: $15 (NW $28) order=0 income=$3  companies=[BD]

**FI**: $9 income=$5

**Auction Row** [1]: PR (fv=$19, 2★, inc=$3)

**Corporations**
  SI: $18 price=$13(idx 9) shares=bank:1/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 12 remaining


### Step 23: P0 [INVEST]

  NN Values: P0=+0.192, P1=+0.087, P2=-0.299
  NN Priors (top 2 of 2 legal):
     1.  99.6% ( -4.1pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.4% ( +4.1pp)  SELL SI share

  MCTS Visits (top 2, 6400 total):
     1.  6391 (99.9%) Q=-0.005 ███████████████████████████████████████ PASS (INVEST)
     2.     9 ( 0.1%) Q=-0.636  SELL SI share
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 56, vbackups: 5866)

  **Action: PASS (INVEST)**

### Step 24: P2 [INVEST]

  NN Values: P0=+0.145, P1=+0.059, P2=-0.204
  NN Priors (top 2 of 2 legal):
     1.  96.2% ( -5.5pp) ██████████████████████████████████████ BUY SI share
     2.   3.8% ( +5.5pp) █ PASS (INVEST)

  MCTS Visits (top 2, 6400 total):
     1.  6297 (98.4%) Q=-0.028 ███████████████████████████████████████ BUY SI share
     2.   103 ( 1.6%) Q=-0.141  PASS (INVEST)
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 55, vbackups: 6384)

  **Action: BUY SI share**

### Step 25: P1 [INVEST]

  NN Values: P0=+0.039, P1=-0.024, P2=-0.047
  NN Priors (top 2 of 2 legal):
     1.  94.7% ( -8.5pp) █████████████████████████████████████ AUCTION slot 0 (PR, face $19)
     2.   5.3% ( +8.5pp) ██ PASS (INVEST)

  MCTS Visits (top 2, 6400 total):
     1.  6217 (97.1%) Q=+0.011 ██████████████████████████████████████ AUCTION slot 0 (PR, face $19)
     2.   183 ( 2.9%) Q=-0.073 █ PASS (INVEST)
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 54, vbackups: 6280)

  **Action: AUCTION slot 0 (PR, face $19)**

  ↳ auto: BID $19
  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

### Step 26: P0 [INVEST]

  NN Values: P0=-0.099, P1=+0.128, P2=-0.073
  NN Priors (top 2 of 2 legal):
     1.  99.0% ( -7.5pp) ███████████████████████████████████████ PASS (INVEST)
     2.   1.0% ( +7.5pp)  SELL SI share

  MCTS Visits (top 2, 6400 total):
     1.  6377 (99.6%) Q=-0.004 ███████████████████████████████████████ PASS (INVEST)
     2.    23 ( 0.4%) Q=-0.498  SELL SI share
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 53, vbackups: 6238)

  **Action: PASS (INVEST)**

### Step 27: P2 [INVEST]

  NN Values: P0=-0.150, P1=+0.180, P2=-0.089
  NN Priors (top 2 of 2 legal):
     1.  98.3% (-11.2pp) ███████████████████████████████████████ PASS (INVEST)
     2.   1.7% (+11.2pp)  SELL SI share

  MCTS Visits (top 2, 6400 total):
     1.  6313 (98.6%) Q=-0.030 ███████████████████████████████████████ PASS (INVEST)
     2.    87 ( 1.4%) Q=-0.227  SELL SI share
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 52, vbackups: 6336)

  **Action: PASS (INVEST)**

  ↳ auto: PASS (INVEST)
  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 2  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $26) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $0 (NW $27) order=2 income=$6  companies=[BSE, AKE, PR]
  P2: $1 (NW $28) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $9 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SI: $18 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**Acquisition — Select Corp**: P0 may buy with SI($18)

### Step 28: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with SI($18)

  NN Values: P0=-0.006, P1=+0.032, P2=-0.023
  NN Priors (top 2 of 2 legal):
     1.  94.5% ( -8.6pp) █████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   5.5% ( +8.6pp) ██ ACQ select SI

  MCTS Visits (top 2, 6400 total):
     1.  5678 (88.7%) Q=-0.002 ███████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   722 (11.3%) Q=-0.008 ████ ACQ select SI
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 51, vbackups: 5797)

  **Action: PASS (ACQ_SELECT_CORP)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 2  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $9 (NW $28) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $6 (NW $33) order=2 income=$6  companies=[BSE, AKE, PR]
  P2: $4 (NW $31) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SI: $20 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**Dividends**: SI

### Step 29: P0 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.046, P1=+0.130, P2=-0.142
  NN Priors (top 5 of 5 legal):
     1.  79.6% (-10.6pp) ███████████████████████████████ DIVIDEND $4
     2.  12.1% ( +0.9pp) ████ DIVIDEND $3
     3.   5.6% ( +2.0pp) ██ DIVIDEND $2
     4.   1.6% ( +2.3pp)  DIVIDEND $0
     5.   1.2% ( +5.4pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  5906 (92.3%) Q=+0.000 ████████████████████████████████████ DIVIDEND $4
     2.   401 ( 6.3%) Q=-0.029 ██ DIVIDEND $3
     3.    46 ( 0.7%) Q=-0.224  DIVIDEND $2
     4.    30 ( 0.5%) Q=-0.339  DIVIDEND $1
     5.    17 ( 0.3%) Q=-0.360  DIVIDEND $0
  A0GB Value: P0=-0.432, P1=+0.204, P2=+0.252 (depth: 50, vbackups: 6232)

  **Action: DIVIDEND $4**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 2  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $13 (NW $34) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $6 (NW $33) order=2 income=$6  companies=[BSE, AKE, PR]
  P2: $8 (NW $37) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SI: $12 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$2 stars=4 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**Issue**: SI

### Step 30: P0 [ISSUE_SHARES]

  **Issue**: SI

  NN Values: P0=+0.039, P1=+0.051, P2=-0.152
  NN Priors (top 2 of 2 legal):
     1.  78.8% ( +0.4pp) ███████████████████████████████ ISSUE SI shares
     2.  21.2% ( -0.4pp) ████████ PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6227 (97.3%) Q=+0.004 ██████████████████████████████████████ ISSUE SI shares
     2.   173 ( 2.7%) Q=-0.143 █ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.371, P1=-0.191, P2=+0.455 (depth: 51, vbackups: 5905)

  **Action: ISSUE SI shares**

Phase: IPO  |  Turn: 2  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $6 (NW $33) order=2 income=$6  companies=[BSE, AKE, PR]
  P2: $8 (NW $35) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**IPO**: PR

### Step 31: P1 [IPO]

  **IPO**: PR

  NN Values: P0=+0.028, P1=+0.066, P2=-0.148
  NN Priors (top 8 of 8 legal):
     1.  96.6% ( -9.2pp) ██████████████████████████████████████ IPO PR → float DA
     2.   2.5% ( +1.6pp) █ IPO PR → float PR
     3.   0.3% ( +0.4pp)  IPO PR → float SM
     4.   0.2% ( +1.7pp)  PASS (IPO)
     5.   0.1% ( +0.9pp)  IPO PR → float VM
     6.   0.1% ( +2.5pp)  IPO PR → float S
     7.   0.1% ( +0.6pp)  IPO PR → float OS
     8.   0.0% ( +1.4pp)  IPO PR → float JS

  MCTS Visits (top 8, 6400 total):
     1.  5708 (89.2%) Q=-0.021 ███████████████████████████████████ IPO PR → float DA
     2.   525 ( 8.2%) Q=-0.011 ███ IPO PR → float PR
     3.    48 ( 0.8%) Q=-0.049  IPO PR → float SM
     4.    36 ( 0.6%) Q=-0.110  IPO PR → float S
     5.    34 ( 0.5%) Q=-0.090  PASS (IPO)
     6.    26 ( 0.4%) Q=-0.052  IPO PR → float VM
     7.    12 ( 0.2%) Q=-0.275  IPO PR → float JS
     8.    11 ( 0.2%) Q=-0.149  IPO PR → float OS
  A0GB Value: P0=-0.371, P1=-0.191, P2=+0.455 (depth: 50, vbackups: 6200)

  **Action: IPO PR → float DA**

Phase: PAR  |  Turn: 2  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $6 (NW $33) order=2 income=$6  companies=[BSE, AKE, PR]
  P2: $8 (NW $35) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**PAR**: PR -> DA

### Step 32: P1 [PAR]

  **PAR**: PR -> DA

  NN Values: P0=+0.082, P1=+0.025, P2=-0.188
  NN Priors (top 4 of 4 legal):
     1.  44.6% ( -2.2pp) █████████████████ PAR DA @$11 (IPO PR)
     2.  22.8% ( -0.1pp) █████████ PAR DA @$12 (IPO PR)
     3.  21.9% ( +0.3pp) ████████ PAR DA @$10 (IPO PR)
     4.  10.7% ( +2.0pp) ████ PAR DA @$20 (IPO PR)

  MCTS Visits (top 4, 6400 total):
     1.  4433 (69.3%) Q=-0.011 ███████████████████████████ PAR DA @$11 (IPO PR)
     2.   910 (14.2%) Q=-0.034 █████ PAR DA @$12 (IPO PR)
     3.   616 ( 9.6%) Q=-0.028 ███ PAR DA @$20 (IPO PR)
     4.   441 ( 6.9%) Q=-0.067 ██ PAR DA @$10 (IPO PR)
  A0GB Value: P0=-0.334, P1=-0.289, P2=+0.543 (depth: 50, vbackups: 5733)

  **Action: PAR DA @$11 (IPO PR)**

Phase: IPO  |  Turn: 2  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $8 (NW $35) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**IPO**: BD

### Step 33: P2 [IPO]

  **IPO**: BD

  NN Values: P0=-0.093, P1=+0.307, P2=-0.128
  NN Priors (top 7 of 7 legal):
     1.  57.7% ( -9.6pp) ███████████████████████ IPO BD → float SM
     2.  27.6% ( -3.0pp) ███████████ PASS (IPO)
     3.  11.2% ( +6.6pp) ████ IPO BD → float PR
     4.   2.1% ( +1.8pp)  IPO BD → float VM
     5.   0.5% ( +0.1pp)  IPO BD → float OS
     6.   0.5% ( +4.0pp)  IPO BD → float S
     7.   0.4% ( -0.0pp)  IPO BD → float JS

  MCTS Visits (top 7, 6400 total):
     1.  6207 (97.0%) Q=+0.083 ██████████████████████████████████████ IPO BD → float SM
     2.    93 ( 1.5%) Q=-0.265  PASS (IPO)
     3.    76 ( 1.2%) Q=-0.223  IPO BD → float PR
     4.    12 ( 0.2%) Q=-0.340  IPO BD → float VM
     5.    10 ( 0.2%) Q=-0.471  IPO BD → float S
     6.     1 ( 0.0%) Q=-0.465  IPO BD → float JS
     7.     1 ( 0.0%) Q=-0.373  IPO BD → float OS
  A0GB Value: P0=-0.224, P1=-0.281, P2=+0.490 (depth: 63, vbackups: 4424)

  **Action: IPO BD → float SM**

Phase: PAR  |  Turn: 2  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $8 (NW $35) order=1 income=$3  companies=[BD]  shares=[SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**PAR**: BD -> SM

### Step 34: P2 [PAR]

  **PAR**: BD -> SM

  NN Values: P0=-0.067, P1=+0.252, P2=-0.131
  NN Priors (top 5 of 5 legal):
     1.  75.5% (-12.0pp) ██████████████████████████████ PAR SM @$16 (IPO BD)
     2.   9.1% ( +0.6pp) ███ PAR SM @$18 (IPO BD)
     3.   7.7% ( +6.6pp) ███ PAR SM @$13 (IPO BD)
     4.   4.7% ( +4.0pp) █ PAR SM @$10 (IPO BD)
     5.   3.0% ( +0.7pp) █ PAR SM @$20 (IPO BD)

  MCTS Visits (top 5, 6400 total):
     1.  5978 (93.4%) Q=+0.104 █████████████████████████████████████ PAR SM @$18 (IPO BD)
     2.   328 ( 5.1%) Q=-0.157 ██ PAR SM @$16 (IPO BD)
     3.    45 ( 0.7%) Q=-0.187  PAR SM @$10 (IPO BD)
     4.    36 ( 0.6%) Q=-0.462  PAR SM @$13 (IPO BD)
     5.    13 ( 0.2%) Q=-0.336  PAR SM @$20 (IPO BD)
  A0GB Value: P0=-0.224, P1=-0.281, P2=+0.490 (depth: 62, vbackups: 6150)

  **Action: PAR SM @$18 (IPO BD)**

  ↳ auto: PASS (IPO)

Phase: IPO  |  Turn: 2  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $3 (NW $35) order=1 income=$0  shares=[SM=1 (pres), SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**IPO**: KME

### Step 35: P0 [IPO]

  **IPO**: KME

  NN Values: P0=-0.017, P1=+0.330, P2=-0.227
  NN Priors (top 6 of 6 legal):
     1.  98.7% (-14.3pp) ███████████████████████████████████████ PASS (IPO)
     2.   0.6% ( +0.5pp)  IPO KME → float VM
     3.   0.4% ( +2.2pp)  IPO KME → float PR
     4.   0.1% ( +8.9pp)  IPO KME → float JS
     5.   0.1% ( +2.4pp)  IPO KME → float S
     6.   0.1% ( +0.3pp)  IPO KME → float OS

  MCTS Visits (top 6, 6400 total):
     1.  6335 (99.0%) Q=-0.107 ███████████████████████████████████████ PASS (IPO)
     2.    31 ( 0.5%) Q=-0.537  IPO KME → float JS
     3.    15 ( 0.2%) Q=-0.412  IPO KME → float PR
     4.    13 ( 0.2%) Q=-0.495  IPO KME → float S
     5.     5 ( 0.1%) Q=-0.375  IPO KME → float VM
     6.     1 ( 0.0%) Q=-0.407  IPO KME → float OS
  A0GB Value: P0=-0.224, P1=-0.281, P2=+0.490 (depth: 61, vbackups: 5973)

  **Action: PASS (IPO)**

  ↳ auto: PASS (IPO)

--- Turn 3 ---

Phase: INVEST  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $3 (NW $35) order=1 income=$0  shares=[SM=1 (pres), SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining


### Step 36: P0 [INVEST]

  NN Values: P0=+0.081, P1=+0.215, P2=-0.205
  NN Priors (top 4 of 4 legal):
     1.  98.7% (-13.7pp) ███████████████████████████████████████ AUCTION slot 0 (WT, face $11)
     2.   0.7% ( -0.0pp)  SELL SI share
     3.   0.5% ( +7.2pp)  PASS (INVEST)
     4.   0.1% ( +6.5pp)  BUY DA share

  MCTS Visits (top 4, 6400 total):
     1.  6317 (98.7%) Q=-0.106 ███████████████████████████████████████ AUCTION slot 0 (WT, face $11)
     2.    61 ( 1.0%) Q=-0.262  PASS (INVEST)
     3.    18 ( 0.3%) Q=-0.586  BUY DA share
     4.     4 ( 0.1%) Q=-0.290  SELL SI share
  A0GB Value: P0=-0.224, P1=-0.281, P2=+0.490 (depth: 60, vbackups: 6334)

  **Action: AUCTION slot 0 (WT, face $11)**

Phase: BID_IN_AUCTION  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $13 (NW $32) order=0 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $3 (NW $35) order=1 income=$0  shares=[SM=1 (pres), SI=1]

**FI**: $14 income=$5

**Auction Row** [3]: WT (fv=$11, 2★, inc=$3), SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 11 remaining

**Auction**: WT current bid=$0 high bidder=P-1 starter=P0

### Step 37: P0 [BID_IN_AUCTION]

  **Auction**: WT current bid=$0 high bidder=P-1 starter=P0

  NN Values: P0=+0.058, P1=+0.225, P2=-0.221
  NN Priors (top 3 of 3 legal):
     1.  97.9% (-13.8pp) ███████████████████████████████████████ BID $11
     2.   1.7% ( +8.3pp)  BID $12
     3.   0.4% ( +5.5pp)  BID $13

  MCTS Visits (top 3, 6400 total):
     1.  6086 (95.1%) Q=-0.106 ██████████████████████████████████████ BID $11
     2.   272 ( 4.2%) Q=-0.138 █ BID $12
     3.    42 ( 0.7%) Q=-0.276  BID $13
  A0GB Value: P0=-0.224, P1=-0.281, P2=+0.490 (depth: 59, vbackups: 6160)

  **Action: BID $11**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 3  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $2 (NW $32) order=0 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=2 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $3 (NW $35) order=1 income=$0  shares=[SM=1 (pres), SI=1]

**FI**: $14 income=$5

**Auction Row** [2]: SX (fv=$16, 2★, inc=$3), NS (fv=$22, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$14(idx 10) shares=bank:1/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 10 remaining


### Step 38: P2 [INVEST]

  NN Values: P0=+0.177, P1=+0.037, P2=-0.220
  NN Priors (top 3 of 3 legal):
     1.  45.7% ( -1.3pp) ██████████████████ SELL SI share
     2.  36.2% ( -2.4pp) ██████████████ PASS (INVEST)
     3.  18.1% ( +3.8pp) ███████ SELL SM share

  MCTS Visits (top 3, 6400 total):
     1.  6237 (97.5%) Q=+0.114 ██████████████████████████████████████ SELL SI share
     2.   120 ( 1.9%) Q=-0.258  PASS (INVEST)
     3.    43 ( 0.7%) Q=-0.578  SELL SM share
  A0GB Value: P0=-0.314, P1=-0.203, P2=+0.451 (depth: 52, vbackups: 6240)

  **Action: SELL SI share**

### Step 39: P1 [INVEST]

  NN Values: P0=+0.059, P1=-0.119, P2=+0.048
  NN Priors (top 2 of 2 legal):
     1.  93.7% ( -5.8pp) █████████████████████████████████████ PASS (INVEST)
     2.   6.3% ( +5.8pp) ██ SELL DA share

  MCTS Visits (top 2, 6400 total):
     1.  6321 (98.8%) Q=-0.034 ███████████████████████████████████████ PASS (INVEST)
     2.    79 ( 1.2%) Q=-0.232  SELL DA share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 52, vbackups: 6236)

  **Action: PASS (INVEST)**

### Step 40: P0 [INVEST]

  NN Values: P0=+0.143, P1=-0.153, P2=-0.017
  NN Priors (top 2 of 2 legal):
     1.  98.4% ( -5.7pp) ███████████████████████████████████████ PASS (INVEST)
     2.   1.6% ( +5.7pp)  SELL SI share

  MCTS Visits (top 2, 6400 total):
     1.  6296 (98.4%) Q=-0.117 ███████████████████████████████████████ PASS (INVEST)
     2.   104 ( 1.6%) Q=-0.193  SELL SI share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 51, vbackups: 6312)

  **Action: PASS (INVEST)**

### Step 41: P2 [INVEST]

  NN Values: P0=+0.097, P1=-0.011, P2=-0.115
  NN Priors (top 5 of 5 legal):
     1.  99.3% (-14.4pp) ███████████████████████████████████████ AUCTION slot 0 (SX, face $16)
     2.   0.3% ( +0.5pp)  BUY SI share
     3.   0.2% ( +9.2pp)  BUY DA share
     4.   0.1% ( +1.6pp)  PASS (INVEST)
     5.   0.1% ( +3.0pp)  SELL SM share

  MCTS Visits (top 5, 6400 total):
     1.  6366 (99.5%) Q=+0.114 ███████████████████████████████████████ AUCTION slot 0 (SX, face $16)
     2.    21 ( 0.3%) Q=-0.510  BUY DA share
     3.     7 ( 0.1%) Q=-0.469  SELL SM share
     4.     3 ( 0.0%) Q=-0.522  PASS (INVEST)
     5.     3 ( 0.0%) Q=-0.396  BUY SI share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 50, vbackups: 6303)

  **Action: AUCTION slot 0 (SX, face $16)**

  ↳ auto: BID $16
  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

### Step 42: P1 [INVEST]

  NN Values: P0=-0.053, P1=-0.145, P2=+0.185
  NN Priors (top 2 of 2 legal):
     1.  92.3% (-10.0pp) ████████████████████████████████████ PASS (INVEST)
     2.   7.7% (+10.0pp) ███ SELL DA share

  MCTS Visits (top 2, 6400 total):
     1.  6279 (98.1%) Q=-0.033 ███████████████████████████████████████ PASS (INVEST)
     2.   121 ( 1.9%) Q=-0.223  SELL DA share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 49, vbackups: 6328)

  **Action: PASS (INVEST)**

### Step 43: P0 [INVEST]

  NN Values: P0=-0.089, P1=-0.133, P2=+0.190
  NN Priors (top 2 of 2 legal):
     1.  98.1% ( -9.4pp) ███████████████████████████████████████ PASS (INVEST)
     2.   1.9% ( +9.4pp)  SELL SI share

  MCTS Visits (top 2, 6400 total):
     1.  6181 (96.6%) Q=-0.119 ██████████████████████████████████████ PASS (INVEST)
     2.   219 ( 3.4%) Q=-0.169 █ SELL SI share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 48, vbackups: 6200)

  **Action: PASS (INVEST)**

### Step 44: P2 [INVEST]

  NN Values: P0=-0.058, P1=-0.075, P2=+0.106
  NN Priors (top 2 of 2 legal):
     1.  99.5% ( -5.1pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.5% ( +5.1pp)  SELL SM share

  MCTS Visits (top 2, 6400 total):
     1.  6385 (99.8%) Q=+0.113 ███████████████████████████████████████ PASS (INVEST)
     2.    15 ( 0.2%) Q=-0.546  SELL SM share
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 47, vbackups: 6295)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 3  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=0 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Corp**: P1 may buy with DA($25)

### Step 45: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with DA($25)

  NN Values: P0=-0.093, P1=-0.120, P2=+0.167
  NN Priors (top 2 of 2 legal):
     1.  96.5% ( -1.4pp) ██████████████████████████████████████ ACQ select DA
     2.   3.5% ( +1.4pp) █ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6366 (99.5%) Q=-0.031 ███████████████████████████████████████ ACQ select DA
     2.    34 ( 0.5%) Q=-0.232  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.324, P1=-0.287, P2=+0.527 (depth: 46, vbackups: 6384)

  **Action: ACQ select DA**

Phase: ACQ_SELECT_COMPANY  |  Turn: 3  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=0 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Company**: P1 buying with DA ($25)

### Step 46: P1 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P1 buying with DA ($25)

  NN Values: P0=-0.108, P1=-0.056, P2=+0.136
  NN Priors (top 2 of 2 legal):
     1.  58.9% ( +2.6pp) ███████████████████████ ACQ target BSE (with DA)
     2.  41.1% ( -2.6pp) ████████████████ ACQ target AKE (with DA)

  MCTS Visits (top 2, 6400 total):
     1.  3895 (60.9%) Q=-0.040 ████████████████████████ ACQ target BSE (with DA)
     2.  2505 (39.1%) Q=-0.040 ███████████████ ACQ target AKE (with DA)
  A0GB Value: P0=-0.256, P1=-0.222, P2=+0.477 (depth: 48, vbackups: 5856)

  **Action: ACQ target BSE (with DA)**

Phase: ACQ_SELECT_PRICE  |  Turn: 3  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $3 (NW $33) order=0 income=$3  companies=[BSE, AKE]  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$6 stars=4 pres=P1  companies=[PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Price**: P1 DA -> BSE (price range $1-$3)

### Step 47: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 DA -> BSE (price range $1-$3)

  NN Values: P0=-0.078, P1=-0.146, P2=+0.190
  NN Priors (top 3 of 3 legal):
     1.  94.7% (-12.8pp) █████████████████████████████████████ ACQUIRE BSE with DA @ $3
     2.   4.4% (+11.9pp) █ ACQUIRE BSE with DA @ $2
     3.   0.9% ( +0.9pp)  ACQUIRE BSE with DA @ $1

  MCTS Visits (top 3, 6400 total):
     1.  6171 (96.4%) Q=-0.073 ██████████████████████████████████████ ACQUIRE BSE with DA @ $3
     2.   218 ( 3.4%) Q=-0.157 █ ACQUIRE BSE with DA @ $2
     3.    11 ( 0.2%) Q=-0.326  ACQUIRE BSE with DA @ $1
  A0GB Value: P0=-0.220, P1=-0.330, P2=+0.566 (depth: 56, vbackups: 3894)

  **Action: ACQUIRE BSE with DA @ $3**

Phase: ACQ_SELECT_CORP  |  Turn: 3  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $6 (NW $34) order=0 income=$2  companies=[AKE]  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $22 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$8 stars=5 pres=P1  companies=[BSE*, PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Corp**: P1 may buy with DA($22)

### Step 48: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with DA($22)

  NN Values: P0=-0.044, P1=-0.064, P2=+0.092
  NN Priors (top 2 of 2 legal):
     1.  86.0% (-10.9pp) ██████████████████████████████████ ACQ select DA
     2.  14.0% (+10.9pp) █████ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  4763 (74.4%) Q=-0.076 █████████████████████████████ ACQ select DA
     2.  1637 (25.6%) Q=-0.076 ██████████ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.220, P1=-0.330, P2=+0.566 (depth: 55, vbackups: 6170)

  **Action: ACQ select DA**

  ↳ auto: ACQ target AKE (with DA)

Phase: ACQ_SELECT_PRICE  |  Turn: 3  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $6 (NW $34) order=0 income=$2  companies=[AKE]  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $22 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$8 stars=5 pres=P1  companies=[BSE*, PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Price**: P1 DA -> AKE (price range $3-$8)

### Step 49: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 DA -> AKE (price range $3-$8)

  NN Values: P0=-0.053, P1=-0.114, P2=+0.144
  NN Priors (top 6 of 6 legal):
     1.  89.7% (-10.6pp) ███████████████████████████████████ ACQUIRE AKE with DA @ $8
     2.   7.2% ( -0.6pp) ██ ACQUIRE AKE with DA @ $7
     3.   1.5% ( +1.2pp)  ACQUIRE AKE with DA @ $6
     4.   0.8% ( +4.0pp)  ACQUIRE AKE with DA @ $5
     5.   0.5% ( +4.6pp)  ACQUIRE AKE with DA @ $4
     6.   0.2% ( +1.3pp)  ACQUIRE AKE with DA @ $3

  MCTS Visits (top 6, 6400 total):
     1.  6140 (95.9%) Q=-0.096 ██████████████████████████████████████ ACQUIRE AKE with DA @ $8
     2.   125 ( 2.0%) Q=-0.153  ACQUIRE AKE with DA @ $7
     3.    47 ( 0.7%) Q=-0.242  ACQUIRE AKE with DA @ $5
     4.    46 ( 0.7%) Q=-0.200  ACQUIRE AKE with DA @ $6
     5.    35 ( 0.5%) Q=-0.286  ACQUIRE AKE with DA @ $4
     6.     7 ( 0.1%) Q=-0.382  ACQUIRE AKE with DA @ $3
  A0GB Value: P0=-0.250, P1=-0.299, P2=+0.559 (depth: 59, vbackups: 4762)

  **Action: ACQUIRE AKE with DA @ $8**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $14 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE*, AKE*, PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Corp**: P0 may buy with SI($26)

### Step 50: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with SI($26)

  NN Values: P0=-0.050, P1=+0.123, P2=-0.069
  NN Priors (top 2 of 2 legal):
     1.  97.3% ( -6.8pp) ██████████████████████████████████████ ACQ select SI
     2.   2.7% ( +6.8pp) █ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6304 (98.5%) Q=-0.095 ███████████████████████████████████████ ACQ select SI
     2.    96 ( 1.5%) Q=-0.220  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.314, P1=+0.073, P2=+0.352 (depth: 60, vbackups: 6139)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $14 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE*, AKE*, PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Company**: P0 buying with SI ($26)

### Step 51: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with SI ($26)

  NN Values: P0=-0.037, P1=+0.113, P2=-0.074
  NN Priors (top 2 of 2 legal):
     1.  81.6% ( -4.2pp) ████████████████████████████████ ACQ target WT (with SI)
     2.  18.4% ( +4.2pp) ███████ ACQ target KME (with SI)

  MCTS Visits (top 2, 6400 total):
     1.  5945 (92.9%) Q=-0.086 █████████████████████████████████████ ACQ target WT (with SI)
     2.   455 ( 7.1%) Q=-0.139 ██ ACQ target KME (with SI)
  A0GB Value: P0=-0.314, P1=+0.073, P2=+0.352 (depth: 59, vbackups: 6177)

  **Action: ACQ target WT (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $31) order=1 income=$5  companies=[KME, WT]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $14 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE*, AKE*, PR]
  SI: $26 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$2 stars=5 pres=P0  companies=[MHE]

**Deck**: 9 remaining

**Acquisition — Select Price**: P0 SI -> WT (price range $6-$14)

### Step 52: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 SI -> WT (price range $6-$14)

  NN Values: P0=-0.073, P1=+0.083, P2=-0.002
  NN Priors (top 9 of 9 legal):
     1.  96.3% (-10.6pp) ██████████████████████████████████████ ACQUIRE WT with SI @ $14
     2.   2.4% ( +3.5pp)  ACQUIRE WT with SI @ $13
     3.   0.4% ( +1.6pp)  ACQUIRE WT with SI @ $12
     4.   0.3% ( +0.5pp)  ACQUIRE WT with SI @ $11
     5.   0.2% ( +1.6pp)  ACQUIRE WT with SI @ $10
     6.   0.2% ( -0.0pp)  ACQUIRE WT with SI @ $9
     7.   0.1% ( +1.3pp)  ACQUIRE WT with SI @ $8
     8.   0.1% ( +2.2pp)  ACQUIRE WT with SI @ $6
     9.   0.1% ( +0.0pp)  ACQUIRE WT with SI @ $7

  MCTS Visits (top 9, 6400 total):
     1.  6244 (97.6%) Q=-0.072 ███████████████████████████████████████ ACQUIRE WT with SI @ $14
     2.    86 ( 1.3%) Q=-0.163  ACQUIRE WT with SI @ $13
     3.    42 ( 0.7%) Q=-0.133  ACQUIRE WT with SI @ $12
     4.    10 ( 0.2%) Q=-0.746  ACQUIRE WT with SI @ $6
     5.    10 ( 0.2%) Q=-0.461  ACQUIRE WT with SI @ $10
     6.     3 ( 0.0%) Q=-0.557  ACQUIRE WT with SI @ $8
     7.     3 ( 0.0%) Q=-0.375  ACQUIRE WT with SI @ $11
     8.     1 ( 0.0%) Q=-0.395  ACQUIRE WT with SI @ $9
     9.     1 ( 0.0%) Q=-0.629  ACQUIRE WT with SI @ $7
  A0GB Value: P0=-0.314, P1=+0.073, P2=+0.352 (depth: 58, vbackups: 5937)

  **Action: ACQUIRE WT with SI @ $14**

Phase: ACQ_SELECT_CORP  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $34) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $14 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE*, AKE*, PR]
  SI: $12 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$5 stars=6 pres=P0  companies=[MHE, WT*]

**Deck**: 9 remaining

**Acquisition — Select Corp**: P0 may buy with SI($12)

### Step 53: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with SI($12)

  NN Values: P0=+0.145, P1=+0.011, P2=-0.188
  NN Priors (top 2 of 2 legal):
     1.  69.9% ( -7.3pp) ███████████████████████████ PASS (ACQ_SELECT_CORP)
     2.  30.1% ( +7.3pp) ████████████ ACQ select SI

  MCTS Visits (top 2, 6400 total):
     1.  5898 (92.2%) Q=-0.071 ████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   502 ( 7.8%) Q=-0.158 ███ ACQ select SI
  A0GB Value: P0=-0.330, P1=-0.043, P2=+0.439 (depth: 58, vbackups: 6243)

  **Action: PASS (ACQ_SELECT_CORP)**

### Step 54: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with SM($23)

  NN Values: P0=+0.017, P1=+0.073, P2=-0.124
  NN Priors (top 2 of 2 legal):
     1.  99.5% ( -3.4pp) ███████████████████████████████████████ ACQ select SM
     2.   0.5% ( +3.4pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6392 (99.9%) Q=+0.203 ███████████████████████████████████████ ACQ select SM
     2.     8 ( 0.1%) Q=-0.525  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.316, P1=-0.252, P2=+0.570 (depth: 58, vbackups: 5897)

  **Action: ACQ select SM**

  ↳ auto: ACQ target SX (with SM)

Phase: ACQ_SELECT_PRICE  |  Turn: 3  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $16 (NW $34) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $0 (NW $34) order=2 income=$3  companies=[SX]  shares=[SM=1 (pres)]

**FI**: $14 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $23 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$3 stars=4 pres=P2  companies=[BD]
  DA: $14 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE*, AKE*, PR]
  SI: $12 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$5 stars=6 pres=P0  companies=[MHE, WT*]

**Deck**: 9 remaining

**Acquisition — Select Price**: P2 SM -> SX (price range $8-$21)

### Step 55: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SM -> SX (price range $8-$21)

  NN Values: P0=-0.009, P1=-0.027, P2=-0.031
  NN Priors (top 10 of 14 legal):
     1.  94.2% (-12.8pp) █████████████████████████████████████ ACQUIRE SX with SM @ $21
     2.   4.5% ( +3.4pp) █ ACQUIRE SX with SM @ $20
     3.   0.5% ( -0.0pp)  ACQUIRE SX with SM @ $19
     4.   0.2% ( +0.1pp)  ACQUIRE SX with SM @ $18
     5.   0.1% ( +1.4pp)  ACQUIRE SX with SM @ $17
     6.   0.1% ( +0.2pp)  ACQUIRE SX with SM @ $16
     7.   0.1% ( +0.8pp)  ACQUIRE SX with SM @ $15
     8.   0.1% ( +1.9pp)  ACQUIRE SX with SM @ $8
     9.   0.1% ( -0.0pp)  ACQUIRE SX with SM @ $14
    10.   0.0% ( +4.3pp)  ACQUIRE SX with SM @ $11

  MCTS Visits (top 10, 6400 total):
     1.  6335 (99.0%) Q=+0.204 ███████████████████████████████████████ ACQUIRE SX with SM @ $21
     2.    43 ( 0.7%) Q=-0.031  ACQUIRE SX with SM @ $20
     3.     8 ( 0.1%) Q=-0.797  ACQUIRE SX with SM @ $11
     4.     5 ( 0.1%) Q=-0.838  ACQUIRE SX with SM @ $8
     5.     4 ( 0.1%) Q=-0.429  ACQUIRE SX with SM @ $17
     6.     1 ( 0.0%) Q=-0.617  ACQUIRE SX with SM @ $16
     7.     1 ( 0.0%) Q=-0.805  ACQUIRE SX with SM @ $10
     8.     1 ( 0.0%) Q=-0.656  ACQUIRE SX with SM @ $15
     9.     1 ( 0.0%) Q=-0.156  ACQUIRE SX with SM @ $19
    10.     1 ( 0.0%) Q=-0.344  ACQUIRE SX with SM @ $18
  A0GB Value: P0=-0.316, P1=-0.252, P2=+0.570 (depth: 57, vbackups: 6362)

  **Action: ACQUIRE SX with SM @ $21**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 3  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $18 (NW $36) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $14 (NW $36) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $21 (NW $39) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $8 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$6 stars=4 pres=P2  companies=[BD, SX]
  DA: $25 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$11 stars=6 pres=P1  companies=[BSE, AKE, PR]
  SI: $17 price=$13(idx 9) shares=bank:2/unissued:1/issued:3 income=$5 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining

**Dividends**: SM

### Step 56: P2 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=+0.123, P1=-0.202, P2=+0.038
  NN Priors (top 5 of 5 legal):
     1.  85.6% (-11.5pp) ██████████████████████████████████ DIVIDEND $4
     2.  13.3% ( -0.3pp) █████ DIVIDEND $3
     3.   0.9% ( +0.9pp)  DIVIDEND $2
     4.   0.1% ( +4.2pp)  DIVIDEND $1
     5.   0.0% ( +6.7pp)  DIVIDEND $0

  MCTS Visits (top 5, 6400 total):
     1.  6190 (96.7%) Q=+0.207 ██████████████████████████████████████ DIVIDEND $4
     2.   171 ( 2.7%) Q=+0.119 █ DIVIDEND $3
     3.    16 ( 0.2%) Q=-0.420  DIVIDEND $0
     4.    14 ( 0.2%) Q=-0.203  DIVIDEND $1
     5.     9 ( 0.1%) Q=-0.066  DIVIDEND $2
  A0GB Value: P0=-0.316, P1=-0.252, P2=+0.570 (depth: 56, vbackups: 6347)

  **Action: DIVIDEND $4**

### Step 57: P0 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=+0.052, P1=-0.132, P2=+0.053
  NN Priors (top 5 of 5 legal):
     1.  91.7% (-10.3pp) ████████████████████████████████████ DIVIDEND $4
     2.   5.0% ( +0.1pp) █ DIVIDEND $2
     3.   2.9% ( +1.2pp) █ DIVIDEND $3
     4.   0.3% ( +0.7pp)  DIVIDEND $1
     5.   0.1% ( +8.3pp)  DIVIDEND $0

  MCTS Visits (top 5, 6400 total):
     1.  6106 (95.4%) Q=-0.087 ██████████████████████████████████████ DIVIDEND $4
     2.   107 ( 1.7%) Q=-0.133  DIVIDEND $2
     3.    86 ( 1.3%) Q=-0.147  DIVIDEND $3
     4.    77 ( 1.2%) Q=-0.215  DIVIDEND $0
     5.    24 ( 0.4%) Q=-0.126  DIVIDEND $1
  A0GB Value: P0=-0.416, P1=-0.201, P2=+0.594 (depth: 56, vbackups: 6157)

  **Action: DIVIDEND $4**

### Step 58: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=+0.031, P1=-0.151, P2=+0.058
  NN Priors (top 4 of 4 legal):
     1.  89.2% ( -9.2pp) ███████████████████████████████████ DIVIDEND $3
     2.   8.6% ( +4.2pp) ███ DIVIDEND $1
     3.   1.7% ( +0.9pp)  DIVIDEND $2
     4.   0.6% ( +4.0pp)  DIVIDEND $0

  MCTS Visits (top 4, 6400 total):
     1.  6056 (94.6%) Q=-0.155 █████████████████████████████████████ DIVIDEND $3
     2.   287 ( 4.5%) Q=-0.199 █ DIVIDEND $1
     3.    31 ( 0.5%) Q=-0.363  DIVIDEND $0
     4.    26 ( 0.4%) Q=-0.281  DIVIDEND $2
  A0GB Value: P0=-0.416, P1=-0.201, P2=+0.594 (depth: 55, vbackups: 6105)

  **Action: DIVIDEND $3**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 3  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $22 (NW $41) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $20 (NW $44) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $25 (NW $43) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $0 price=$18(idx 12) shares=bank:1/unissued:4/issued:2 income=$6 stars=4 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE, AKE, PR]
  SI: $5 price=$14(idx 10) shares=bank:2/unissued:1/issued:3 income=$5 stars=5 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining

**Issue**: SM

### Step 59: P2 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.011, P1=-0.100, P2=+0.099
  NN Priors (top 2 of 2 legal):
     1.  98.1% ( -5.3pp) ███████████████████████████████████████ ISSUE SM shares
     2.   1.9% ( +5.3pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6371 (99.5%) Q=+0.221 ███████████████████████████████████████ ISSUE SM shares
     2.    29 ( 0.5%) Q=-0.096  PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.320, P1=-0.346, P2=+0.590 (depth: 55, vbackups: 6055)

  **Action: ISSUE SM shares**

### Step 60: P0 [ISSUE_SHARES]

  **Issue**: SI

  NN Values: P0=-0.052, P1=-0.140, P2=+0.186
  NN Priors (top 2 of 2 legal):
     1.  69.3% ( +1.4pp) ███████████████████████████ PASS (ISSUE_SHARES)
     2.  30.7% ( -1.4pp) ████████████ ISSUE SI shares

  MCTS Visits (top 2, 6400 total):
     1.  5453 (85.2%) Q=-0.087 ██████████████████████████████████ ISSUE SI shares
     2.   947 (14.8%) Q=-0.181 █████ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.320, P1=-0.346, P2=+0.590 (depth: 54, vbackups: 6370)

  **Action: ISSUE SI shares**

### Step 61: P1 [ISSUE_SHARES]

  **Issue**: DA

  NN Values: P0=-0.208, P1=+0.089, P2=+0.145
  NN Priors (top 2 of 2 legal):
     1.  91.6% (-11.1pp) ████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   8.4% (+11.1pp) ███ ISSUE DA shares

  MCTS Visits (top 2, 6400 total):
     1.  5121 (80.0%) Q=-0.176 ████████████████████████████████ PASS (ISSUE_SHARES)
     2.  1279 (20.0%) Q=-0.175 ███████ ISSUE DA shares
  A0GB Value: P0=-0.332, P1=-0.318, P2=+0.625 (depth: 54, vbackups: 5452)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $22 (NW $40) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $20 (NW $44) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $25 (NW $43) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE, AKE, PR]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$5 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining

**IPO**: KME

### Step 62: P0 [IPO]

  **IPO**: KME

  NN Values: P0=-0.130, P1=-0.007, P2=+0.150
  NN Priors (top 6 of 6 legal):
     1.  98.3% (-11.3pp) ███████████████████████████████████████ IPO KME → float VM
     2.   0.8% ( +6.1pp)  IPO KME → float PR
     3.   0.7% ( +0.7pp)  PASS (IPO)
     4.   0.1% ( +4.2pp)  IPO KME → float S
     5.   0.1% ( +0.4pp)  IPO KME → float JS
     6.   0.0% ( -0.0pp)  IPO KME → float OS

  MCTS Visits (top 6, 6400 total):
     1.  6302 (98.5%) Q=-0.079 ███████████████████████████████████████ IPO KME → float VM
     2.    54 ( 0.8%) Q=-0.232  IPO KME → float PR
     3.    27 ( 0.4%) Q=-0.350  IPO KME → float S
     4.    13 ( 0.2%) Q=-0.197  PASS (IPO)
     5.     4 ( 0.1%) Q=-0.424  IPO KME → float JS
  A0GB Value: P0=-0.307, P1=-0.330, P2=+0.609 (depth: 55, vbackups: 5119)

  **Action: IPO KME → float VM**

Phase: PAR  |  Turn: 3  |  CoO Level: 3  |  Active Player: 0  |  End Card: no

**Players**
  P0: $22 (NW $40) order=1 income=$2  companies=[KME]  shares=[SI=1 (pres)]
  P1: $20 (NW $44) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $25 (NW $43) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE, AKE, PR]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$5 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining

**PAR**: KME -> VM

### Step 63: P0 [PAR]

  **PAR**: KME -> VM

  NN Values: P0=-0.099, P1=-0.027, P2=+0.123
  NN Priors (top 3 of 3 legal):
     1.  66.2% ( -2.9pp) ██████████████████████████ PAR VM @$10 (IPO KME)
     2.  31.8% ( +2.9pp) ████████████ PAR VM @$11 (IPO KME)
     3.   1.9% ( +0.0pp)  PAR VM @$14 (IPO KME)

  MCTS Visits (top 3, 6400 total):
     1.  6043 (94.4%) Q=-0.059 █████████████████████████████████████ PAR VM @$11 (IPO KME)
     2.   332 ( 5.2%) Q=-0.314 ██ PAR VM @$10 (IPO KME)
     3.    25 ( 0.4%) Q=-0.158  PAR VM @$14 (IPO KME)
  A0GB Value: P0=-0.307, P1=-0.330, P2=+0.609 (depth: 54, vbackups: 6274)

  **Action: PAR VM @$11 (IPO KME)**

--- Turn 4 ---

Phase: INVEST  |  Turn: 4  |  CoO Level: 3  |  Active Player: 1  |  End Card: no

**Players**
  P0: $16 (NW $40) order=1 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $20 (NW $44) order=0 income=$0  shares=[DA=2 (pres)]
  P2: $25 (NW $43) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$11(idx 7) shares=bank:1/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$5 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining


### Step 64: P1 [INVEST]

  NN Values: P0=-0.027, P1=+0.048, P2=-0.085
  NN Priors (top 6 of 6 legal):
     1.  44.0% ( +1.2pp) █████████████████ PASS (INVEST)
     2.  33.6% ( -0.0pp) █████████████ SELL DA share
     3.  20.1% ( -3.6pp) ████████ BUY VM share
     4.   1.2% ( +0.7pp)  BUY SI share
     5.   0.7% ( +0.1pp)  BUY DA share
     6.   0.5% ( +1.6pp)  BUY SM share

  MCTS Visits (top 6, 6400 total):
     1.  4123 (64.4%) Q=-0.186 █████████████████████████ BUY VM share
     2.  2042 (31.9%) Q=-0.210 ████████████ PASS (INVEST)
     3.   185 ( 2.9%) Q=-0.434 █ SELL DA share
     4.    20 ( 0.3%) Q=-0.306  BUY SI share
     5.    15 ( 0.2%) Q=-0.381  BUY DA share
     6.    15 ( 0.2%) Q=-0.418  BUY SM share
  A0GB Value: P0=-0.307, P1=-0.330, P2=+0.609 (depth: 53, vbackups: 6039)

  **Action: BUY VM share**

### Step 65: P0 [INVEST]

  NN Values: P0=-0.128, P1=+0.067, P2=+0.015
  NN Priors (top 5 of 5 legal):
     1.  81.5% (-11.5pp) ████████████████████████████████ PASS (INVEST)
     2.  14.8% ( -2.3pp) █████ SELL VM share
     3.   2.2% ( +4.7pp)  BUY SI share
     4.   1.1% ( +6.5pp)  SELL SI share
     5.   0.4% ( +2.6pp)  BUY DA share

  MCTS Visits (top 5, 6400 total):
     1.  5898 (92.2%) Q=-0.227 ████████████████████████████████████ PASS (INVEST)
     2.   324 ( 5.1%) Q=-0.247 ██ BUY SI share
     3.    86 ( 1.3%) Q=-0.407  SELL VM share
     4.    60 ( 0.9%) Q=-0.399  SELL SI share
     5.    32 ( 0.5%) Q=-0.383  BUY DA share
  A0GB Value: P0=-0.307, P1=-0.320, P2=+0.660 (depth: 49, vbackups: 4122)

  **Action: PASS (INVEST)**

### Step 66: P2 [INVEST]

  NN Values: P0=-0.136, P1=+0.011, P2=+0.108
  NN Priors (top 7 of 7 legal):
     1.  64.3% (-10.3pp) █████████████████████████ AUCTION slot 0 (KK, face $21)
     2.  32.3% ( -1.2pp) ████████████ AUCTION slot 1 (NS, face $22)
     3.   3.0% ( +2.2pp) █ PASS (INVEST)
     4.   0.1% ( +1.0pp)  SELL SM share
     5.   0.1% ( +3.9pp)  BUY SM share
     6.   0.1% ( +2.1pp)  BUY SI share
     7.   0.0% ( +2.3pp)  BUY DA share

  MCTS Visits (top 7, 6400 total):
     1.  6120 (95.6%) Q=+0.445 ██████████████████████████████████████ AUCTION slot 0 (KK, face $21)
     2.   255 ( 4.0%) Q=+0.290 █ AUCTION slot 1 (NS, face $22)
     3.    11 ( 0.2%) Q=-0.163  PASS (INVEST)
     4.     6 ( 0.1%) Q=-0.498  BUY SM share
     5.     4 ( 0.1%) Q=-0.560  BUY SI share
     6.     3 ( 0.0%) Q=-0.690  BUY DA share
     7.     1 ( 0.0%) Q=-0.350  SELL SM share
  A0GB Value: P0=-0.049, P1=-0.527, P2=+0.609 (depth: 51, vbackups: 5890)

  **Action: AUCTION slot 0 (KK, face $21)**

Phase: BID_IN_AUCTION  |  Turn: 4  |  CoO Level: 3  |  Active Player: 2  |  End Card: no

**Players**
  P0: $16 (NW $43) order=1 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=0 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $25 (NW $43) order=2 income=$0  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: KK (fv=$21, 3★, inc=$5), NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$11 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$5 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 9 remaining

**Auction**: KK current bid=$0 high bidder=P-1 starter=P2

### Step 67: P2 [BID_IN_AUCTION]

  **Auction**: KK current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.146, P1=+0.002, P2=+0.083
  NN Priors (top 5 of 5 legal):
     1.  97.2% ( -8.7pp) ██████████████████████████████████████ BID $21
     2.   2.1% ( +3.9pp)  BID $22
     3.   0.3% ( +2.9pp)  BID $23
     4.   0.2% ( +1.8pp)  BID $24
     5.   0.1% ( +0.1pp)  BID $25

  MCTS Visits (top 5, 6400 total):
     1.  6364 (99.4%) Q=+0.445 ███████████████████████████████████████ BID $21
     2.    25 ( 0.4%) Q=+0.142  BID $22
     3.     7 ( 0.1%) Q=-0.111  BID $23
     4.     4 ( 0.1%) Q=-0.233  BID $24
  A0GB Value: P0=-0.119, P1=-0.492, P2=+0.598 (depth: 52, vbackups: 6119)

  **Action: BID $21**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 4  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $16 (NW $43) order=1 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=0 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [2]: NS (fv=$22, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$3 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 8 remaining


### Step 68: P1 [INVEST]

  NN Values: P0=-0.113, P1=-0.079, P2=+0.146
  NN Priors (top 3 of 3 legal):
     1.  66.3% (-10.3pp) ██████████████████████████ PASS (INVEST)
     2.  33.2% ( -2.0pp) █████████████ SELL DA share
     3.   0.5% (+12.3pp)  SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  5603 (87.5%) Q=-0.237 ███████████████████████████████████ PASS (INVEST)
     2.   702 (11.0%) Q=-0.284 ████ SELL DA share
     3.    95 ( 1.5%) Q=-0.432  SELL VM share
  A0GB Value: P0=-0.119, P1=-0.492, P2=+0.598 (depth: 51, vbackups: 6320)

  **Action: PASS (INVEST)**

### Step 69: P0 [INVEST]

  NN Values: P0=-0.096, P1=-0.061, P2=+0.197
  NN Priors (top 5 of 5 legal):
     1.  68.9% ( -8.7pp) ███████████████████████████ PASS (INVEST)
     2.  22.8% ( +0.0pp) █████████ SELL VM share
     3.   6.7% ( -1.1pp) ██ BUY SI share
     4.   1.4% ( +0.6pp)  SELL SI share
     5.   0.2% ( +9.2pp)  BUY DA share

  MCTS Visits (top 5, 6400 total):
     1.  5621 (87.8%) Q=-0.252 ███████████████████████████████████ PASS (INVEST)
     2.   552 ( 8.6%) Q=-0.261 ███ BUY SI share
     3.   142 ( 2.2%) Q=-0.454  SELL VM share
     4.    73 ( 1.1%) Q=-0.421  BUY DA share
     5.    12 ( 0.2%) Q=-0.467  SELL SI share
  A0GB Value: P0=+0.090, P1=-0.680, P2=+0.594 (depth: 59, vbackups: 5490)

  **Action: PASS (INVEST)**

### Step 70: P2 [INVEST]

  NN Values: P0=-0.052, P1=-0.072, P2=+0.102
  NN Priors (top 2 of 2 legal):
     1.  99.2% (-10.5pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.8% (+10.5pp)  SELL SM share

  MCTS Visits (top 2, 6400 total):
     1.  6379 (99.7%) Q=+0.563 ███████████████████████████████████████ PASS (INVEST)
     2.    21 ( 0.3%) Q=-0.130  SELL SM share
  A0GB Value: P0=+0.043, P1=-0.676, P2=+0.629 (depth: 60, vbackups: 5620)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$3 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 8 remaining

**Acquisition — Select Corp**: P0 may buy with VM($17), SI($18)

### Step 71: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with VM($17), SI($18)

  NN Values: P0=-0.299, P1=-0.289, P2=+0.520
  NN Priors (top 2 of 2 legal):
     1.  97.6% ( -3.5pp) ███████████████████████████████████████ ACQ select VM
     2.   2.4% ( +3.5pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6214 (97.1%) Q=-0.248 ██████████████████████████████████████ ACQ select VM
     2.   186 ( 2.9%) Q=-0.271 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=+0.043, P1=-0.676, P2=+0.629 (depth: 59, vbackups: 6263)

  **Action: ACQ select VM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$3 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 8 remaining

**Acquisition — Select Company**: P0 buying with VM ($17)

### Step 72: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with VM ($17)

  NN Values: P0=-0.314, P1=-0.273, P2=+0.504
  NN Priors (top 2 of 2 legal):
     1.  51.2% ( -0.6pp) ████████████████████ ACQ target WT (with VM)
     2.  48.8% ( +0.6pp) ███████████████████ ACQ target MHE (with VM)

  MCTS Visits (top 2, 6400 total):
     1.  5012 (78.3%) Q=-0.239 ███████████████████████████████ ACQ target WT (with VM)
     2.  1388 (21.7%) Q=-0.274 ████████ ACQ target MHE (with VM)
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 59, vbackups: 6001)

  **Action: ACQ target WT (with VM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[KME]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$3 stars=6 pres=P0  companies=[MHE, WT]

**Deck**: 8 remaining

**Acquisition — Select Price**: P0 VM -> WT (price range $6-$14)

### Step 73: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> WT (price range $6-$14)

  NN Values: P0=-0.297, P1=-0.332, P2=+0.527
  NN Priors (top 9 of 9 legal):
     1.  92.7% (-12.8pp) █████████████████████████████████████ ACQUIRE WT with VM @ $6
     2.   3.4% ( -0.4pp) █ ACQUIRE WT with VM @ $7
     3.   0.9% ( +0.2pp)  ACQUIRE WT with VM @ $8
     4.   0.8% ( +3.2pp)  ACQUIRE WT with VM @ $9
     5.   0.7% ( +4.1pp)  ACQUIRE WT with VM @ $10
     6.   0.5% ( +1.2pp)  ACQUIRE WT with VM @ $11
     7.   0.4% ( +2.2pp)  ACQUIRE WT with VM @ $14
     8.   0.4% ( +1.9pp)  ACQUIRE WT with VM @ $13
     9.   0.3% ( +0.5pp)  ACQUIRE WT with VM @ $12

  MCTS Visits (top 9, 6400 total):
     1.  6160 (96.2%) Q=-0.234 ██████████████████████████████████████ ACQUIRE WT with VM @ $6
     2.    59 ( 0.9%) Q=-0.308  ACQUIRE WT with VM @ $9
     3.    50 ( 0.8%) Q=-0.346  ACQUIRE WT with VM @ $10
     4.    34 ( 0.5%) Q=-0.345  ACQUIRE WT with VM @ $14
     5.    32 ( 0.5%) Q=-0.342  ACQUIRE WT with VM @ $7
     6.    26 ( 0.4%) Q=-0.329  ACQUIRE WT with VM @ $11
     7.    21 ( 0.3%) Q=-0.410  ACQUIRE WT with VM @ $13
     8.    10 ( 0.2%) Q=-0.391  ACQUIRE WT with VM @ $12
     9.     8 ( 0.1%) Q=-0.386  ACQUIRE WT with VM @ $8
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 58, vbackups: 5007)

  **Action: ACQUIRE WT with VM @ $6**

Phase: ACQ_SELECT_CORP  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $11 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[KME, WT*]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$0 stars=4 pres=P0  companies=[MHE]

**Deck**: 8 remaining

**Acquisition — Select Corp**: P0 may buy with VM($11), SI($18)

### Step 74: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with VM($11), SI($18)

  NN Values: P0=-0.342, P1=-0.151, P2=+0.428
  NN Priors (top 2 of 2 legal):
     1.  94.9% ( -6.2pp) █████████████████████████████████████ ACQ select SI
     2.   5.1% ( +6.2pp) ██ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6136 (95.9%) Q=-0.233 ██████████████████████████████████████ ACQ select SI
     2.   264 ( 4.1%) Q=-0.273 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 57, vbackups: 6159)

  **Action: ACQ select SI**

  ↳ auto: ACQ target KME (with SI)

Phase: ACQ_SELECT_PRICE  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $11 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[KME, WT*]
  SI: $18 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$0 stars=4 pres=P0  companies=[MHE]

**Deck**: 8 remaining

**Acquisition — Select Price**: P0 SI -> KME (price range $3-$7)

### Step 75: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 SI -> KME (price range $3-$7)

  NN Values: P0=-0.340, P1=-0.193, P2=+0.463
  NN Priors (top 5 of 5 legal):
     1.  93.6% (-12.2pp) █████████████████████████████████████ ACQUIRE KME with SI @ $7
     2.   4.4% ( +2.8pp) █ ACQUIRE KME with SI @ $6
     3.   0.8% ( +0.3pp)  ACQUIRE KME with SI @ $5
     4.   0.6% ( +6.2pp)  ACQUIRE KME with SI @ $4
     5.   0.6% ( +2.9pp)  ACQUIRE KME with SI @ $3

  MCTS Visits (top 5, 6400 total):
     1.  5974 (93.3%) Q=-0.232 █████████████████████████████████████ ACQUIRE KME with SI @ $7
     2.   242 ( 3.8%) Q=-0.257 █ ACQUIRE KME with SI @ $6
     3.   131 ( 2.0%) Q=-0.286  ACQUIRE KME with SI @ $4
     4.    43 ( 0.7%) Q=-0.322  ACQUIRE KME with SI @ $3
     5.    10 ( 0.2%) Q=-0.348  ACQUIRE KME with SI @ $5
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 56, vbackups: 6096)

  **Action: ACQUIRE KME with SI @ $7**

Phase: ACQ_SELECT_CORP  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $11 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$3 stars=3 pres=P0  companies=[WT*]
  SI: $11 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$1 stars=5 pres=P0  companies=[KME*, MHE]

**Deck**: 8 remaining

**Acquisition — Select Corp**: P0 may buy with VM($11), SI($11)

### Step 76: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with VM($11), SI($11)

  NN Values: P0=-0.373, P1=-0.133, P2=+0.480
  NN Priors (top 2 of 2 legal):
     1.  98.5% (-10.9pp) ███████████████████████████████████████ ACQ select VM
     2.   1.5% (+10.9pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6265 (97.9%) Q=-0.229 ███████████████████████████████████████ ACQ select VM
     2.   135 ( 2.1%) Q=-0.338  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 55, vbackups: 6011)

  **Action: ACQ select VM**

  ↳ auto: ACQ target MHE (with VM)

Phase: ACQ_SELECT_PRICE  |  Turn: 4  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $11 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$3 stars=3 pres=P0  companies=[WT*]
  SI: $11 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$1 stars=5 pres=P0  companies=[KME*, MHE]

**Deck**: 8 remaining

**Acquisition — Select Price**: P0 VM -> MHE (price range $4-$10)

### Step 77: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> MHE (price range $4-$10)

  NN Values: P0=-0.348, P1=-0.210, P2=+0.475
  NN Priors (top 7 of 7 legal):
     1.  83.3% (-12.4pp) █████████████████████████████████ ACQUIRE MHE with VM @ $4
     2.   7.8% ( +3.8pp) ███ ACQUIRE MHE with VM @ $5
     3.   2.7% ( +0.6pp) █ ACQUIRE MHE with VM @ $6
     4.   1.8% ( +0.7pp)  ACQUIRE MHE with VM @ $7
     5.   1.8% ( +4.4pp)  ACQUIRE MHE with VM @ $10
     6.   1.3% ( +1.9pp)  ACQUIRE MHE with VM @ $8
     7.   1.2% ( +1.0pp)  ACQUIRE MHE with VM @ $9

  MCTS Visits (top 7, 6400 total):
     1.  5792 (90.5%) Q=-0.225 ████████████████████████████████████ ACQUIRE MHE with VM @ $4
     2.   385 ( 6.0%) Q=-0.252 ██ ACQUIRE MHE with VM @ $5
     3.    88 ( 1.4%) Q=-0.305  ACQUIRE MHE with VM @ $10
     4.    48 ( 0.8%) Q=-0.318  ACQUIRE MHE with VM @ $8
     5.    33 ( 0.5%) Q=-0.329  ACQUIRE MHE with VM @ $7
     6.    29 ( 0.5%) Q=-0.361  ACQUIRE MHE with VM @ $6
     7.    25 ( 0.4%) Q=-0.326  ACQUIRE MHE with VM @ $9
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 54, vbackups: 6144)

  **Action: ACQUIRE MHE with VM @ $4**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 4  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $7 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$5 stars=3 pres=P0  companies=[MHE*, WT*]
  SI: $11 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$0 stars=4 pres=P0  companies=[KME*]

**Deck**: 8 remaining

**Acquisition — Select Corp**: P2 may buy with SM($18)

### Step 78: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with SM($18)

  NN Values: P0=-0.307, P1=-0.160, P2=+0.414
  NN Priors (top 2 of 2 legal):
     1.  99.6% (-12.2pp) ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   0.4% (+12.2pp)  ACQ select SM

  MCTS Visits (top 2, 6400 total):
     1.  6375 (99.6%) Q=+0.598 ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.    25 ( 0.4%) Q=-0.066  ACQ select SM
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 53, vbackups: 5909)

  **Action: PASS (ACQ_SELECT_CORP)**

Phase: CLOSING  |  Turn: 4  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $4 (NW $43) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $19 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $18 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $14 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $21 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$0 stars=5 pres=P0  companies=[KME]

**Deck**: 8 remaining

**Closing**: P1 may close BSE (DA), AKE (DA), PR (DA)

### Step 79: P1 [CLOSING]

  **Closing**: P1 may close BSE (DA), AKE (DA), PR (DA)

  NN Values: P0=-0.271, P1=-0.197, P2=+0.389
  NN Priors (top 2 of 2 legal):
     1.  99.5% ( -8.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.5% ( +8.4pp)  CLOSE BSE

  MCTS Visits (top 2, 6400 total):
     1.  6064 (94.8%) Q=-0.379 █████████████████████████████████████ PASS (CLOSING)
     2.   336 ( 5.2%) Q=-0.396 ██ CLOSE BSE
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 52, vbackups: 6088)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 4  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $16 (NW $43) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $6 (NW $44) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $9 (NW $48) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $24 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=6 pres=P2  companies=[BD, SX]
  DA: $20 price=$12(idx 8) shares=bank:2/unissued:1/issued:4 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $19 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $21 price=$13(idx 9) shares=bank:3/unissued:0/issued:4 income=$0 stars=5 pres=P0  companies=[KME]

**Deck**: 8 remaining

**Dividends**: SM

### Step 80: P2 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.352, P1=-0.199, P2=+0.424
  NN Priors (top 7 of 7 legal):
     1.  75.4% (-10.0pp) ██████████████████████████████ DIVIDEND $4
     2.  14.3% ( -1.9pp) █████ DIVIDEND $6
     3.   5.2% ( +0.0pp) ██ DIVIDEND $1
     4.   2.8% ( +0.6pp) █ DIVIDEND $5
     5.   2.0% ( +2.0pp)  DIVIDEND $3
     6.   0.3% ( +7.2pp)  DIVIDEND $0
     7.   0.2% ( +2.1pp)  DIVIDEND $2

  MCTS Visits (top 7, 6400 total):
     1.  5981 (93.5%) Q=+0.603 █████████████████████████████████████ DIVIDEND $4
     2.   309 ( 4.8%) Q=+0.563 █ DIVIDEND $6
     3.    30 ( 0.5%) Q=+0.467  DIVIDEND $5
     4.    28 ( 0.4%) Q=+0.424  DIVIDEND $3
     5.    23 ( 0.4%) Q=+0.184  DIVIDEND $0
     6.    22 ( 0.3%) Q=+0.310  DIVIDEND $1
     7.     7 ( 0.1%) Q=+0.227  DIVIDEND $2
  A0GB Value: P0=+0.050, P1=-0.691, P2=+0.590 (depth: 51, vbackups: 6220)

  **Action: DIVIDEND $4**

### Step 81: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.375, P1=-0.188, P2=+0.451
  NN Priors (top 5 of 5 legal):
     1.  44.4% ( +4.6pp) █████████████████ DIVIDEND $4
     2.  21.4% ( -2.7pp) ████████ DIVIDEND $3
     3.  14.5% ( -2.1pp) █████ DIVIDEND $0
     4.  10.2% ( +1.0pp) ████ DIVIDEND $1
     5.   9.5% ( -0.8pp) ███ DIVIDEND $2

  MCTS Visits (top 5, 6400 total):
     1.  2674 (41.8%) Q=-0.205 ████████████████ DIVIDEND $1
     2.  2100 (32.8%) Q=-0.207 █████████████ DIVIDEND $0
     3.   984 (15.4%) Q=-0.267 ██████ DIVIDEND $4
     4.   411 ( 6.4%) Q=-0.261 ██ DIVIDEND $3
     5.   231 ( 3.6%) Q=-0.253 █ DIVIDEND $2
  A0GB Value: P0=+0.047, P1=-0.688, P2=+0.605 (depth: 49, vbackups: 5472)

  **Action: DIVIDEND $1**

### Step 82: P0 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.346, P1=-0.262, P2=+0.500
  NN Priors (top 5 of 5 legal):
     1.  95.8% (-12.9pp) ██████████████████████████████████████ DIVIDEND $4
     2.   2.5% ( +0.6pp) █ DIVIDEND $3
     3.   1.3% ( +0.0pp)  DIVIDEND $2
     4.   0.3% ( +5.4pp)  DIVIDEND $0
     5.   0.1% ( +6.9pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  6236 (97.4%) Q=-0.230 ██████████████████████████████████████ DIVIDEND $4
     2.    54 ( 0.8%) Q=-0.425  DIVIDEND $1
     3.    47 ( 0.7%) Q=-0.315  DIVIDEND $3
     4.    41 ( 0.6%) Q=-0.405  DIVIDEND $0
     5.    22 ( 0.3%) Q=-0.298  DIVIDEND $2
  A0GB Value: P0=+0.139, P1=-0.422, P2=+0.258 (depth: 64, vbackups: 2673)

  **Action: DIVIDEND $4**

### Step 83: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.268, P1=-0.342, P2=+0.504
  NN Priors (top 5 of 5 legal):
     1.  88.5% ( -7.3pp) ███████████████████████████████████ DIVIDEND $4
     2.   9.6% ( -2.0pp) ███ DIVIDEND $2
     3.   1.1% ( +0.9pp)  DIVIDEND $3
     4.   0.5% ( +5.4pp)  DIVIDEND $0
     5.   0.2% ( +3.0pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  5398 (84.3%) Q=-0.417 █████████████████████████████████ DIVIDEND $4
     2.   761 (11.9%) Q=-0.411 ████ DIVIDEND $2
     3.   187 ( 2.9%) Q=-0.442 █ DIVIDEND $0
     4.    31 ( 0.5%) Q=-0.536  DIVIDEND $1
     5.    23 ( 0.4%) Q=-0.508  DIVIDEND $3
  A0GB Value: P0=+0.131, P1=-0.471, P2=+0.312 (depth: 64, vbackups: 6160)

  **Action: DIVIDEND $4**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 4  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $48) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $15 (NW $51) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $13 (NW $52) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $12 price=$18(idx 12) shares=bank:2/unissued:3/issued:3 income=$6 stars=5 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$11(idx 7) shares=bank:3/unissued:0/issued:4 income=$0 stars=3 pres=P0  companies=[KME]

**Deck**: 8 remaining

**Issue**: SM

### Step 84: P2 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.225, P1=-0.412, P2=+0.633
  NN Priors (top 2 of 2 legal):
     1.  93.4% (-11.7pp) █████████████████████████████████████ ISSUE SM shares
     2.   6.6% (+11.7pp) ██ PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  4289 (67.0%) Q=+0.657 ██████████████████████████ ISSUE SM shares
     2.  2111 (33.0%) Q=+0.671 █████████████ PASS (ISSUE_SHARES)
  A0GB Value: P0=+0.131, P1=-0.471, P2=+0.312 (depth: 63, vbackups: 5397)

  **Action: ISSUE SM shares**

### Step 85: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.173, P1=-0.371, P2=+0.578
  NN Priors (top 2 of 2 legal):
     1.  98.4% ( -7.7pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   1.6% ( +7.7pp)  ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6256 (97.8%) Q=-0.258 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   144 ( 2.2%) Q=-0.329  ISSUE VM shares
  A0GB Value: P0=+0.131, P1=-0.471, P2=+0.312 (depth: 62, vbackups: 4288)

  **Action: PASS (ISSUE_SHARES)**

### Step 86: P1 [ISSUE_SHARES]

  **Issue**: DA

  NN Values: P0=-0.228, P1=-0.322, P2=+0.555
  NN Priors (top 2 of 2 legal):
     1.  99.6% ( -2.6pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   0.4% ( +2.6pp)  ISSUE DA shares

  MCTS Visits (top 2, 6400 total):
     1.  6227 (97.3%) Q=-0.403 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   173 ( 2.7%) Q=-0.410 █ ISSUE DA shares
  A0GB Value: P0=+0.131, P1=-0.471, P2=+0.312 (depth: 61, vbackups: 6240)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 4  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $48) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $15 (NW $51) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $13 (NW $52) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$11(idx 7) shares=bank:3/unissued:0/issued:4 income=$0 stars=3 pres=P0  companies=[KME]

**Deck**: 8 remaining

**IPO**: KK

### Step 87: P2 [IPO]

  **IPO**: KK

  NN Values: P0=-0.212, P1=-0.395, P2=+0.590
  NN Priors (top 5 of 5 legal):
     1.  84.4% (-11.5pp) █████████████████████████████████ IPO KK → float OS
     2.  15.0% ( +2.8pp) ██████ IPO KK → float PR
     3.   0.3% ( +1.3pp)  PASS (IPO)
     4.   0.1% ( +2.7pp)  IPO KK → float S
     5.   0.1% ( +4.7pp)  IPO KK → float JS

  MCTS Visits (top 5, 6400 total):
     1.  4182 (65.3%) Q=+0.656 ██████████████████████████ IPO KK → float OS
     2.  2161 (33.8%) Q=+0.671 █████████████ IPO KK → float PR
     3.    27 ( 0.4%) Q=+0.539  IPO KK → float S
     4.    24 ( 0.4%) Q=+0.413  IPO KK → float JS
     5.     6 ( 0.1%) Q=+0.372  PASS (IPO)
  A0GB Value: P0=+0.131, P1=-0.471, P2=+0.312 (depth: 60, vbackups: 6073)

  **Action: IPO KK → float OS**

Phase: PAR  |  Turn: 4  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $48) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $15 (NW $51) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $13 (NW $52) order=2 income=$5  companies=[KK]  shares=[SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$11(idx 7) shares=bank:3/unissued:0/issued:4 income=$0 stars=3 pres=P0  companies=[KME]

**Deck**: 8 remaining

**PAR**: KK -> OS

### Step 88: P2 [PAR]

  **PAR**: KK -> OS

  NN Values: P0=-0.177, P1=-0.439, P2=+0.598
  NN Priors (top 3 of 3 legal):
     1.  84.7% ( -7.1pp) █████████████████████████████████ PAR OS @$27 (IPO KK)
     2.  12.0% ( +4.1pp) ████ PAR OS @$24 (IPO KK)
     3.   3.2% ( +3.0pp) █ PAR OS @$22 (IPO KK)

  MCTS Visits (top 3, 6400 total):
     1.  4131 (64.5%) Q=+0.676 █████████████████████████ PAR OS @$24 (IPO KK)
     2.  2088 (32.6%) Q=+0.630 █████████████ PAR OS @$27 (IPO KK)
     3.   181 ( 2.8%) Q=+0.633 █ PAR OS @$22 (IPO KK)
  A0GB Value: P0=-0.332, P1=-0.441, P2=+0.672 (depth: 48, vbackups: 4349)

  **Action: PAR OS @$24 (IPO KK)**

--- Turn 5 ---

Phase: INVEST  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $21 (NW $48) order=0 income=$0  shares=[VM=1 (pres), SI=1 (pres)]
  P1: $15 (NW $51) order=1 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $10 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$11(idx 7) shares=bank:3/unissued:0/issued:4 income=$0 stars=3 pres=P0  companies=[KME]

**Deck**: 8 remaining


### Step 89: P0 [INVEST]

  NN Values: P0=-0.342, P1=-0.318, P2=+0.582
  NN Priors (top 6 of 6 legal):
     1.  84.6% ( -9.1pp) █████████████████████████████████ SELL SI share
     2.  14.9% ( -0.3pp) █████ PASS (INVEST)
     3.   0.2% ( +0.5pp)  BUY SM share
     4.   0.2% ( +5.6pp)  BUY SI share
     5.   0.1% ( +0.7pp)  BUY DA share
     6.   0.1% ( +2.6pp)  SELL VM share

  MCTS Visits (top 6, 6400 total):
     1.  5890 (92.0%) Q=-0.337 ████████████████████████████████████ SELL SI share
     2.   394 ( 6.2%) Q=-0.372 ██ PASS (INVEST)
     3.    81 ( 1.3%) Q=-0.416  BUY SI share
     4.    15 ( 0.2%) Q=-0.755  SELL VM share
     5.    14 ( 0.2%) Q=-0.474  BUY DA share
     6.     6 ( 0.1%) Q=-0.472  BUY SM share
  A0GB Value: P0=-0.312, P1=-0.609, P2=+0.676 (depth: 52, vbackups: 4130)

  **Action: SELL SI share**

### Step 90: P1 [INVEST]

  NN Values: P0=-0.344, P1=-0.326, P2=+0.617
  NN Priors (top 5 of 5 legal):
     1.  83.5% (-11.6pp) █████████████████████████████████ PASS (INVEST)
     2.   8.5% ( -0.5pp) ███ SELL DA share
     3.   4.9% ( +1.6pp) █ SELL VM share
     4.   1.7% ( +5.4pp)  BUY SI share
     5.   1.4% ( +5.2pp)  BUY DA share

  MCTS Visits (top 5, 6400 total):
     1.  4874 (76.2%) Q=-0.327 ██████████████████████████████ SELL DA share
     2.  1169 (18.3%) Q=-0.409 ███████ PASS (INVEST)
     3.   157 ( 2.5%) Q=-0.387  BUY SI share
     4.   152 ( 2.4%) Q=-0.392  BUY DA share
     5.    48 ( 0.8%) Q=-0.520  SELL VM share
  A0GB Value: P0=-0.283, P1=-0.598, P2=+0.680 (depth: 52, vbackups: 5727)

  **Action: SELL DA share**

### Step 91: P2 [INVEST]

  NN Values: P0=-0.260, P1=-0.408, P2=+0.617
  NN Priors (top 5 of 5 legal):
     1.  92.7% (-13.1pp) █████████████████████████████████████ PASS (INVEST)
     2.   6.9% ( +0.9pp) ██ BUY DA share
     3.   0.3% ( +2.1pp)  BUY SI share
     4.   0.1% ( +2.3pp)  SELL SM share
     5.   0.0% ( +7.7pp)  SELL OS share

  MCTS Visits (top 5, 6400 total):
     1.  5909 (92.3%) Q=+0.681 ████████████████████████████████████ PASS (INVEST)
     2.   448 ( 7.0%) Q=+0.676 ██ BUY DA share
     3.    20 ( 0.3%) Q=+0.540  BUY SI share
     4.    17 ( 0.3%) Q=+0.077  SELL OS share
     5.     6 ( 0.1%) Q=+0.205  SELL SM share
  A0GB Value: P0=-0.192, P1=-0.629, P2=+0.770 (depth: 51, vbackups: 4873)

  **Action: PASS (INVEST)**

### Step 92: P0 [INVEST]

  NN Values: P0=-0.226, P1=-0.406, P2=+0.609
  NN Priors (top 9 of 9 legal):
     1.  92.0% ( -8.5pp) ████████████████████████████████████ AUCTION slot 0 (NS, face $22)
     2.   4.3% ( -0.6pp) █ PASS (INVEST)
     3.   1.2% ( +1.6pp)  AUCTION slot 1 (PKP, face $25)
     4.   1.2% ( +0.5pp)  BUY OS share
     5.   1.0% ( +5.7pp)  BUY DA share
     6.   0.2% ( +0.0pp)  BUY SI share
     7.   0.1% ( +0.5pp)  AUCTION slot 2 (DR, face $29)
     8.   0.0% ( +0.0pp)  BUY SM share
     9.   0.0% ( +0.7pp)  SELL VM share

  MCTS Visits (top 9, 6400 total):
     1.  5511 (86.1%) Q=-0.369 ██████████████████████████████████ AUCTION slot 0 (NS, face $22)
     2.   614 ( 9.6%) Q=-0.358 ███ PASS (INVEST)
     3.   118 ( 1.8%) Q=-0.381  AUCTION slot 1 (PKP, face $25)
     4.    78 ( 1.2%) Q=-0.471  BUY DA share
     5.    44 ( 0.7%) Q=-0.423  BUY OS share
     6.    19 ( 0.3%) Q=-0.424  AUCTION slot 2 (DR, face $29)
     7.     9 ( 0.1%) Q=-0.688  SELL VM share
     8.     6 ( 0.1%) Q=-0.392  BUY SI share
     9.     1 ( 0.0%) Q=-0.527  BUY SM share
  A0GB Value: P0=-0.192, P1=-0.629, P2=+0.770 (depth: 50, vbackups: 5890)

  **Action: AUCTION slot 0 (NS, face $22)**

Phase: BID_IN_AUCTION  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $30 (NW $46) order=0 income=$0  shares=[VM=1 (pres)]
  P1: $23 (NW $47) order=1 income=$0  shares=[DA=1 (pres), VM=1]
  P2: $10 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [3]: NS (fv=$22, 3★, inc=$5), PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$8(idx 4) shares=bank:3/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 8 remaining

**Auction**: NS current bid=$0 high bidder=P-1 starter=P0

### Step 93: P0 [BID_IN_AUCTION]

  **Auction**: NS current bid=$0 high bidder=P-1 starter=P0

  NN Values: P0=-0.232, P1=-0.395, P2=+0.605
  NN Priors (top 9 of 9 legal):
     1.  89.9% (-13.7pp) ███████████████████████████████████ BID $22
     2.   9.6% ( -0.1pp) ███ BID $23
     3.   0.3% ( +2.3pp)  BID $24
     4.   0.1% ( +0.1pp)  BID $25
     5.   0.0% ( +0.4pp)  BID $26
     6.   0.0% ( +1.4pp)  BID $27
     7.   0.0% ( +3.0pp)  BID $28
     8.   0.0% ( +6.6pp)  BID $29
     9.   0.0% ( +0.0pp)  BID $30

  MCTS Visits (top 9, 6400 total):
     1.  4546 (71.0%) Q=-0.375 ████████████████████████████ BID $22
     2.  1224 (19.1%) Q=-0.363 ███████ BID $23
     3.   512 ( 8.0%) Q=-0.359 ███ BID $24
     4.    47 ( 0.7%) Q=-0.564  BID $29
     5.    27 ( 0.4%) Q=-0.374  BID $25
     6.    21 ( 0.3%) Q=-0.555  BID $28
     7.    15 ( 0.2%) Q=-0.485  BID $27
     8.     7 ( 0.1%) Q=-0.444  BID $26
     9.     1 ( 0.0%) Q=-0.688  BID $30
  A0GB Value: P0=-0.233, P1=-0.570, P2=+0.688 (depth: 49, vbackups: 5510)

  **Action: BID $22**

### Step 94: P1 [BID_IN_AUCTION]

  **Auction**: NS current bid=$22 high bidder=P0 starter=P0

  NN Values: P0=-0.250, P1=-0.379, P2=+0.609
  NN Priors (top 2 of 2 legal):
     1.  62.5% ( -1.6pp) █████████████████████████ PASS (BID_IN_AUCTION)
     2.  37.5% ( +1.6pp) ██████████████ BID $23

  MCTS Visits (top 2, 6400 total):
     1.  5740 (89.7%) Q=-0.308 ███████████████████████████████████ BID $23
     2.   660 (10.3%) Q=-0.425 ████ PASS (BID_IN_AUCTION)
  A0GB Value: P0=-0.220, P1=-0.668, P2=+0.691 (depth: 51, vbackups: 4545)

  **Action: BID $23**

  ↳ auto: PASS (BID_IN_AUCTION)

### Step 95: P0 [BID_IN_AUCTION]

  **Auction**: NS current bid=$23 high bidder=P1 starter=P0

  NN Values: P0=-0.273, P1=-0.379, P2=+0.625
  NN Priors (top 8 of 8 legal):
     1.  99.4% ( -9.9pp) ███████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   0.5% ( +1.8pp)  BID $24
     3.   0.1% ( +1.4pp)  BID $25
     4.   0.0% ( +0.7pp)  BID $26
     5.   0.0% ( +0.4pp)  BID $27
     6.   0.0% ( +1.5pp)  BID $28
     7.   0.0% ( +0.0pp)  BID $29
     8.   0.0% ( +4.1pp)  BID $30

  MCTS Visits (top 8, 6400 total):
     1.  5826 (91.0%) Q=-0.390 ████████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   424 ( 6.6%) Q=-0.378 ██ BID $24
     3.    79 ( 1.2%) Q=-0.404  BID $25
     4.    23 ( 0.4%) Q=-0.461  BID $26
     5.    23 ( 0.4%) Q=-0.660  BID $30
     6.    17 ( 0.3%) Q=-0.558  BID $28
     7.     7 ( 0.1%) Q=-0.506  BID $27
     8.     1 ( 0.0%) Q=-0.641  BID $29
  A0GB Value: P0=-0.195, P1=-0.656, P2=+0.703 (depth: 51, vbackups: 5739)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 5  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $30 (NW $46) order=0 income=$0  shares=[VM=1 (pres)]
  P1: $0 (NW $46) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $10 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [2]: PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$8(idx 4) shares=bank:3/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 7 remaining


### Step 96: P1 [INVEST]

  NN Values: P0=-0.268, P1=-0.449, P2=+0.684
  NN Priors (top 3 of 3 legal):
     1.  98.7% (-10.1pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.9% ( +8.3pp)  SELL VM share
     3.   0.4% ( +1.8pp)  SELL DA share

  MCTS Visits (top 3, 6400 total):
     1.  6340 (99.1%) Q=-0.308 ███████████████████████████████████████ PASS (INVEST)
     2.    46 ( 0.7%) Q=-0.558  SELL VM share
     3.    14 ( 0.2%) Q=-0.536  SELL DA share
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 48, vbackups: 5825)

  **Action: PASS (INVEST)**

### Step 97: P2 [INVEST]

  NN Values: P0=-0.314, P1=-0.400, P2=+0.664
  NN Priors (top 5 of 5 legal):
     1.  93.4% (-10.1pp) █████████████████████████████████████ PASS (INVEST)
     2.   6.3% ( -0.7pp) ██ BUY DA share
     3.   0.2% ( +4.3pp)  BUY SI share
     4.   0.1% ( +2.3pp)  SELL OS share
     5.   0.0% ( +4.2pp)  SELL SM share

  MCTS Visits (top 5, 6400 total):
     1.  5401 (84.4%) Q=+0.670 █████████████████████████████████ PASS (INVEST)
     2.   948 (14.8%) Q=+0.685 █████ BUY DA share
     3.    40 ( 0.6%) Q=+0.575  BUY SI share
     4.     7 ( 0.1%) Q=-0.034  SELL SM share
     5.     4 ( 0.1%) Q=-0.042  SELL OS share
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 47, vbackups: 6337)

  **Action: PASS (INVEST)**

### Step 98: P0 [INVEST]

  NN Values: P0=-0.299, P1=-0.408, P2=+0.676
  NN Priors (top 8 of 8 legal):
     1.  83.4% (-10.0pp) █████████████████████████████████ AUCTION slot 0 (PKP, face $25)
     2.  14.5% ( -1.1pp) █████ BUY OS share
     3.   0.9% ( +0.4pp)  BUY DA share
     4.   0.6% ( +0.0pp)  AUCTION slot 1 (DR, face $29)
     5.   0.5% ( +0.2pp)  BUY SI share
     6.   0.1% ( +1.8pp)  BUY SM share
     7.   0.0% ( +2.9pp)  SELL VM share
     8.   0.0% ( +5.8pp)  PASS (INVEST)

  MCTS Visits (top 8, 6400 total):
     1.  6100 (95.3%) Q=-0.386 ██████████████████████████████████████ AUCTION slot 0 (PKP, face $25)
     2.   108 ( 1.7%) Q=-0.550  BUY OS share
     3.    64 ( 1.0%) Q=-0.400  BUY SI share
     4.    47 ( 0.7%) Q=-0.553  PASS (INVEST)
     5.    30 ( 0.5%) Q=-0.424  AUCTION slot 1 (DR, face $29)
     6.    23 ( 0.4%) Q=-0.445  BUY DA share
     7.    16 ( 0.2%) Q=-0.530  BUY SM share
     8.    12 ( 0.2%) Q=-0.697  SELL VM share
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 46, vbackups: 5394)

  **Action: AUCTION slot 0 (PKP, face $25)**

Phase: BID_IN_AUCTION  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $30 (NW $46) order=0 income=$0  shares=[VM=1 (pres)]
  P1: $0 (NW $46) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $10 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [2]: PKP (fv=$25, 3★, inc=$5), DR (fv=$29, 3★, inc=$5)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$8(idx 4) shares=bank:3/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 7 remaining

**Auction**: PKP current bid=$0 high bidder=P-1 starter=P0

### Step 99: P0 [BID_IN_AUCTION]

  **Auction**: PKP current bid=$0 high bidder=P-1 starter=P0

  NN Values: P0=-0.256, P1=-0.414, P2=+0.672
  NN Priors (top 6 of 6 legal):
     1.  99.2% (-14.4pp) ███████████████████████████████████████ BID $25
     2.   0.6% ( +0.4pp)  BID $26
     3.   0.1% ( +4.6pp)  BID $27
     4.   0.0% ( +0.7pp)  BID $28
     5.   0.0% ( +5.0pp)  BID $29
     6.   0.0% ( +3.7pp)  BID $30

  MCTS Visits (top 6, 6400 total):
     1.  6252 (97.7%) Q=-0.387 ███████████████████████████████████████ BID $25
     2.    52 ( 0.8%) Q=-0.496  BID $27
     3.    36 ( 0.6%) Q=-0.610  BID $29
     4.    24 ( 0.4%) Q=-0.434  BID $26
     5.    24 ( 0.4%) Q=-0.649  BID $30
     6.    12 ( 0.2%) Q=-0.462  BID $28
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 45, vbackups: 6099)

  **Action: BID $25**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 5  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $46) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $10 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres)]

**FI**: $24 income=$5

**Auction Row** [1]: DR (fv=$29, 3★, inc=$5)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$8(idx 4) shares=bank:3/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining


### Step 100: P1 [INVEST]

  NN Values: P0=-0.338, P1=-0.326, P2=+0.688
  NN Priors (top 3 of 3 legal):
     1.  98.0% (-11.0pp) ███████████████████████████████████████ PASS (INVEST)
     2.   1.3% ( +2.1pp)  SELL VM share
     3.   0.8% ( +8.9pp)  SELL DA share

  MCTS Visits (top 3, 6400 total):
     1.  6284 (98.2%) Q=-0.314 ███████████████████████████████████████ PASS (INVEST)
     2.    86 ( 1.3%) Q=-0.459  SELL DA share
     3.    30 ( 0.5%) Q=-0.504  SELL VM share
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 44, vbackups: 6251)

  **Action: PASS (INVEST)**

### Step 101: P2 [INVEST]

  NN Values: P0=-0.387, P1=-0.312, P2=+0.695
  NN Priors (top 5 of 5 legal):
     1.  61.9% ( -7.5pp) ████████████████████████ PASS (INVEST)
     2.  37.9% ( -3.9pp) ███████████████ BUY DA share
     3.   0.2% ( +4.9pp)  BUY SI share
     4.   0.0% ( +3.5pp)  SELL OS share
     5.   0.0% ( +3.0pp)  SELL SM share

  MCTS Visits (top 5, 6400 total):
     1.  3940 (61.6%) Q=+0.680 ████████████████████████ BUY DA share
     2.  2313 (36.1%) Q=+0.659 ██████████████ PASS (INVEST)
     3.    99 ( 1.5%) Q=+0.628  BUY SI share
     4.    33 ( 0.5%) Q=+0.547  SELL OS share
     5.    15 ( 0.2%) Q=+0.437  SELL SM share
  A0GB Value: P0=-0.350, P1=-0.582, P2=+0.711 (depth: 43, vbackups: 6169)

  **Action: BUY DA share**

### Step 102: P0 [INVEST]

  NN Values: P0=-0.268, P1=-0.398, P2=+0.680
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -8.1pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.0% ( +8.1pp)  SELL VM share

  MCTS Visits (top 2, 6400 total):
     1.  6364 (99.4%) Q=-0.368 ███████████████████████████████████████ PASS (INVEST)
     2.    36 ( 0.6%) Q=-0.684  SELL VM share
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 42, vbackups: 3939)

  **Action: PASS (INVEST)**

### Step 103: P1 [INVEST]

  NN Values: P0=-0.330, P1=-0.311, P2=+0.672
  NN Priors (top 3 of 3 legal):
     1.  99.0% ( -9.5pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.5% ( +7.5pp)  SELL VM share
     3.   0.5% ( +2.0pp)  SELL DA share

  MCTS Visits (top 3, 6400 total):
     1.  6295 (98.4%) Q=-0.328 ███████████████████████████████████████ PASS (INVEST)
     2.    59 ( 0.9%) Q=-0.504  SELL VM share
     3.    46 ( 0.7%) Q=-0.388  SELL DA share
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 41, vbackups: 6304)

  **Action: PASS (INVEST)**

### Step 104: P2 [INVEST]

  NN Values: P0=-0.312, P1=-0.361, P2=+0.684
  NN Priors (top 4 of 4 legal):
     1.  99.3% (-13.0pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.6% ( +4.2pp)  SELL DA share
     3.   0.1% ( +1.0pp)  SELL OS share
     4.   0.0% ( +7.8pp)  SELL SM share

  MCTS Visits (top 4, 6400 total):
     1.  6330 (98.9%) Q=+0.680 ███████████████████████████████████████ PASS (INVEST)
     2.    46 ( 0.7%) Q=+0.627  SELL DA share
     3.    20 ( 0.3%) Q=+0.183  SELL SM share
     4.     4 ( 0.1%) Q=+0.239  SELL OS share
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 40, vbackups: 6347)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Corp**: P0 may buy with VM($17)

### Step 105: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with VM($17)

  NN Values: P0=-0.243, P1=-0.426, P2=+0.688
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -7.6pp) ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   0.0% ( +7.6pp)  ACQ select VM

  MCTS Visits (top 2, 6400 total):
     1.  6371 (99.5%) Q=-0.368 ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.    29 ( 0.5%) Q=-0.719  ACQ select VM
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 39, vbackups: 6335)

  **Action: PASS (ACQ_SELECT_CORP)**

  ↳ auto: PASS (ACQ_SELECT_CORP)

### Step 106: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($27), SM($30)

  NN Values: P0=-0.340, P1=-0.369, P2=+0.715
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -8.1pp) ███████████████████████████████████████ ACQ select OS
     2.   0.0% ( +8.1pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6379 (99.7%) Q=+0.680 ███████████████████████████████████████ ACQ select OS
     2.    21 ( 0.3%) Q=+0.183  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 38, vbackups: 6370)

  **Action: ACQ select OS**

Phase: ACQ_SELECT_COMPANY  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Company**: P2 buying with OS ($27)

### Step 107: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with OS ($27)

  NN Values: P0=-0.346, P1=-0.357, P2=+0.711
  NN Priors (top 2 of 2 legal):
     1.  98.2% ( -6.0pp) ███████████████████████████████████████ ACQ target SX (with OS)
     2.   1.8% ( +6.0pp)  ACQ target BD (with OS)

  MCTS Visits (top 2, 6400 total):
     1.  6167 (96.4%) Q=+0.680 ██████████████████████████████████████ ACQ target SX (with OS)
     2.   233 ( 3.6%) Q=+0.676 █ ACQ target BD (with OS)
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 37, vbackups: 6273)

  **Action: ACQ target SX (with OS)**

Phase: ACQ_SELECT_PRICE  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $27 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$5 stars=5 pres=P2  companies=[KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$6 stars=7 pres=P2  companies=[BD, SX]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Price**: P2 OS -> SX (price range $8-$21)

### Step 108: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> SX (price range $8-$21)

  NN Values: P0=-0.355, P1=-0.328, P2=+0.699
  NN Priors (top 10 of 14 legal):
     1.  99.2% (-14.8pp) ███████████████████████████████████████ ACQUIRE SX with OS @ $8
     2.   0.5% ( -0.1pp)  ACQUIRE SX with OS @ $9
     3.   0.0% ( +0.2pp)  ACQUIRE SX with OS @ $13
     4.   0.0% ( +1.0pp)  ACQUIRE SX with OS @ $19
     5.   0.0% ( +0.0pp)  ACQUIRE SX with OS @ $12
     6.   0.0% ( +0.0pp)  ACQUIRE SX with OS @ $10
     7.   0.0% ( +0.5pp)  ACQUIRE SX with OS @ $21
     8.   0.0% ( -0.0pp)  ACQUIRE SX with OS @ $20
     9.   0.0% ( +0.0pp)  ACQUIRE SX with OS @ $18
    10.   0.0% ( +6.5pp)  ACQUIRE SX with OS @ $14

  MCTS Visits (top 10, 6400 total):
     1.  6176 (96.5%) Q=+0.680 ██████████████████████████████████████ ACQUIRE SX with OS @ $8
     2.    73 ( 1.1%) Q=+0.592  ACQUIRE SX with OS @ $14
     3.    44 ( 0.7%) Q=+0.645  ACQUIRE SX with OS @ $11
     4.    24 ( 0.4%) Q=+0.562  ACQUIRE SX with OS @ $17
     5.    21 ( 0.3%) Q=+0.672  ACQUIRE SX with OS @ $9
     6.    14 ( 0.2%) Q=+0.581  ACQUIRE SX with OS @ $16
     7.     9 ( 0.1%) Q=+0.682  ACQUIRE SX with OS @ $10
     8.     9 ( 0.1%) Q=+0.592  ACQUIRE SX with OS @ $15
     9.     8 ( 0.1%) Q=+0.529  ACQUIRE SX with OS @ $19
    10.     8 ( 0.1%) Q=+0.641  ACQUIRE SX with OS @ $13
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 36, vbackups: 6234)

  **Action: ACQUIRE SX with OS @ $8**

Phase: ACQ_SELECT_CORP  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $19 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$10 stars=6 pres=P2  companies=[SX*, KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$3 stars=5 pres=P2  companies=[BD]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Corp**: P2 may buy with OS($19), SM($30)

### Step 109: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($19), SM($30)

  NN Values: P0=-0.344, P1=-0.373, P2=+0.727
  NN Priors (top 2 of 2 legal):
     1.  94.7% ( -1.0pp) █████████████████████████████████████ ACQ select SM
     2.   5.3% ( +1.0pp) ██ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6276 (98.1%) Q=+0.681 ███████████████████████████████████████ ACQ select SM
     2.   124 ( 1.9%) Q=+0.632  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 35, vbackups: 6208)

  **Action: ACQ select SM**

  ↳ auto: ACQ target KK (with SM)

Phase: ACQ_SELECT_PRICE  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $19 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$10 stars=6 pres=P2  companies=[SX*, KK]
  SM: $30 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$3 stars=5 pres=P2  companies=[BD]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Price**: P2 SM -> KK (price range $11-$28)

### Step 110: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SM -> KK (price range $11-$28)

  NN Values: P0=-0.352, P1=-0.348, P2=+0.730
  NN Priors (top 10 of 18 legal):
     1.  97.8% (-14.6pp) ███████████████████████████████████████ ACQUIRE KK with SM @ $28
     2.   2.0% ( -0.4pp)  ACQUIRE KK with SM @ $27
     3.   0.1% ( +0.4pp)  ACQUIRE KK with SM @ $26
     4.   0.0% ( +0.2pp)  ACQUIRE KK with SM @ $23
     5.   0.0% ( +3.3pp)  ACQUIRE KK with SM @ $22
     6.   0.0% ( +0.0pp)  ACQUIRE KK with SM @ $24
     7.   0.0% ( +0.5pp)  ACQUIRE KK with SM @ $25
     8.   0.0% ( +2.2pp)  ACQUIRE KK with SM @ $11
     9.   0.0% ( +0.3pp)  ACQUIRE KK with SM @ $17
    10.   0.0% ( +3.4pp)  ACQUIRE KK with SM @ $21

  MCTS Visits (top 10, 6400 total):
     1.  6164 (96.3%) Q=+0.681 ██████████████████████████████████████ ACQUIRE KK with SM @ $28
     2.    49 ( 0.8%) Q=+0.655  ACQUIRE KK with SM @ $27
     3.    41 ( 0.6%) Q=+0.603  ACQUIRE KK with SM @ $22
     4.    36 ( 0.6%) Q=+0.581  ACQUIRE KK with SM @ $21
     5.    31 ( 0.5%) Q=+0.568  ACQUIRE KK with SM @ $19
     6.    20 ( 0.3%) Q=+0.676  ACQUIRE KK with SM @ $26
     7.    18 ( 0.3%) Q=+0.661  ACQUIRE KK with SM @ $25
     8.     9 ( 0.1%) Q=+0.382  ACQUIRE KK with SM @ $11
     9.     6 ( 0.1%) Q=+0.626  ACQUIRE KK with SM @ $23
    10.     6 ( 0.1%) Q=+0.517  ACQUIRE KK with SM @ $15
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 34, vbackups: 6233)

  **Action: ACQUIRE KK with SM @ $28**

Phase: ACQ_SELECT_CORP  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $19 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$3 stars=3 pres=P2  companies=[SX*]
  SM: $2 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$8 stars=5 pres=P2  companies=[BD, KK*]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Corp**: P2 may buy with OS($19), SM($2)

### Step 111: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($19), SM($2)

  NN Values: P0=-0.330, P1=-0.361, P2=+0.691
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -9.3pp) ███████████████████████████████████████ ACQ select OS
     2.   0.1% ( +9.3pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6318 (98.7%) Q=+0.682 ███████████████████████████████████████ ACQ select OS
     2.    82 ( 1.3%) Q=+0.545  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 33, vbackups: 6201)

  **Action: ACQ select OS**

  ↳ auto: ACQ target BD (with OS)

Phase: ACQ_SELECT_PRICE  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $19 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$3 stars=3 pres=P2  companies=[SX*]
  SM: $2 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$8 stars=5 pres=P2  companies=[BD, KK*]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Acquisition — Select Price**: P2 OS -> BD (price range $7-$17)

### Step 112: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> BD (price range $7-$17)

  NN Values: P0=-0.346, P1=-0.344, P2=+0.691
  NN Priors (top 10 of 11 legal):
     1.  96.4% (-14.5pp) ██████████████████████████████████████ ACQUIRE BD with OS @ $7
     2.   2.1% ( +0.3pp)  ACQUIRE BD with OS @ $8
     3.   0.4% ( +4.3pp)  ACQUIRE BD with OS @ $9
     4.   0.3% ( +4.2pp)  ACQUIRE BD with OS @ $10
     5.   0.2% ( +0.1pp)  ACQUIRE BD with OS @ $11
     6.   0.1% ( +0.2pp)  ACQUIRE BD with OS @ $12
     7.   0.1% ( +1.1pp)  ACQUIRE BD with OS @ $16
     8.   0.1% ( +0.9pp)  ACQUIRE BD with OS @ $13
     9.   0.1% ( +0.2pp)  ACQUIRE BD with OS @ $17
    10.   0.1% ( -0.0pp)  ACQUIRE BD with OS @ $15

  MCTS Visits (top 10, 6400 total):
     1.  6059 (94.7%) Q=+0.682 █████████████████████████████████████ ACQUIRE BD with OS @ $7
     2.   123 ( 1.9%) Q=+0.663  ACQUIRE BD with OS @ $9
     3.    85 ( 1.3%) Q=+0.641  ACQUIRE BD with OS @ $10
     4.    65 ( 1.0%) Q=+0.653  ACQUIRE BD with OS @ $8
     5.    30 ( 0.5%) Q=+0.554  ACQUIRE BD with OS @ $14
     6.    10 ( 0.2%) Q=+0.658  ACQUIRE BD with OS @ $11
     7.    10 ( 0.2%) Q=+0.572  ACQUIRE BD with OS @ $13
     8.    10 ( 0.2%) Q=+0.548  ACQUIRE BD with OS @ $16
     9.     4 ( 0.1%) Q=+0.603  ACQUIRE BD with OS @ $12
    10.     3 ( 0.0%) Q=+0.592  ACQUIRE BD with OS @ $17
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 32, vbackups: 6162)

  **Action: ACQUIRE BD with OS @ $7**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 5  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $5 (NW $46) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $0 (NW $48) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $24 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $40 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$6 stars=8 pres=P2  companies=[BD, SX]
  SM: $17 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$5 stars=4 pres=P2  companies=[KK]
  DA: $4 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=4 pres=P1  companies=[BSE, AKE, PR]
  VM: $17 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Closing**: P1 may close BSE (DA), AKE (DA), PR (DA), NS

### Step 113: P1 [CLOSING]

  **Closing**: P1 may close BSE (DA), AKE (DA), PR (DA), NS

  NN Values: P0=-0.237, P1=-0.434, P2=+0.750
  NN Priors (top 2 of 2 legal):
     1.  98.6% ( -5.5pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   1.4% ( +5.5pp)  CLOSE BSE

  MCTS Visits (top 2, 6400 total):
     1.  6165 (96.3%) Q=-0.331 ██████████████████████████████████████ PASS (CLOSING)
     2.   235 ( 3.7%) Q=-0.352 █ CLOSE BSE
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 31, vbackups: 6192)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $10 (NW $51) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $5 (NW $53) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $0 (NW $52) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $46 price=$24(idx 15) shares=bank:1/unissued:4/issued:2 income=$6 stars=8 pres=P2  companies=[BD, SX]
  SM: $22 price=$18(idx 12) shares=bank:3/unissued:2/issued:4 income=$5 stars=5 pres=P2  companies=[KK]
  DA: $11 price=$10(idx 6) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $22 price=$16(idx 11) shares=bank:0/unissued:2/issued:2 income=$5 stars=5 pres=P0  companies=[MHE, WT]
  SI: $5 price=$9(idx 5) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Dividends**: OS

### Step 114: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.242, P1=-0.377, P2=+0.715
  NN Priors (top 9 of 9 legal):
     1.  99.1% (-14.6pp) ███████████████████████████████████████ DIVIDEND $8
     2.   0.8% ( +0.2pp)  DIVIDEND $7
     3.   0.1% ( +0.2pp)  DIVIDEND $5
     4.   0.0% ( +1.3pp)  DIVIDEND $6
     5.   0.0% ( +3.0pp)  DIVIDEND $2
     6.   0.0% ( +0.6pp)  DIVIDEND $1
     7.   0.0% ( +8.8pp)  DIVIDEND $4
     8.   0.0% ( +0.0pp)  DIVIDEND $3
     9.   0.0% ( +0.5pp)  DIVIDEND $0

  MCTS Visits (top 9, 6400 total):
     1.  6189 (96.7%) Q=+0.682 ██████████████████████████████████████ DIVIDEND $8
     2.   110 ( 1.7%) Q=+0.591  DIVIDEND $4
     3.    32 ( 0.5%) Q=+0.642  DIVIDEND $6
     4.    31 ( 0.5%) Q=+0.657  DIVIDEND $7
     5.    30 ( 0.5%) Q=+0.569  DIVIDEND $2
     6.     3 ( 0.0%) Q=+0.486  DIVIDEND $1
     7.     3 ( 0.0%) Q=+0.600  DIVIDEND $5
     8.     1 ( 0.0%) Q=+0.520  DIVIDEND $3
     9.     1 ( 0.0%) Q=+0.299  DIVIDEND $0
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 30, vbackups: 6183)

  **Action: DIVIDEND $8**

### Step 115: P2 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.188, P1=-0.385, P2=+0.680
  NN Priors (top 6 of 6 legal):
     1.  99.1% (-11.4pp) ███████████████████████████████████████ DIVIDEND $5
     2.   0.8% ( +1.3pp)  DIVIDEND $4
     3.   0.0% ( +0.7pp)  DIVIDEND $1
     4.   0.0% ( +0.6pp)  DIVIDEND $2
     5.   0.0% ( +0.7pp)  DIVIDEND $0
     6.   0.0% ( +8.2pp)  DIVIDEND $3

  MCTS Visits (top 6, 6400 total):
     1.  6176 (96.5%) Q=+0.682 ██████████████████████████████████████ DIVIDEND $5
     2.   138 ( 2.2%) Q=+0.630  DIVIDEND $3
     3.    52 ( 0.8%) Q=+0.658  DIVIDEND $4
     4.    14 ( 0.2%) Q=+0.646  DIVIDEND $2
     5.    12 ( 0.2%) Q=+0.620  DIVIDEND $1
     6.     8 ( 0.1%) Q=+0.589  DIVIDEND $0
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 29, vbackups: 6188)

  **Action: DIVIDEND $5**

### Step 116: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.352, P1=-0.277, P2=+0.641
  NN Priors (top 6 of 6 legal):
     1.  94.3% (-12.0pp) █████████████████████████████████████ DIVIDEND $5
     2.   3.1% ( -0.2pp) █ DIVIDEND $1
     3.   1.7% ( +4.2pp)  DIVIDEND $4
     4.   0.6% ( +7.5pp)  DIVIDEND $0
     5.   0.2% ( +0.2pp)  DIVIDEND $2
     6.   0.1% ( +0.3pp)  DIVIDEND $3

  MCTS Visits (top 6, 6400 total):
     1.  5603 (87.5%) Q=-0.371 ███████████████████████████████████ DIVIDEND $5
     2.   382 ( 6.0%) Q=-0.373 ██ DIVIDEND $4
     3.   230 ( 3.6%) Q=-0.401 █ DIVIDEND $0
     4.   120 ( 1.9%) Q=-0.383  DIVIDEND $1
     5.    33 ( 0.5%) Q=-0.365  DIVIDEND $3
     6.    32 ( 0.5%) Q=-0.370  DIVIDEND $2
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 28, vbackups: 5968)

  **Action: DIVIDEND $5**

### Step 117: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.293, P1=-0.283, P2=+0.648
  NN Priors (top 3 of 3 legal):
     1.  70.2% ( -3.5pp) ████████████████████████████ DIVIDEND $0
     2.  28.3% ( -2.2pp) ███████████ DIVIDEND $2
     3.   1.5% ( +5.7pp)  DIVIDEND $1

  MCTS Visits (top 3, 6400 total):
     1.  3334 (52.1%) Q=-0.336 ████████████████████ DIVIDEND $0
     2.  1753 (27.4%) Q=-0.329 ██████████ DIVIDEND $2
     3.  1313 (20.5%) Q=-0.315 ████████ DIVIDEND $1
  A0GB Value: P0=-0.336, P1=-0.570, P2=+0.707 (depth: 27, vbackups: 5655)

  **Action: DIVIDEND $0**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 5  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $15 (NW $58) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $10 (NW $61) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $13 (NW $68) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $30 price=$30(idx 17) shares=bank:1/unissued:4/issued:2 income=$6 stars=7 pres=P2  companies=[BD, SX]
  SM: $2 price=$14(idx 10) shares=bank:3/unissued:2/issued:4 income=$5 stars=3 pres=P2  companies=[KK]
  DA: $11 price=$11(idx 7) shares=bank:2/unissued:1/issued:4 income=$7 stars=5 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**Issue**: OS

### Step 118: P2 [ISSUE_SHARES]

  **Issue**: OS

  NN Values: P0=-0.293, P1=-0.373, P2=+0.695
  NN Priors (top 2 of 2 legal):
     1.  97.1% (-11.1pp) ██████████████████████████████████████ ISSUE OS shares
     2.   2.9% (+11.1pp) █ PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  5993 (93.6%) Q=+0.680 █████████████████████████████████████ ISSUE OS shares
     2.   407 ( 6.4%) Q=+0.652 ██ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.203, P1=-0.648, P2=+0.738 (depth: 27, vbackups: 3352)

  **Action: ISSUE OS shares**

### Step 119: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.344, P1=-0.309, P2=+0.676
  NN Priors (top 2 of 2 legal):
     1.  92.9% ( -5.7pp) █████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   7.1% ( +5.7pp) ██ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6172 (96.4%) Q=-0.342 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   228 ( 3.6%) Q=-0.399 █ ISSUE VM shares
  A0GB Value: P0=-0.289, P1=-0.516, P2=+0.734 (depth: 26, vbackups: 5992)

  **Action: PASS (ISSUE_SHARES)**

### Step 120: P2 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.326, P1=-0.359, P2=+0.688
  NN Priors (top 2 of 2 legal):
     1.  98.7% ( -5.9pp) ███████████████████████████████████████ ISSUE SM shares
     2.   1.3% ( +5.9pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6322 (98.8%) Q=+0.683 ███████████████████████████████████████ ISSUE SM shares
     2.    78 ( 1.2%) Q=+0.576  PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.289, P1=-0.516, P2=+0.734 (depth: 25, vbackups: 6171)

  **Action: ISSUE SM shares**

### Step 121: P1 [ISSUE_SHARES]

  **Issue**: DA

  NN Values: P0=-0.324, P1=-0.348, P2=+0.688
  NN Priors (top 2 of 2 legal):
     1.  91.7% ( -5.1pp) ████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   8.3% ( +5.1pp) ███ ISSUE DA shares

  MCTS Visits (top 2, 6400 total):
     1.  3203 (50.0%) Q=-0.336 ████████████████████ ISSUE DA shares
     2.  3197 (50.0%) Q=-0.367 ███████████████████ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.344, P1=-0.449, P2=+0.668 (depth: 25, vbackups: 6292)

  **Action: ISSUE DA shares**

Phase: IPO  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $15 (NW $58) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**IPO**: PKP

### Step 122: P0 [IPO]

  **IPO**: PKP

  NN Values: P0=-0.338, P1=-0.281, P2=+0.648
  NN Priors (top 4 of 4 legal):
     1.  96.2% (-13.2pp) ██████████████████████████████████████ IPO PKP → float PR
     2.   2.1% ( +8.2pp)  IPO PKP → float S
     3.   1.0% ( +2.1pp)  PASS (IPO)
     4.   0.8% ( +2.9pp)  IPO PKP → float JS

  MCTS Visits (top 4, 6400 total):
     1.  6253 (97.7%) Q=-0.319 ███████████████████████████████████████ IPO PKP → float PR
     2.    93 ( 1.5%) Q=-0.451  IPO PKP → float S
     3.    33 ( 0.5%) Q=-0.454  IPO PKP → float JS
     4.    21 ( 0.3%) Q=-0.503  PASS (IPO)
  A0GB Value: P0=-0.192, P1=-0.500, P2=+0.770 (depth: 26, vbackups: 3202)

  **Action: IPO PKP → float PR**

Phase: PAR  |  Turn: 5  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $15 (NW $58) order=0 income=$5  companies=[PKP]  shares=[VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**PAR**: PKP -> PR

### Step 123: P0 [PAR]

  **PAR**: PKP -> PR

  NN Values: P0=-0.352, P1=-0.301, P2=+0.664
  NN Priors (top 2 of 2 legal):
     1.  56.3% ( +3.1pp) ██████████████████████ PAR PR @$20 (IPO PKP)
     2.  43.7% ( -3.1pp) █████████████████ PAR PR @$16 (IPO PKP)

  MCTS Visits (top 2, 6400 total):
     1.  3504 (54.8%) Q=-0.313 █████████████████████ PAR PR @$16 (IPO PKP)
     2.  2896 (45.2%) Q=-0.324 ██████████████████ PAR PR @$20 (IPO PKP)
  A0GB Value: P0=-0.192, P1=-0.500, P2=+0.770 (depth: 25, vbackups: 5998)

  **Action: PAR PR @$16 (IPO PKP)**

Phase: IPO  |  Turn: 5  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=0 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**IPO**: NS

### Step 124: P1 [IPO]

  **IPO**: NS

  NN Values: P0=-0.198, P1=-0.398, P2=+0.625
  NN Priors (top 3 of 3 legal):
     1.  77.9% ( -7.5pp) ███████████████████████████████ IPO NS → float S
     2.  13.3% ( +4.8pp) █████ PASS (IPO)
     3.   8.8% ( +2.7pp) ███ IPO NS → float JS

  MCTS Visits (top 3, 6400 total):
     1.  3134 (49.0%) Q=-0.445 ███████████████████ IPO NS → float S
     2.  1811 (28.3%) Q=-0.428 ███████████ PASS (IPO)
     3.  1455 (22.7%) Q=-0.426 █████████ IPO NS → float JS
  A0GB Value: P0=-0.236, P1=-0.605, P2=+0.824 (depth: 30, vbackups: 3757)

  **Action: IPO NS → float S**

Phase: PAR  |  Turn: 5  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=0 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$5  companies=[NS]  shares=[DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining

**PAR**: NS -> S

### Step 125: P1 [PAR]

  **PAR**: NS -> S

  NN Values: P0=-0.157, P1=-0.402, P2=+0.617
  NN Priors (top 2 of 2 legal):
     1.  81.8% ( -2.0pp) ████████████████████████████████ PAR S @$22 (IPO NS)
     2.  18.2% ( +2.0pp) ███████ PAR S @$24 (IPO NS)

  MCTS Visits (top 2, 6400 total):
     1.  5480 (85.6%) Q=-0.453 ██████████████████████████████████ PAR S @$22 (IPO NS)
     2.   920 (14.4%) Q=-0.463 █████ PAR S @$24 (IPO NS)
  A0GB Value: P0=-0.439, P1=-0.311, P2=+0.832 (depth: 74, vbackups: 3133)

  **Action: PAR S @$22 (IPO NS)**

--- Turn 6 ---

Phase: INVEST  |  Turn: 6  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $8 (NW $58) order=0 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=2 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: DR (fv=$29, 3★, inc=$5), SZD (fv=$30, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 6 remaining


### Step 126: P0 [INVEST]

  NN Values: P0=-0.469, P1=-0.303, P2=+0.656
  NN Priors (top 3 of 3 legal):
     1.  91.2% ( -9.6pp) ████████████████████████████████████ PASS (INVEST)
     2.   8.6% ( +7.7pp) ███ SELL PR share
     3.   0.2% ( +1.9pp)  SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  6281 (98.1%) Q=-0.312 ███████████████████████████████████████ PASS (INVEST)
     2.   105 ( 1.6%) Q=-0.522  SELL PR share
     3.    14 ( 0.2%) Q=-0.648  SELL VM share
  A0GB Value: P0=-0.559, P1=-0.309, P2=+0.828 (depth: 77, vbackups: 5479)

  **Action: PASS (INVEST)**

### Step 127: P1 [INVEST]

  NN Values: P0=-0.389, P1=-0.424, P2=+0.723
  NN Priors (top 5 of 5 legal):
     1.  86.9% ( -8.8pp) ██████████████████████████████████ PASS (INVEST)
     2.   6.6% ( +7.4pp) ██ BUY SI share
     3.   3.2% ( -0.3pp) █ SELL DA share
     4.   2.1% ( -0.3pp)  SELL S share
     5.   1.2% ( +2.0pp)  SELL VM share

  MCTS Visits (top 5, 6400 total):
     1.  5475 (85.5%) Q=-0.456 ██████████████████████████████████ PASS (INVEST)
     2.   648 (10.1%) Q=-0.466 ████ BUY SI share
     3.   105 ( 1.6%) Q=-0.474  SELL DA share
     4.    93 ( 1.5%) Q=-0.462  SELL S share
     5.    79 ( 1.2%) Q=-0.502  SELL VM share
  A0GB Value: P0=-0.559, P1=-0.309, P2=+0.828 (depth: 76, vbackups: 6232)

  **Action: PASS (INVEST)**

### Step 128: P2 [INVEST]

  NN Values: P0=-0.512, P1=-0.252, P2=+0.688
  NN Priors (top 6 of 6 legal):
     1.  85.8% ( -8.7pp) ██████████████████████████████████ BUY DA share
     2.   8.4% ( -0.7pp) ███ PASS (INVEST)
     3.   3.0% ( +3.2pp) █ SELL SM share
     4.   2.1% ( +1.5pp)  SELL DA share
     5.   0.7% ( +0.2pp)  BUY SI share
     6.   0.1% ( +4.5pp)  SELL OS share

  MCTS Visits (top 6, 6400 total):
     1.  3737 (58.4%) Q=+0.782 ███████████████████████ PASS (INVEST)
     2.  2590 (40.5%) Q=+0.744 ████████████████ BUY DA share
     3.    34 ( 0.5%) Q=+0.544  SELL SM share
     4.    23 ( 0.4%) Q=+0.576  SELL DA share
     5.    12 ( 0.2%) Q=+0.284  SELL OS share
     6.     4 ( 0.1%) Q=+0.495  BUY SI share
  A0GB Value: P0=-0.492, P1=-0.332, P2=+0.809 (depth: 76, vbackups: 5480)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Corp**: P2 may buy with OS($57), SM($16)

### Step 129: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($57), SM($16)

  NN Values: P0=-0.338, P1=-0.455, P2=+0.652
  NN Priors (top 3 of 3 legal):
     1.  95.2% (-11.2pp) ██████████████████████████████████████ ACQ select SM
     2.   4.7% (+10.9pp) █ ACQ select OS
     3.   0.0% ( +0.3pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 3, 6400 total):
     1.  5084 (79.4%) Q=+0.780 ███████████████████████████████ ACQ select SM
     2.  1315 (20.5%) Q=+0.787 ████████ ACQ select OS
     3.     1 ( 0.0%) Q=+0.559  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.512, P1=-0.291, P2=+0.789 (depth: 75, vbackups: 3736)

  **Action: ACQ select SM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Company**: P2 buying with SM ($16)

### Step 130: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SM ($16)

  NN Values: P0=-0.342, P1=-0.447, P2=+0.641
  NN Priors (top 2 of 2 legal):
     1.  98.8% ( -3.4pp) ███████████████████████████████████████ ACQ target BD (with SM)
     2.   1.2% ( +3.4pp)  ACQ target SX (with SM)

  MCTS Visits (top 2, 6400 total):
     1.  6245 (97.6%) Q=+0.779 ███████████████████████████████████████ ACQ target BD (with SM)
     2.   155 ( 2.4%) Q=+0.759  ACQ target SX (with SM)
  A0GB Value: P0=-0.512, P1=-0.262, P2=+0.781 (depth: 76, vbackups: 5083)

  **Action: ACQ target BD (with SM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$6 stars=9 pres=P2  companies=[BD, SX]
  SM: $16 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$5 stars=4 pres=P2  companies=[KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Price**: P2 SM -> BD (price range $7-$17)

### Step 131: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SM -> BD (price range $7-$17)

  NN Values: P0=-0.371, P1=-0.412, P2=+0.656
  NN Priors (top 10 of 10 legal):
     1.  91.3% (-13.6pp) ████████████████████████████████████ ACQUIRE BD with SM @ $16
     2.   6.0% ( +0.2pp) ██ ACQUIRE BD with SM @ $15
     3.   1.1% ( +2.2pp)  ACQUIRE BD with SM @ $14
     4.   0.5% ( +2.5pp)  ACQUIRE BD with SM @ $13
     5.   0.3% ( +1.1pp)  ACQUIRE BD with SM @ $7
     6.   0.3% ( +1.3pp)  ACQUIRE BD with SM @ $12
     7.   0.2% ( +0.4pp)  ACQUIRE BD with SM @ $11
     8.   0.1% ( +0.8pp)  ACQUIRE BD with SM @ $8
     9.   0.1% ( +4.3pp)  ACQUIRE BD with SM @ $10
    10.   0.1% ( +0.8pp)  ACQUIRE BD with SM @ $9

  MCTS Visits (top 10, 6400 total):
     1.  4900 (76.6%) Q=+0.778 ██████████████████████████████ ACQUIRE BD with SM @ $16
     2.   845 (13.2%) Q=+0.791 █████ ACQUIRE BD with SM @ $15
     3.   194 ( 3.0%) Q=+0.782 █ ACQUIRE BD with SM @ $14
     4.   180 ( 2.8%) Q=+0.772 █ ACQUIRE BD with SM @ $10
     5.   145 ( 2.3%) Q=+0.776  ACQUIRE BD with SM @ $13
     6.    64 ( 1.0%) Q=+0.766  ACQUIRE BD with SM @ $12
     7.    35 ( 0.5%) Q=+0.744  ACQUIRE BD with SM @ $7
     8.    15 ( 0.2%) Q=+0.715  ACQUIRE BD with SM @ $8
     9.    14 ( 0.2%) Q=+0.717  ACQUIRE BD with SM @ $9
    10.     8 ( 0.1%) Q=+0.716  ACQUIRE BD with SM @ $11
  A0GB Value: P0=-0.512, P1=-0.262, P2=+0.781 (depth: 75, vbackups: 5705)

  **Action: ACQUIRE BD with SM @ $16**

Phase: ACQ_SELECT_CORP  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$3 stars=7 pres=P2  companies=[SX]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$8 stars=5 pres=P2  companies=[BD*, KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Corp**: P2 may buy with OS($57), SM($0)

### Step 132: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($57), SM($0)

  NN Values: P0=-0.377, P1=-0.432, P2=+0.656
  NN Priors (top 2 of 2 legal):
     1. 100.0% (-11.8pp) ███████████████████████████████████████ ACQ select OS
     2.   0.0% (+11.8pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6333 (99.0%) Q=+0.778 ███████████████████████████████████████ ACQ select OS
     2.    67 ( 1.0%) Q=+0.554  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 75, vbackups: 5438)

  **Action: ACQ select OS**

Phase: ACQ_SELECT_COMPANY  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$3 stars=7 pres=P2  companies=[SX]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$8 stars=5 pres=P2  companies=[BD*, KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Company**: P2 buying with OS ($57)

### Step 133: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with OS ($57)

  NN Values: P0=-0.355, P1=-0.471, P2=+0.676
  NN Priors (top 2 of 2 legal):
     1.  93.8% (-10.0pp) █████████████████████████████████████ ACQ target KK (with OS)
     2.   6.2% (+10.0pp) ██ ACQ target DR (with OS)

  MCTS Visits (top 2, 6400 total):
     1.  5133 (80.2%) Q=+0.776 ████████████████████████████████ ACQ target KK (with OS)
     2.  1267 (19.8%) Q=+0.785 ███████ ACQ target DR (with OS)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 74, vbackups: 5969)

  **Action: ACQ target KK (with OS)**

Phase: ACQ_SELECT_PRICE  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $57 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$3 stars=7 pres=P2  companies=[SX]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$8 stars=5 pres=P2  companies=[BD*, KK]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Price**: P2 OS -> KK (price range $11-$28)

### Step 134: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> KK (price range $11-$28)

  NN Values: P0=-0.404, P1=-0.428, P2=+0.652
  NN Priors (top 10 of 18 legal):
     1.  98.7% (-14.3pp) ███████████████████████████████████████ ACQUIRE KK with OS @ $11
     2.   0.9% ( +0.2pp)  ACQUIRE KK with OS @ $12
     3.   0.1% ( +0.2pp)  ACQUIRE KK with OS @ $13
     4.   0.1% ( +0.6pp)  ACQUIRE KK with OS @ $14
     5.   0.1% ( +2.2pp)  ACQUIRE KK with OS @ $15
     6.   0.0% ( +3.3pp)  ACQUIRE KK with OS @ $16
     7.   0.0% ( -0.0pp)  ACQUIRE KK with OS @ $17
     8.   0.0% ( +0.3pp)  ACQUIRE KK with OS @ $20
     9.   0.0% ( +0.1pp)  ACQUIRE KK with OS @ $21
    10.   0.0% ( +0.2pp)  ACQUIRE KK with OS @ $19

  MCTS Visits (top 10, 6400 total):
     1.  5722 (89.4%) Q=+0.776 ███████████████████████████████████ ACQUIRE KK with OS @ $11
     2.   205 ( 3.2%) Q=+0.776 █ ACQUIRE KK with OS @ $16
     3.   155 ( 2.4%) Q=+0.778  ACQUIRE KK with OS @ $15
     4.   151 ( 2.4%) Q=+0.788  ACQUIRE KK with OS @ $12
     5.    52 ( 0.8%) Q=+0.762  ACQUIRE KK with OS @ $18
     6.    30 ( 0.5%) Q=+0.721  ACQUIRE KK with OS @ $23
     7.    26 ( 0.4%) Q=+0.760  ACQUIRE KK with OS @ $14
     8.    25 ( 0.4%) Q=+0.679  ACQUIRE KK with OS @ $27
     9.    13 ( 0.2%) Q=+0.674  ACQUIRE KK with OS @ $24
    10.     6 ( 0.1%) Q=+0.734  ACQUIRE KK with OS @ $13
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 73, vbackups: 5495)

  **Action: ACQUIRE KK with OS @ $11**

Phase: ACQ_SELECT_CORP  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $0 income=$10  companies=[DR]

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $46 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$10 stars=9 pres=P2  companies=[SX, KK*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Corp**: P2 may buy with OS($46), SM($0)

### Step 135: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($46), SM($0)

  NN Values: P0=-0.334, P1=-0.523, P2=+0.746
  NN Priors (top 2 of 2 legal):
     1. 100.0% (-11.8pp) ███████████████████████████████████████ ACQ select OS
     2.   0.0% (+11.8pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6296 (98.4%) Q=+0.774 ███████████████████████████████████████ ACQ select OS
     2.   104 ( 1.6%) Q=+0.639  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 72, vbackups: 5721)

  **Action: ACQ select OS**

  ↳ auto: ACQ target DR (with OS)
  ↳ auto: PASS (ACQ_SELECT_CORP)

### Step 136: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($22), DA($21)

  NN Values: P0=-0.471, P1=-0.512, P2=+0.746
  NN Priors (top 2 of 2 legal):
     1.  99.6% ( -9.4pp) ███████████████████████████████████████ ACQ select S
     2.   0.4% ( +9.4pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6297 (98.4%) Q=-0.393 ███████████████████████████████████████ ACQ select S
     2.   103 ( 1.6%) Q=-0.503  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 71, vbackups: 6295)

  **Action: ACQ select S**

Phase: ACQ_SELECT_COMPANY  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Company**: P1 buying with S ($22)

### Step 137: P1 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P1 buying with S ($22)

  NN Values: P0=-0.498, P1=-0.512, P2=+0.754
  NN Priors (top 3 of 3 legal):
     1.  95.5% ( -5.7pp) ██████████████████████████████████████ ACQ target PR (with S)
     2.   2.6% ( +3.0pp) █ ACQ target AKE (with S)
     3.   1.9% ( +2.7pp)  ACQ target BSE (with S)

  MCTS Visits (top 3, 6400 total):
     1.  6089 (95.1%) Q=-0.392 ██████████████████████████████████████ ACQ target PR (with S)
     2.   166 ( 2.6%) Q=-0.418 █ ACQ target AKE (with S)
     3.   145 ( 2.3%) Q=-0.415  ACQ target BSE (with S)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 70, vbackups: 6200)

  **Action: ACQ target PR (with S)**

Phase: ACQ_SELECT_PRICE  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $22 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$5 stars=5 pres=P1  companies=[NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, AKE, PR]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Price**: P1 S -> PR (price range $10-$25)

### Step 138: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 S -> PR (price range $10-$25)

  NN Values: P0=-0.480, P1=-0.555, P2=+0.738
  NN Priors (top 10 of 13 legal):
     1.  97.5% (-11.1pp) ██████████████████████████████████████ ACQUIRE PR with S @ $10
     2.   1.5% ( -0.2pp)  ACQUIRE PR with S @ $11
     3.   0.3% ( -0.0pp)  ACQUIRE PR with S @ $12
     4.   0.3% ( -0.0pp)  ACQUIRE PR with S @ $13
     5.   0.2% ( +2.7pp)  ACQUIRE PR with S @ $14
     6.   0.1% ( +0.5pp)  ACQUIRE PR with S @ $15
     7.   0.1% ( +0.0pp)  ACQUIRE PR with S @ $19
     8.   0.0% ( +0.1pp)  ACQUIRE PR with S @ $18
     9.   0.0% ( +1.2pp)  ACQUIRE PR with S @ $16
    10.   0.0% ( +3.9pp)  ACQUIRE PR with S @ $20

  MCTS Visits (top 10, 6400 total):
     1.  6241 (97.5%) Q=-0.391 ███████████████████████████████████████ ACQUIRE PR with S @ $10
     2.    36 ( 0.6%) Q=-0.523  ACQUIRE PR with S @ $14
     3.    36 ( 0.6%) Q=-0.540  ACQUIRE PR with S @ $20
     4.    25 ( 0.4%) Q=-0.494  ACQUIRE PR with S @ $17
     5.    23 ( 0.4%) Q=-0.472  ACQUIRE PR with S @ $16
     6.    19 ( 0.3%) Q=-0.463  ACQUIRE PR with S @ $11
     7.    12 ( 0.2%) Q=-0.593  ACQUIRE PR with S @ $21
     8.     4 ( 0.1%) Q=-0.560  ACQUIRE PR with S @ $15
     9.     2 ( 0.0%) Q=-0.503  ACQUIRE PR with S @ $12
    10.     1 ( 0.0%) Q=-0.559  ACQUIRE PR with S @ $13
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 69, vbackups: 6180)

  **Action: ACQUIRE PR with S @ $10**

Phase: ACQ_SELECT_CORP  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $12 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$10 stars=6 pres=P1  companies=[PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$1 stars=4 pres=P1  companies=[BSE, AKE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Corp**: P1 may buy with S($12), DA($21)

### Step 139: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($12), DA($21)

  NN Values: P0=-0.477, P1=-0.469, P2=+0.730
  NN Priors (top 3 of 3 legal):
     1.  99.2% ( -8.9pp) ███████████████████████████████████████ ACQ select S
     2.   0.7% ( +3.6pp)  PASS (ACQ_SELECT_CORP)
     3.   0.2% ( +5.3pp)  ACQ select DA

  MCTS Visits (top 3, 6400 total):
     1.  6283 (98.2%) Q=-0.391 ███████████████████████████████████████ ACQ select S
     2.    84 ( 1.3%) Q=-0.443  PASS (ACQ_SELECT_CORP)
     3.    33 ( 0.5%) Q=-0.638  ACQ select DA
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 68, vbackups: 6240)

  **Action: ACQ select S**

Phase: ACQ_SELECT_COMPANY  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $12 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$10 stars=6 pres=P1  companies=[PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$1 stars=4 pres=P1  companies=[BSE, AKE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Company**: P1 buying with S ($12)

### Step 140: P1 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P1 buying with S ($12)

  NN Values: P0=-0.512, P1=-0.420, P2=+0.730
  NN Priors (top 2 of 2 legal):
     1.  94.7% ( -8.1pp) █████████████████████████████████████ ACQ target AKE (with S)
     2.   5.3% ( +8.1pp) ██ ACQ target BSE (with S)

  MCTS Visits (top 2, 6400 total):
     1.  5323 (83.2%) Q=-0.390 █████████████████████████████████ ACQ target AKE (with S)
     2.  1077 (16.8%) Q=-0.384 ██████ ACQ target BSE (with S)
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 67, vbackups: 5475)

  **Action: ACQ target AKE (with S)**

Phase: ACQ_SELECT_PRICE  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $12 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$10 stars=6 pres=P1  companies=[PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$1 stars=4 pres=P1  companies=[BSE, AKE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Price**: P1 S -> AKE (price range $3-$8)

### Step 141: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 S -> AKE (price range $3-$8)

  NN Values: P0=-0.498, P1=-0.418, P2=+0.727
  NN Priors (top 6 of 6 legal):
     1.  97.0% ( -9.6pp) ██████████████████████████████████████ ACQUIRE AKE with S @ $3
     2.   1.8% ( +3.4pp)  ACQUIRE AKE with S @ $4
     3.   0.6% ( +4.2pp)  ACQUIRE AKE with S @ $5
     4.   0.4% ( +0.3pp)  ACQUIRE AKE with S @ $6
     5.   0.2% ( +1.5pp)  ACQUIRE AKE with S @ $7
     6.   0.0% ( +0.2pp)  ACQUIRE AKE with S @ $8

  MCTS Visits (top 6, 6400 total):
     1.  6064 (94.8%) Q=-0.389 █████████████████████████████████████ ACQUIRE AKE with S @ $3
     2.   139 ( 2.2%) Q=-0.417  ACQUIRE AKE with S @ $5
     3.   136 ( 2.1%) Q=-0.427  ACQUIRE AKE with S @ $4
     4.    39 ( 0.6%) Q=-0.402  ACQUIRE AKE with S @ $6
     5.    11 ( 0.2%) Q=-0.558  ACQUIRE AKE with S @ $7
     6.    11 ( 0.2%) Q=-0.560  ACQUIRE AKE with S @ $8
  A0GB Value: P0=-0.438, P1=-0.396, P2=+0.824 (depth: 66, vbackups: 6129)

  **Action: ACQUIRE AKE with S @ $3**

Phase: ACQ_SELECT_CORP  |  Turn: 6  |  CoO Level: 4  |  Active Player: 1  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $9 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE*, PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$0 stars=3 pres=P1  companies=[BSE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Corp**: P1 may buy with S($9), DA($21)

### Step 142: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($9), DA($21)

  NN Values: P0=-0.512, P1=-0.348, P2=+0.703
  NN Priors (top 2 of 2 legal):
     1.  94.8% ( -2.7pp) █████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   5.2% ( +2.7pp) ██ ACQ select DA

  MCTS Visits (top 2, 6400 total):
     1.  6267 (97.9%) Q=-0.388 ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   133 ( 2.1%) Q=-0.449  ACQ select DA
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 65, vbackups: 6063)

  **Action: PASS (ACQ_SELECT_CORP)**

### Step 143: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with PR($39), VM($12)

  NN Values: P0=-0.443, P1=-0.457, P2=+0.758
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -9.7pp) ███████████████████████████████████████ ACQ select PR
     2.   0.0% ( +9.7pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6335 (99.0%) Q=-0.365 ███████████████████████████████████████ ACQ select PR
     2.    65 ( 1.0%) Q=-0.558  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 64, vbackups: 6266)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 6  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $9 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE*, PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$0 stars=3 pres=P1  companies=[BSE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Company**: P0 buying with PR ($39)

### Step 144: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with PR ($39)

  NN Values: P0=-0.475, P1=-0.447, P2=+0.754
  NN Priors (top 2 of 2 legal):
     1.  99.7% ( -6.9pp) ███████████████████████████████████████ ACQ target WT (with PR)
     2.   0.3% ( +6.9pp)  ACQ target MHE (with PR)

  MCTS Visits (top 2, 6400 total):
     1.  6347 (99.2%) Q=-0.365 ███████████████████████████████████████ ACQ target WT (with PR)
     2.    53 ( 0.8%) Q=-0.541  ACQ target MHE (with PR)
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 63, vbackups: 6334)

  **Action: ACQ target WT (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 6  |  CoO Level: 4  |  Active Player: 0  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $29 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $9 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE*, PR*, NS]
  OS: $17 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=9 pres=P2  companies=[SX, KK*, DR*]
  SM: $0 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD*]
  PR: $39 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$6 stars=6 pres=P0  companies=[PKP]
  DA: $21 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$0 stars=3 pres=P1  companies=[BSE]
  VM: $12 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[MHE, WT]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Acquisition — Select Price**: P0 PR -> WT (price range $6-$14)

### Step 145: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 PR -> WT (price range $6-$14)

  NN Values: P0=-0.482, P1=-0.406, P2=+0.746
  NN Priors (top 9 of 9 legal):
     1.  95.5% (-12.2pp) ██████████████████████████████████████ ACQUIRE WT with PR @ $6
     2.   2.2% ( +0.5pp)  ACQUIRE WT with PR @ $7
     3.   0.6% ( +1.0pp)  ACQUIRE WT with PR @ $8
     4.   0.5% ( +2.9pp)  ACQUIRE WT with PR @ $9
     5.   0.5% ( +0.1pp)  ACQUIRE WT with PR @ $10
     6.   0.3% ( +3.2pp)  ACQUIRE WT with PR @ $14
     7.   0.2% ( +3.0pp)  ACQUIRE WT with PR @ $11
     8.   0.2% ( +1.2pp)  ACQUIRE WT with PR @ $13
     9.   0.1% ( +0.2pp)  ACQUIRE WT with PR @ $12

  MCTS Visits (top 9, 6400 total):
     1.  6108 (95.4%) Q=-0.364 ██████████████████████████████████████ ACQUIRE WT with PR @ $6
     2.   122 ( 1.9%) Q=-0.331  ACQUIRE WT with PR @ $7
     3.    42 ( 0.7%) Q=-0.462  ACQUIRE WT with PR @ $9
     4.    38 ( 0.6%) Q=-0.474  ACQUIRE WT with PR @ $11
     5.    29 ( 0.5%) Q=-0.524  ACQUIRE WT with PR @ $14
     6.    22 ( 0.3%) Q=-0.449  ACQUIRE WT with PR @ $8
     7.    17 ( 0.3%) Q=-0.466  ACQUIRE WT with PR @ $13
     8.    13 ( 0.2%) Q=-0.442  ACQUIRE WT with PR @ $10
     9.     9 ( 0.1%) Q=-0.469  ACQUIRE WT with PR @ $12
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 62, vbackups: 6171)

  **Action: ACQUIRE WT with PR @ $6**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $8 (NW $58) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $10 (NW $60) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $13 (NW $64) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $21 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$12 stars=8 pres=P1  companies=[AKE, PR, NS]
  OS: $54 price=$27(idx 16) shares=bank:2/unissued:3/issued:3 income=$21 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $14 price=$14(idx 10) shares=bank:4/unissued:1/issued:5 income=$3 stars=3 pres=P2  companies=[BD]
  PR: $43 price=$16(idx 11) shares=bank:2/unissued:1/issued:4 income=$10 stars=9 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:3/unissued:0/issued:5 income=$0 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$18(idx 12) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Dividends**: OS

### Step 146: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.453, P1=-0.404, P2=+0.820
  NN Priors (top 10 of 10 legal):
     1.  98.1% (-14.4pp) ███████████████████████████████████████ DIVIDEND $9
     2.   1.8% ( +0.0pp)  DIVIDEND $8
     3.   0.0% ( +4.0pp)  DIVIDEND $4
     4.   0.0% ( +1.6pp)  DIVIDEND $5
     5.   0.0% ( +1.4pp)  DIVIDEND $7
     6.   0.0% ( +0.1pp)  DIVIDEND $3
     7.   0.0% ( +4.9pp)  DIVIDEND $6
     8.   0.0% ( +0.2pp)  DIVIDEND $0
     9.   0.0% ( +0.5pp)  DIVIDEND $2
    10.   0.0% ( +1.6pp)  DIVIDEND $1

  MCTS Visits (top 10, 6400 total):
     1.  5340 (83.4%) Q=+0.771 █████████████████████████████████ DIVIDEND $9
     2.   490 ( 7.7%) Q=+0.790 ███ DIVIDEND $8
     3.   216 ( 3.4%) Q=+0.769 █ DIVIDEND $6
     4.   129 ( 2.0%) Q=+0.755  DIVIDEND $4
     5.    86 ( 1.3%) Q=+0.778  DIVIDEND $7
     6.    67 ( 1.0%) Q=+0.760  DIVIDEND $5
     7.    36 ( 0.6%) Q=+0.733  DIVIDEND $1
     8.    18 ( 0.3%) Q=+0.755  DIVIDEND $2
     9.    11 ( 0.2%) Q=+0.776  DIVIDEND $3
    10.     7 ( 0.1%) Q=+0.752  DIVIDEND $0
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 61, vbackups: 5809)

  **Action: DIVIDEND $9**

### Step 147: P1 [DIVIDENDS]

  **Dividends**: S

  NN Values: P0=-0.426, P1=-0.412, P2=+0.770
  NN Priors (top 8 of 8 legal):
     1.  98.9% ( -9.7pp) ███████████████████████████████████████ DIVIDEND $7
     2.   0.7% ( +1.0pp)  DIVIDEND $6
     3.   0.1% ( +0.6pp)  DIVIDEND $4
     4.   0.1% ( +3.1pp)  DIVIDEND $5
     5.   0.1% ( +0.9pp)  DIVIDEND $3
     6.   0.0% ( +0.4pp)  DIVIDEND $2
     7.   0.0% ( +3.0pp)  DIVIDEND $1
     8.   0.0% ( +0.7pp)  DIVIDEND $0

  MCTS Visits (top 8, 6400 total):
     1.  6247 (97.6%) Q=-0.383 ███████████████████████████████████████ DIVIDEND $7
     2.    49 ( 0.8%) Q=-0.455  DIVIDEND $5
     3.    28 ( 0.4%) Q=-0.447  DIVIDEND $6
     4.    25 ( 0.4%) Q=-0.560  DIVIDEND $1
     5.    18 ( 0.3%) Q=-0.435  DIVIDEND $4
     6.    17 ( 0.3%) Q=-0.480  DIVIDEND $3
     7.     8 ( 0.1%) Q=-0.512  DIVIDEND $2
     8.     8 ( 0.1%) Q=-0.570  DIVIDEND $0
  A0GB Value: P0=-0.371, P1=-0.283, P2=+0.590 (depth: 60, vbackups: 5812)

  **Action: DIVIDEND $7**

### Step 148: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.391, P1=-0.424, P2=+0.738
  NN Priors (top 7 of 7 legal):
     1.  64.3% ( -3.0pp) █████████████████████████ DIVIDEND $0
     2.  29.0% ( -4.3pp) ███████████ DIVIDEND $6
     3.   3.3% ( +3.9pp) █ DIVIDEND $5
     4.   1.3% ( +1.0pp)  DIVIDEND $3
     5.   0.9% ( +1.5pp)  DIVIDEND $4
     6.   0.8% ( +1.0pp)  DIVIDEND $1
     7.   0.4% ( -0.1pp)  DIVIDEND $2

  MCTS Visits (top 7, 6400 total):
     1.  2892 (45.2%) Q=-0.373 ██████████████████ DIVIDEND $0
     2.  2885 (45.1%) Q=-0.356 ██████████████████ DIVIDEND $6
     3.   338 ( 5.3%) Q=-0.353 ██ DIVIDEND $1
     4.   119 ( 1.9%) Q=-0.431  DIVIDEND $5
     5.    91 ( 1.4%) Q=-0.383  DIVIDEND $3
     6.    72 ( 1.1%) Q=-0.390  DIVIDEND $4
     7.     3 ( 0.0%) Q=-0.482  DIVIDEND $2
  A0GB Value: P0=-0.590, P1=-0.181, P2=+0.805 (depth: 57, vbackups: 5864)

  **Action: DIVIDEND $0**

### Step 149: P0 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=-0.500, P1=-0.324, P2=+0.809
  NN Priors (top 6 of 6 legal):
     1.  63.6% ( -8.6pp) █████████████████████████ DIVIDEND $5
     2.  35.1% ( +1.4pp) ██████████████ DIVIDEND $3
     3.   0.8% ( +1.6pp)  DIVIDEND $4
     4.   0.4% ( +0.5pp)  DIVIDEND $2
     5.   0.0% ( +4.5pp)  DIVIDEND $0
     6.   0.0% ( +0.7pp)  DIVIDEND $1

  MCTS Visits (top 6, 6400 total):
     1.  3401 (53.1%) Q=-0.383 █████████████████████ DIVIDEND $5
     2.  2684 (41.9%) Q=-0.380 ████████████████ DIVIDEND $3
     3.   127 ( 2.0%) Q=-0.390  DIVIDEND $4
     4.   124 ( 1.9%) Q=-0.380  DIVIDEND $2
     5.    60 ( 0.9%) Q=-0.471  DIVIDEND $0
     6.     4 ( 0.1%) Q=-0.553  DIVIDEND $1
  A0GB Value: P0=-0.527, P1=-0.299, P2=+0.812 (depth: 61, vbackups: 2964)

  **Action: DIVIDEND $5**

### Step 150: P2 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.393, P1=-0.420, P2=+0.809
  NN Priors (top 3 of 3 legal):
     1.  99.4% (-10.2pp) ███████████████████████████████████████ DIVIDEND $2
     2.   0.5% ( +6.0pp)  DIVIDEND $1
     3.   0.2% ( +4.2pp)  DIVIDEND $0

  MCTS Visits (top 3, 6400 total):
     1.  5680 (88.8%) Q=+0.787 ███████████████████████████████████ DIVIDEND $2
     2.   533 ( 8.3%) Q=+0.791 ███ DIVIDEND $1
     3.   187 ( 2.9%) Q=+0.776 █ DIVIDEND $0
  A0GB Value: P0=-0.496, P1=+0.295, P2=+0.215 (depth: 70, vbackups: 3400)

  **Action: DIVIDEND $2**

### Step 151: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.404, P1=-0.428, P2=+0.828
  NN Priors (top 4 of 4 legal):
     1.  90.4% ( -6.5pp) ████████████████████████████████████ DIVIDEND $0
     2.   8.4% ( -0.4pp) ███ DIVIDEND $3
     3.   0.8% ( +3.9pp)  DIVIDEND $2
     4.   0.5% ( +2.9pp)  DIVIDEND $1

  MCTS Visits (top 4, 6400 total):
     1.  6016 (94.0%) Q=-0.381 █████████████████████████████████████ DIVIDEND $0
     2.   194 ( 3.0%) Q=-0.421 █ DIVIDEND $3
     3.   111 ( 1.7%) Q=-0.423  DIVIDEND $2
     4.    79 ( 1.2%) Q=-0.421  DIVIDEND $1
  A0GB Value: P0=-0.496, P1=+0.295, P2=+0.215 (depth: 69, vbackups: 5679)

  **Action: DIVIDEND $0**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 6  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $18 (NW $67) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $17 (NW $66) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $24 (NW $78) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $27 price=$33(idx 18) shares=bank:2/unissued:3/issued:3 income=$21 stars=10 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$12(idx 8) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD]
  PR: $23 price=$18(idx 12) shares=bank:2/unissued:1/issued:4 income=$10 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$9(idx 5) shares=bank:3/unissued:0/issued:5 income=$0 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Issue**: OS

### Step 152: P2 [ISSUE_SHARES]

  **Issue**: OS

  NN Values: P0=-0.227, P1=-0.500, P2=+0.805
  NN Priors (top 2 of 2 legal):
     1.  99.8% ( -5.4pp) ███████████████████████████████████████ ISSUE OS shares
     2.   0.2% ( +5.4pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6150 (96.1%) Q=+0.790 ██████████████████████████████████████ ISSUE OS shares
     2.   250 ( 3.9%) Q=+0.785 █ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.496, P1=+0.295, P2=+0.215 (depth: 68, vbackups: 6015)

  **Action: ISSUE OS shares**

### Step 153: P1 [ISSUE_SHARES]

  **Issue**: S

  NN Values: P0=-0.247, P1=-0.461, P2=+0.801
  NN Priors (top 2 of 2 legal):
     1.  95.3% (-10.8pp) ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   4.7% (+10.8pp) █ ISSUE S shares

  MCTS Visits (top 2, 6400 total):
     1.  6164 (96.3%) Q=-0.381 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   236 ( 3.7%) Q=-0.452 █ ISSUE S shares
  A0GB Value: P0=-0.243, P1=-0.504, P2=+0.852 (depth: 71, vbackups: 6149)

  **Action: PASS (ISSUE_SHARES)**

### Step 154: P0 [ISSUE_SHARES]

  **Issue**: PR

  NN Values: P0=-0.254, P1=-0.500, P2=+0.805
  NN Priors (top 2 of 2 legal):
     1.  98.7% ( -3.9pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   1.3% ( +3.9pp)  ISSUE PR shares

  MCTS Visits (top 2, 6400 total):
     1.  6271 (98.0%) Q=-0.386 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   129 ( 2.0%) Q=-0.432  ISSUE PR shares
  A0GB Value: P0=-0.243, P1=-0.504, P2=+0.852 (depth: 70, vbackups: 6163)

  **Action: PASS (ISSUE_SHARES)**

### Step 155: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.352, P1=-0.426, P2=+0.797
  NN Priors (top 2 of 2 legal):
     1.  64.9% ( +0.4pp) █████████████████████████ PASS (ISSUE_SHARES)
     2.  35.1% ( -0.4pp) ██████████████ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  3218 (50.3%) Q=-0.394 ████████████████████ PASS (ISSUE_SHARES)
     2.  3182 (49.7%) Q=-0.381 ███████████████████ ISSUE VM shares
  A0GB Value: P0=-0.516, P1=-0.332, P2=+0.852 (depth: 70, vbackups: 6270)

  **Action: PASS (ISSUE_SHARES)**

### Step 156: P2 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.305, P1=-0.451, P2=+0.801
  NN Priors (top 2 of 2 legal):
     1.  97.1% ( -5.6pp) ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   2.9% ( +5.6pp) █ ISSUE SM shares

  MCTS Visits (top 2, 6400 total):
     1.  5966 (93.2%) Q=+0.827 █████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   434 ( 6.8%) Q=+0.821 ██ ISSUE SM shares
  A0GB Value: P0=-0.471, P1=-0.328, P2=+0.828 (depth: 78, vbackups: 3217)

  **Action: PASS (ISSUE_SHARES)**

--- Turn 7 ---

Phase: INVEST  |  Turn: 7  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $18 (NW $67) order=2 income=$0  shares=[PR=2 (pres), VM=1 (pres)]
  P1: $17 (NW $66) order=1 income=$0  shares=[S=1 (pres), DA=1 (pres), VM=1]
  P2: $24 (NW $75) order=0 income=$0  shares=[OS=1 (pres), SM=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$21 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$12(idx 8) shares=bank:4/unissued:1/issued:5 income=$3 stars=2 pres=P2  companies=[BD]
  PR: $23 price=$18(idx 12) shares=bank:2/unissued:1/issued:4 income=$10 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$9(idx 5) shares=bank:3/unissued:0/issued:5 income=$0 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining


### Step 157: P2 [INVEST]

  NN Values: P0=-0.254, P1=-0.395, P2=+0.777
  NN Priors (top 8 of 8 legal):
     1.  99.8% (-11.6pp) ███████████████████████████████████████ SELL SM share
     2.   0.1% ( +5.0pp)  SELL DA share
     3.   0.0% ( +0.5pp)  BUY DA share
     4.   0.0% ( +2.0pp)  PASS (INVEST)
     5.   0.0% ( +1.8pp)  BUY SI share
     6.   0.0% ( +0.5pp)  BUY SM share
     7.   0.0% ( +1.6pp)  BUY PR share
     8.   0.0% ( +0.2pp)  SELL OS share

  MCTS Visits (top 8, 6400 total):
     1.  6196 (96.8%) Q=+0.827 ██████████████████████████████████████ SELL SM share
     2.    87 ( 1.4%) Q=+0.767  SELL DA share
     3.    63 ( 1.0%) Q=+0.803  PASS (INVEST)
     4.    26 ( 0.4%) Q=+0.751  BUY SI share
     5.    12 ( 0.2%) Q=+0.665  BUY PR share
     6.     9 ( 0.1%) Q=+0.766  BUY DA share
     7.     6 ( 0.1%) Q=+0.740  BUY SM share
     8.     1 ( 0.0%) Q=+0.013  SELL OS share
  A0GB Value: P0=-0.471, P1=-0.328, P2=+0.828 (depth: 77, vbackups: 5965)

  **Action: SELL SM share**

### Step 158: P1 [INVEST]

  NN Values: P0=-0.357, P1=-0.486, P2=+0.793
  NN Priors (top 7 of 7 legal):
     1.  57.7% ( -3.3pp) ███████████████████████ BUY DA share
     2.  40.9% ( -5.5pp) ████████████████ PASS (INVEST)
     3.   0.5% ( +0.4pp)  SELL S share
     4.   0.3% ( +1.6pp)  BUY SM share
     5.   0.3% ( +0.7pp)  SELL VM share
     6.   0.2% ( +5.9pp)  SELL DA share
     7.   0.1% ( +0.3pp)  BUY SI share

  MCTS Visits (top 7, 6400 total):
     1.  4094 (64.0%) Q=-0.406 █████████████████████████ BUY DA share
     2.  2134 (33.3%) Q=-0.411 █████████████ PASS (INVEST)
     3.   117 ( 1.8%) Q=-0.461  SELL DA share
     4.    17 ( 0.3%) Q=-0.541  BUY SM share
     5.    16 ( 0.2%) Q=-0.593  SELL VM share
     6.    15 ( 0.2%) Q=-0.491  SELL S share
     7.     7 ( 0.1%) Q=-0.496  BUY SI share
  A0GB Value: P0=-0.361, P1=-0.475, P2=+0.828 (depth: 77, vbackups: 6032)

  **Action: BUY DA share**

### Step 159: P0 [INVEST]

  NN Values: P0=-0.344, P1=-0.379, P2=+0.781
  NN Priors (top 6 of 6 legal):
     1.  99.0% (-14.8pp) ███████████████████████████████████████ SELL PR share
     2.   0.7% ( +1.7pp)  PASS (INVEST)
     3.   0.1% ( +0.8pp)  BUY SI share
     4.   0.1% ( +3.8pp)  BUY DA share
     5.   0.1% ( +7.9pp)  BUY SM share
     6.   0.0% ( +0.7pp)  SELL VM share

  MCTS Visits (top 6, 6400 total):
     1.  6124 (95.7%) Q=-0.383 ██████████████████████████████████████ SELL PR share
     2.   121 ( 1.9%) Q=-0.455  BUY SM share
     3.    88 ( 1.4%) Q=-0.410  PASS (INVEST)
     4.    41 ( 0.6%) Q=-0.499  BUY DA share
     5.    17 ( 0.3%) Q=-0.510  BUY SI share
     6.     9 ( 0.1%) Q=-0.651  SELL VM share
  A0GB Value: P0=-0.271, P1=-0.598, P2=+0.832 (depth: 81, vbackups: 4093)

  **Action: SELL PR share**

### Step 160: P2 [INVEST]

  NN Values: P0=-0.256, P1=-0.516, P2=+0.816
  NN Priors (top 10 of 11 legal):
     1.  98.2% (-13.8pp) ███████████████████████████████████████ AUCTION slot 0 (SZD, face $30)
     2.   1.3% ( +0.4pp)  SELL DA share
     3.   0.4% ( -0.0pp)  AUCTION slot 1 (BR, face $34)
     4.   0.2% ( +3.3pp)  PASS (INVEST)
     5.   0.0% ( +0.4pp)  BUY OS share
     6.   0.0% ( +2.4pp)  BUY SM share
     7.   0.0% ( +0.3pp)  BUY SI share
     8.   0.0% ( +0.2pp)  BUY PR share
     9.   0.0% ( +0.1pp)  BUY DA share
    10.   0.0% ( +6.0pp)  BUY S share

  MCTS Visits (top 10, 6400 total):
     1.  6095 (95.2%) Q=+0.831 ██████████████████████████████████████ AUCTION slot 0 (SZD, face $30)
     2.    96 ( 1.5%) Q=+0.813  PASS (INVEST)
     3.    61 ( 1.0%) Q=+0.803  BUY SM share
     4.    58 ( 0.9%) Q=+0.710  BUY S share
     5.    56 ( 0.9%) Q=+0.810  SELL DA share
     6.    12 ( 0.2%) Q=+0.812  AUCTION slot 1 (BR, face $34)
     7.     9 ( 0.1%) Q=+0.786  BUY OS share
     8.     6 ( 0.1%) Q=+0.781  BUY SI share
     9.     3 ( 0.0%) Q=+0.803  BUY DA share
    10.     3 ( 0.0%) Q=+0.788  BUY PR share
  A0GB Value: P0=-0.271, P1=-0.598, P2=+0.832 (depth: 80, vbackups: 6123)

  **Action: AUCTION slot 0 (SZD, face $30)**

Phase: BID_IN_AUCTION  |  Turn: 7  |  CoO Level: 4  |  Active Player: 2  |  End Card: no

**Players**
  P0: $34 (NW $63) order=2 income=$0  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=1 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $35 (NW $75) order=0 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SZD (fv=$30, 4★, inc=$7), BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$12 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$21 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$3 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$10 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$0 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$0 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 5 remaining

**Auction**: SZD current bid=$0 high bidder=P-1 starter=P2

### Step 161: P2 [BID_IN_AUCTION]

  **Auction**: SZD current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.252, P1=-0.512, P2=+0.797
  NN Priors (top 6 of 6 legal):
     1.  98.6% (-12.1pp) ███████████████████████████████████████ BID $30
     2.   1.3% ( +1.2pp)  BID $31
     3.   0.1% ( +3.3pp)  BID $32
     4.   0.0% ( +5.7pp)  BID $33
     5.   0.0% ( +0.7pp)  BID $34
     6.   0.0% ( +1.1pp)  BID $35

  MCTS Visits (top 6, 6400 total):
     1.  6044 (94.4%) Q=+0.831 █████████████████████████████████████ BID $30
     2.   129 ( 2.0%) Q=+0.805  BID $33
     3.   101 ( 1.6%) Q=+0.825  BID $31
     4.    86 ( 1.3%) Q=+0.812  BID $32
     5.    24 ( 0.4%) Q=+0.787  BID $35
     6.    16 ( 0.2%) Q=+0.792  BID $34
  A0GB Value: P0=-0.271, P1=-0.598, P2=+0.832 (depth: 79, vbackups: 6094)

  **Action: BID $30**

  ↳ auto: PASS (BID_IN_AUCTION)

### Step 162: P0 [BID_IN_AUCTION]

  **Auction**: SZD current bid=$30 high bidder=P2 starter=P2

  NN Values: P0=-0.307, P1=-0.453, P2=+0.805
  NN Priors (top 5 of 5 legal):
     1.  74.9% (-10.6pp) █████████████████████████████ PASS (BID_IN_AUCTION)
     2.  24.6% ( -3.9pp) █████████ BID $31
     3.   0.4% ( +8.6pp)  BID $32
     4.   0.1% ( +0.6pp)  BID $33
     5.   0.0% ( +5.3pp)  BID $34

  MCTS Visits (top 5, 6400 total):
     1.  5416 (84.6%) Q=-0.378 █████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   639 (10.0%) Q=-0.407 ███ BID $31
     3.   268 ( 4.2%) Q=-0.412 █ BID $32
     4.    59 ( 0.9%) Q=-0.504  BID $34
     5.    18 ( 0.3%) Q=-0.468  BID $33
  A0GB Value: P0=-0.271, P1=-0.598, P2=+0.832 (depth: 78, vbackups: 5964)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $34 (NW $63) order=2 income=$0  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=1 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=0 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [2]: BR (fv=$34, 4★, inc=$7), E (fv=$43, 4★, inc=$7)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$6 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$-2 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 4 remaining


### Step 163: P1 [INVEST]

  NN Values: P0=-0.340, P1=-0.480, P2=+0.816
  NN Priors (top 4 of 4 legal):
     1.  97.4% (-12.0pp) ██████████████████████████████████████ PASS (INVEST)
     2.   2.4% ( +1.9pp)  SELL DA share
     3.   0.1% ( +3.3pp)  SELL S share
     4.   0.0% ( +6.9pp)  SELL VM share

  MCTS Visits (top 4, 6400 total):
     1.  6260 (97.8%) Q=-0.414 ███████████████████████████████████████ PASS (INVEST)
     2.    67 ( 1.0%) Q=-0.490  SELL DA share
     3.    41 ( 0.6%) Q=-0.640  SELL VM share
     4.    32 ( 0.5%) Q=-0.537  SELL S share
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 79, vbackups: 5415)

  **Action: PASS (INVEST)**

### Step 164: P0 [INVEST]

  NN Values: P0=-0.381, P1=-0.455, P2=+0.812
  NN Priors (top 10 of 10 legal):
     1.  99.9% (-14.8pp) ███████████████████████████████████████ AUCTION slot 0 (BR, face $34)
     2.   0.0% ( +0.0pp)  BUY OS share
     3.   0.0% ( +0.2pp)  BUY SI share
     4.   0.0% ( +5.3pp)  BUY SM share
     5.   0.0% ( +0.3pp)  BUY DA share
     6.   0.0% ( +0.6pp)  BUY PR share
     7.   0.0% ( +0.0pp)  SELL PR share
     8.   0.0% ( +4.3pp)  PASS (INVEST)
     9.   0.0% ( +2.8pp)  SELL VM share
    10.   0.0% ( +1.3pp)  BUY S share

  MCTS Visits (top 10, 6400 total):
     1.  6258 (97.8%) Q=-0.380 ███████████████████████████████████████ AUCTION slot 0 (BR, face $34)
     2.    65 ( 1.0%) Q=-0.457  BUY SM share
     3.    27 ( 0.4%) Q=-0.558  PASS (INVEST)
     4.    20 ( 0.3%) Q=-0.465  BUY PR share
     5.    14 ( 0.2%) Q=-0.652  SELL VM share
     6.    12 ( 0.2%) Q=-0.598  BUY S share
     7.     1 ( 0.0%) Q=-0.645  SELL PR share
     8.     1 ( 0.0%) Q=-0.451  BUY OS share
     9.     1 ( 0.0%) Q=-0.527  BUY DA share
    10.     1 ( 0.0%) Q=-0.664  BUY SI share
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 78, vbackups: 6259)

  **Action: AUCTION slot 0 (BR, face $34)**

  ↳ auto: BID $34
  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

### Step 165: P2 [INVEST]

  NN Values: P0=-0.387, P1=-0.398, P2=+0.805
  NN Priors (top 3 of 3 legal):
     1.  99.8% ( -6.0pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.2% ( +4.2pp)  SELL DA share
     3.   0.0% ( +1.7pp)  SELL OS share

  MCTS Visits (top 3, 6400 total):
     1.  6310 (98.6%) Q=+0.833 ███████████████████████████████████████ PASS (INVEST)
     2.    83 ( 1.3%) Q=+0.806  SELL DA share
     3.     7 ( 0.1%) Q=+0.525  SELL OS share
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 77, vbackups: 6257)

  **Action: PASS (INVEST)**

### Step 166: P1 [INVEST]

  NN Values: P0=-0.381, P1=-0.459, P2=+0.809
  NN Priors (top 4 of 4 legal):
     1.  96.6% (-11.4pp) ██████████████████████████████████████ PASS (INVEST)
     2.   3.0% ( +1.2pp) █ SELL DA share
     3.   0.2% ( +6.8pp)  SELL VM share
     4.   0.1% ( +3.4pp)  SELL S share

  MCTS Visits (top 4, 6400 total):
     1.  6195 (96.8%) Q=-0.413 ██████████████████████████████████████ PASS (INVEST)
     2.   103 ( 1.6%) Q=-0.488  SELL VM share
     3.    61 ( 1.0%) Q=-0.501  SELL DA share
     4.    41 ( 0.6%) Q=-0.549  SELL S share
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 76, vbackups: 6248)

  **Action: PASS (INVEST)**

### Step 167: P0 [INVEST]

  NN Values: P0=-0.414, P1=-0.398, P2=+0.809
  NN Priors (top 3 of 3 legal):
     1.  99.6% (-11.6pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.2% ( +9.6pp)  SELL PR share
     3.   0.1% ( +2.0pp)  SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  6344 (99.1%) Q=-0.380 ███████████████████████████████████████ PASS (INVEST)
     2.    44 ( 0.7%) Q=-0.664  SELL PR share
     3.    12 ( 0.2%) Q=-0.586  SELL VM share
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 75, vbackups: 6255)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$6 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$-2 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P1 may buy with S($7), DA($34)

### Step 168: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($7), DA($34)

  NN Values: P0=-0.355, P1=-0.471, P2=+0.820
  NN Priors (top 2 of 2 legal):
     1.  98.8% ( -4.4pp) ███████████████████████████████████████ ACQ select DA
     2.   1.2% ( +4.4pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6276 (98.1%) Q=-0.413 ███████████████████████████████████████ ACQ select DA
     2.   124 ( 1.9%) Q=-0.464  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 74, vbackups: 6320)

  **Action: ACQ select DA**

Phase: ACQ_SELECT_COMPANY  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$6 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$-2 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Company**: P1 buying with DA ($34)

### Step 169: P1 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P1 buying with DA ($34)

  NN Values: P0=-0.357, P1=-0.475, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  99.5% ( -8.0pp) ███████████████████████████████████████ ACQ target NS (with DA)
     2.   0.3% ( +1.5pp)  ACQ target PR (with DA)
     3.   0.2% ( +6.5pp)  ACQ target AKE (with DA)

  MCTS Visits (top 3, 6400 total):
     1.  6289 (98.3%) Q=-0.413 ███████████████████████████████████████ ACQ target NS (with DA)
     2.    87 ( 1.4%) Q=-0.499  ACQ target AKE (with DA)
     3.    24 ( 0.4%) Q=-0.537  ACQ target PR (with DA)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 73, vbackups: 6298)

  **Action: ACQ target NS (with DA)**

Phase: ACQ_SELECT_PRICE  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$6 stars=6 pres=P1  companies=[AKE, PR, NS]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $34 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$-2 stars=4 pres=P1  companies=[BSE]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Price**: P1 DA -> NS (price range $11-$29)

### Step 170: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 DA -> NS (price range $11-$29)

  NN Values: P0=-0.357, P1=-0.471, P2=+0.824
  NN Priors (top 10 of 19 legal):
     1.  99.1% (-13.0pp) ███████████████████████████████████████ ACQUIRE NS with DA @ $11
     2.   0.6% ( +1.3pp)  ACQUIRE NS with DA @ $12
     3.   0.1% ( +2.9pp)  ACQUIRE NS with DA @ $13
     4.   0.0% ( +0.2pp)  ACQUIRE NS with DA @ $15
     5.   0.0% ( +0.3pp)  ACQUIRE NS with DA @ $14
     6.   0.0% ( +0.2pp)  ACQUIRE NS with DA @ $16
     7.   0.0% ( -0.0pp)  ACQUIRE NS with DA @ $21
     8.   0.0% ( +0.9pp)  ACQUIRE NS with DA @ $28
     9.   0.0% ( +0.8pp)  ACQUIRE NS with DA @ $20
    10.   0.0% ( +0.6pp)  ACQUIRE NS with DA @ $27

  MCTS Visits (top 10, 6400 total):
     1.  6170 (96.4%) Q=-0.413 ██████████████████████████████████████ ACQUIRE NS with DA @ $11
     2.    59 ( 0.9%) Q=-0.467  ACQUIRE NS with DA @ $13
     3.    43 ( 0.7%) Q=-0.456  ACQUIRE NS with DA @ $12
     4.    33 ( 0.5%) Q=-0.483  ACQUIRE NS with DA @ $19
     5.    30 ( 0.5%) Q=-0.473  ACQUIRE NS with DA @ $17
     6.    12 ( 0.2%) Q=-0.596  ACQUIRE NS with DA @ $29
     7.    11 ( 0.2%) Q=-0.556  ACQUIRE NS with DA @ $23
     8.    11 ( 0.2%) Q=-0.503  ACQUIRE NS with DA @ $20
     9.    11 ( 0.2%) Q=-0.590  ACQUIRE NS with DA @ $28
    10.     4 ( 0.1%) Q=-0.531  ACQUIRE NS with DA @ $16
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 72, vbackups: 6184)

  **Action: ACQUIRE NS with DA @ $11**

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-2 stars=3 pres=P1  companies=[AKE, PR]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $23 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P1 may buy with S($7), DA($23)

### Step 171: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($7), DA($23)

  NN Values: P0=-0.334, P1=-0.512, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  99.1% ( -9.0pp) ███████████████████████████████████████ ACQ select DA
     2.   0.7% ( +4.0pp)  ACQ select S
     3.   0.3% ( +5.0pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 3, 6400 total):
     1.  6204 (96.9%) Q=-0.412 ██████████████████████████████████████ ACQ select DA
     2.   106 ( 1.7%) Q=-0.457  ACQ select S
     3.    90 ( 1.4%) Q=-0.487  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 71, vbackups: 6224)

  **Action: ACQ select DA**

Phase: ACQ_SELECT_COMPANY  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-2 stars=3 pres=P1  companies=[AKE, PR]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $23 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Company**: P1 buying with DA ($23)

### Step 172: P1 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P1 buying with DA ($23)

  NN Values: P0=-0.328, P1=-0.516, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  99.5% (-10.0pp) ███████████████████████████████████████ ACQ target PR (with DA)
     2.   0.5% (+10.0pp)  ACQ target AKE (with DA)

  MCTS Visits (top 2, 6400 total):
     1.  4593 (71.8%) Q=-0.412 ████████████████████████████ ACQ target PR (with DA)
     2.  1807 (28.2%) Q=-0.393 ███████████ ACQ target AKE (with DA)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 70, vbackups: 4606)

  **Action: ACQ target PR (with DA)**

Phase: ACQ_SELECT_PRICE  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-2 stars=3 pres=P1  companies=[AKE, PR]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $23 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$7 stars=6 pres=P1  companies=[BSE, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Price**: P1 DA -> PR (price range $10-$25)

### Step 173: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 DA -> PR (price range $10-$25)

  NN Values: P0=-0.305, P1=-0.520, P2=+0.824
  NN Priors (top 10 of 14 legal):
     1.  98.7% (-14.2pp) ███████████████████████████████████████ ACQUIRE PR with DA @ $10
     2.   0.9% ( +0.2pp)  ACQUIRE PR with DA @ $11
     3.   0.1% ( +0.5pp)  ACQUIRE PR with DA @ $12
     4.   0.1% ( -0.0pp)  ACQUIRE PR with DA @ $13
     5.   0.1% ( +0.3pp)  ACQUIRE PR with DA @ $14
     6.   0.0% ( +0.3pp)  ACQUIRE PR with DA @ $15
     7.   0.0% ( +0.2pp)  ACQUIRE PR with DA @ $19
     8.   0.0% ( -0.0pp)  ACQUIRE PR with DA @ $18
     9.   0.0% ( +4.4pp)  ACQUIRE PR with DA @ $20
    10.   0.0% ( +6.1pp)  ACQUIRE PR with DA @ $16

  MCTS Visits (top 10, 6400 total):
     1.  5979 (93.4%) Q=-0.412 █████████████████████████████████████ ACQUIRE PR with DA @ $10
     2.   177 ( 2.8%) Q=-0.439 █ ACQUIRE PR with DA @ $16
     3.   122 ( 1.9%) Q=-0.445  ACQUIRE PR with DA @ $20
     4.    29 ( 0.5%) Q=-0.461  ACQUIRE PR with DA @ $11
     5.    16 ( 0.2%) Q=-0.505  ACQUIRE PR with DA @ $21
     6.    16 ( 0.2%) Q=-0.509  ACQUIRE PR with DA @ $23
     7.    16 ( 0.2%) Q=-0.445  ACQUIRE PR with DA @ $12
     8.    15 ( 0.2%) Q=-0.507  ACQUIRE PR with DA @ $22
     9.    14 ( 0.2%) Q=-0.462  ACQUIRE PR with DA @ $14
    10.     8 ( 0.1%) Q=-0.463  ACQUIRE PR with DA @ $19
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 69, vbackups: 6008)

  **Action: ACQUIRE PR with DA @ $10**

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-2 stars=1 pres=P1  companies=[AKE]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$9 stars=7 pres=P1  companies=[BSE, PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P1 may buy with S($7), DA($13)

### Step 174: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($7), DA($13)

  NN Values: P0=-0.426, P1=-0.414, P2=+0.812
  NN Priors (top 2 of 2 legal):
     1.  99.8% ( -8.5pp) ███████████████████████████████████████ ACQ select S
     2.   0.2% ( +8.5pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  5907 (92.3%) Q=-0.412 ████████████████████████████████████ ACQ select S
     2.   493 ( 7.7%) Q=-0.411 ███ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 68, vbackups: 5916)

  **Action: ACQ select S**

  ↳ auto: ACQ target BSE (with S)

Phase: ACQ_SELECT_PRICE  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-2 stars=1 pres=P1  companies=[AKE]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$9 stars=7 pres=P1  companies=[BSE, PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Price**: P1 S -> BSE (price range $1-$3)

### Step 175: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 S -> BSE (price range $1-$3)

  NN Values: P0=-0.430, P1=-0.385, P2=+0.816
  NN Priors (top 3 of 3 legal):
     1.  98.5% ( -7.8pp) ███████████████████████████████████████ ACQUIRE BSE with S @ $3
     2.   1.5% ( +7.3pp)  ACQUIRE BSE with S @ $2
     3.   0.0% ( +0.5pp)  ACQUIRE BSE with S @ $1

  MCTS Visits (top 3, 6400 total):
     1.  4845 (75.7%) Q=-0.412 ██████████████████████████████ ACQUIRE BSE with S @ $3
     2.  1544 (24.1%) Q=-0.393 █████████ ACQUIRE BSE with S @ $2
     3.    11 ( 0.2%) Q=-0.469  ACQUIRE BSE with S @ $1
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 67, vbackups: 4909)

  **Action: ACQUIRE BSE with S @ $3**

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $4 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-5 stars=2 pres=P1  companies=[BSE*, AKE]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$11 stars=6 pres=P1  companies=[PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P1 may buy with S($4), DA($13)

### Step 176: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with S($4), DA($13)

  NN Values: P0=-0.355, P1=-0.465, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  99.1% ( -7.7pp) ███████████████████████████████████████ ACQ select DA
     2.   0.9% ( +7.7pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6232 (97.4%) Q=-0.413 ██████████████████████████████████████ ACQ select DA
     2.   168 ( 2.6%) Q=-0.466 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 66, vbackups: 6133)

  **Action: ACQ select DA**

  ↳ auto: ACQ target AKE (with DA)

Phase: ACQ_SELECT_PRICE  |  Turn: 7  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $4 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-5 stars=2 pres=P1  companies=[BSE*, AKE]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$11 stars=6 pres=P1  companies=[PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Price**: P1 DA -> AKE (price range $3-$8)

### Step 177: P1 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P1 DA -> AKE (price range $3-$8)

  NN Values: P0=-0.357, P1=-0.445, P2=+0.820
  NN Priors (top 6 of 6 legal):
     1.  97.2% (-11.9pp) ██████████████████████████████████████ ACQUIRE AKE with DA @ $3
     2.   1.6% ( +0.1pp)  ACQUIRE AKE with DA @ $4
     3.   0.5% ( +2.4pp)  ACQUIRE AKE with DA @ $5
     4.   0.4% ( +0.6pp)  ACQUIRE AKE with DA @ $6
     5.   0.2% ( +6.8pp)  ACQUIRE AKE with DA @ $7
     6.   0.1% ( +2.1pp)  ACQUIRE AKE with DA @ $8

  MCTS Visits (top 6, 6400 total):
     1.  5836 (91.2%) Q=-0.412 ████████████████████████████████████ ACQUIRE AKE with DA @ $3
     2.   249 ( 3.9%) Q=-0.430 █ ACQUIRE AKE with DA @ $7
     3.   193 ( 3.0%) Q=-0.413 █ ACQUIRE AKE with DA @ $5
     4.    52 ( 0.8%) Q=-0.436  ACQUIRE AKE with DA @ $4
     5.    51 ( 0.8%) Q=-0.453  ACQUIRE AKE with DA @ $8
     6.    19 ( 0.3%) Q=-0.468  ACQUIRE AKE with DA @ $6
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 65, vbackups: 5912)

  **Action: ACQUIRE AKE with DA @ $3**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $4 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE*]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $10 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE*, PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P2 may buy with OS($57)

### Step 178: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($57)

  NN Values: P0=-0.395, P1=-0.441, P2=+0.820
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -3.4pp) ███████████████████████████████████████ ACQ select OS
     2.   0.0% ( +3.4pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6325 (98.8%) Q=+0.833 ███████████████████████████████████████ ACQ select OS
     2.    75 ( 1.2%) Q=+0.791  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 64, vbackups: 6142)

  **Action: ACQ select OS**

  ↳ auto: ACQ target SZD (with OS)

Phase: ACQ_SELECT_PRICE  |  Turn: 7  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $5 (NW $75) order=1 income=$7  companies=[SZD]  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $4 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE*]
  OS: $57 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$17 stars=13 pres=P2  companies=[SX, KK, DR]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $10 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE*, PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Price**: P2 OS -> SZD (price range $15-$40)

### Step 179: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> SZD (price range $15-$40)

  NN Values: P0=-0.387, P1=-0.465, P2=+0.816
  NN Priors (top 10 of 26 legal):
     1.  99.9% (-15.0pp) ███████████████████████████████████████ ACQUIRE SZD with OS @ $40
     2.   0.1% ( +1.8pp)  ACQUIRE SZD with OS @ $39
     3.   0.0% ( +0.0pp)  ACQUIRE SZD with OS @ $35
     4.   0.0% ( -0.0pp)  ACQUIRE SZD with OS @ $36
     5.   0.0% ( +0.4pp)  ACQUIRE SZD with OS @ $38
     6.   0.0% ( +0.1pp)  ACQUIRE SZD with OS @ $37
     7.   0.0% ( +0.2pp)  ACQUIRE SZD with OS @ $30
     8.   0.0% ( +0.0pp)  ACQUIRE SZD with OS @ $34
     9.   0.0% ( +0.0pp)  ACQUIRE SZD with OS @ $29
    10.   0.0% ( +0.0pp)  ACQUIRE SZD with OS @ $31

  MCTS Visits (top 10, 6400 total):
     1.  6093 (95.2%) Q=+0.833 ██████████████████████████████████████ ACQUIRE SZD with OS @ $40
     2.   181 ( 2.8%) Q=+0.809 █ ACQUIRE SZD with OS @ $33
     3.    67 ( 1.0%) Q=+0.831  ACQUIRE SZD with OS @ $39
     4.    24 ( 0.4%) Q=+0.795  ACQUIRE SZD with OS @ $32
     5.    17 ( 0.3%) Q=+0.819  ACQUIRE SZD with OS @ $38
     6.     6 ( 0.1%) Q=+0.595  ACQUIRE SZD with OS @ $20
     7.     4 ( 0.1%) Q=+0.775  ACQUIRE SZD with OS @ $30
     8.     2 ( 0.0%) Q=+0.245  ACQUIRE SZD with OS @ $16
     9.     2 ( 0.0%) Q=+0.805  ACQUIRE SZD with OS @ $37
    10.     1 ( 0.0%) Q=+0.801  ACQUIRE SZD with OS @ $35
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 63, vbackups: 6097)

  **Action: ACQUIRE SZD with OS @ $40**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 7  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $45 (NW $85) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $4 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE*]
  OS: $17 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$24 stars=13 pres=P2  companies=[SX, KK, DR, SZD*]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $10 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE*, PR*, NS*]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Acquisition — Select Corp**: P0 may buy with PR($23), VM($20)

### Step 180: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with PR($23), VM($20)

  NN Values: P0=-0.328, P1=-0.422, P2=+0.820
  NN Priors (top 3 of 3 legal):
     1.  75.1% ( -9.3pp) ██████████████████████████████ ACQ select VM
     2.  24.6% ( -0.0pp) █████████ PASS (ACQ_SELECT_CORP)
     3.   0.2% ( +9.4pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  5107 (79.8%) Q=-0.366 ███████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.  1256 (19.6%) Q=-0.431 ███████ ACQ select VM
     3.    37 ( 0.6%) Q=-0.716  ACQ select PR
  A0GB Value: P0=-0.328, P1=-0.512, P2=+0.832 (depth: 62, vbackups: 6161)

  **Action: PASS (ACQ_SELECT_CORP)**

Phase: CLOSING  |  Turn: 7  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $0 (NW $63) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $45 (NW $85) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $34 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $28 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-3 stars=3 pres=P1  companies=[BSE]
  OS: $17 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$24 stars=13 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $4 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $23 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $5 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Closing**: P0 may close WT (PR), PKP (PR), BR

### Step 181: P0 [CLOSING]

  **Closing**: P0 may close WT (PR), PKP (PR), BR

  NN Values: P0=-0.447, P1=-0.344, P2=+0.840
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -7.2pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +7.2pp)  CLOSE WT

  MCTS Visits (top 2, 6400 total):
     1.  6338 (99.0%) Q=-0.368 ███████████████████████████████████████ PASS (CLOSING)
     2.    62 ( 1.0%) Q=-0.514  CLOSE WT
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 63, vbackups: 5106)

  **Action: PASS (CLOSING)**

### Step 182: P1 [CLOSING]

  **Closing**: P1 may close AKE (DA), PR (DA), NS (DA)

  NN Values: P0=-0.432, P1=-0.344, P2=+0.844
  NN Priors (top 3 of 3 legal):
     1.  98.5% ( -9.2pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   1.4% ( +5.6pp)  CLOSE AKE
     3.   0.1% ( +3.6pp)  CLOSE PR

  MCTS Visits (top 3, 6400 total):
     1.  6312 (98.6%) Q=-0.425 ███████████████████████████████████████ PASS (CLOSING)
     2.    65 ( 1.0%) Q=-0.557  CLOSE AKE
     3.    23 ( 0.4%) Q=-0.631  CLOSE PR
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 62, vbackups: 6328)

  **Action: PASS (CLOSING)**

### Step 183: P2 [CLOSING]

  **Closing**: P2 may close SX (OS), KK (OS), DR (OS), SZD (OS)

  NN Values: P0=-0.432, P1=-0.312, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -6.6pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.0% ( +6.6pp)  CLOSE SX

  MCTS Visits (top 2, 6400 total):
     1.  6263 (97.9%) Q=+0.833 ███████████████████████████████████████ PASS (CLOSING)
     2.   137 ( 2.1%) Q=+0.829  CLOSE SX
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 61, vbackups: 6265)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 7  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $70) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $7 (NW $67) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $45 (NW $85) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $25 price=$27(idx 16) shares=bank:1/unissued:5/issued:2 income=$-3 stars=3 pres=P1  companies=[BSE]
  OS: $41 price=$30(idx 17) shares=bank:3/unissued:2/issued:4 income=$24 stars=16 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $3 price=$11(idx 7) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $29 price=$16(idx 11) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $23 price=$10(idx 6) shares=bank:2/unissued:0/issued:5 income=$10 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $22 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Dividends**: OS

### Step 184: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.389, P1=-0.426, P2=+0.848
  NN Priors (top 10 of 11 legal):
     1.  99.4% ( -7.2pp) ███████████████████████████████████████ DIVIDEND $5
     2.   0.3% ( +1.5pp)  DIVIDEND $6
     3.   0.2% ( +0.0pp)  DIVIDEND $4
     4.   0.1% ( +0.2pp)  DIVIDEND $8
     5.   0.0% ( +1.0pp)  DIVIDEND $2
     6.   0.0% ( +0.0pp)  DIVIDEND $0
     7.   0.0% ( +0.2pp)  DIVIDEND $9
     8.   0.0% ( +0.8pp)  DIVIDEND $7
     9.   0.0% ( +0.1pp)  DIVIDEND $3
    10.   0.0% ( +0.8pp)  DIVIDEND $10

  MCTS Visits (top 10, 6400 total):
     1.  5854 (91.5%) Q=+0.832 ████████████████████████████████████ DIVIDEND $5
     2.   157 ( 2.5%) Q=+0.844  DIVIDEND $1
     3.    90 ( 1.4%) Q=+0.835  DIVIDEND $6
     4.    65 ( 1.0%) Q=+0.844  DIVIDEND $2
     5.    57 ( 0.9%) Q=+0.838  DIVIDEND $7
     6.    48 ( 0.8%) Q=+0.831  DIVIDEND $10
     7.    42 ( 0.7%) Q=+0.848  DIVIDEND $4
     8.    30 ( 0.5%) Q=+0.840  DIVIDEND $8
     9.    28 ( 0.4%) Q=+0.848  DIVIDEND $3
    10.    18 ( 0.3%) Q=+0.834  DIVIDEND $9
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 60, vbackups: 5954)

  **Action: DIVIDEND $5**

### Step 185: P1 [DIVIDENDS]

  **Dividends**: S

  NN Values: P0=-0.486, P1=-0.328, P2=+0.855
  NN Priors (top 10 of 10 legal):
     1.  99.3% (-12.5pp) ███████████████████████████████████████ DIVIDEND $9
     2.   0.7% ( +0.9pp)  DIVIDEND $8
     3.   0.0% ( +1.4pp)  DIVIDEND $7
     4.   0.0% ( +0.6pp)  DIVIDEND $6
     5.   0.0% ( +3.7pp)  DIVIDEND $5
     6.   0.0% ( +0.0pp)  DIVIDEND $4
     7.   0.0% ( +0.3pp)  DIVIDEND $0
     8.   0.0% ( +2.6pp)  DIVIDEND $3
     9.   0.0% ( +0.1pp)  DIVIDEND $2
    10.   0.0% ( +2.8pp)  DIVIDEND $1

  MCTS Visits (top 10, 6400 total):
     1.  6253 (97.7%) Q=-0.425 ███████████████████████████████████████ DIVIDEND $9
     2.    39 ( 0.6%) Q=-0.563  DIVIDEND $5
     3.    37 ( 0.6%) Q=-0.467  DIVIDEND $8
     4.    23 ( 0.4%) Q=-0.511  DIVIDEND $7
     5.    19 ( 0.3%) Q=-0.638  DIVIDEND $3
     6.    15 ( 0.2%) Q=-0.684  DIVIDEND $1
     7.    11 ( 0.2%) Q=-0.558  DIVIDEND $6
     8.     1 ( 0.0%) Q=-0.629  DIVIDEND $0
     9.     1 ( 0.0%) Q=-0.475  DIVIDEND $4
    10.     1 ( 0.0%) Q=-0.535  DIVIDEND $2
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 59, vbackups: 6207)

  **Action: DIVIDEND $9**

### Step 186: P0 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=-0.451, P1=-0.312, P2=+0.855
  NN Priors (top 6 of 6 legal):
     1.  88.1% ( -9.7pp) ███████████████████████████████████ DIVIDEND $2
     2.   4.4% ( +1.2pp) █ DIVIDEND $5
     3.   3.1% ( +0.3pp) █ DIVIDEND $3
     4.   2.3% ( +1.1pp)  DIVIDEND $4
     5.   2.1% ( +0.0pp)  DIVIDEND $1
     6.   0.1% ( +7.0pp)  DIVIDEND $0

  MCTS Visits (top 6, 6400 total):
     1.  5769 (90.1%) Q=-0.366 ████████████████████████████████████ DIVIDEND $2
     2.   179 ( 2.8%) Q=-0.392 █ DIVIDEND $5
     3.   144 ( 2.2%) Q=-0.381  DIVIDEND $4
     4.   134 ( 2.1%) Q=-0.422  DIVIDEND $0
     5.   117 ( 1.8%) Q=-0.390  DIVIDEND $3
     6.    57 ( 0.9%) Q=-0.398  DIVIDEND $1
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 58, vbackups: 6200)

  **Action: DIVIDEND $2**

### Step 187: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.404, P1=-0.389, P2=+0.855
  NN Priors (top 5 of 5 legal):
     1.  73.6% ( -5.4pp) █████████████████████████████ DIVIDEND $1
     2.  12.0% ( -1.2pp) ████ DIVIDEND $4
     3.   5.9% ( +0.3pp) ██ DIVIDEND $0
     4.   4.5% ( +4.8pp) █ DIVIDEND $3
     5.   4.1% ( +1.5pp) █ DIVIDEND $2

  MCTS Visits (top 5, 6400 total):
     1.  2306 (36.0%) Q=-0.383 ██████████████ DIVIDEND $1
     2.  1738 (27.2%) Q=-0.350 ██████████ DIVIDEND $3
     3.  1509 (23.6%) Q=-0.353 █████████ DIVIDEND $4
     4.   732 (11.4%) Q=-0.353 ████ DIVIDEND $2
     5.   115 ( 1.8%) Q=-0.417  DIVIDEND $0
  A0GB Value: P0=-0.322, P1=-0.504, P2=+0.836 (depth: 57, vbackups: 5360)

  **Action: DIVIDEND $1**

### Step 188: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.342, P1=-0.465, P2=+0.859
  NN Priors (top 4 of 4 legal):
     1.  92.3% (-11.9pp) ████████████████████████████████████ DIVIDEND $2
     2.   5.9% ( +1.7pp) ██ DIVIDEND $3
     3.   0.9% ( +5.7pp)  DIVIDEND $1
     4.   0.8% ( +4.4pp)  DIVIDEND $0

  MCTS Visits (top 4, 6400 total):
     1.  5259 (82.2%) Q=-0.405 ████████████████████████████████ DIVIDEND $2
     2.   575 ( 9.0%) Q=-0.397 ███ DIVIDEND $0
     3.   296 ( 4.6%) Q=-0.415 █ DIVIDEND $1
     4.   270 ( 4.2%) Q=-0.424 █ DIVIDEND $3
  A0GB Value: P0=-0.332, P1=-0.404, P2=+0.836 (depth: 59, vbackups: 2469)

  **Action: DIVIDEND $2**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 7  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $10 (NW $75) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $21 (NW $80) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $52 (NW $101) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE]
  OS: $21 price=$37(idx 19) shares=bank:3/unissued:2/issued:4 income=$24 stars=14 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $3 price=$9(idx 5) shares=bank:5/unissued:1/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Issue**: OS

### Step 189: P2 [ISSUE_SHARES]

  **Issue**: OS

  NN Values: P0=-0.344, P1=-0.445, P2=+0.859
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -8.9pp) ███████████████████████████████████████ ISSUE OS shares
     2.   0.1% ( +8.9pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  5893 (92.1%) Q=+0.834 ████████████████████████████████████ ISSUE OS shares
     2.   507 ( 7.9%) Q=+0.830 ███ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 63, vbackups: 5258)

  **Action: ISSUE OS shares**

### Step 190: P1 [ISSUE_SHARES]

  **Issue**: S

  NN Values: P0=-0.428, P1=-0.334, P2=+0.852
  NN Priors (top 2 of 2 legal):
     1.  98.0% (-11.9pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   2.0% (+11.9pp)  ISSUE S shares

  MCTS Visits (top 2, 6400 total):
     1.  6206 (97.0%) Q=-0.403 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   194 ( 3.0%) Q=-0.483 █ ISSUE S shares
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 62, vbackups: 5892)

  **Action: PASS (ISSUE_SHARES)**

### Step 191: P0 [ISSUE_SHARES]

  **Issue**: PR

  NN Values: P0=-0.371, P1=-0.422, P2=+0.852
  NN Priors (top 2 of 2 legal):
     1.  97.3% ( -5.8pp) ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   2.7% ( +5.8pp) █ ISSUE PR shares

  MCTS Visits (top 2, 6400 total):
     1.  6242 (97.5%) Q=-0.393 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   158 ( 2.5%) Q=-0.451  ISSUE PR shares
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 61, vbackups: 6205)

  **Action: PASS (ISSUE_SHARES)**

### Step 192: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.369, P1=-0.400, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1.  92.4% (-10.8pp) ████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   7.6% (+10.8pp) ███ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6236 (97.4%) Q=-0.391 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   164 ( 2.6%) Q=-0.530 █ ISSUE VM shares
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 60, vbackups: 6241)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 7  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $10 (NW $75) order=2 income=$7  companies=[BR]  shares=[PR=1 (pres), VM=1 (pres)]
  P1: $21 (NW $80) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $52 (NW $97) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  S: $7 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**IPO**: BR

### Step 193: P0 [IPO]

  **IPO**: BR

  NN Values: P0=-0.357, P1=-0.393, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -5.1pp) ███████████████████████████████████████ IPO BR → float JS
     2.   0.1% ( +5.1pp)  PASS (IPO)

  MCTS Visits (top 2, 6400 total):
     1.  6335 (99.0%) Q=-0.391 ███████████████████████████████████████ IPO BR → float JS
     2.    65 ( 1.0%) Q=-0.481  PASS (IPO)
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 59, vbackups: 6235)

  **Action: IPO BR → float JS**

  ↳ auto: PAR JS @$37 (IPO BR)

--- Turn 8 ---

Phase: INVEST  |  Turn: 8  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $21 (NW $80) order=0 income=$0  shares=[S=1 (pres), DA=2 (pres), VM=1]
  P2: $52 (NW $97) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$22(idx 14) shares=bank:1/unissued:5/issued:2 income=$-3 stars=1 pres=P1  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining


### Step 194: P1 [INVEST]

  NN Values: P0=-0.490, P1=-0.357, P2=+0.848
  NN Priors (top 8 of 8 legal):
     1.  88.1% (-12.3pp) ███████████████████████████████████ SELL S share
     2.   6.8% ( +0.2pp) ██ PASS (INVEST)
     3.   4.3% ( +1.1pp) █ SELL DA share
     4.   0.3% ( +1.2pp)  BUY DA share
     5.   0.1% ( +0.5pp)  SELL VM share
     6.   0.1% ( +0.2pp)  BUY SM share
     7.   0.1% ( +1.7pp)  BUY PR share
     8.   0.1% ( +7.4pp)  BUY SI share

  MCTS Visits (top 8, 6400 total):
     1.  6022 (94.1%) Q=-0.404 █████████████████████████████████████ SELL S share
     2.   148 ( 2.3%) Q=-0.455  PASS (INVEST)
     3.    95 ( 1.5%) Q=-0.476  SELL DA share
     4.    89 ( 1.4%) Q=-0.509  BUY SI share
     5.    17 ( 0.3%) Q=-0.526  BUY DA share
     6.    17 ( 0.3%) Q=-0.580  BUY PR share
     7.     9 ( 0.1%) Q=-0.516  SELL VM share
     8.     3 ( 0.0%) Q=-0.518  BUY SM share
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 58, vbackups: 6256)

  **Action: SELL S share**

### Step 195: P2 [INVEST]

  NN Values: P0=-0.406, P1=-0.387, P2=+0.840
  NN Priors (top 10 of 12 legal):
     1.  98.7% (-14.2pp) ███████████████████████████████████████ AUCTION slot 0 (SJ, face $31)
     2.   0.9% ( -0.2pp)  SELL DA share
     3.   0.1% ( -0.0pp)  BUY DA share
     4.   0.1% ( +0.4pp)  BUY SM share
     5.   0.1% ( +2.5pp)  BUY SI share
     6.   0.1% ( +1.8pp)  BUY S share
     7.   0.0% ( +0.2pp)  PASS (INVEST)
     8.   0.0% ( +0.2pp)  AUCTION slot 1 (E, face $43)
     9.   0.0% ( +0.5pp)  BUY JS share
    10.   0.0% ( +6.4pp)  BUY PR share

  MCTS Visits (top 10, 6400 total):
     1.  5954 (93.0%) Q=+0.834 █████████████████████████████████████ AUCTION slot 0 (SJ, face $31)
     2.   167 ( 2.6%) Q=+0.814 █ BUY PR share
     3.    89 ( 1.4%) Q=+0.824  BUY SI share
     4.    58 ( 0.9%) Q=+0.825  BUY S share
     5.    37 ( 0.6%) Q=+0.827  SELL DA share
     6.    24 ( 0.4%) Q=+0.828  BUY SM share
     7.    20 ( 0.3%) Q=+0.838  PASS (INVEST)
     8.    14 ( 0.2%) Q=+0.811  BUY JS share
     9.    12 ( 0.2%) Q=+0.824  AUCTION slot 1 (E, face $43)
    10.     9 ( 0.1%) Q=+0.841  BUY DA share
  A0GB Value: P0=-0.383, P1=-0.320, P2=+0.844 (depth: 57, vbackups: 6018)

  **Action: AUCTION slot 0 (SJ, face $31)**

Phase: BID_IN_AUCTION  |  Turn: 8  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $41 (NW $78) order=0 income=$0  shares=[DA=2 (pres), VM=1]
  P2: $52 (NW $97) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: SJ (fv=$31, 4★, inc=$7), E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 3 remaining

**Auction**: SJ current bid=$0 high bidder=P-1 starter=P2

### Step 196: P2 [BID_IN_AUCTION]

  **Auction**: SJ current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.344, P1=-0.447, P2=+0.848
  NN Priors (top 10 of 15 legal):
     1.  99.9% (-12.5pp) ███████████████████████████████████████ BID $31
     2.   0.1% ( +0.9pp)  BID $32
     3.   0.0% ( +1.5pp)  BID $33
     4.   0.0% ( +0.0pp)  BID $34
     5.   0.0% ( +0.1pp)  BID $38
     6.   0.0% ( -0.0pp)  BID $43
     7.   0.0% ( +6.6pp)  BID $36
     8.   0.0% ( +0.6pp)  BID $35
     9.   0.0% ( +0.1pp)  BID $42
    10.   0.0% ( +0.0pp)  BID $37

  MCTS Visits (top 10, 6400 total):
     1.  6052 (94.6%) Q=+0.834 █████████████████████████████████████ BID $31
     2.   180 ( 2.8%) Q=+0.813 █ BID $36
     3.    62 ( 1.0%) Q=+0.829  BID $33
     4.    53 ( 0.8%) Q=+0.830  BID $32
     5.    22 ( 0.3%) Q=+0.816  BID $35
     6.    18 ( 0.3%) Q=+0.712  BID $44
     7.     4 ( 0.1%) Q=+0.680  BID $45
     8.     3 ( 0.0%) Q=+0.745  BID $41
     9.     2 ( 0.0%) Q=+0.832  BID $34
    10.     2 ( 0.0%) Q=+0.811  BID $38
  A0GB Value: P0=-0.400, P1=-0.324, P2=+0.852 (depth: 58, vbackups: 6027)

  **Action: BID $31**

  ↳ auto: PASS (BID_IN_AUCTION)

### Step 197: P1 [BID_IN_AUCTION]

  **Auction**: SJ current bid=$31 high bidder=P2 starter=P2

  NN Values: P0=-0.338, P1=-0.432, P2=+0.852
  NN Priors (top 10 of 11 legal):
     1.  81.2% ( -9.9pp) ████████████████████████████████ BID $32
     2.  15.3% ( -0.7pp) ██████ PASS (BID_IN_AUCTION)
     3.   2.4% ( +0.0pp)  BID $33
     4.   0.4% ( +1.8pp)  BID $34
     5.   0.4% ( +4.4pp)  BID $35
     6.   0.1% ( -0.0pp)  BID $36
     7.   0.1% ( -0.0pp)  BID $37
     8.   0.1% ( +2.5pp)  BID $39
     9.   0.0% ( +1.8pp)  BID $38
    10.   0.0% ( +0.1pp)  BID $40

  MCTS Visits (top 10, 6400 total):
     1.  5804 (90.7%) Q=-0.400 ████████████████████████████████████ BID $32
     2.   170 ( 2.7%) Q=-0.500 █ PASS (BID_IN_AUCTION)
     3.   118 ( 1.8%) Q=-0.411  BID $33
     4.   111 ( 1.7%) Q=-0.447  BID $35
     5.   105 ( 1.6%) Q=-0.413  BID $34
     6.    48 ( 0.8%) Q=-0.488  BID $39
     7.    41 ( 0.6%) Q=-0.446  BID $38
     8.     1 ( 0.0%) Q=-0.486  BID $36
     9.     1 ( 0.0%) Q=-0.498  BID $37
    10.     1 ( 0.0%) Q=-0.562  BID $40
  A0GB Value: P0=-0.582, P1=-0.147, P2=+0.848 (depth: 58, vbackups: 6020)

  **Action: BID $32**

### Step 198: P2 [BID_IN_AUCTION]

  **Auction**: SJ current bid=$32 high bidder=P1 starter=P2

  NN Values: P0=-0.354, P1=-0.424, P2=+0.852
  NN Priors (top 10 of 14 legal):
     1.  87.0% (-15.0pp) ██████████████████████████████████ PASS (BID_IN_AUCTION)
     2.  12.9% ( +0.2pp) █████ BID $33
     3.   0.0% ( +0.0pp)  BID $34
     4.   0.0% ( +0.0pp)  BID $44
     5.   0.0% ( +0.3pp)  BID $36
     6.   0.0% ( +0.0pp)  BID $37
     7.   0.0% ( +1.4pp)  BID $40
     8.   0.0% ( +0.0pp)  BID $43
     9.   0.0% ( +0.2pp)  BID $35
    10.   0.0% ( +0.0pp)  BID $38

  MCTS Visits (top 10, 6400 total):
     1.  5256 (82.1%) Q=+0.835 ████████████████████████████████ PASS (BID_IN_AUCTION)
     2.   924 (14.4%) Q=+0.834 █████ BID $33
     3.    93 ( 1.5%) Q=+0.763  BID $41
     4.    70 ( 1.1%) Q=+0.740  BID $42
     5.    24 ( 0.4%) Q=+0.776  BID $40
     6.     8 ( 0.1%) Q=+0.809  BID $36
     7.     7 ( 0.1%) Q=+0.783  BID $39
     8.     6 ( 0.1%) Q=+0.814  BID $35
     9.     4 ( 0.1%) Q=+0.825  BID $34
    10.     3 ( 0.0%) Q=+0.676  BID $45
  A0GB Value: P0=-0.363, P1=-0.367, P2=+0.848 (depth: 58, vbackups: 5803)

  **Action: PASS (BID_IN_AUCTION)**

Phase: INVEST  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $52 (NW $97) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [2]: E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 2 remaining


### Step 199: P0 [INVEST]

  NN Values: P0=-0.363, P1=-0.469, P2=+0.840
  NN Priors (top 4 of 4 legal):
     1.  98.6% (-12.7pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.9% ( +1.2pp)  SELL PR share
     3.   0.3% ( +4.4pp)  SELL JS share
     4.   0.2% ( +7.1pp)  SELL VM share

  MCTS Visits (top 4, 6400 total):
     1.  6297 (98.4%) Q=-0.408 ███████████████████████████████████████ PASS (INVEST)
     2.    41 ( 0.6%) Q=-0.581  SELL JS share
     3.    36 ( 0.6%) Q=-0.673  SELL VM share
     4.    26 ( 0.4%) Q=-0.533  SELL PR share
  A0GB Value: P0=-0.516, P1=-0.210, P2=+0.840 (depth: 59, vbackups: 5255)

  **Action: PASS (INVEST)**

### Step 200: P1 [INVEST]

  NN Values: P0=-0.412, P1=-0.416, P2=+0.836
  NN Priors (top 3 of 3 legal):
     1.  83.8% ( -7.2pp) █████████████████████████████████ PASS (INVEST)
     2.  16.1% ( +5.7pp) ██████ SELL DA share
     3.   0.1% ( +1.5pp)  SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  5946 (92.9%) Q=-0.383 █████████████████████████████████████ PASS (INVEST)
     2.   437 ( 6.8%) Q=-0.435 ██ SELL DA share
     3.    17 ( 0.3%) Q=-0.499  SELL VM share
  A0GB Value: P0=-0.516, P1=-0.210, P2=+0.840 (depth: 58, vbackups: 6248)

  **Action: PASS (INVEST)**

### Step 201: P2 [INVEST]

  NN Values: P0=-0.426, P1=-0.424, P2=+0.852
  NN Priors (top 10 of 11 legal):
     1.  81.3% (-11.1pp) ████████████████████████████████ AUCTION slot 0 (E, face $43)
     2.  18.2% ( -2.7pp) ███████ SELL DA share
     3.   0.2% ( +0.4pp)  BUY SM share
     4.   0.1% ( -0.0pp)  BUY SI share
     5.   0.1% ( +0.1pp)  BUY DA share
     6.   0.1% ( +1.1pp)  BUY S share
     7.   0.0% ( +0.0pp)  BUY JS share
     8.   0.0% ( +0.9pp)  BUY PR share
     9.   0.0% ( +0.0pp)  BUY OS share
    10.   0.0% (+11.1pp)  SELL OS share

  MCTS Visits (top 10, 6400 total):
     1.  5392 (84.2%) Q=+0.838 █████████████████████████████████ AUCTION slot 0 (E, face $43)
     2.   585 ( 9.1%) Q=+0.819 ███ SELL DA share
     3.   283 ( 4.4%) Q=+0.811 █ SELL OS share
     4.    56 ( 0.9%) Q=+0.828  BUY S share
     5.    36 ( 0.6%) Q=+0.820  BUY PR share
     6.    28 ( 0.4%) Q=+0.826  BUY SM share
     7.    12 ( 0.2%) Q=+0.833  BUY DA share
     8.     5 ( 0.1%) Q=+0.828  BUY SI share
     9.     1 ( 0.0%) Q=+0.797  BUY OS share
    10.     1 ( 0.0%) Q=+0.785  BUY JS share
  A0GB Value: P0=-0.516, P1=-0.210, P2=+0.840 (depth: 57, vbackups: 5953)

  **Action: AUCTION slot 0 (E, face $43)**

Phase: BID_IN_AUCTION  |  Turn: 8  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $52 (NW $97) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [2]: E (fv=$43, 4★, inc=$7), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 2 remaining

**Auction**: E current bid=$0 high bidder=P-1 starter=P2

### Step 202: P2 [BID_IN_AUCTION]

  **Auction**: E current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.404, P1=-0.463, P2=+0.848
  NN Priors (top 10 of 10 legal):
     1.  99.9% (-13.8pp) ███████████████████████████████████████ BID $43
     2.   0.0% ( +7.2pp)  BID $44
     3.   0.0% ( +3.8pp)  BID $45
     4.   0.0% ( +0.1pp)  BID $50
     5.   0.0% ( +0.3pp)  BID $46
     6.   0.0% ( +0.8pp)  BID $47
     7.   0.0% ( +0.0pp)  BID $48
     8.   0.0% ( +0.2pp)  BID $49
     9.   0.0% ( +0.3pp)  BID $52
    10.   0.0% ( +1.1pp)  BID $51

  MCTS Visits (top 10, 6400 total):
     1.  5765 (90.1%) Q=+0.838 ████████████████████████████████████ BID $43
     2.   379 ( 5.9%) Q=+0.832 ██ BID $44
     3.   186 ( 2.9%) Q=+0.829 █ BID $45
     4.    26 ( 0.4%) Q=+0.818  BID $47
     5.    24 ( 0.4%) Q=+0.796  BID $51
     6.     8 ( 0.1%) Q=+0.814  BID $46
     7.     6 ( 0.1%) Q=+0.787  BID $52
     8.     4 ( 0.1%) Q=+0.806  BID $49
     9.     2 ( 0.0%) Q=+0.805  BID $50
  A0GB Value: P0=-0.352, P1=-0.436, P2=+0.844 (depth: 44, vbackups: 5391)

  **Action: BID $43**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $9 (NW $97) order=1 income=$7  companies=[E]  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [1]: FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining


### Step 203: P0 [INVEST]

  NN Values: P0=-0.408, P1=-0.443, P2=+0.828
  NN Priors (top 4 of 4 legal):
     1.  99.2% (-12.3pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.6% ( +2.2pp)  SELL PR share
     3.   0.2% ( +4.7pp)  SELL VM share
     4.   0.1% ( +5.5pp)  SELL JS share

  MCTS Visits (top 4, 6400 total):
     1.  6303 (98.5%) Q=-0.401 ███████████████████████████████████████ PASS (INVEST)
     2.    36 ( 0.6%) Q=-0.615  SELL JS share
     3.    33 ( 0.5%) Q=-0.543  SELL PR share
     4.    28 ( 0.4%) Q=-0.628  SELL VM share
  A0GB Value: P0=-0.262, P1=-0.490, P2=+0.840 (depth: 44, vbackups: 5764)

  **Action: PASS (INVEST)**

### Step 204: P1 [INVEST]

  NN Values: P0=-0.416, P1=-0.475, P2=+0.840
  NN Priors (top 3 of 3 legal):
     1.  89.4% (-12.3pp) ███████████████████████████████████ PASS (INVEST)
     2.  10.4% ( +4.0pp) ████ SELL DA share
     3.   0.2% ( +8.3pp)  SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  5990 (93.6%) Q=-0.395 █████████████████████████████████████ PASS (INVEST)
     2.   246 ( 3.8%) Q=-0.458 █ SELL DA share
     3.   164 ( 2.6%) Q=-0.451 █ SELL VM share
  A0GB Value: P0=-0.262, P1=-0.490, P2=+0.840 (depth: 43, vbackups: 6184)

  **Action: PASS (INVEST)**

### Step 205: P2 [INVEST]

  NN Values: P0=-0.428, P1=-0.459, P2=+0.836
  NN Priors (top 3 of 3 legal):
     1.  99.9% ( -7.8pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.1% ( +1.3pp)  SELL DA share
     3.   0.0% ( +6.5pp)  SELL OS share

  MCTS Visits (top 3, 6400 total):
     1.  6231 (97.4%) Q=+0.838 ██████████████████████████████████████ PASS (INVEST)
     2.   111 ( 1.7%) Q=+0.785  SELL OS share
     3.    58 ( 0.9%) Q=+0.828  SELL DA share
  A0GB Value: P0=-0.154, P1=-0.617, P2=+0.836 (depth: 33, vbackups: 6107)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 8  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $9 (NW $97) order=1 income=$7  companies=[E]  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Corp**: P2 may buy with OS($54)

### Step 206: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($54)

  NN Values: P0=-0.490, P1=-0.414, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -7.6pp) ███████████████████████████████████████ ACQ select OS
     2.   0.0% ( +7.6pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6242 (97.5%) Q=+0.838 ███████████████████████████████████████ ACQ select OS
     2.   158 ( 2.5%) Q=+0.824  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.314, P1=-0.424, P2=+0.824 (depth: 42, vbackups: 6230)

  **Action: ACQ select OS**

  ↳ auto: ACQ target E (with OS)

Phase: ACQ_SELECT_PRICE  |  Turn: 8  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $9 (NW $97) order=1 income=$7  companies=[E]  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $54 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$24 stars=17 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Price**: P2 OS -> E (price range $22-$57)

### Step 207: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> E (price range $22-$57)

  NN Values: P0=-0.508, P1=-0.393, P2=+0.828
  NN Priors (top 10 of 33 legal):
     1.  89.5% (-12.4pp) ███████████████████████████████████ ACQUIRE E with OS @ $54
     2.   9.4% ( -0.5pp) ███ ACQUIRE E with OS @ $53
     3.   1.0% ( -0.3pp)  ACQUIRE E with OS @ $52
     4.   0.0% ( +0.2pp)  ACQUIRE E with OS @ $51
     5.   0.0% ( +3.0pp)  ACQUIRE E with OS @ $50
     6.   0.0% ( +0.0pp)  ACQUIRE E with OS @ $49
     7.   0.0% ( +0.2pp)  ACQUIRE E with OS @ $48
     8.   0.0% ( +0.2pp)  ACQUIRE E with OS @ $47
     9.   0.0% ( +0.0pp)  ACQUIRE E with OS @ $46
    10.   0.0% ( +0.0pp)  ACQUIRE E with OS @ $45

  MCTS Visits (top 10, 6400 total):
     1.  5687 (88.9%) Q=+0.838 ███████████████████████████████████ ACQUIRE E with OS @ $54
     2.   544 ( 8.5%) Q=+0.835 ███ ACQUIRE E with OS @ $53
     3.    58 ( 0.9%) Q=+0.823  ACQUIRE E with OS @ $50
     4.    46 ( 0.7%) Q=+0.754  ACQUIRE E with OS @ $39
     5.    32 ( 0.5%) Q=+0.829  ACQUIRE E with OS @ $52
     6.     8 ( 0.1%) Q=+0.666  ACQUIRE E with OS @ $35
     7.     5 ( 0.1%) Q=+0.619  ACQUIRE E with OS @ $34
     8.     4 ( 0.1%) Q=+0.802  ACQUIRE E with OS @ $51
     9.     4 ( 0.1%) Q=+0.794  ACQUIRE E with OS @ $48
    10.     3 ( 0.0%) Q=+0.781  ACQUIRE E with OS @ $47
  A0GB Value: P0=-0.314, P1=-0.424, P2=+0.824 (depth: 41, vbackups: 6239)

  **Action: ACQUIRE E with OS @ $54**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $0 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=16 pres=P2  companies=[SX, KK, DR, SZD, E*]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Corp**: P0 may buy with JS($40), PR($21), VM($20)

### Step 208: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($40), PR($21), VM($20)

  NN Values: P0=-0.574, P1=-0.270, P2=+0.812
  NN Priors (top 3 of 3 legal):
     1.  90.3% (-10.3pp) ████████████████████████████████████ ACQ select JS
     2.   9.5% ( +6.7pp) ███ ACQ select VM
     3.   0.2% ( +3.5pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 3, 6400 total):
     1.  5889 (92.0%) Q=-0.397 ████████████████████████████████████ ACQ select JS
     2.   469 ( 7.3%) Q=-0.427 ██ ACQ select VM
     3.    42 ( 0.7%) Q=-0.518  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.363, P1=-0.414, P2=+0.832 (depth: 41, vbackups: 5686)

  **Action: ACQ select JS**

Phase: ACQ_SELECT_COMPANY  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $0 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=16 pres=P2  companies=[SX, KK, DR, SZD, E*]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Company**: P0 buying with JS ($40)

### Step 209: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with JS ($40)

  NN Values: P0=-0.543, P1=-0.299, P2=+0.820
  NN Priors (top 2 of 2 legal):
     1.  98.1% ( -8.3pp) ███████████████████████████████████████ ACQ target PKP (with JS)
     2.   1.9% ( +8.3pp)  ACQ target WT (with JS)

  MCTS Visits (top 2, 6400 total):
     1.  6185 (96.6%) Q=-0.395 ██████████████████████████████████████ ACQ target PKP (with JS)
     2.   215 ( 3.4%) Q=-0.443 █ ACQ target WT (with JS)
  A0GB Value: P0=-0.389, P1=-0.396, P2=+0.840 (depth: 41, vbackups: 5888)

  **Action: ACQ target PKP (with JS)**

Phase: ACQ_SELECT_PRICE  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $40 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$7 stars=8 pres=P0  companies=[BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $0 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=16 pres=P2  companies=[SX, KK, DR, SZD, E*]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$6 stars=7 pres=P0  companies=[WT, PKP]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Price**: P0 JS -> PKP (price range $13-$33)

### Step 210: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 JS -> PKP (price range $13-$33)

  NN Values: P0=-0.574, P1=-0.311, P2=+0.812
  NN Priors (top 10 of 21 legal):
     1.  96.6% (-14.3pp) ██████████████████████████████████████ ACQUIRE PKP with JS @ $13
     2.   1.9% ( +0.3pp)  ACQUIRE PKP with JS @ $14
     3.   0.3% ( -0.0pp)  ACQUIRE PKP with JS @ $15
     4.   0.2% ( +0.1pp)  ACQUIRE PKP with JS @ $17
     5.   0.2% ( +0.0pp)  ACQUIRE PKP with JS @ $16
     6.   0.2% ( -0.0pp)  ACQUIRE PKP with JS @ $18
     7.   0.1% ( -0.0pp)  ACQUIRE PKP with JS @ $23
     8.   0.1% ( +1.9pp)  ACQUIRE PKP with JS @ $22
     9.   0.1% ( +0.0pp)  ACQUIRE PKP with JS @ $19
    10.   0.1% ( -0.0pp)  ACQUIRE PKP with JS @ $21

  MCTS Visits (top 10, 6400 total):
     1.  5579 (87.2%) Q=-0.397 ██████████████████████████████████ ACQUIRE PKP with JS @ $13
     2.   311 ( 4.9%) Q=-0.387 █ ACQUIRE PKP with JS @ $14
     3.   137 ( 2.1%) Q=-0.421  ACQUIRE PKP with JS @ $33
     4.   109 ( 1.7%) Q=-0.410  ACQUIRE PKP with JS @ $20
     5.    81 ( 1.3%) Q=-0.427  ACQUIRE PKP with JS @ $30
     6.    49 ( 0.8%) Q=-0.437  ACQUIRE PKP with JS @ $22
     7.    34 ( 0.5%) Q=-0.387  ACQUIRE PKP with JS @ $15
     8.    28 ( 0.4%) Q=-0.441  ACQUIRE PKP with JS @ $29
     9.    21 ( 0.3%) Q=-0.399  ACQUIRE PKP with JS @ $17
    10.    12 ( 0.2%) Q=-0.431  ACQUIRE PKP with JS @ $25
  A0GB Value: P0=-0.389, P1=-0.396, P2=+0.840 (depth: 40, vbackups: 5768)

  **Action: ACQUIRE PKP with JS @ $13**

Phase: ACQ_SELECT_CORP  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $27 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$12 stars=9 pres=P0  companies=[PKP*, BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $0 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=16 pres=P2  companies=[SX, KK, DR, SZD, E*]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $21 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$0 stars=4 pres=P0  companies=[WT]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Acquisition — Select Corp**: P0 may buy with JS($27), PR($21), VM($20)

### Step 211: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($27), PR($21), VM($20)

  NN Values: P0=-0.395, P1=-0.361, P2=+0.816
  NN Priors (top 3 of 3 legal):
     1.  77.8% ( -9.5pp) ███████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.  20.0% ( +0.9pp) ███████ ACQ select PR
     3.   2.3% ( +8.6pp)  ACQ select VM

  MCTS Visits (top 3, 6400 total):
     1.  6091 (95.2%) Q=-0.390 ██████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   229 ( 3.6%) Q=-0.507 █ ACQ select PR
     3.    80 ( 1.2%) Q=-0.564  ACQ select VM
  A0GB Value: P0=-0.299, P1=-0.469, P2=+0.848 (depth: 41, vbackups: 5578)

  **Action: PASS (ACQ_SELECT_CORP)**

Phase: CLOSING  |  Turn: 8  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $9 (NW $77) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $39 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $27 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$12 stars=9 pres=P0  companies=[PKP, BR]
  S: $7 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $0 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=16 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $12 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $34 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$0 stars=5 pres=P0  companies=[WT]
  DA: $13 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=7 pres=P1  companies=[AKE, PR, NS]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $3 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Closing**: P1 may close AKE (DA), PR (DA), NS (DA), SJ

### Step 212: P1 [CLOSING]

  **Closing**: P1 may close AKE (DA), PR (DA), NS (DA), SJ

  NN Values: P0=-0.334, P1=-0.424, P2=+0.820
  NN Priors (top 3 of 3 legal):
     1.  97.0% (-11.3pp) ██████████████████████████████████████ PASS (CLOSING)
     2.   2.7% ( +7.8pp) █ CLOSE AKE
     3.   0.3% ( +3.6pp)  CLOSE PR

  MCTS Visits (top 3, 6400 total):
     1.  6171 (96.4%) Q=-0.407 ██████████████████████████████████████ PASS (CLOSING)
     2.   198 ( 3.1%) Q=-0.462 █ CLOSE AKE
     3.    31 ( 0.5%) Q=-0.577  CLOSE PR
  A0GB Value: P0=-0.256, P1=-0.469, P2=+0.836 (depth: 39, vbackups: 6090)

  **Action: PASS (CLOSING)**

### Step 213: P2 [CLOSING]

  **Closing**: P2 may close SX (OS), KK (OS), DR (OS), SZD (OS), E (OS)

  NN Values: P0=-0.375, P1=-0.383, P2=+0.828
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -0.5pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +0.5pp)  CLOSE SX

  MCTS Visits (top 2, 6400 total):
     1.  6388 (99.8%) Q=+0.838 ███████████████████████████████████████ PASS (CLOSING)
     2.    12 ( 0.2%) Q=+0.799  CLOSE SX
  A0GB Value: P0=-0.249, P1=-0.609, P2=+0.836 (depth: 40, vbackups: 6170)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $7 (NW $75) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $16 (NW $84) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $108) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $39 price=$37(idx 19) shares=bank:1/unissued:5/issued:2 income=$12 stars=10 pres=P0  companies=[PKP, BR]
  S: $4 price=$20(idx 13) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $31 price=$33(idx 18) shares=bank:4/unissued:1/issued:5 income=$31 stars=19 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$9(idx 5) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $34 price=$18(idx 12) shares=bank:3/unissued:1/issued:4 income=$0 stars=5 pres=P0  companies=[WT]
  DA: $23 price=$12(idx 8) shares=bank:2/unissued:0/issued:5 income=$10 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $22 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$2 stars=3 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Dividends**: JS

### Step 214: P0 [DIVIDENDS]

  **Dividends**: JS

  NN Values: P0=-0.305, P1=-0.482, P2=+0.828
  NN Priors (top 10 of 13 legal):
     1.  89.9% (-13.0pp) ███████████████████████████████████ DIVIDEND $9
     2.   7.9% ( -0.7pp) ███ DIVIDEND $12
     3.   1.2% ( +0.8pp)  DIVIDEND $10
     4.   0.7% ( +0.1pp)  DIVIDEND $8
     5.   0.2% ( +1.4pp)  DIVIDEND $11
     6.   0.1% ( +1.2pp)  DIVIDEND $6
     7.   0.0% ( +0.1pp)  DIVIDEND $3
     8.   0.0% ( +0.4pp)  DIVIDEND $7
     9.   0.0% ( +1.0pp)  DIVIDEND $2
    10.   0.0% ( +0.8pp)  DIVIDEND $5

  MCTS Visits (top 10, 6400 total):
     1.  5434 (84.9%) Q=-0.388 █████████████████████████████████ DIVIDEND $9
     2.   316 ( 4.9%) Q=-0.399 █ DIVIDEND $12
     3.   166 ( 2.6%) Q=-0.381 █ DIVIDEND $11
     4.   137 ( 2.1%) Q=-0.427  DIVIDEND $4
     5.   115 ( 1.8%) Q=-0.392  DIVIDEND $10
     6.    72 ( 1.1%) Q=-0.384  DIVIDEND $8
     7.    43 ( 0.7%) Q=-0.407  DIVIDEND $6
     8.    33 ( 0.5%) Q=-0.411  DIVIDEND $7
     9.    18 ( 0.3%) Q=-0.454  DIVIDEND $1
    10.    18 ( 0.3%) Q=-0.429  DIVIDEND $5
  A0GB Value: P0=-0.249, P1=-0.609, P2=+0.836 (depth: 39, vbackups: 6060)

  **Action: DIVIDEND $9**

### Step 215: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.334, P1=-0.432, P2=+0.828
  NN Priors (top 7 of 7 legal):
     1.  99.6% (-14.8pp) ███████████████████████████████████████ DIVIDEND $0
     2.   0.2% ( +3.2pp)  DIVIDEND $2
     3.   0.1% ( +1.9pp)  DIVIDEND $1
     4.   0.0% ( +2.4pp)  DIVIDEND $5
     5.   0.0% ( +2.4pp)  DIVIDEND $3
     6.   0.0% ( +0.9pp)  DIVIDEND $6
     7.   0.0% ( +4.0pp)  DIVIDEND $4

  MCTS Visits (top 7, 6400 total):
     1.  5445 (85.1%) Q=+0.837 ██████████████████████████████████ DIVIDEND $0
     2.   250 ( 3.9%) Q=+0.841 █ DIVIDEND $4
     3.   240 ( 3.8%) Q=+0.842 █ DIVIDEND $2
     4.   153 ( 2.4%) Q=+0.840  DIVIDEND $3
     5.   133 ( 2.1%) Q=+0.837  DIVIDEND $5
     6.   128 ( 2.0%) Q=+0.840  DIVIDEND $1
     7.    51 ( 0.8%) Q=+0.834  DIVIDEND $6
  A0GB Value: P0=-0.249, P1=-0.609, P2=+0.836 (depth: 38, vbackups: 5433)

  **Action: DIVIDEND $0**

### Step 216: P0 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=-0.340, P1=-0.402, P2=+0.840
  NN Priors (top 7 of 7 legal):
     1.  96.6% (-13.9pp) ██████████████████████████████████████ DIVIDEND $6
     2.   1.1% ( +4.5pp)  DIVIDEND $0
     3.   1.1% ( +0.4pp)  DIVIDEND $5
     4.   0.4% ( +0.2pp)  DIVIDEND $3
     5.   0.4% ( +1.2pp)  DIVIDEND $1
     6.   0.2% ( +0.0pp)  DIVIDEND $2
     7.   0.2% ( +7.6pp)  DIVIDEND $4

  MCTS Visits (top 7, 6400 total):
     1.  5884 (91.9%) Q=-0.386 ████████████████████████████████████ DIVIDEND $6
     2.   249 ( 3.9%) Q=-0.413 █ DIVIDEND $4
     3.   103 ( 1.6%) Q=-0.445  DIVIDEND $0
     4.    83 ( 1.3%) Q=-0.392  DIVIDEND $5
     5.    39 ( 0.6%) Q=-0.420  DIVIDEND $1
     6.    34 ( 0.5%) Q=-0.392  DIVIDEND $3
     7.     8 ( 0.1%) Q=-0.409  DIVIDEND $2
  A0GB Value: P0=-0.311, P1=-0.508, P2=+0.844 (depth: 38, vbackups: 5408)

  **Action: DIVIDEND $6**

### Step 217: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.383, P1=-0.369, P2=+0.848
  NN Priors (top 5 of 5 legal):
     1.  71.4% ( -9.5pp) ████████████████████████████ DIVIDEND $1
     2.  18.1% ( +0.9pp) ███████ DIVIDEND $0
     3.   4.6% ( +1.4pp) █ DIVIDEND $4
     4.   3.8% ( +4.0pp) █ DIVIDEND $2
     5.   2.2% ( +3.2pp)  DIVIDEND $3

  MCTS Visits (top 5, 6400 total):
     1.  4737 (74.0%) Q=-0.360 █████████████████████████████ DIVIDEND $2
     2.   856 (13.4%) Q=-0.456 █████ DIVIDEND $1
     3.   363 ( 5.7%) Q=-0.429 ██ DIVIDEND $0
     4.   239 ( 3.7%) Q=-0.394 █ DIVIDEND $4
     5.   205 ( 3.2%) Q=-0.400 █ DIVIDEND $3
  A0GB Value: P0=-0.201, P1=-0.641, P2=+0.840 (depth: 38, vbackups: 5609)

  **Action: DIVIDEND $2**

### Step 218: P1 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.283, P1=-0.490, P2=+0.824
  NN Priors (top 5 of 5 legal):
     1.  75.7% ( -8.8pp) ██████████████████████████████ DIVIDEND $4
     2.  15.4% ( -2.6pp) ██████ DIVIDEND $0
     3.   7.7% ( +5.3pp) ███ DIVIDEND $2
     4.   0.8% ( +4.7pp)  DIVIDEND $3
     5.   0.4% ( +1.4pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  3502 (54.7%) Q=-0.418 █████████████████████ DIVIDEND $0
     2.  1847 (28.9%) Q=-0.463 ███████████ DIVIDEND $4
     3.   573 ( 9.0%) Q=-0.427 ███ DIVIDEND $3
     4.   383 ( 6.0%) Q=-0.463 ██ DIVIDEND $2
     5.    95 ( 1.5%) Q=-0.442  DIVIDEND $1
  A0GB Value: P0=-0.201, P1=-0.641, P2=+0.840 (depth: 37, vbackups: 4736)

  **Action: DIVIDEND $0**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 8  |  CoO Level: 5  |  Active Player: 0  |  End Card: no

**Players**
  P0: $24 (NW $94) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $18 (NW $96) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $122) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$12 stars=9 pres=P0  companies=[PKP, BR]
  S: $4 price=$16(idx 11) shares=bank:2/unissued:5/issued:2 income=$-3 stars=1 RECEIVERSHIP  companies=[BSE]
  OS: $31 price=$41(idx 20) shares=bank:4/unissued:1/issued:5 income=$31 stars=19 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$14(idx 10) shares=bank:3/unissued:1/issued:4 income=$0 stars=3 pres=P0  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Issue**: JS

### Step 219: P0 [ISSUE_SHARES]

  **Issue**: JS

  NN Values: P0=-0.297, P1=-0.500, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1.  98.6% (-10.6pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   1.4% (+10.6pp)  ISSUE JS shares

  MCTS Visits (top 2, 6400 total):
     1.  6303 (98.5%) Q=-0.367 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.    97 ( 1.5%) Q=-0.531  ISSUE JS shares
  A0GB Value: P0=-0.355, P1=-0.543, P2=+0.855 (depth: 41, vbackups: 3501)

  **Action: PASS (ISSUE_SHARES)**

### Step 220: P2 [ISSUE_SHARES]

  **Issue**: OS

  NN Values: P0=-0.373, P1=-0.439, P2=+0.828
  NN Priors (top 2 of 2 legal):
     1. 100.0% ( -4.5pp) ███████████████████████████████████████ ISSUE OS shares
     2.   0.0% ( +4.5pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6311 (98.6%) Q=+0.836 ███████████████████████████████████████ ISSUE OS shares
     2.    89 ( 1.4%) Q=+0.840  PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.355, P1=-0.543, P2=+0.855 (depth: 40, vbackups: 6302)

  **Action: ISSUE OS shares**

### Step 221: P0 [ISSUE_SHARES]

  **Issue**: PR

  NN Values: P0=-0.270, P1=-0.523, P2=+0.844
  NN Priors (top 2 of 2 legal):
     1.  99.7% ( -6.2pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   0.3% ( +6.2pp)  ISSUE PR shares

  MCTS Visits (top 2, 6400 total):
     1.  6309 (98.6%) Q=-0.367 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.    91 ( 1.4%) Q=-0.444  ISSUE PR shares
  A0GB Value: P0=-0.355, P1=-0.543, P2=+0.855 (depth: 39, vbackups: 6310)

  **Action: PASS (ISSUE_SHARES)**

### Step 222: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.312, P1=-0.500, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1.  75.0% ( -5.9pp) █████████████████████████████ ISSUE VM shares
     2.  25.0% ( +5.9pp) ██████████ PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  5475 (85.5%) Q=-0.354 ██████████████████████████████████ PASS (ISSUE_SHARES)
     2.   925 (14.5%) Q=-0.448 █████ ISSUE VM shares
  A0GB Value: P0=-0.355, P1=-0.543, P2=+0.855 (depth: 38, vbackups: 6236)

  **Action: PASS (ISSUE_SHARES)**

  ↳ auto: PASS (IPO)

--- Turn 9 ---

Phase: INVEST  |  Turn: 9  |  CoO Level: 5  |  Active Player: 1  |  End Card: no

**Players**
  P0: $24 (NW $94) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $18 (NW $96) order=0 income=$7  companies=[SJ]  shares=[DA=2 (pres), VM=1]
  P2: $63 (NW $118) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$12 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-3 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $68 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$31 stars=22 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$14(idx 10) shares=bank:3/unissued:1/issued:4 income=$0 stars=3 pres=P0  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining


### Step 223: P1 [INVEST]

  NN Values: P0=-0.387, P1=-0.406, P2=+0.844
  NN Priors (top 7 of 7 legal):
     1.  99.6% (-12.3pp) ███████████████████████████████████████ SELL DA share
     2.   0.4% ( +0.1pp)  PASS (INVEST)
     3.   0.0% ( +0.3pp)  BUY SI share
     4.   0.0% ( +7.4pp)  BUY S share
     5.   0.0% ( +3.3pp)  SELL VM share
     6.   0.0% ( +0.6pp)  BUY SM share
     7.   0.0% ( +0.6pp)  BUY PR share

  MCTS Visits (top 7, 6400 total):
     1.  6186 (96.7%) Q=-0.442 ██████████████████████████████████████ SELL DA share
     2.   105 ( 1.6%) Q=-0.531  BUY S share
     3.    33 ( 0.5%) Q=-0.599  SELL VM share
     4.    33 ( 0.5%) Q=-0.467  BUY SM share
     5.    25 ( 0.4%) Q=-0.453  PASS (INVEST)
     6.     9 ( 0.1%) Q=-0.504  BUY PR share
     7.     9 ( 0.1%) Q=-0.484  BUY SI share
  A0GB Value: P0=-0.578, P1=-0.182, P2=+0.848 (depth: 43, vbackups: 5474)

  **Action: SELL DA share**

### Step 224: P2 [INVEST]

  NN Values: P0=-0.404, P1=-0.383, P2=+0.848
  NN Priors (top 10 of 13 legal):
     1.  68.9% ( -7.3pp) ███████████████████████████ AUCTION slot 0 (HR, face $47)
     2.  30.6% ( -5.1pp) ████████████ AUCTION slot 2 (FRA, face $56)
     3.   0.2% ( +0.1pp)  SELL DA share
     4.   0.1% ( +0.9pp)  BUY DA share
     5.   0.1% ( +0.1pp)  BUY SM share
     6.   0.0% ( -0.0pp)  BUY SI share
     7.   0.0% ( +7.4pp)  AUCTION slot 1 (MAD, face $50)
     8.   0.0% ( +0.2pp)  BUY OS share
     9.   0.0% ( +0.5pp)  BUY S share
    10.   0.0% ( +2.2pp)  PASS (INVEST)

  MCTS Visits (top 10, 6400 total):
     1.  4234 (66.2%) Q=+0.838 ██████████████████████████ AUCTION slot 0 (HR, face $47)
     2.  1650 (25.8%) Q=+0.836 ██████████ AUCTION slot 2 (FRA, face $56)
     3.   272 ( 4.2%) Q=+0.833 █ AUCTION slot 1 (MAD, face $50)
     4.    97 ( 1.5%) Q=+0.835  PASS (INVEST)
     5.    67 ( 1.0%) Q=+0.839  BUY DA share
     6.    24 ( 0.4%) Q=+0.831  BUY S share
     7.    19 ( 0.3%) Q=+0.819  BUY PR share
     8.    16 ( 0.2%) Q=+0.835  SELL DA share
     9.     9 ( 0.1%) Q=+0.832  BUY SM share
    10.     8 ( 0.1%) Q=+0.830  BUY OS share
  A0GB Value: P0=-0.578, P1=-0.182, P2=+0.848 (depth: 42, vbackups: 5909)

  **Action: AUCTION slot 0 (HR, face $47)**

Phase: BID_IN_AUCTION  |  Turn: 9  |  CoO Level: 5  |  Active Player: 2  |  End Card: no

**Players**
  P0: $24 (NW $94) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $34 (NW $92) order=0 income=$7  companies=[SJ]  shares=[DA=1 (pres), VM=1]
  P2: $63 (NW $116) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: HR (fv=$47, 5★, inc=$10), MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$12 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-3 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $68 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$31 stars=22 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-1 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$14(idx 10) shares=bank:3/unissued:1/issued:4 income=$0 stars=3 pres=P0  companies=[WT]
  DA: $23 price=$16(idx 11) shares=bank:3/unissued:0/issued:5 income=$10 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-2 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 1 remaining

**Auction**: HR current bid=$0 high bidder=P-1 starter=P2

### Step 225: P2 [BID_IN_AUCTION]

  **Auction**: HR current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.342, P1=-0.424, P2=+0.848
  NN Priors (top 10 of 15 legal):
     1.  99.9% (-14.3pp) ███████████████████████████████████████ BID $47
     2.   0.0% ( +1.3pp)  BID $48
     3.   0.0% ( +0.6pp)  BID $49
     4.   0.0% ( -0.0pp)  BID $50
     5.   0.0% ( +0.7pp)  BID $54
     6.   0.0% ( +2.0pp)  BID $53
     7.   0.0% ( +0.8pp)  BID $51
     8.   0.0% ( +0.5pp)  BID $52
     9.   0.0% ( +0.3pp)  BID $58
    10.   0.0% ( +0.0pp)  BID $57

  MCTS Visits (top 10, 6400 total):
     1.  5881 (91.9%) Q=+0.837 ████████████████████████████████████ BID $47
     2.    92 ( 1.4%) Q=+0.819  BID $55
     3.    85 ( 1.3%) Q=+0.804  BID $61
     4.    84 ( 1.3%) Q=+0.824  BID $53
     5.    74 ( 1.2%) Q=+0.817  BID $56
     6.    72 ( 1.1%) Q=+0.832  BID $48
     7.    32 ( 0.5%) Q=+0.823  BID $51
     8.    30 ( 0.5%) Q=+0.827  BID $49
     9.    21 ( 0.3%) Q=+0.813  BID $54
    10.    17 ( 0.3%) Q=+0.818  BID $52
  A0GB Value: P0=-0.455, P1=-0.328, P2=+0.852 (depth: 38, vbackups: 4502)

  **Action: BID $47**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $24 (NW $94) order=2 income=$0  shares=[JS=1 (pres), PR=1 (pres), VM=1 (pres)]
  P1: $34 (NW $92) order=0 income=$7  companies=[SJ]  shares=[DA=1 (pres), VM=1]
  P2: $16 (NW $116) order=1 income=$10  companies=[HR]  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [2]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $68 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$14 stars=22 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$14(idx 10) shares=bank:3/unissued:1/issued:4 income=$-3 stars=3 pres=P0  companies=[WT]
  DA: $23 price=$16(idx 11) shares=bank:3/unissued:0/issued:5 income=$-3 stars=8 pres=P1  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining


### Step 226: P0 [INVEST]

  NN Values: P0=-0.406, P1=-0.357, P2=+0.828
  NN Priors (top 9 of 9 legal):
     1.  96.0% (-14.3pp) ██████████████████████████████████████ SELL PR share
     2.   1.5% ( +5.8pp)  SELL JS share
     3.   0.8% ( -0.0pp)  PASS (INVEST)
     4.   0.6% ( +3.3pp)  BUY SI share
     5.   0.4% ( +1.6pp)  BUY SM share
     6.   0.3% ( +0.4pp)  SELL VM share
     7.   0.2% ( +0.2pp)  BUY DA share
     8.   0.1% ( +0.6pp)  BUY S share
     9.   0.1% ( +2.4pp)  BUY PR share

  MCTS Visits (top 9, 6400 total):
     1.  3583 (56.0%) Q=-0.371 ██████████████████████ SELL PR share
     2.  2729 (42.6%) Q=-0.340 █████████████████ SELL JS share
     3.    23 ( 0.4%) Q=-0.586  BUY SI share
     4.    17 ( 0.3%) Q=-0.551  BUY PR share
     5.    16 ( 0.2%) Q=-0.532  BUY SM share
     6.    13 ( 0.2%) Q=-0.411  PASS (INVEST)
     7.     9 ( 0.1%) Q=-0.602  SELL VM share
     8.     9 ( 0.1%) Q=-0.565  BUY S share
     9.     1 ( 0.0%) Q=-0.613  BUY DA share
  A0GB Value: P0=-0.455, P1=-0.328, P2=+0.852 (depth: 37, vbackups: 3640)

  **Action: SELL PR share**

### Step 227: P1 [INVEST]

  NN Values: P0=-0.398, P1=-0.379, P2=+0.836
  NN Priors (top 8 of 8 legal):
     1.  83.2% (-11.0pp) █████████████████████████████████ PASS (INVEST)
     2.  15.7% ( -1.5pp) ██████ SELL DA share
     3.   0.3% ( +4.5pp)  BUY DA share
     4.   0.3% ( +0.6pp)  BUY SI share
     5.   0.2% ( +0.6pp)  BUY S share
     6.   0.1% ( +2.8pp)  BUY SM share
     7.   0.1% ( +3.6pp)  BUY PR share
     8.   0.0% ( +0.3pp)  SELL VM share

  MCTS Visits (top 8, 6400 total):
     1.  4550 (71.1%) Q=-0.417 ████████████████████████████ SELL DA share
     2.  1651 (25.8%) Q=-0.472 ██████████ PASS (INVEST)
     3.    63 ( 1.0%) Q=-0.518  BUY DA share
     4.    62 ( 1.0%) Q=-0.501  BUY PR share
     5.    52 ( 0.8%) Q=-0.492  BUY SM share
     6.     9 ( 0.1%) Q=-0.588  SELL VM share
     7.     7 ( 0.1%) Q=-0.558  BUY S share
     8.     6 ( 0.1%) Q=-0.600  BUY SI share
  A0GB Value: P0=-0.641, P1=-0.264, P2=+0.836 (depth: 38, vbackups: 5429)

  **Action: SELL DA share**

### Step 228: P2 [INVEST]

  NN Values: P0=-0.350, P1=-0.391, P2=+0.840
  NN Priors (top 8 of 8 legal):
     1.  99.7% (-13.6pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.1% ( +2.7pp)  SELL DA share
     3.   0.0% ( +0.2pp)  BUY SM share
     4.   0.0% ( +3.4pp)  BUY SI share
     5.   0.0% ( +4.2pp)  BUY DA share
     6.   0.0% ( +0.6pp)  BUY S share
     7.   0.0% ( +2.3pp)  BUY PR share
     8.   0.0% ( +0.2pp)  SELL OS share

  MCTS Visits (top 8, 6400 total):
     1.  5665 (88.5%) Q=+0.838 ███████████████████████████████████ PASS (INVEST)
     2.   291 ( 4.5%) Q=+0.837 █ BUY DA share
     3.   194 ( 3.0%) Q=+0.833 █ BUY SI share
     4.   101 ( 1.6%) Q=+0.819  SELL DA share
     5.   100 ( 1.6%) Q=+0.826  BUY PR share
     6.    33 ( 0.5%) Q=+0.830  BUY S share
     7.    14 ( 0.2%) Q=+0.836  BUY SM share
     8.     2 ( 0.0%) Q=+0.750  SELL OS share
  A0GB Value: P0=-0.641, P1=-0.264, P2=+0.836 (depth: 37, vbackups: 4549)

  **Action: PASS (INVEST)**

### Step 229: P0 [INVEST]

  NN Values: P0=-0.414, P1=-0.414, P2=+0.840
  NN Priors (top 8 of 8 legal):
     1.  48.4% ( -9.3pp) ███████████████████ BUY DA share
     2.  33.2% ( +1.9pp) █████████████ PASS (INVEST)
     3.  12.6% ( -0.6pp) █████ SELL JS share
     4.   3.3% ( +1.2pp) █ BUY SI share
     5.   1.2% ( +1.0pp)  SELL VM share
     6.   0.6% ( +0.0pp)  BUY SM share
     7.   0.5% ( +0.8pp)  BUY S share
     8.   0.2% ( +5.0pp)  BUY PR share

  MCTS Visits (top 8, 6400 total):
     1.  4439 (69.4%) Q=-0.392 ███████████████████████████ BUY DA share
     2.  1400 (21.9%) Q=-0.391 ████████ SELL JS share
     3.   322 ( 5.0%) Q=-0.530 ██ PASS (INVEST)
     4.   110 ( 1.7%) Q=-0.448  BUY SI share
     5.    65 ( 1.0%) Q=-0.520  BUY PR share
     6.    23 ( 0.4%) Q=-0.513  BUY S share
     7.    23 ( 0.4%) Q=-0.565  SELL VM share
     8.    18 ( 0.3%) Q=-0.428  BUY SM share
  A0GB Value: P0=-0.594, P1=-0.247, P2=+0.836 (depth: 37, vbackups: 5664)

  **Action: BUY DA share**

### Step 230: P1 [INVEST]

  NN Values: P0=-0.314, P1=-0.531, P2=+0.844
  NN Priors (top 8 of 8 legal):
     1.  86.0% (-10.0pp) ██████████████████████████████████ PASS (INVEST)
     2.   5.0% ( +3.4pp) ██ BUY DA share
     3.   3.3% ( +0.3pp) █ BUY S share
     4.   2.7% ( -0.0pp) █ BUY SI share
     5.   1.0% ( +1.0pp)  BUY SM share
     6.   1.0% ( +1.7pp)  BUY OS share
     7.   0.7% ( +3.4pp)  BUY PR share
     8.   0.3% ( +0.1pp)  SELL VM share

  MCTS Visits (top 8, 6400 total):
     1.  6075 (94.9%) Q=-0.428 █████████████████████████████████████ PASS (INVEST)
     2.   120 ( 1.9%) Q=-0.520  BUY DA share
     3.    59 ( 0.9%) Q=-0.493  BUY S share
     4.    50 ( 0.8%) Q=-0.492  BUY OS share
     5.    46 ( 0.7%) Q=-0.535  BUY PR share
     6.    23 ( 0.4%) Q=-0.579  BUY SM share
     7.    18 ( 0.3%) Q=-0.606  BUY SI share
     8.     9 ( 0.1%) Q=-0.599  SELL VM share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 36, vbackups: 4435)

  **Action: PASS (INVEST)**

### Step 231: P2 [INVEST]

  NN Values: P0=-0.309, P1=-0.531, P2=+0.828
  NN Priors (top 7 of 7 legal):
     1.  99.7% (-14.7pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.1% ( +0.4pp)  SELL DA share
     3.   0.0% ( +1.4pp)  BUY SI share
     4.   0.0% ( +2.9pp)  BUY SM share
     5.   0.0% ( +4.8pp)  BUY S share
     6.   0.0% ( +2.3pp)  BUY PR share
     7.   0.0% ( +3.0pp)  SELL OS share

  MCTS Visits (top 7, 6400 total):
     1.  5977 (93.4%) Q=+0.839 █████████████████████████████████████ PASS (INVEST)
     2.   158 ( 2.5%) Q=+0.833  BUY S share
     3.    90 ( 1.4%) Q=+0.823  BUY SM share
     4.    75 ( 1.2%) Q=+0.828  BUY PR share
     5.    53 ( 0.8%) Q=+0.827  BUY SI share
     6.    25 ( 0.4%) Q=+0.828  SELL DA share
     7.    22 ( 0.3%) Q=+0.682  SELL OS share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 35, vbackups: 5986)

  **Action: PASS (INVEST)**

### Step 232: P0 [INVEST]

  NN Values: P0=-0.322, P1=-0.496, P2=+0.840
  NN Priors (top 9 of 9 legal):
     1.  99.7% (-11.3pp) ███████████████████████████████████████ BUY DA share
     2.   0.1% ( +0.7pp)  SELL DA share
     3.   0.0% ( +1.7pp)  SELL JS share
     4.   0.0% ( +2.7pp)  BUY SI share
     5.   0.0% ( +0.5pp)  PASS (INVEST)
     6.   0.0% ( +4.4pp)  BUY S share
     7.   0.0% ( +0.1pp)  BUY SM share
     8.   0.0% ( +0.2pp)  BUY PR share
     9.   0.0% ( +0.9pp)  SELL VM share

  MCTS Visits (top 9, 6400 total):
     1.  6180 (96.6%) Q=-0.394 ██████████████████████████████████████ BUY DA share
     2.    98 ( 1.5%) Q=-0.436  BUY S share
     3.    49 ( 0.8%) Q=-0.453  BUY SI share
     4.    33 ( 0.5%) Q=-0.453  SELL JS share
     5.    21 ( 0.3%) Q=-0.456  SELL DA share
     6.     9 ( 0.1%) Q=-0.509  SELL VM share
     7.     8 ( 0.1%) Q=-0.451  PASS (INVEST)
     8.     1 ( 0.0%) Q=-0.508  BUY SM share
     9.     1 ( 0.0%) Q=-0.539  BUY PR share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 34, vbackups: 6064)

  **Action: BUY DA share**

### Step 233: P1 [INVEST]

  NN Values: P0=-0.395, P1=-0.391, P2=+0.828
  NN Priors (top 8 of 8 legal):
     1.  91.8% (-12.6pp) ████████████████████████████████████ PASS (INVEST)
     2.   3.9% ( +1.6pp) █ SELL VM share
     3.   1.4% ( +4.9pp)  BUY S share
     4.   0.9% ( +2.5pp)  BUY SI share
     5.   0.7% ( +1.6pp)  BUY DA share
     6.   0.5% ( +0.8pp)  BUY SM share
     7.   0.4% ( +0.2pp)  BUY OS share
     8.   0.3% ( +1.0pp)  BUY PR share

  MCTS Visits (top 8, 6400 total):
     1.  5814 (90.8%) Q=-0.427 ████████████████████████████████████ PASS (INVEST)
     2.   395 ( 6.2%) Q=-0.429 ██ SELL VM share
     3.   103 ( 1.6%) Q=-0.492  BUY S share
     4.    23 ( 0.4%) Q=-0.557  BUY DA share
     5.    22 ( 0.3%) Q=-0.632  BUY SI share
     6.    16 ( 0.2%) Q=-0.553  BUY PR share
     7.    15 ( 0.2%) Q=-0.461  BUY OS share
     8.    12 ( 0.2%) Q=-0.600  BUY SM share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 33, vbackups: 6016)

  **Action: PASS (INVEST)**

### Step 234: P2 [INVEST]

  NN Values: P0=-0.424, P1=-0.389, P2=+0.824
  NN Priors (top 7 of 7 legal):
     1.  99.5% (-10.2pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.3% ( +2.8pp)  SELL DA share
     3.   0.1% ( +0.1pp)  BUY SI share
     4.   0.1% ( +0.3pp)  BUY SM share
     5.   0.1% ( +3.4pp)  BUY S share
     6.   0.0% ( +0.0pp)  BUY PR share
     7.   0.0% ( +3.6pp)  SELL OS share

  MCTS Visits (top 7, 6400 total):
     1.  6086 (95.1%) Q=+0.840 ██████████████████████████████████████ PASS (INVEST)
     2.   138 ( 2.2%) Q=+0.834  SELL DA share
     3.   129 ( 2.0%) Q=+0.828  BUY S share
     4.    30 ( 0.5%) Q=+0.698  SELL OS share
     5.    11 ( 0.2%) Q=+0.819  BUY SM share
     6.     4 ( 0.1%) Q=+0.817  BUY SI share
     7.     2 ( 0.0%) Q=+0.822  BUY PR share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 32, vbackups: 5976)

  **Action: PASS (INVEST)**

### Step 235: P0 [INVEST]

  NN Values: P0=-0.471, P1=-0.338, P2=+0.828
  NN Priors (top 4 of 4 legal):
     1.  80.2% ( -9.6pp) ████████████████████████████████ PASS (INVEST)
     2.  18.3% ( +0.8pp) ███████ SELL DA share
     3.   1.0% ( +7.7pp)  SELL JS share
     4.   0.5% ( +1.1pp)  SELL VM share

  MCTS Visits (top 4, 6400 total):
     1.  5192 (81.1%) Q=-0.392 ████████████████████████████████ PASS (INVEST)
     2.   940 (14.7%) Q=-0.378 █████ SELL JS share
     3.   251 ( 3.9%) Q=-0.474 █ SELL DA share
     4.    17 ( 0.3%) Q=-0.537  SELL VM share
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 31, vbackups: 5466)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 9  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $16 (NW $118) order=1 income=$10  companies=[HR]  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $68 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$14 stars=22 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$-3 stars=8 pres=P0  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($68)

### Step 236: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($68)

  NN Values: P0=-0.338, P1=-0.414, P2=+0.836
  NN Priors (top 2 of 2 legal):
     1.  94.3% ( -1.5pp) █████████████████████████████████████ ACQ select OS
     2.   5.7% ( +1.5pp) ██ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  5805 (90.7%) Q=+0.839 ████████████████████████████████████ ACQ select OS
     2.   595 ( 9.3%) Q=+0.845 ███ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 30, vbackups: 5810)

  **Action: ACQ select OS**

  ↳ auto: ACQ target HR (with OS)

Phase: ACQ_SELECT_PRICE  |  Turn: 9  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $16 (NW $118) order=1 income=$10  companies=[HR]  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $68 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$14 stars=22 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$-3 stars=8 pres=P0  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 OS -> HR (price range $24-$62)

### Step 237: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 OS -> HR (price range $24-$62)

  NN Values: P0=-0.352, P1=-0.404, P2=+0.832
  NN Priors (top 10 of 39 legal):
     1.  99.7% (-14.9pp) ███████████████████████████████████████ ACQUIRE HR with OS @ $62
     2.   0.3% ( -0.1pp)  ACQUIRE HR with OS @ $61
     3.   0.0% ( +0.0pp)  ACQUIRE HR with OS @ $60
     4.   0.0% ( -0.0pp)  ACQUIRE HR with OS @ $59
     5.   0.0% ( +0.0pp)  ACQUIRE HR with OS @ $58
     6.   0.0% ( -0.0pp)  ACQUIRE HR with OS @ $57
     7.   0.0% ( +0.0pp)  ACQUIRE HR with OS @ $56
     8.   0.0% ( +0.1pp)  ACQUIRE HR with OS @ $55
     9.   0.0% ( +0.6pp)  ACQUIRE HR with OS @ $54
    10.   0.0% ( -0.0pp)  ACQUIRE HR with OS @ $26

  MCTS Visits (top 10, 6400 total):
     1.  5928 (92.6%) Q=+0.839 █████████████████████████████████████ ACQUIRE HR with OS @ $62
     2.   265 ( 4.1%) Q=+0.815 █ ACQUIRE HR with OS @ $36
     3.    99 ( 1.5%) Q=+0.823  ACQUIRE HR with OS @ $49
     4.    28 ( 0.4%) Q=+0.818  ACQUIRE HR with OS @ $44
     5.    19 ( 0.3%) Q=+0.819  ACQUIRE HR with OS @ $54
     6.    12 ( 0.2%) Q=+0.825  ACQUIRE HR with OS @ $39
     7.     8 ( 0.1%) Q=+0.825  ACQUIRE HR with OS @ $61
     8.     8 ( 0.1%) Q=+0.824  ACQUIRE HR with OS @ $42
     9.     7 ( 0.1%) Q=+0.819  ACQUIRE HR with OS @ $52
    10.     7 ( 0.1%) Q=+0.819  ACQUIRE HR with OS @ $50
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 29, vbackups: 5800)

  **Action: ACQUIRE HR with OS @ $62**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$-3 stars=8 pres=P0  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with JS($21), DA($23), VM($18)

### Step 238: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($21), DA($23), VM($18)

  NN Values: P0=-0.428, P1=-0.398, P2=+0.824
  NN Priors (top 4 of 4 legal):
     1.  91.8% (-12.7pp) ████████████████████████████████████ ACQ select DA
     2.   6.6% ( +6.3pp) ██ ACQ select JS
     3.   1.4% ( +5.4pp)  ACQ select VM
     4.   0.1% ( +1.1pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 4, 6400 total):
     1.  5863 (91.6%) Q=-0.395 ████████████████████████████████████ ACQ select DA
     2.   341 ( 5.3%) Q=-0.409 ██ ACQ select VM
     3.   182 ( 2.8%) Q=-0.476 █ ACQ select JS
     4.    14 ( 0.2%) Q=-0.504  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 28, vbackups: 5927)

  **Action: ACQ select DA**

Phase: ACQ_SELECT_COMPANY  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$-3 stars=8 pres=P0  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with DA ($23)

### Step 239: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with DA ($23)

  NN Values: P0=-0.410, P1=-0.471, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  99.7% ( -6.6pp) ███████████████████████████████████████ ACQ target BR (with DA)
     2.   0.3% ( +6.6pp)  ACQ target PKP (with DA)

  MCTS Visits (top 2, 6400 total):
     1.  6289 (98.3%) Q=-0.395 ███████████████████████████████████████ ACQ target BR (with DA)
     2.   111 ( 1.7%) Q=-0.466  ACQ target PKP (with DA)
  A0GB Value: P0=-0.574, P1=-0.229, P2=+0.840 (depth: 29, vbackups: 5862)

  **Action: ACQ target BR (with DA)**

Phase: ACQ_SELECT_PRICE  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$5 stars=9 pres=P0  companies=[PKP, BR]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $23 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$-3 stars=8 pres=P0  companies=[AKE, PR, NS]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 DA -> BR (price range $17-$45)

### Step 240: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 DA -> BR (price range $17-$45)

  NN Values: P0=-0.414, P1=-0.463, P2=+0.824
  NN Priors (top 7 of 7 legal):
     1.  97.8% (-14.2pp) ███████████████████████████████████████ ACQUIRE BR with DA @ $17
     2.   1.0% ( +0.5pp)  ACQUIRE BR with DA @ $18
     3.   0.4% ( +1.8pp)  ACQUIRE BR with DA @ $21
     4.   0.3% ( +0.1pp)  ACQUIRE BR with DA @ $20
     5.   0.2% ( +6.6pp)  ACQUIRE BR with DA @ $19
     6.   0.2% ( +1.5pp)  ACQUIRE BR with DA @ $22
     7.   0.1% ( +3.7pp)  ACQUIRE BR with DA @ $23

  MCTS Visits (top 7, 6400 total):
     1.  5099 (79.7%) Q=-0.396 ███████████████████████████████ ACQUIRE BR with DA @ $17
     2.   440 ( 6.9%) Q=-0.395 ██ ACQUIRE BR with DA @ $19
     3.   355 ( 5.5%) Q=-0.380 ██ ACQUIRE BR with DA @ $18
     4.   193 ( 3.0%) Q=-0.389 █ ACQUIRE BR with DA @ $21
     5.   172 ( 2.7%) Q=-0.406 █ ACQUIRE BR with DA @ $23
     6.   123 ( 1.9%) Q=-0.393  ACQUIRE BR with DA @ $22
     7.    18 ( 0.3%) Q=-0.408  ACQUIRE BR with DA @ $20
  A0GB Value: P0=-0.574, P1=-0.229, P2=+0.840 (depth: 28, vbackups: 5472)

  **Action: ACQUIRE BR with DA @ $17**

Phase: ACQ_SELECT_CORP  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-2 stars=5 pres=P0  companies=[PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$6 stars=10 pres=P0  companies=[AKE, PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with JS($21), DA($6), VM($18)

### Step 241: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($21), DA($6), VM($18)

  NN Values: P0=-0.404, P1=-0.449, P2=+0.828
  NN Priors (top 3 of 3 legal):
     1.  50.1% ( +2.2pp) ████████████████████ ACQ select VM
     2.  49.3% ( -7.1pp) ███████████████████ ACQ select JS
     3.   0.5% ( +4.9pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 3, 6400 total):
     1.  3398 (53.1%) Q=-0.394 █████████████████████ ACQ select JS
     2.  2885 (45.1%) Q=-0.401 ██████████████████ ACQ select VM
     3.   117 ( 1.8%) Q=-0.444  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.412, P1=-0.441, P2=+0.848 (depth: 25, vbackups: 5914)

  **Action: ACQ select JS**

Phase: ACQ_SELECT_COMPANY  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-2 stars=5 pres=P0  companies=[PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$6 stars=10 pres=P0  companies=[AKE, PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with JS ($21)

### Step 242: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with JS ($21)

  NN Values: P0=-0.396, P1=-0.449, P2=+0.828
  NN Priors (top 3 of 3 legal):
     1.  99.6% (-14.1pp) ███████████████████████████████████████ ACQ target AKE (with JS)
     2.   0.4% ( +8.8pp)  ACQ target PR (with JS)
     3.   0.0% ( +5.2pp)  ACQ target NS (with JS)

  MCTS Visits (top 3, 6400 total):
     1.  5957 (93.1%) Q=-0.397 █████████████████████████████████████ ACQ target AKE (with JS)
     2.   354 ( 5.5%) Q=-0.413 ██ ACQ target PR (with JS)
     3.    89 ( 1.4%) Q=-0.466  ACQ target NS (with JS)
  A0GB Value: P0=-0.660, P1=-0.192, P2=+0.840 (depth: 30, vbackups: 3397)

  **Action: ACQ target AKE (with JS)**

Phase: ACQ_SELECT_PRICE  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $21 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-2 stars=5 pres=P0  companies=[PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$6 stars=10 pres=P0  companies=[AKE, PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 JS -> AKE (price range $3-$8)

### Step 243: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 JS -> AKE (price range $3-$8)

  NN Values: P0=-0.391, P1=-0.455, P2=+0.832
  NN Priors (top 6 of 6 legal):
     1.  96.7% (-14.2pp) ██████████████████████████████████████ ACQUIRE AKE with JS @ $8
     2.   1.2% ( +0.7pp)  ACQUIRE AKE with JS @ $7
     3.   1.0% ( +1.7pp)  ACQUIRE AKE with JS @ $4
     4.   0.6% (+10.0pp)  ACQUIRE AKE with JS @ $3
     5.   0.4% ( +0.1pp)  ACQUIRE AKE with JS @ $5
     6.   0.2% ( +1.9pp)  ACQUIRE AKE with JS @ $6

  MCTS Visits (top 6, 6400 total):
     1.  5364 (83.8%) Q=-0.398 █████████████████████████████████ ACQUIRE AKE with JS @ $8
     2.   305 ( 4.8%) Q=-0.425 █ ACQUIRE AKE with JS @ $3
     3.   219 ( 3.4%) Q=-0.390 █ ACQUIRE AKE with JS @ $6
     4.   219 ( 3.4%) Q=-0.394 █ ACQUIRE AKE with JS @ $4
     5.   195 ( 3.0%) Q=-0.391 █ ACQUIRE AKE with JS @ $7
     6.    98 ( 1.5%) Q=-0.385  ACQUIRE AKE with JS @ $5
  A0GB Value: P0=-0.660, P1=-0.192, P2=+0.840 (depth: 29, vbackups: 5800)

  **Action: ACQUIRE AKE with JS @ $8**

Phase: ACQ_SELECT_CORP  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-7 stars=5 pres=P0  companies=[AKE*, PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with JS($13), DA($6), VM($18)

### Step 244: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($13), DA($6), VM($18)

  NN Values: P0=-0.371, P1=-0.461, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  99.1% (-13.7pp) ███████████████████████████████████████ ACQ select VM
     2.   0.9% ( +7.7pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +6.1pp)  ACQ select JS

  MCTS Visits (top 3, 6400 total):
     1.  5495 (85.9%) Q=-0.398 ██████████████████████████████████ ACQ select VM
     2.   559 ( 8.7%) Q=-0.399 ███ PASS (ACQ_SELECT_CORP)
     3.   346 ( 5.4%) Q=-0.402 ██ ACQ select JS
  A0GB Value: P0=-0.660, P1=-0.192, P2=+0.840 (depth: 28, vbackups: 5519)

  **Action: ACQ select VM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-7 stars=5 pres=P0  companies=[AKE*, PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with VM ($18)

### Step 245: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with VM ($18)

  NN Values: P0=-0.332, P1=-0.492, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  96.9% ( -7.1pp) ██████████████████████████████████████ ACQ target PKP (with VM)
     2.   2.9% ( +1.3pp) █ ACQ target PR (with VM)
     3.   0.2% ( +5.7pp)  ACQ target NS (with VM)

  MCTS Visits (top 3, 6400 total):
     1.  3388 (52.9%) Q=-0.408 █████████████████████ ACQ target PKP (with VM)
     2.  2828 (44.2%) Q=-0.371 █████████████████ ACQ target PR (with VM)
     3.   184 ( 2.9%) Q=-0.416 █ ACQ target NS (with VM)
  A0GB Value: P0=-0.660, P1=-0.192, P2=+0.840 (depth: 27, vbackups: 5149)

  **Action: ACQ target PKP (with VM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-7 stars=5 pres=P0  companies=[AKE*, PKP]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $18 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$2 stars=2 pres=P0  companies=[MHE]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 VM -> PKP (price range $13-$33)

### Step 246: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> PKP (price range $13-$33)

  NN Values: P0=-0.355, P1=-0.477, P2=+0.820
  NN Priors (top 6 of 6 legal):
     1.  91.2% (-13.6pp) ████████████████████████████████████ ACQUIRE PKP with VM @ $13
     2.   3.5% ( +0.2pp) █ ACQUIRE PKP with VM @ $14
     3.   1.7% ( +2.0pp)  ACQUIRE PKP with VM @ $17
     4.   1.5% ( +4.8pp)  ACQUIRE PKP with VM @ $18
     5.   1.1% ( +6.5pp)  ACQUIRE PKP with VM @ $16
     6.   1.0% ( +0.1pp)  ACQUIRE PKP with VM @ $15

  MCTS Visits (top 6, 6400 total):
     1.  3269 (51.1%) Q=-0.418 ████████████████████ ACQUIRE PKP with VM @ $13
     2.   910 (14.2%) Q=-0.396 █████ ACQUIRE PKP with VM @ $18
     3.   816 (12.8%) Q=-0.399 █████ ACQUIRE PKP with VM @ $16
     4.   689 (10.8%) Q=-0.393 ████ ACQUIRE PKP with VM @ $17
     5.   411 ( 6.4%) Q=-0.399 ██ ACQUIRE PKP with VM @ $14
     6.   305 ( 4.8%) Q=-0.392 █ ACQUIRE PKP with VM @ $15
  A0GB Value: P0=-0.660, P1=-0.192, P2=+0.840 (depth: 26, vbackups: 3732)

  **Action: ACQUIRE PKP with VM @ $13**

Phase: ACQ_SELECT_CORP  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-5 stars=2 pres=P0  companies=[AKE*]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $5 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$3 stars=4 pres=P0  companies=[MHE, PKP*]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with JS($13), DA($6), VM($5)

### Step 247: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with JS($13), DA($6), VM($5)

  NN Values: P0=-0.346, P1=-0.520, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  55.4% ( +0.8pp) ██████████████████████ PASS (ACQ_SELECT_CORP)
     2.  22.6% ( -1.7pp) █████████ ACQ select DA
     3.  21.9% ( +0.9pp) ████████ ACQ select JS

  MCTS Visits (top 3, 6400 total):
     1.  2849 (44.5%) Q=-0.411 █████████████████ ACQ select JS
     2.  2594 (40.5%) Q=-0.429 ████████████████ PASS (ACQ_SELECT_CORP)
     3.   957 (15.0%) Q=-0.429 █████ ACQ select DA
  A0GB Value: P0=-0.455, P1=-0.455, P2=+0.820 (depth: 29, vbackups: 3268)

  **Action: ACQ select JS**

Phase: ACQ_SELECT_COMPANY  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-5 stars=2 pres=P0  companies=[AKE*]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $5 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$3 stars=4 pres=P0  companies=[MHE, PKP*]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with JS ($13)

### Step 248: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with JS ($13)

  NN Values: P0=-0.371, P1=-0.477, P2=+0.828
  NN Priors (top 3 of 3 legal):
     1.  99.4% (-13.0pp) ███████████████████████████████████████ ACQ target MHE (with JS)
     2.   0.6% ( +0.8pp)  ACQ target PR (with JS)
     3.   0.0% (+12.2pp)  ACQ target NS (with JS)

  MCTS Visits (top 3, 6400 total):
     1.  5783 (90.4%) Q=-0.416 ████████████████████████████████████ ACQ target MHE (with JS)
     2.   319 ( 5.0%) Q=-0.404 █ ACQ target PR (with JS)
     3.   298 ( 4.7%) Q=-0.455 █ ACQ target NS (with JS)
  A0GB Value: P0=-0.559, P1=-0.377, P2=+0.828 (depth: 37, vbackups: 2848)

  **Action: ACQ target MHE (with JS)**

Phase: ACQ_SELECT_PRICE  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $13 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-5 stars=2 pres=P0  companies=[AKE*]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR*]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $6 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=9 pres=P0  companies=[PR, NS, BR*]
  VM: $5 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$3 stars=4 pres=P0  companies=[MHE, PKP*]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 JS -> MHE (price range $4-$10)

### Step 249: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 JS -> MHE (price range $4-$10)

  NN Values: P0=-0.387, P1=-0.465, P2=+0.832
  NN Priors (top 7 of 7 legal):
     1.  43.1% ( -2.8pp) █████████████████ ACQUIRE MHE with JS @ $10
     2.  33.6% ( -4.3pp) █████████████ ACQUIRE MHE with JS @ $4
     3.  10.2% ( -1.1pp) ████ ACQUIRE MHE with JS @ $9
     4.   4.5% ( +8.0pp) █ ACQUIRE MHE with JS @ $5
     5.   4.0% ( +0.7pp) █ ACQUIRE MHE with JS @ $8
     6.   2.4% ( -0.3pp)  ACQUIRE MHE with JS @ $7
     7.   2.1% ( -0.2pp)  ACQUIRE MHE with JS @ $6

  MCTS Visits (top 7, 6400 total):
     1.  2697 (42.1%) Q=-0.415 ████████████████ ACQUIRE MHE with JS @ $10
     2.  1314 (20.5%) Q=-0.425 ████████ ACQUIRE MHE with JS @ $4
     3.   772 (12.1%) Q=-0.417 ████ ACQUIRE MHE with JS @ $5
     4.   724 (11.3%) Q=-0.412 ████ ACQUIRE MHE with JS @ $9
     5.   339 ( 5.3%) Q=-0.414 ██ ACQUIRE MHE with JS @ $8
     6.   286 ( 4.5%) Q=-0.407 █ ACQUIRE MHE with JS @ $6
     7.   268 ( 4.2%) Q=-0.406 █ ACQUIRE MHE with JS @ $7
  A0GB Value: P0=-0.613, P1=-0.256, P2=+0.836 (depth: 38, vbackups: 5711)

  **Action: ACQUIRE MHE with JS @ $10**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $48 (NW $90) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $44 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $33 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-9 stars=5 pres=P0  companies=[AKE, MHE]
  S: $17 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $6 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=21 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $11 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=3 RECEIVERSHIP  companies=[BD]
  PR: $10 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $14 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=10 pres=P0  companies=[PR, NS, BR]
  VM: $15 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $1 price=$8(idx 4) shares=bank:4/unissued:0/issued:4 income=$-5 stars=3 RECEIVERSHIP  companies=[KME]

**Deck**: 0 remaining

**Closing**: P0 may close AKE (JS), MHE (JS), PR (DA), NS (DA), BR (DA)

### Step 250: P0 [CLOSING]

  **Closing**: P0 may close AKE (JS), MHE (JS), PR (DA), NS (DA), BR (DA)

  NN Values: P0=-0.350, P1=-0.449, P2=+0.824
  NN Priors (top 5 of 5 legal):
     1.  98.3% ( -6.5pp) ███████████████████████████████████████ CLOSE AKE
     2.   1.4% ( +1.9pp)  CLOSE MHE
     3.   0.2% ( +0.6pp)  PASS (CLOSING)
     4.   0.0% ( +2.3pp)  CLOSE PR
     5.   0.0% ( +1.6pp)  CLOSE NS

  MCTS Visits (top 5, 6400 total):
     1.  5535 (86.5%) Q=-0.427 ██████████████████████████████████ CLOSE AKE
     2.   802 (12.5%) Q=-0.410 █████ CLOSE MHE
     3.    29 ( 0.5%) Q=-0.519  CLOSE PR
     4.    21 ( 0.3%) Q=-0.459  PASS (CLOSING)
     5.    13 ( 0.2%) Q=-0.582  CLOSE NS
  A0GB Value: P0=-0.455, P1=-0.389, P2=+0.852 (depth: 47, vbackups: 2696)

  **Action: CLOSE AKE**

### Step 251: P0 [CLOSING]

  **Closing**: P0 may close PR (DA), NS (DA), BR (DA)

  NN Values: P0=-0.270, P1=-0.504, P2=+0.828
  NN Priors (top 3 of 3 legal):
     1.  99.8% (-12.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.2% ( +9.4pp)  CLOSE PR
     3.   0.0% ( +3.0pp)  CLOSE NS

  MCTS Visits (top 3, 6400 total):
     1.  6256 (97.8%) Q=-0.427 ███████████████████████████████████████ PASS (CLOSING)
     2.   117 ( 1.8%) Q=-0.524  CLOSE PR
     3.    27 ( 0.4%) Q=-0.598  CLOSE NS
  A0GB Value: P0=-0.455, P1=-0.389, P2=+0.852 (depth: 46, vbackups: 5534)

  **Action: PASS (CLOSING)**

### Step 252: P2 [CLOSING]

  **Closing**: P2 may close SX (OS), KK (OS), DR (OS), SZD (OS), E (OS), HR (OS)

  NN Values: P0=-0.219, P1=-0.512, P2=+0.828
  NN Priors (top 4 of 4 legal):
     1.  99.8% ( -5.9pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +0.6pp)  CLOSE SX
     3.   0.0% ( +1.9pp)  CLOSE KK
     4.   0.0% ( +3.3pp)  CLOSE DR

  MCTS Visits (top 4, 6400 total):
     1.  6263 (97.9%) Q=+0.843 ███████████████████████████████████████ PASS (CLOSING)
     2.    71 ( 1.1%) Q=+0.834  CLOSE DR
     3.    45 ( 0.7%) Q=+0.833  CLOSE KK
     4.    21 ( 0.3%) Q=+0.834  CLOSE SX
  A0GB Value: P0=-0.455, P1=-0.389, P2=+0.852 (depth: 45, vbackups: 6255)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $2 (NW $94) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $55 (NW $97) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $133) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $32 price=$45(idx 21) shares=bank:1/unissued:5/issued:2 income=$-5 stars=4 pres=P0  companies=[MHE]
  S: $11 price=$13(idx 9) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$7(idx 3) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $7 price=$12(idx 8) shares=bank:4/unissued:1/issued:4 income=$-3 stars=2 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$18(idx 12) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $20 price=$11(idx 7) shares=bank:0/unissued:2/issued:2 income=$5 stars=5 pres=P0  companies=[PKP]

**Deck**: 0 remaining

**Dividends**: JS

### Step 253: P0 [DIVIDENDS]

  **Dividends**: JS

  NN Values: P0=-0.285, P1=-0.512, P2=+0.836
  NN Priors (top 10 of 16 legal):
     1.  99.4% (-13.6pp) ███████████████████████████████████████ DIVIDEND $15
     2.   0.6% ( +3.9pp)  DIVIDEND $14
     3.   0.0% ( +1.6pp)  DIVIDEND $13
     4.   0.0% ( +0.0pp)  DIVIDEND $12
     5.   0.0% ( +0.0pp)  DIVIDEND $9
     6.   0.0% ( +1.8pp)  DIVIDEND $11
     7.   0.0% ( +0.7pp)  DIVIDEND $10
     8.   0.0% ( +0.0pp)  DIVIDEND $8
     9.   0.0% ( +0.4pp)  DIVIDEND $7
    10.   0.0% ( +0.6pp)  DIVIDEND $6

  MCTS Visits (top 10, 6400 total):
     1.  5786 (90.4%) Q=-0.427 ████████████████████████████████████ DIVIDEND $15
     2.   360 ( 5.6%) Q=-0.424 ██ DIVIDEND $14
     3.    89 ( 1.4%) Q=-0.432  DIVIDEND $13
     4.    57 ( 0.9%) Q=-0.453  DIVIDEND $11
     5.    33 ( 0.5%) Q=-0.447  DIVIDEND $10
     6.    25 ( 0.4%) Q=-0.534  DIVIDEND $5
     7.     9 ( 0.1%) Q=-0.567  DIVIDEND $4
     8.     9 ( 0.1%) Q=-0.625  DIVIDEND $1
     9.     9 ( 0.1%) Q=-0.585  DIVIDEND $3
    10.     9 ( 0.1%) Q=-0.534  DIVIDEND $6
  A0GB Value: P0=-0.455, P1=-0.389, P2=+0.852 (depth: 44, vbackups: 5880)

  **Action: DIVIDEND $15**

### Step 254: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.400, P1=-0.406, P2=+0.828
  NN Priors (top 8 of 8 legal):
     1.  99.0% (-14.8pp) ███████████████████████████████████████ DIVIDEND $0
     2.   0.9% ( +3.9pp)  DIVIDEND $2
     3.   0.1% ( +0.4pp)  DIVIDEND $1
     4.   0.0% ( +2.1pp)  DIVIDEND $3
     5.   0.0% ( +7.1pp)  DIVIDEND $5
     6.   0.0% ( +1.0pp)  DIVIDEND $6
     7.   0.0% ( +0.2pp)  DIVIDEND $4
     8.   0.0% ( +0.1pp)  DIVIDEND $7

  MCTS Visits (top 8, 6400 total):
     1.  5507 (86.0%) Q=+0.843 ██████████████████████████████████ DIVIDEND $0
     2.   376 ( 5.9%) Q=+0.844 ██ DIVIDEND $5
     3.   287 ( 4.5%) Q=+0.847 █ DIVIDEND $2
     4.   120 ( 1.9%) Q=+0.847  DIVIDEND $3
     5.    59 ( 0.9%) Q=+0.841  DIVIDEND $6
     6.    41 ( 0.6%) Q=+0.847  DIVIDEND $1
     7.     8 ( 0.1%) Q=+0.833  DIVIDEND $4
     8.     2 ( 0.0%) Q=+0.830  DIVIDEND $7
  A0GB Value: P0=-0.455, P1=-0.389, P2=+0.852 (depth: 43, vbackups: 5586)

  **Action: DIVIDEND $0**

### Step 255: P0 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.377, P1=-0.465, P2=+0.836
  NN Priors (top 5 of 5 legal):
     1.  39.3% ( -1.2pp) ███████████████ DIVIDEND $0
     2.  29.7% ( +3.3pp) ███████████ DIVIDEND $4
     3.  21.7% ( -3.4pp) ████████ DIVIDEND $2
     4.   7.3% ( -1.0pp) ██ DIVIDEND $3
     5.   2.0% ( +2.3pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  3593 (56.1%) Q=-0.421 ██████████████████████ DIVIDEND $0
     2.  1606 (25.1%) Q=-0.433 ██████████ DIVIDEND $4
     3.   866 (13.5%) Q=-0.435 █████ DIVIDEND $2
     4.   181 ( 2.8%) Q=-0.440 █ DIVIDEND $1
     5.   154 ( 2.4%) Q=-0.461  DIVIDEND $3
  A0GB Value: P0=-0.531, P1=-0.307, P2=+0.863 (depth: 43, vbackups: 5959)

  **Action: DIVIDEND $0**

### Step 256: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.283, P1=-0.520, P2=+0.836
  NN Priors (top 4 of 4 legal):
     1.  95.0% (-12.1pp) █████████████████████████████████████ DIVIDEND $3
     2.   2.3% ( +4.8pp)  DIVIDEND $0
     3.   2.1% ( +1.7pp)  DIVIDEND $2
     4.   0.6% ( +5.7pp)  DIVIDEND $1

  MCTS Visits (top 4, 6400 total):
     1.  3517 (55.0%) Q=-0.448 █████████████████████ DIVIDEND $3
     2.  1481 (23.1%) Q=-0.424 █████████ DIVIDEND $0
     3.   808 (12.6%) Q=-0.428 █████ DIVIDEND $1
     4.   594 ( 9.3%) Q=-0.425 ███ DIVIDEND $2
  A0GB Value: P0=-0.641, P1=-0.273, P2=+0.855 (depth: 46, vbackups: 3592)

  **Action: DIVIDEND $3**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 9  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $20 (NW $110) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $58 (NW $102) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $145) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$33(idx 18) shares=bank:1/unissued:5/issued:2 income=$-5 stars=1 pres=P0  companies=[MHE]
  S: $11 price=$10(idx 6) shares=bank:3/unissued:4/issued:3 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $7 price=$9(idx 5) shares=bank:4/unissued:1/issued:4 income=$-3 stars=2 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]

**Deck**: 0 remaining

**Issue**: JS

### Step 257: P0 [ISSUE_SHARES]

  **Issue**: JS

  NN Values: P0=-0.402, P1=-0.426, P2=+0.844
  NN Priors (top 2 of 2 legal):
     1.  97.4% ( -6.8pp) ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   2.6% ( +6.8pp) █ ISSUE JS shares

  MCTS Visits (top 2, 6400 total):
     1.  6179 (96.5%) Q=-0.463 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   221 ( 3.5%) Q=-0.504 █ ISSUE JS shares
  A0GB Value: P0=-0.387, P1=-0.490, P2=+0.840 (depth: 53, vbackups: 3516)

  **Action: PASS (ISSUE_SHARES)**

### Step 258: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.414, P1=-0.404, P2=+0.844
  NN Priors (top 2 of 2 legal):
     1.  83.4% (-11.1pp) █████████████████████████████████ PASS (ISSUE_SHARES)
     2.  16.6% (+11.1pp) ██████ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6152 (96.1%) Q=-0.461 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   248 ( 3.9%) Q=-0.597 █ ISSUE VM shares
  A0GB Value: P0=-0.365, P1=-0.383, P2=+0.848 (depth: 53, vbackups: 6178)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 9  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $20 (NW $110) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $58 (NW $102) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $145) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$33(idx 18) shares=bank:1/unissued:5/issued:2 income=$-5 stars=1 pres=P0  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]

**Deck**: 0 remaining

**IPO**: SJ

### Step 259: P1 [IPO]

  **IPO**: SJ

  NN Values: P0=-0.426, P1=-0.402, P2=+0.848
  NN Priors (top 2 of 2 legal):
     1.  94.7% ( -6.9pp) █████████████████████████████████████ IPO SJ → float SI
     2.   5.3% ( +6.9pp) ██ PASS (IPO)

  MCTS Visits (top 2, 6400 total):
     1.  6280 (98.1%) Q=-0.377 ███████████████████████████████████████ IPO SJ → float SI
     2.   120 ( 1.9%) Q=-0.507  PASS (IPO)
  A0GB Value: P0=-0.365, P1=-0.383, P2=+0.848 (depth: 52, vbackups: 6151)

  **Action: IPO SJ → float SI**

Phase: PAR  |  Turn: 9  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $20 (NW $110) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $58 (NW $102) order=0 income=$7  companies=[SJ]  shares=[VM=1]
  P2: $78 (NW $145) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$33(idx 18) shares=bank:1/unissued:5/issued:2 income=$-5 stars=1 pres=P0  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]

**Deck**: 0 remaining

**PAR**: SJ -> SI

### Step 260: P1 [PAR]

  **PAR**: SJ -> SI

  NN Values: P0=-0.443, P1=-0.387, P2=+0.848
  NN Priors (top 4 of 4 legal):
     1.  92.6% (-10.2pp) █████████████████████████████████████ PAR SI @$24 (IPO SJ)
     2.   5.4% ( +4.4pp) ██ PAR SI @$37 (IPO SJ)
     3.   1.2% ( +0.2pp)  PAR SI @$30 (IPO SJ)
     4.   0.7% ( +5.5pp)  PAR SI @$27 (IPO SJ)

  MCTS Visits (top 4, 6400 total):
     1.  6105 (95.4%) Q=-0.375 ██████████████████████████████████████ PAR SI @$24 (IPO SJ)
     2.   190 ( 3.0%) Q=-0.430 █ PAR SI @$37 (IPO SJ)
     3.    89 ( 1.4%) Q=-0.457  PAR SI @$27 (IPO SJ)
     4.    16 ( 0.2%) Q=-0.475  PAR SI @$30 (IPO SJ)
  A0GB Value: P0=-0.365, P1=-0.383, P2=+0.848 (depth: 51, vbackups: 6200)

  **Action: PAR SI @$24 (IPO SJ)**

--- Turn 10 ---

Phase: INVEST  |  Turn: 10  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $20 (NW $110) order=2 income=$0  shares=[JS=1 (pres), DA=2 (pres), VM=1 (pres)]
  P1: $41 (NW $102) order=0 income=$0  shares=[VM=1, SI=2 (pres)]
  P2: $78 (NW $145) order=1 income=$0  shares=[OS=1 (pres), DA=1]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$33(idx 18) shares=bank:1/unissued:5/issued:2 income=$-5 stars=1 pres=P0  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$24(idx 15) shares=bank:2/unissued:0/issued:4 income=$7 stars=12 pres=P1  companies=[SJ]

**Deck**: 0 remaining


### Step 261: P1 [INVEST]

  NN Values: P0=-0.393, P1=-0.438, P2=+0.848
  NN Priors (top 9 of 9 legal):
     1.  47.5% ( -6.4pp) ███████████████████ PASS (INVEST)
     2.  47.2% ( -7.6pp) ██████████████████ SELL SI share
     3.   1.4% ( +2.5pp)  BUY SI share
     4.   1.2% ( +0.2pp)  BUY SM share
     5.   0.8% ( +4.6pp)  BUY S share
     6.   0.7% ( +1.6pp)  SELL VM share
     7.   0.6% ( +0.1pp)  BUY DA share
     8.   0.6% ( +1.3pp)  BUY PR share
     9.   0.1% ( +3.5pp)  BUY JS share

  MCTS Visits (top 9, 6400 total):
     1.  4578 (71.5%) Q=-0.366 ████████████████████████████ SELL SI share
     2.  1653 (25.8%) Q=-0.388 ██████████ PASS (INVEST)
     3.    49 ( 0.8%) Q=-0.505  BUY S share
     4.    40 ( 0.6%) Q=-0.501  BUY SI share
     5.    25 ( 0.4%) Q=-0.540  SELL VM share
     6.    25 ( 0.4%) Q=-0.549  BUY JS share
     7.    17 ( 0.3%) Q=-0.518  BUY PR share
     8.     9 ( 0.1%) Q=-0.555  BUY SM share
     9.     4 ( 0.1%) Q=-0.569  BUY DA share
  A0GB Value: P0=-0.645, P1=-0.178, P2=+0.840 (depth: 53, vbackups: 5768)

  **Action: SELL SI share**

### Step 262: P2 [INVEST]

  NN Values: P0=-0.326, P1=-0.455, P2=+0.859
  NN Priors (top 10 of 13 legal):
     1. 100.0% (-14.8pp) ███████████████████████████████████████ BUY SI share
     2.   0.0% ( +0.0pp)  AUCTION slot 2 (CDG, face $60)
     3.   0.0% ( +0.0pp)  SELL DA share
     4.   0.0% ( +5.1pp)  AUCTION slot 0 (MAD, face $50)
     5.   0.0% ( +1.9pp)  BUY PR share
     6.   0.0% ( +0.6pp)  BUY DA share
     7.   0.0% ( +1.4pp)  BUY S share
     8.   0.0% ( +0.1pp)  SELL OS share
     9.   0.0% ( +0.3pp)  BUY OS share
    10.   0.0% ( +0.2pp)  BUY SM share

  MCTS Visits (top 10, 6400 total):
     1.  5519 (86.2%) Q=+0.841 ██████████████████████████████████ BUY SI share
     2.   303 ( 4.7%) Q=+0.839 █ AUCTION slot 0 (MAD, face $50)
     3.   239 ( 3.7%) Q=+0.837 █ AUCTION slot 1 (FRA, face $56)
     4.   114 ( 1.8%) Q=+0.838  BUY PR share
     5.    84 ( 1.3%) Q=+0.838  BUY S share
     6.    63 ( 1.0%) Q=+0.841  PASS (INVEST)
     7.    42 ( 0.7%) Q=+0.843  BUY DA share
     8.    13 ( 0.2%) Q=+0.836  BUY OS share
     9.     9 ( 0.1%) Q=+0.840  BUY JS share
    10.     9 ( 0.1%) Q=+0.833  BUY SM share
  A0GB Value: P0=-0.408, P1=-0.365, P2=+0.840 (depth: 50, vbackups: 4576)

  **Action: BUY SI share**

### Step 263: P0 [INVEST]

  NN Values: P0=-0.285, P1=-0.451, P2=+0.863
  NN Priors (top 7 of 7 legal):
     1.  75.2% (-10.3pp) ██████████████████████████████ SELL JS share
     2.  19.0% ( -0.4pp) ███████ SELL DA share
     3.   3.5% ( +0.2pp) █ PASS (INVEST)
     4.   1.4% ( +5.0pp)  SELL VM share
     5.   0.4% ( +2.8pp)  BUY SM share
     6.   0.3% ( +1.6pp)  BUY S share
     7.   0.2% ( +1.0pp)  BUY PR share

  MCTS Visits (top 7, 6400 total):
     1.  3213 (50.2%) Q=-0.492 ████████████████████ SELL JS share
     2.  2808 (43.9%) Q=-0.474 █████████████████ SELL DA share
     3.   157 ( 2.5%) Q=-0.499  PASS (INVEST)
     4.    66 ( 1.0%) Q=-0.537  BUY SM share
     5.    65 ( 1.0%) Q=-0.512  BUY S share
     6.    50 ( 0.8%) Q=-0.517  BUY PR share
     7.    41 ( 0.6%) Q=-0.704  SELL VM share
  A0GB Value: P0=-0.523, P1=-0.283, P2=+0.828 (depth: 53, vbackups: 5518)

  **Action: SELL JS share**

### Step 264: P1 [INVEST]

  NN Values: P0=-0.143, P1=-0.586, P2=+0.844
  NN Priors (top 10 of 13 legal):
     1.  70.1% ( -8.5pp) ████████████████████████████ PASS (INVEST)
     2.  17.7% ( -2.4pp) ███████ AUCTION slot 0 (MAD, face $50)
     3.   6.8% ( -1.3pp) ██ BUY SI share
     4.   1.0% ( +2.4pp)  AUCTION slot 1 (FRA, face $56)
     5.   1.0% ( -0.2pp)  BUY SM share
     6.   0.8% ( -0.1pp)  BUY S share
     7.   0.7% ( +1.4pp)  AUCTION slot 2 (CDG, face $60)
     8.   0.6% ( +4.9pp)  BUY PR share
     9.   0.4% ( +3.0pp)  BUY JS share
    10.   0.3% ( +0.8pp)  SELL VM share

  MCTS Visits (top 10, 6400 total):
     1.  6010 (93.9%) Q=-0.365 █████████████████████████████████████ PASS (INVEST)
     2.   243 ( 3.8%) Q=-0.438 █ AUCTION slot 0 (MAD, face $50)
     3.    55 ( 0.9%) Q=-0.503  BUY SI share
     4.    33 ( 0.5%) Q=-0.592  BUY PR share
     5.    17 ( 0.3%) Q=-0.625  BUY JS share
     6.    15 ( 0.2%) Q=-0.641  AUCTION slot 1 (FRA, face $56)
     7.    11 ( 0.2%) Q=-0.620  AUCTION slot 2 (CDG, face $60)
     8.     9 ( 0.1%) Q=-0.592  SELL VM share
     9.     3 ( 0.0%) Q=-0.643  BUY S share
    10.     3 ( 0.0%) Q=-0.651  BUY SM share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 67, vbackups: 3200)

  **Action: PASS (INVEST)**

### Step 265: P2 [INVEST]

  NN Values: P0=-0.223, P1=-0.539, P2=+0.844
  NN Priors (top 10 of 12 legal):
     1. 100.0% (-14.7pp) ███████████████████████████████████████ BUY SI share
     2.   0.0% ( +0.0pp)  SELL DA share
     3.   0.0% ( +0.7pp)  AUCTION slot 0 (MAD, face $50)
     4.   0.0% ( +7.7pp)  PASS (INVEST)
     5.   0.0% ( +0.0pp)  BUY S share
     6.   0.0% ( +0.7pp)  BUY PR share
     7.   0.0% ( +0.0pp)  BUY SM share
     8.   0.0% ( +3.1pp)  BUY OS share
     9.   0.0% ( +1.6pp)  BUY JS share
    10.   0.0% ( +0.0pp)  BUY DA share

  MCTS Visits (top 10, 6400 total):
     1.  5639 (88.1%) Q=+0.836 ███████████████████████████████████ BUY SI share
     2.   381 ( 6.0%) Q=+0.838 ██ PASS (INVEST)
     3.   146 ( 2.3%) Q=+0.834  BUY OS share
     4.    95 ( 1.5%) Q=+0.837  BUY JS share
     5.    61 ( 1.0%) Q=+0.841  AUCTION slot 0 (MAD, face $50)
     6.    49 ( 0.8%) Q=+0.836  BUY PR share
     7.    20 ( 0.3%) Q=+0.822  SELL SI share
     8.     4 ( 0.1%) Q=+0.770  SELL OS share
     9.     4 ( 0.1%) Q=+0.846  SELL DA share
    10.     1 ( 0.0%) Q=+0.840  BUY DA share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 66, vbackups: 5642)

  **Action: BUY SI share**

### Step 266: P0 [INVEST]

  NN Values: P0=-0.480, P1=-0.330, P2=+0.840
  NN Priors (top 10 of 11 legal):
     1.  98.9% (-14.7pp) ███████████████████████████████████████ BUY SI share
     2.   0.6% ( +0.6pp)  PASS (INVEST)
     3.   0.3% ( -0.0pp)  AUCTION slot 0 (MAD, face $50)
     4.   0.1% ( +2.0pp)  SELL DA share
     5.   0.0% ( -0.0pp)  BUY SM share
     6.   0.0% ( +4.7pp)  BUY S share
     7.   0.0% ( +0.0pp)  BUY PR share
     8.   0.0% ( +0.3pp)  BUY DA share
     9.   0.0% ( +5.3pp)  BUY OS share
    10.   0.0% ( +0.0pp)  SELL VM share

  MCTS Visits (top 10, 6400 total):
     1.  6233 (97.4%) Q=-0.471 ██████████████████████████████████████ BUY SI share
     2.    46 ( 0.7%) Q=-0.608  BUY S share
     3.    41 ( 0.6%) Q=-0.626  BUY OS share
     4.    25 ( 0.4%) Q=-0.618  BUY JS share
     5.    25 ( 0.4%) Q=-0.616  SELL DA share
     6.    18 ( 0.3%) Q=-0.539  PASS (INVEST)
     7.     8 ( 0.1%) Q=-0.510  AUCTION slot 0 (MAD, face $50)
     8.     1 ( 0.0%) Q=-0.574  BUY SM share
     9.     1 ( 0.0%) Q=-0.641  BUY DA share
    10.     1 ( 0.0%) Q=-0.602  BUY PR share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 65, vbackups: 5977)

  **Action: BUY SI share**

### Step 267: P1 [INVEST]

  NN Values: P0=-0.414, P1=-0.439, P2=+0.852
  NN Priors (top 10 of 12 legal):
     1.  90.1% (-12.2pp) ████████████████████████████████████ AUCTION slot 0 (MAD, face $50)
     2.   5.7% ( -0.1pp) ██ AUCTION slot 2 (CDG, face $60)
     3.   2.0% ( -0.3pp)  AUCTION slot 1 (FRA, face $56)
     4.   1.2% ( -0.1pp)  PASS (INVEST)
     5.   0.3% ( -0.0pp)  BUY S share
     6.   0.3% ( +0.1pp)  BUY SM share
     7.   0.2% ( +3.0pp)  BUY PR share
     8.   0.0% ( +3.7pp)  BUY DA share
     9.   0.0% ( +0.0pp)  SELL VM share
    10.   0.0% ( +0.3pp)  BUY JS share

  MCTS Visits (top 10, 6400 total):
     1.  6182 (96.6%) Q=-0.368 ██████████████████████████████████████ AUCTION slot 0 (MAD, face $50)
     2.    78 ( 1.2%) Q=-0.447  AUCTION slot 2 (CDG, face $60)
     3.    39 ( 0.6%) Q=-0.553  BUY OS share
     4.    26 ( 0.4%) Q=-0.438  AUCTION slot 1 (FRA, face $56)
     5.    25 ( 0.4%) Q=-0.585  BUY DA share
     6.    23 ( 0.4%) Q=-0.415  PASS (INVEST)
     7.    23 ( 0.4%) Q=-0.622  BUY PR share
     8.     1 ( 0.0%) Q=-0.719  BUY JS share
     9.     1 ( 0.0%) Q=-0.648  BUY SM share
    10.     1 ( 0.0%) Q=-0.648  BUY S share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 64, vbackups: 6220)

  **Action: AUCTION slot 0 (MAD, face $50)**

Phase: BID_IN_AUCTION  |  Turn: 10  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $17 (NW $107) order=2 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $61 (NW $107) order=0 income=$0  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=1 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [3]: MAD (fv=$50, 5★, inc=$10), FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$7 stars=12 pres=P2  companies=[SJ]

**Deck**: 0 remaining

**Auction**: MAD current bid=$0 high bidder=P-1 starter=P1

### Step 268: P1 [BID_IN_AUCTION]

  **Auction**: MAD current bid=$0 high bidder=P-1 starter=P1

  NN Values: P0=-0.492, P1=-0.451, P2=+0.848
  NN Priors (top 10 of 12 legal):
     1.  97.7% ( -6.8pp) ███████████████████████████████████████ BID $50
     2.   1.2% ( +0.1pp)  BID $51
     3.   0.6% ( -0.1pp)  BID $52
     4.   0.2% ( -0.0pp)  BID $53
     5.   0.1% ( +0.0pp)  BID $54
     6.   0.1% ( +0.0pp)  BID $55
     7.   0.0% ( +0.5pp)  BID $56
     8.   0.0% ( +0.1pp)  BID $57
     9.   0.0% ( -0.0pp)  BID $61
    10.   0.0% ( +1.1pp)  BID $60

  MCTS Visits (top 10, 6400 total):
     1.  6324 (98.8%) Q=-0.368 ███████████████████████████████████████ BID $50
     2.    22 ( 0.3%) Q=-0.629  BID $59
     3.    20 ( 0.3%) Q=-0.598  BID $58
     4.    13 ( 0.2%) Q=-0.482  BID $51
     5.     9 ( 0.1%) Q=-0.669  BID $60
     6.     6 ( 0.1%) Q=-0.617  BID $56
     7.     3 ( 0.0%) Q=-0.537  BID $52
     8.     1 ( 0.0%) Q=-0.555  BID $53
     9.     1 ( 0.0%) Q=-0.469  BID $54
    10.     1 ( 0.0%) Q=-0.551  BID $57
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 63, vbackups: 6155)

  **Action: BID $50**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=2 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=0 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=1 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$7 stars=12 pres=P2  companies=[SJ]

**Deck**: 0 remaining


### Step 269: P2 [INVEST]

  NN Values: P0=-0.455, P1=-0.361, P2=+0.836
  NN Priors (top 8 of 8 legal):
     1.  98.4% (-14.7pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.7% ( +0.3pp)  SELL SI share
     3.   0.6% ( -0.1pp)  SELL DA share
     4.   0.1% ( +1.7pp)  BUY S share
     5.   0.1% (+10.1pp)  BUY PR share
     6.   0.1% ( +0.9pp)  BUY SM share
     7.   0.0% ( +0.8pp)  SELL OS share
     8.   0.0% ( +1.0pp)  BUY DA share

  MCTS Visits (top 8, 6400 total):
     1.  5390 (84.2%) Q=+0.835 █████████████████████████████████ PASS (INVEST)
     2.   434 ( 6.8%) Q=+0.832 ██ BUY PR share
     3.   163 ( 2.5%) Q=+0.850 █ SELL SI share
     4.   142 ( 2.2%) Q=+0.852  SELL DA share
     5.    93 ( 1.5%) Q=+0.832  BUY S share
     6.    80 ( 1.2%) Q=+0.840  BUY DA share
     7.    61 ( 1.0%) Q=+0.834  BUY SM share
     8.    37 ( 0.6%) Q=+0.828  SELL OS share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 62, vbackups: 5675)

  **Action: PASS (INVEST)**

### Step 270: P0 [INVEST]

  NN Values: P0=-0.359, P1=-0.475, P2=+0.836
  NN Priors (top 7 of 7 legal):
     1.  65.8% ( +2.5pp) ██████████████████████████ PASS (INVEST)
     2.  32.5% ( -3.0pp) ████████████ SELL DA share
     3.   0.5% ( -0.1pp)  BUY SM share
     4.   0.5% ( -0.1pp)  BUY S share
     5.   0.3% ( -0.0pp)  BUY PR share
     6.   0.2% ( +0.7pp)  SELL VM share
     7.   0.2% ( -0.0pp)  SELL SI share

  MCTS Visits (top 7, 6400 total):
     1.  5841 (91.3%) Q=-0.462 ████████████████████████████████████ PASS (INVEST)
     2.   499 ( 7.8%) Q=-0.527 ███ SELL DA share
     3.    22 ( 0.3%) Q=-0.479  BUY SM share
     4.    15 ( 0.2%) Q=-0.638  SELL VM share
     5.    11 ( 0.2%) Q=-0.480  BUY PR share
     6.     8 ( 0.1%) Q=-0.518  BUY S share
     7.     4 ( 0.1%) Q=-0.505  SELL SI share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 61, vbackups: 5541)

  **Action: PASS (INVEST)**

### Step 271: P1 [INVEST]

  NN Values: P0=-0.357, P1=-0.426, P2=+0.840
  NN Priors (top 6 of 6 legal):
     1.  87.9% (-13.2pp) ███████████████████████████████████ PASS (INVEST)
     2.   5.1% ( +3.2pp) ██ BUY SM share
     3.   3.6% ( +0.6pp) █ BUY S share
     4.   1.9% ( +4.4pp)  BUY PR share
     5.   1.1% ( +4.6pp)  SELL VM share
     6.   0.4% ( +0.4pp)  SELL SI share

  MCTS Visits (top 6, 6400 total):
     1.  6224 (97.2%) Q=-0.368 ██████████████████████████████████████ PASS (INVEST)
     2.    60 ( 0.9%) Q=-0.538  BUY SM share
     3.    49 ( 0.8%) Q=-0.564  BUY PR share
     4.    33 ( 0.5%) Q=-0.619  SELL VM share
     5.    25 ( 0.4%) Q=-0.636  BUY S share
     6.     9 ( 0.1%) Q=-0.658  SELL SI share
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 60, vbackups: 5840)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$7 stars=12 pres=P2  companies=[SJ]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($42), SI($65)

### Step 272: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($42), SI($65)

  NN Values: P0=-0.512, P1=-0.357, P2=+0.828
  NN Priors (top 2 of 2 legal):
     1.  99.7% ( -8.3pp) ███████████████████████████████████████ ACQ select SI
     2.   0.3% ( +8.3pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6156 (96.2%) Q=+0.834 ██████████████████████████████████████ ACQ select SI
     2.   244 ( 3.8%) Q=+0.831 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 59, vbackups: 6169)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$7 stars=12 pres=P2  companies=[SJ]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($65)

### Step 273: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($65)

  NN Values: P0=-0.504, P1=-0.363, P2=+0.828
  NN Priors (top 6 of 6 legal):
     1.  99.2% (-12.2pp) ███████████████████████████████████████ ACQ target HR (with SI)
     2.   0.7% ( +4.0pp)  ACQ target SZD (with SI)
     3.   0.0% ( +0.8pp)  ACQ target SX (with SI)
     4.   0.0% ( +5.6pp)  ACQ target KK (with SI)
     5.   0.0% ( +0.2pp)  ACQ target E (with SI)
     6.   0.0% ( +1.6pp)  ACQ target DR (with SI)

  MCTS Visits (top 6, 6400 total):
     1.  5637 (88.1%) Q=+0.834 ███████████████████████████████████ ACQ target HR (with SI)
     2.   301 ( 4.7%) Q=+0.838 █ ACQ target KK (with SI)
     3.   292 ( 4.6%) Q=+0.840 █ ACQ target SZD (with SI)
     4.    92 ( 1.4%) Q=+0.839  ACQ target DR (with SI)
     5.    68 ( 1.1%) Q=+0.838  ACQ target SX (with SI)
     6.    10 ( 0.2%) Q=+0.829  ACQ target E (with SI)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 58, vbackups: 5666)

  **Action: ACQ target HR (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$36 stars=25 pres=P2  companies=[SX, KK, DR, SZD, E, HR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $65 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$7 stars=12 pres=P2  companies=[SJ]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> HR (price range $24-$62)

### Step 274: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> HR (price range $24-$62)

  NN Values: P0=-0.520, P1=-0.332, P2=+0.828
  NN Priors (top 10 of 39 legal):
     1.  99.7% (-14.8pp) ███████████████████████████████████████ ACQUIRE HR with SI @ $24
     2.   0.2% ( -0.0pp)  ACQUIRE HR with SI @ $25
     3.   0.0% ( -0.0pp)  ACQUIRE HR with SI @ $42
     4.   0.0% ( +0.3pp)  ACQUIRE HR with SI @ $60
     5.   0.0% ( +0.1pp)  ACQUIRE HR with SI @ $43
     6.   0.0% ( +0.7pp)  ACQUIRE HR with SI @ $46
     7.   0.0% ( -0.0pp)  ACQUIRE HR with SI @ $47
     8.   0.0% ( +0.0pp)  ACQUIRE HR with SI @ $29
     9.   0.0% ( +2.4pp)  ACQUIRE HR with SI @ $59
    10.   0.0% ( +4.5pp)  ACQUIRE HR with SI @ $41

  MCTS Visits (top 10, 6400 total):
     1.  5779 (90.3%) Q=+0.834 ████████████████████████████████████ ACQUIRE HR with SI @ $24
     2.   201 ( 3.1%) Q=+0.832 █ ACQUIRE HR with SI @ $41
     3.   103 ( 1.6%) Q=+0.827  ACQUIRE HR with SI @ $51
     4.   101 ( 1.6%) Q=+0.833  ACQUIRE HR with SI @ $36
     5.    87 ( 1.4%) Q=+0.827  ACQUIRE HR with SI @ $59
     6.    38 ( 0.6%) Q=+0.829  ACQUIRE HR with SI @ $46
     7.    17 ( 0.3%) Q=+0.821  ACQUIRE HR with SI @ $48
     8.    16 ( 0.2%) Q=+0.828  ACQUIRE HR with SI @ $30
     9.    11 ( 0.2%) Q=+0.824  ACQUIRE HR with SI @ $34
    10.     8 ( 0.1%) Q=+0.830  ACQUIRE HR with SI @ $25
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 57, vbackups: 5793)

  **Action: ACQUIRE HR with SI @ $24**

Phase: ACQ_SELECT_CORP  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$14 stars=20 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $41 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$17 stars=15 pres=P2  companies=[SJ, HR*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($42), SI($41)

### Step 275: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($42), SI($41)

  NN Values: P0=-0.439, P1=-0.457, P2=+0.832
  NN Priors (top 3 of 3 legal):
     1.  99.7% (-14.9pp) ███████████████████████████████████████ ACQ select SI
     2.   0.2% (+11.3pp)  PASS (ACQ_SELECT_CORP)
     3.   0.1% ( +3.6pp)  ACQ select OS

  MCTS Visits (top 3, 6400 total):
     1.  5663 (88.5%) Q=+0.834 ███████████████████████████████████ ACQ select SI
     2.   551 ( 8.6%) Q=+0.834 ███ PASS (ACQ_SELECT_CORP)
     3.   186 ( 2.9%) Q=+0.836 █ ACQ select OS
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 56, vbackups: 5673)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$14 stars=20 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $41 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$17 stars=15 pres=P2  companies=[SJ, HR*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($41)

### Step 276: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($41)

  NN Values: P0=-0.414, P1=-0.469, P2=+0.836
  NN Priors (top 5 of 5 legal):
     1.  99.9% (-14.9pp) ███████████████████████████████████████ ACQ target E (with SI)
     2.   0.0% (+10.3pp)  ACQ target SZD (with SI)
     3.   0.0% ( +2.9pp)  ACQ target DR (with SI)
     4.   0.0% ( +1.4pp)  ACQ target KK (with SI)
     5.   0.0% ( +0.3pp)  ACQ target SX (with SI)

  MCTS Visits (top 5, 6400 total):
     1.  5419 (84.7%) Q=+0.834 █████████████████████████████████ ACQ target E (with SI)
     2.   650 (10.2%) Q=+0.841 ████ ACQ target SZD (with SI)
     3.   190 ( 3.0%) Q=+0.839 █ ACQ target DR (with SI)
     4.   123 ( 1.9%) Q=+0.841  ACQ target KK (with SI)
     5.    18 ( 0.3%) Q=+0.831  ACQ target SX (with SI)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 55, vbackups: 5425)

  **Action: ACQ target E (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$14 stars=20 pres=P2  companies=[SX, KK, DR, SZD, E]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $41 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$17 stars=15 pres=P2  companies=[SJ, HR*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> E (price range $22-$57)

### Step 277: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> E (price range $22-$57)

  NN Values: P0=-0.393, P1=-0.465, P2=+0.836
  NN Priors (top 10 of 20 legal):
     1.  99.7% (-14.6pp) ███████████████████████████████████████ ACQUIRE E with SI @ $22
     2.   0.2% ( -0.0pp)  ACQUIRE E with SI @ $23
     3.   0.0% ( +1.4pp)  ACQUIRE E with SI @ $27
     4.   0.0% ( +1.0pp)  ACQUIRE E with SI @ $26
     5.   0.0% ( +4.0pp)  ACQUIRE E with SI @ $40
     6.   0.0% ( +0.1pp)  ACQUIRE E with SI @ $24
     7.   0.0% ( -0.0pp)  ACQUIRE E with SI @ $32
     8.   0.0% ( +0.1pp)  ACQUIRE E with SI @ $41
     9.   0.0% ( -0.0pp)  ACQUIRE E with SI @ $28
    10.   0.0% ( +0.0pp)  ACQUIRE E with SI @ $25

  MCTS Visits (top 10, 6400 total):
     1.  5528 (86.4%) Q=+0.834 ██████████████████████████████████ ACQUIRE E with SI @ $22
     2.   250 ( 3.9%) Q=+0.838 █ ACQUIRE E with SI @ $29
     3.   204 ( 3.2%) Q=+0.833 █ ACQUIRE E with SI @ $40
     4.   108 ( 1.7%) Q=+0.839  ACQUIRE E with SI @ $27
     5.    89 ( 1.4%) Q=+0.833  ACQUIRE E with SI @ $39
     6.    80 ( 1.2%) Q=+0.839  ACQUIRE E with SI @ $26
     7.    65 ( 1.0%) Q=+0.836  ACQUIRE E with SI @ $33
     8.    21 ( 0.3%) Q=+0.827  ACQUIRE E with SI @ $34
     9.    13 ( 0.2%) Q=+0.827  ACQUIRE E with SI @ $37
    10.    12 ( 0.2%) Q=+0.826  ACQUIRE E with SI @ $38
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 54, vbackups: 5547)

  **Action: ACQUIRE E with SI @ $22**

Phase: ACQ_SELECT_CORP  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$7 stars=16 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $19 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($42), SI($19)

### Step 278: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($42), SI($19)

  NN Values: P0=-0.447, P1=-0.455, P2=+0.832
  NN Priors (top 3 of 3 legal):
     1.  99.7% (-10.9pp) ███████████████████████████████████████ ACQ select SI
     2.   0.3% ( +1.0pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +9.9pp)  ACQ select OS

  MCTS Visits (top 3, 6400 total):
     1.  5870 (91.7%) Q=+0.834 ████████████████████████████████████ ACQ select SI
     2.   450 ( 7.0%) Q=+0.836 ██ ACQ select OS
     3.    80 ( 1.2%) Q=+0.836  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 53, vbackups: 5889)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$7 stars=16 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $19 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($19)

### Step 279: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($19)

  NN Values: P0=-0.436, P1=-0.449, P2=+0.832
  NN Priors (top 4 of 4 legal):
     1.  99.6% (-13.8pp) ███████████████████████████████████████ ACQ target SZD (with SI)
     2.   0.3% ( +5.5pp)  ACQ target DR (with SI)
     3.   0.0% ( +0.5pp)  ACQ target KK (with SI)
     4.   0.0% ( +7.8pp)  ACQ target SX (with SI)

  MCTS Visits (top 4, 6400 total):
     1.  5539 (86.5%) Q=+0.834 ██████████████████████████████████ ACQ target SZD (with SI)
     2.   479 ( 7.5%) Q=+0.840 ██ ACQ target SX (with SI)
     3.   351 ( 5.5%) Q=+0.840 ██ ACQ target DR (with SI)
     4.    31 ( 0.5%) Q=+0.834  ACQ target KK (with SI)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 52, vbackups: 5561)

  **Action: ACQ target SZD (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$7 stars=16 pres=P2  companies=[SX, KK, DR, SZD]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $19 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> SZD (price range $15-$40)

### Step 280: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> SZD (price range $15-$40)

  NN Values: P0=-0.412, P1=-0.486, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  99.9% (-10.7pp) ███████████████████████████████████████ ACQUIRE SZD with SI @ $15
     2.   0.1% ( +3.5pp)  ACQUIRE SZD with SI @ $16
     3.   0.0% ( +0.0pp)  ACQUIRE SZD with SI @ $19
     4.   0.0% ( +4.5pp)  ACQUIRE SZD with SI @ $18
     5.   0.0% ( +2.7pp)  ACQUIRE SZD with SI @ $17

  MCTS Visits (top 5, 6400 total):
     1.  5830 (91.1%) Q=+0.835 ████████████████████████████████████ ACQUIRE SZD with SI @ $15
     2.   223 ( 3.5%) Q=+0.839 █ ACQUIRE SZD with SI @ $18
     3.   203 ( 3.2%) Q=+0.840 █ ACQUIRE SZD with SI @ $16
     4.   143 ( 2.2%) Q=+0.839  ACQUIRE SZD with SI @ $17
     5.     1 ( 0.0%) Q=+0.828  ACQUIRE SZD with SI @ $19
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 51, vbackups: 5841)

  **Action: ACQUIRE SZD with SI @ $15**

Phase: ACQ_SELECT_CORP  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=12 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD*, SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($42), SI($4)

### Step 281: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($42), SI($4)

  NN Values: P0=-0.443, P1=-0.469, P2=+0.836
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -5.1pp) ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   0.1% ( +5.1pp)  ACQ select OS

  MCTS Visits (top 2, 6400 total):
     1.  6176 (96.5%) Q=+0.834 ██████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   224 ( 3.5%) Q=+0.835 █ ACQ select OS
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 50, vbackups: 6065)

  **Action: PASS (ACQ_SELECT_CORP)**

### Step 282: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with DA($24), VM($14)

  NN Values: P0=-0.455, P1=-0.465, P2=+0.828
  NN Priors (top 2 of 2 legal):
     1.  95.6% ( -6.5pp) ██████████████████████████████████████ ACQ select VM
     2.   4.4% ( +6.5pp) █ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6240 (97.5%) Q=-0.461 ███████████████████████████████████████ ACQ select VM
     2.   160 ( 2.5%) Q=-0.537 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 49, vbackups: 6175)

  **Action: ACQ select VM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=12 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD*, SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with VM ($14)

### Step 283: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with VM ($14)

  NN Values: P0=-0.504, P1=-0.441, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  98.7% ( -8.5pp) ███████████████████████████████████████ ACQ target PR (with VM)
     2.   1.3% ( +8.5pp)  ACQ target NS (with VM)

  MCTS Visits (top 2, 6400 total):
     1.  6204 (96.9%) Q=-0.461 ██████████████████████████████████████ ACQ target PR (with VM)
     2.   196 ( 3.1%) Q=-0.509 █ ACQ target NS (with VM)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 48, vbackups: 6224)

  **Action: ACQ target PR (with VM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=12 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=11 pres=P0  companies=[PR, NS, BR]
  VM: $14 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$5 stars=4 pres=P0  companies=[PKP]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD*, SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 VM -> PR (price range $10-$25)

### Step 284: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> PR (price range $10-$25)

  NN Values: P0=-0.500, P1=-0.477, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  86.9% ( -6.4pp) ██████████████████████████████████ ACQUIRE PR with VM @ $10
     2.   5.2% ( -0.3pp) ██ ACQUIRE PR with VM @ $14
     3.   4.3% ( +1.0pp) █ ACQUIRE PR with VM @ $11
     4.   2.0% ( +5.2pp)  ACQUIRE PR with VM @ $13
     5.   1.5% ( +0.5pp)  ACQUIRE PR with VM @ $12

  MCTS Visits (top 5, 6400 total):
     1.  6121 (95.6%) Q=-0.459 ██████████████████████████████████████ ACQUIRE PR with VM @ $10
     2.   106 ( 1.7%) Q=-0.539  ACQUIRE PR with VM @ $13
     3.    77 ( 1.2%) Q=-0.542  ACQUIRE PR with VM @ $11
     4.    71 ( 1.1%) Q=-0.534  ACQUIRE PR with VM @ $14
     5.    25 ( 0.4%) Q=-0.559  ACQUIRE PR with VM @ $12
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 47, vbackups: 6213)

  **Action: ACQUIRE PR with VM @ $10**

Phase: ACQ_SELECT_CORP  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=12 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$12 stars=9 pres=P0  companies=[NS, BR]
  VM: $4 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$6 stars=5 pres=P0  companies=[PR*, PKP]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD*, SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with DA($24), VM($4)

### Step 285: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with DA($24), VM($4)

  NN Values: P0=-0.551, P1=-0.379, P2=+0.828
  NN Priors (top 2 of 2 legal):
     1.  82.9% ( -8.4pp) █████████████████████████████████ ACQ select DA
     2.  17.1% ( +8.4pp) ██████ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6058 (94.7%) Q=-0.455 █████████████████████████████████████ ACQ select DA
     2.   342 ( 5.3%) Q=-0.542 ██ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 46, vbackups: 6120)

  **Action: ACQ select DA**

  ↳ auto: ACQ target PKP (with DA)

Phase: ACQ_SELECT_PRICE  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $42 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=12 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $24 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$12 stars=9 pres=P0  companies=[NS, BR]
  VM: $4 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$6 stars=5 pres=P0  companies=[PR*, PKP]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD*, SJ, E*, HR*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 DA -> PKP (price range $13-$33)

### Step 286: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 DA -> PKP (price range $13-$33)

  NN Values: P0=-0.480, P1=-0.527, P2=+0.824
  NN Priors (top 10 of 12 legal):
     1.  96.9% (-13.9pp) ██████████████████████████████████████ ACQUIRE PKP with DA @ $13
     2.   1.9% ( -0.2pp)  ACQUIRE PKP with DA @ $14
     3.   0.3% ( -0.0pp)  ACQUIRE PKP with DA @ $15
     4.   0.2% ( +0.0pp)  ACQUIRE PKP with DA @ $16
     5.   0.2% ( +5.3pp)  ACQUIRE PKP with DA @ $23
     6.   0.1% ( +3.7pp)  ACQUIRE PKP with DA @ $17
     7.   0.1% ( +0.8pp)  ACQUIRE PKP with DA @ $22
     8.   0.1% ( +1.4pp)  ACQUIRE PKP with DA @ $18
     9.   0.1% ( +1.2pp)  ACQUIRE PKP with DA @ $24
    10.   0.1% ( +0.1pp)  ACQUIRE PKP with DA @ $21

  MCTS Visits (top 10, 6400 total):
     1.  6151 (96.1%) Q=-0.453 ██████████████████████████████████████ ACQUIRE PKP with DA @ $13
     2.    83 ( 1.3%) Q=-0.526  ACQUIRE PKP with DA @ $23
     3.    62 ( 1.0%) Q=-0.530  ACQUIRE PKP with DA @ $17
     4.    32 ( 0.5%) Q=-0.511  ACQUIRE PKP with DA @ $18
     5.    23 ( 0.4%) Q=-0.536  ACQUIRE PKP with DA @ $14
     6.    12 ( 0.2%) Q=-0.622  ACQUIRE PKP with DA @ $20
     7.    12 ( 0.2%) Q=-0.622  ACQUIRE PKP with DA @ $22
     8.    12 ( 0.2%) Q=-0.622  ACQUIRE PKP with DA @ $24
     9.     4 ( 0.1%) Q=-0.618  ACQUIRE PKP with DA @ $16
    10.     4 ( 0.1%) Q=-0.625  ACQUIRE PKP with DA @ $19
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 45, vbackups: 6051)

  **Action: ACQUIRE PKP with DA @ $13**

  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $11 (NW $107) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $49 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  JS: $2 price=$30(idx 17) shares=bank:2/unissued:5/issued:2 income=$-5 stars=1 RECEIVERSHIP  companies=[MHE]
  S: $19 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $103 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=18 pres=P2  companies=[SX, KK, DR]
  SM: $7 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $14 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $21 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=12 pres=P0  companies=[NS, PKP, BR]
  VM: $17 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$3 stars=3 pres=P0  companies=[PR]
  SI: $4 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Closing**: P0 may close NS (DA), PKP (DA), BR (DA)

### Step 287: P0 [CLOSING]

  **Closing**: P0 may close NS (DA), PKP (DA), BR (DA)

  NN Values: P0=-0.539, P1=-0.344, P2=+0.840
  NN Priors (top 3 of 3 legal):
     1.  99.8% (-12.1pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +5.9pp)  CLOSE NS
     3.   0.1% ( +6.2pp)  CLOSE PKP

  MCTS Visits (top 3, 6400 total):
     1.  6330 (98.9%) Q=-0.451 ███████████████████████████████████████ PASS (CLOSING)
     2.    35 ( 0.5%) Q=-0.680  CLOSE NS
     3.    35 ( 0.5%) Q=-0.679  CLOSE PKP
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 44, vbackups: 6150)

  **Action: PASS (CLOSING)**

### Step 288: P2 [CLOSING]

  **Closing**: P2 may close SX (OS), KK (OS), DR (OS), SZD (SI), SJ (SI), E (SI), HR (SI)

  NN Values: P0=-0.512, P1=-0.344, P2=+0.840
  NN Priors (top 4 of 4 legal):
     1.  99.9% (-13.3pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +2.1pp)  CLOSE SX
     3.   0.0% ( +6.5pp)  CLOSE DR
     4.   0.0% ( +4.7pp)  CLOSE KK

  MCTS Visits (top 4, 6400 total):
     1.  5415 (84.6%) Q=+0.834 █████████████████████████████████ PASS (CLOSING)
     2.   478 ( 7.5%) Q=+0.842 ██ CLOSE DR
     3.   353 ( 5.5%) Q=+0.842 ██ CLOSE KK
     4.   154 ( 2.4%) Q=+0.842  CLOSE SX
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 43, vbackups: 5426)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 10  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $17 (NW $107) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $21 (NW $117) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $27 (NW $160) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $13 price=$8(idx 4) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $103 price=$45(idx 21) shares=bank:5/unissued:0/issued:6 income=$0 stars=18 pres=P2  companies=[SX, KK, DR]
  SM: $3 price=$5(idx 1) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[BD]
  PR: $11 price=$7(idx 3) shares=bank:5/unissued:0/issued:5 income=$-3 stars=3 RECEIVERSHIP  companies=[WT]
  DA: $31 price=$22(idx 14) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $20 price=$13(idx 9) shares=bank:0/unissued:2/issued:2 income=$3 stars=4 pres=P0  companies=[PR]
  SI: $43 price=$33(idx 18) shares=bank:0/unissued:0/issued:4 income=$39 stars=23 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Dividends**: OS

### Step 289: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.439, P1=-0.428, P2=+0.848
  NN Priors (top 10 of 16 legal):
     1.  99.7% (-14.6pp) ███████████████████████████████████████ DIVIDEND $15
     2.   0.3% ( +5.1pp)  DIVIDEND $14
     3.   0.0% ( -0.0pp)  DIVIDEND $13
     4.   0.0% ( +0.5pp)  DIVIDEND $10
     5.   0.0% ( +0.1pp)  DIVIDEND $12
     6.   0.0% ( +2.8pp)  DIVIDEND $9
     7.   0.0% ( +0.2pp)  DIVIDEND $4
     8.   0.0% ( +2.7pp)  DIVIDEND $7
     9.   0.0% ( +0.0pp)  DIVIDEND $6
    10.   0.0% ( +0.4pp)  DIVIDEND $3

  MCTS Visits (top 10, 6400 total):
     1.  5412 (84.6%) Q=+0.834 █████████████████████████████████ DIVIDEND $15
     2.   423 ( 6.6%) Q=+0.844 ██ DIVIDEND $14
     3.   175 ( 2.7%) Q=+0.838 █ DIVIDEND $9
     4.   154 ( 2.4%) Q=+0.837  DIVIDEND $7
     5.   119 ( 1.9%) Q=+0.838  DIVIDEND $8
     6.    34 ( 0.5%) Q=+0.835  DIVIDEND $10
     7.    27 ( 0.4%) Q=+0.830  DIVIDEND $0
     8.    21 ( 0.3%) Q=+0.830  DIVIDEND $3
     9.    12 ( 0.2%) Q=+0.829  DIVIDEND $4
    10.     8 ( 0.1%) Q=+0.832  DIVIDEND $12
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 42, vbackups: 5449)

  **Action: DIVIDEND $15**

### Step 290: P2 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.457, P1=-0.383, P2=+0.844
  NN Priors (top 10 of 11 legal):
     1.  99.4% (-14.6pp) ███████████████████████████████████████ DIVIDEND $9
     2.   0.4% ( +6.7pp)  DIVIDEND $6
     3.   0.1% ( +0.2pp)  DIVIDEND $3
     4.   0.1% ( +0.1pp)  DIVIDEND $8
     5.   0.0% ( +0.0pp)  DIVIDEND $0
     6.   0.0% ( -0.0pp)  DIVIDEND $2
     7.   0.0% ( +4.7pp)  DIVIDEND $10
     8.   0.0% ( +2.3pp)  DIVIDEND $7
     9.   0.0% ( +0.4pp)  DIVIDEND $5
    10.   0.0% ( +0.2pp)  DIVIDEND $1

  MCTS Visits (top 10, 6400 total):
     1.  5319 (83.1%) Q=+0.834 █████████████████████████████████ DIVIDEND $9
     2.   435 ( 6.8%) Q=+0.838 ██ DIVIDEND $6
     3.   350 ( 5.5%) Q=+0.842 ██ DIVIDEND $10
     4.   226 ( 3.5%) Q=+0.846 █ DIVIDEND $7
     5.    23 ( 0.4%) Q=+0.832  DIVIDEND $5
     6.    14 ( 0.2%) Q=+0.830  DIVIDEND $3
     7.    10 ( 0.2%) Q=+0.834  DIVIDEND $1
     8.    10 ( 0.2%) Q=+0.833  DIVIDEND $4
     9.     7 ( 0.1%) Q=+0.831  DIVIDEND $8
    10.     5 ( 0.1%) Q=+0.837  DIVIDEND $0
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 41, vbackups: 5352)

  **Action: DIVIDEND $9**

### Step 291: P0 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.477, P1=-0.469, P2=+0.836
  NN Priors (top 7 of 7 legal):
     1.  83.7% (-11.1pp) █████████████████████████████████ DIVIDEND $0
     2.   6.7% ( -0.7pp) ██ DIVIDEND $2
     3.   2.9% ( +9.4pp) █ DIVIDEND $3
     4.   2.8% ( -0.4pp) █ DIVIDEND $5
     5.   2.0% ( +0.6pp)  DIVIDEND $4
     6.   1.8% ( +1.9pp)  DIVIDEND $1
     7.   0.1% ( +0.3pp)  DIVIDEND $6

  MCTS Visits (top 7, 6400 total):
     1.  6054 (94.6%) Q=-0.447 █████████████████████████████████████ DIVIDEND $0
     2.   131 ( 2.0%) Q=-0.559  DIVIDEND $3
     3.    97 ( 1.5%) Q=-0.513  DIVIDEND $2
     4.    46 ( 0.7%) Q=-0.525  DIVIDEND $4
     5.    38 ( 0.6%) Q=-0.563  DIVIDEND $1
     6.    20 ( 0.3%) Q=-0.593  DIVIDEND $5
     7.    14 ( 0.2%) Q=-0.559  DIVIDEND $6
  A0GB Value: P0=-0.432, P1=-0.264, P2=+0.805 (depth: 40, vbackups: 6227)

  **Action: DIVIDEND $0**

### Step 292: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.492, P1=-0.539, P2=+0.832
  NN Priors (top 5 of 5 legal):
     1.  72.6% ( -7.2pp) █████████████████████████████ DIVIDEND $0
     2.  23.6% ( -2.4pp) █████████ DIVIDEND $4
     3.   1.6% ( -0.0pp)  DIVIDEND $2
     4.   1.5% ( +5.5pp)  DIVIDEND $3
     5.   0.9% ( +4.2pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  5938 (92.8%) Q=-0.440 █████████████████████████████████████ DIVIDEND $0
     2.   251 ( 3.9%) Q=-0.540 █ DIVIDEND $4
     3.    99 ( 1.5%) Q=-0.526  DIVIDEND $3
     4.    97 ( 1.5%) Q=-0.503  DIVIDEND $1
     5.    15 ( 0.2%) Q=-0.556  DIVIDEND $2
  A0GB Value: P0=-0.500, P1=-0.299, P2=+0.809 (depth: 49, vbackups: 6038)

  **Action: DIVIDEND $0**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 10  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $26 (NW $135) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $30 (NW $135) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $60 (NW $206) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $13 price=$6(idx 2) shares=bank:4/unissued:3/issued:4 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $20 price=$14(idx 10) shares=bank:0/unissued:2/issued:2 income=$3 stars=4 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Issue**: VM

### Step 293: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.441, P1=-0.480, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  62.1% ( +4.6pp) ████████████████████████ PASS (ISSUE_SHARES)
     2.  37.9% ( -4.6pp) ███████████████ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  5757 (90.0%) Q=-0.424 ███████████████████████████████████ ISSUE VM shares
     2.   643 (10.0%) Q=-0.558 ████ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.543, P1=-0.218, P2=+0.805 (depth: 49, vbackups: 5937)

  **Action: ISSUE VM shares**

Phase: IPO  |  Turn: 10  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $26 (NW $134) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $30 (NW $134) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $60 (NW $206) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$13(idx 9) shares=bank:1/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**IPO**: MAD

### Step 294: P1 [IPO]

  **IPO**: MAD

  NN Values: P0=-0.539, P1=-0.357, P2=+0.832
  NN Priors (top 4 of 4 legal):
     1.  77.8% ( -5.7pp) ███████████████████████████████ PASS (IPO)
     2.  11.5% ( +0.9pp) ████ IPO MAD → float SM
     3.   9.5% ( +1.7pp) ███ IPO MAD → float PR
     4.   1.2% ( +3.2pp)  IPO MAD → float JS

  MCTS Visits (top 4, 6400 total):
     1.  4031 (63.0%) Q=-0.365 █████████████████████████ IPO MAD → float SM
     2.  2037 (31.8%) Q=-0.409 ████████████ PASS (IPO)
     3.   277 ( 4.3%) Q=-0.416 █ IPO MAD → float PR
     4.    55 ( 0.9%) Q=-0.472  IPO MAD → float JS
  A0GB Value: P0=-0.543, P1=-0.222, P2=+0.801 (depth: 48, vbackups: 5756)

  **Action: IPO MAD → float SM**

Phase: PAR  |  Turn: 10  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $26 (NW $134) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $30 (NW $134) order=2 income=$10  companies=[MAD]  shares=[VM=1, SI=1]
  P2: $60 (NW $206) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$13(idx 9) shares=bank:1/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**PAR**: MAD -> SM

### Step 295: P1 [PAR]

  **PAR**: MAD -> SM

  NN Values: P0=-0.590, P1=-0.305, P2=+0.832
  NN Priors (top 2 of 2 legal):
     1.  79.8% ( -3.1pp) ███████████████████████████████ PAR SM @$30 (IPO MAD)
     2.  20.2% ( +3.1pp) ████████ PAR SM @$33 (IPO MAD)

  MCTS Visits (top 2, 6400 total):
     1.  3799 (59.4%) Q=-0.378 ███████████████████████ PAR SM @$30 (IPO MAD)
     2.  2601 (40.6%) Q=-0.362 ████████████████ PAR SM @$33 (IPO MAD)
  A0GB Value: P0=-0.080, P1=-0.586, P2=+0.801 (depth: 48, vbackups: 4030)

  **Action: PAR SM @$30 (IPO MAD)**

--- Turn 11 ---

Phase: INVEST  |  Turn: 11  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $26 (NW $134) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $20 (NW $134) order=2 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $60 (NW $206) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$13(idx 9) shares=bank:1/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining


### Step 296: P2 [INVEST]

  NN Values: P0=-0.570, P1=-0.271, P2=+0.840
  NN Priors (top 10 of 11 legal):
     1. 100.0% ( -8.5pp) ███████████████████████████████████████ AUCTION slot 0 (FRA, face $56)
     2.   0.0% ( +0.5pp)  SELL OS share
     3.   0.0% ( +0.2pp)  AUCTION slot 1 (CDG, face $60)
     4.   0.0% ( +0.3pp)  BUY VM share
     5.   0.0% ( +0.4pp)  SELL DA share
     6.   0.0% ( +2.7pp)  SELL SI share
     7.   0.0% ( +0.1pp)  BUY S share
     8.   0.0% ( +0.5pp)  BUY OS share
     9.   0.0% ( +0.3pp)  BUY SM share
    10.   0.0% ( +3.5pp)  BUY DA share

  MCTS Visits (top 10, 6400 total):
     1.  5696 (89.0%) Q=+0.827 ███████████████████████████████████ AUCTION slot 0 (FRA, face $56)
     2.   225 ( 3.5%) Q=+0.827 █ BUY DA share
     3.   167 ( 2.6%) Q=+0.827 █ SELL SI share
     4.    99 ( 1.5%) Q=+0.843  SELL DA share
     5.    45 ( 0.7%) Q=+0.834  SELL OS share
     6.    40 ( 0.6%) Q=+0.831  BUY OS share
     7.    38 ( 0.6%) Q=+0.842  AUCTION slot 1 (CDG, face $60)
     8.    26 ( 0.4%) Q=+0.832  BUY SM share
     9.    24 ( 0.4%) Q=+0.840  PASS (INVEST)
    10.    21 ( 0.3%) Q=+0.831  BUY VM share
  A0GB Value: P0=-0.102, P1=-0.551, P2=+0.805 (depth: 53, vbackups: 3798)

  **Action: AUCTION slot 0 (FRA, face $56)**

Phase: BID_IN_AUCTION  |  Turn: 11  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $26 (NW $134) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $20 (NW $134) order=2 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $60 (NW $206) order=0 income=$0  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [2]: FRA (fv=$56, 5★, inc=$10), CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$13(idx 9) shares=bank:1/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Auction**: FRA current bid=$0 high bidder=P-1 starter=P2

### Step 297: P2 [BID_IN_AUCTION]

  **Auction**: FRA current bid=$0 high bidder=P-1 starter=P2

  NN Values: P0=-0.625, P1=-0.217, P2=+0.824
  NN Priors (top 5 of 5 legal):
     1.  99.6% (-10.0pp) ███████████████████████████████████████ BID $56
     2.   0.3% ( +3.7pp)  BID $57
     3.   0.1% ( +3.2pp)  BID $58
     4.   0.0% ( +0.8pp)  BID $59
     5.   0.0% ( +2.2pp)  BID $60

  MCTS Visits (top 5, 6400 total):
     1.  5711 (89.2%) Q=+0.827 ███████████████████████████████████ BID $56
     2.   268 ( 4.2%) Q=+0.830 █ BID $57
     3.   203 ( 3.2%) Q=+0.829 █ BID $58
     4.   147 ( 2.3%) Q=+0.829  BID $60
     5.    71 ( 1.1%) Q=+0.831  BID $59
  A0GB Value: P0=-0.115, P1=-0.613, P2=+0.801 (depth: 53, vbackups: 5695)

  **Action: BID $56**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $26 (NW $134) order=1 income=$0  shares=[DA=2 (pres), VM=1 (pres), SI=1]
  P1: $20 (NW $134) order=2 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $206) order=0 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$27(idx 16) shares=bank:2/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$13(idx 9) shares=bank:1/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining


### Step 298: P0 [INVEST]

  NN Values: P0=-0.520, P1=-0.328, P2=+0.848
  NN Priors (top 6 of 6 legal):
     1.  98.3% (-13.0pp) ███████████████████████████████████████ BUY VM share
     2.   1.0% ( +2.0pp)  SELL DA share
     3.   0.6% ( +8.8pp)  PASS (INVEST)
     4.   0.1% ( +1.0pp)  BUY S share
     5.   0.0% ( +1.2pp)  SELL VM share
     6.   0.0% ( +0.1pp)  SELL SI share

  MCTS Visits (top 6, 6400 total):
     1.  6305 (98.5%) Q=-0.388 ███████████████████████████████████████ BUY VM share
     2.    59 ( 0.9%) Q=-0.613  PASS (INVEST)
     3.    22 ( 0.3%) Q=-0.608  SELL DA share
     4.     7 ( 0.1%) Q=-0.660  BUY S share
     5.     7 ( 0.1%) Q=-0.626  SELL VM share
  A0GB Value: P0=-0.055, P1=-0.656, P2=+0.820 (depth: 53, vbackups: 5709)

  **Action: BUY VM share**

### Step 299: P1 [INVEST]

  NN Values: P0=-0.539, P1=-0.307, P2=+0.855
  NN Priors (top 5 of 5 legal):
     1.  99.3% (-11.1pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.2% ( +1.2pp)  SELL SM share
     3.   0.2% ( +3.5pp)  BUY S share
     4.   0.1% ( +3.5pp)  SELL VM share
     5.   0.1% ( +2.8pp)  SELL SI share

  MCTS Visits (top 5, 6400 total):
     1.  5881 (91.9%) Q=-0.395 ████████████████████████████████████ PASS (INVEST)
     2.   241 ( 3.8%) Q=-0.396 █ BUY S share
     3.   236 ( 3.7%) Q=-0.396 █ SELL VM share
     4.    21 ( 0.3%) Q=-0.528  SELL SM share
     5.    21 ( 0.3%) Q=-0.569  SELL SI share
  A0GB Value: P0=-0.055, P1=-0.656, P2=+0.820 (depth: 52, vbackups: 5888)

  **Action: PASS (INVEST)**

### Step 300: P2 [INVEST]

  NN Values: P0=-0.498, P1=-0.357, P2=+0.848
  NN Priors (top 4 of 4 legal):
     1.  76.5% (-10.3pp) ██████████████████████████████ PASS (INVEST)
     2.  13.1% ( +1.2pp) █████ SELL OS share
     3.   9.4% ( +0.2pp) ███ SELL DA share
     4.   1.0% ( +8.9pp)  SELL SI share

  MCTS Visits (top 4, 6400 total):
     1.  3427 (53.5%) Q=+0.821 █████████████████████ PASS (INVEST)
     2.  1295 (20.2%) Q=+0.834 ████████ SELL OS share
     3.  1239 (19.4%) Q=+0.838 ███████ SELL DA share
     4.   439 ( 6.9%) Q=+0.822 ██ SELL SI share
  A0GB Value: P0=-0.055, P1=-0.656, P2=+0.820 (depth: 51, vbackups: 5953)

  **Action: PASS (INVEST)**

### Step 301: P0 [INVEST]

  NN Values: P0=-0.531, P1=-0.371, P2=+0.848
  NN Priors (top 5 of 5 legal):
     1.  68.0% ( -5.0pp) ███████████████████████████ PASS (INVEST)
     2.  29.3% ( -4.9pp) ███████████ SELL DA share
     3.   1.8% ( +1.9pp)  BUY S share
     4.   0.6% ( +0.7pp)  SELL VM share
     5.   0.2% ( +7.3pp)  SELL SI share

  MCTS Visits (top 5, 6400 total):
     1.  5743 (89.7%) Q=-0.292 ███████████████████████████████████ SELL DA share
     2.   593 ( 9.3%) Q=-0.431 ███ PASS (INVEST)
     3.    40 ( 0.6%) Q=-0.571  SELL SI share
     4.    17 ( 0.3%) Q=-0.614  BUY S share
     5.     7 ( 0.1%) Q=-0.514  SELL VM share
  A0GB Value: P0=-0.122, P1=-0.562, P2=+0.809 (depth: 57, vbackups: 3598)

  **Action: SELL DA share**

### Step 302: P1 [INVEST]

  NN Values: P0=-0.605, P1=-0.239, P2=+0.852
  NN Priors (top 5 of 5 legal):
     1.  96.7% (-13.9pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.9% ( +4.5pp)  SELL SM share
     3.   0.5% ( +3.3pp)  SELL SI share
     4.   0.5% ( +1.8pp)  BUY S share
     5.   0.4% ( +4.3pp)  SELL VM share

  MCTS Visits (top 5, 6400 total):
     1.  4340 (67.8%) Q=-0.474 ███████████████████████████ PASS (INVEST)
     2.  1368 (21.4%) Q=-0.451 ████████ BUY S share
     3.   463 ( 7.2%) Q=-0.469 ██ SELL SM share
     4.   164 ( 2.6%) Q=-0.484 █ SELL SI share
     5.    65 ( 1.0%) Q=-0.550  SELL VM share
  A0GB Value: P0=-0.122, P1=-0.562, P2=+0.809 (depth: 56, vbackups: 5742)

  **Action: PASS (INVEST)**

### Step 303: P2 [INVEST]

  NN Values: P0=-0.531, P1=-0.266, P2=+0.848
  NN Priors (top 4 of 4 legal):
     1.  89.6% ( -7.6pp) ███████████████████████████████████ PASS (INVEST)
     2.   9.2% ( -0.7pp) ███ SELL OS share
     3.   1.0% ( +7.5pp)  SELL DA share
     4.   0.2% ( +0.9pp)  SELL SI share

  MCTS Visits (top 4, 6400 total):
     1.  4148 (64.8%) Q=+0.813 █████████████████████████ PASS (INVEST)
     2.  1186 (18.5%) Q=+0.831 ███████ SELL OS share
     3.   976 (15.2%) Q=+0.829 ██████ SELL DA share
     4.    90 ( 1.4%) Q=+0.823  SELL SI share
  A0GB Value: P0=-0.121, P1=-0.566, P2=+0.797 (depth: 60, vbackups: 4339)

  **Action: PASS (INVEST)**

### Step 304: P0 [INVEST]

  NN Values: P0=-0.520, P1=-0.322, P2=+0.844
  NN Priors (top 7 of 7 legal):
     1.  96.0% (-12.2pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.3% ( +0.4pp)  BUY SM share
     3.   1.2% ( +0.1pp)  BUY S share
     4.   0.8% ( +6.3pp)  SELL VM share
     5.   0.5% ( +4.8pp)  BUY DA share
     6.   0.2% ( +0.1pp)  SELL SI share
     7.   0.0% ( +0.4pp)  SELL DA share

  MCTS Visits (top 7, 6400 total):
     1.  6317 (98.7%) Q=-0.235 ███████████████████████████████████████ PASS (INVEST)
     2.    31 ( 0.5%) Q=-0.611  SELL VM share
     3.    26 ( 0.4%) Q=-0.579  BUY DA share
     4.    14 ( 0.2%) Q=-0.529  BUY SM share
     5.    10 ( 0.2%) Q=-0.680  BUY S share
     6.     1 ( 0.0%) Q=-0.633  SELL DA share
     7.     1 ( 0.0%) Q=-0.703  SELL SI share
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 64, vbackups: 4147)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with DA($31), VM($33)

### Step 305: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with DA($31), VM($33)

  NN Values: P0=-0.586, P1=-0.305, P2=+0.836
  NN Priors (top 2 of 2 legal):
     1.  98.7% ( -6.3pp) ███████████████████████████████████████ ACQ select VM
     2.   1.3% ( +6.3pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6366 (99.5%) Q=-0.234 ███████████████████████████████████████ ACQ select VM
     2.    34 ( 0.5%) Q=-0.533  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 63, vbackups: 6316)

  **Action: ACQ select VM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with VM ($33)

### Step 306: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with VM ($33)

  NN Values: P0=-0.551, P1=-0.312, P2=+0.840
  NN Priors (top 3 of 3 legal):
     1.  76.9% ( -8.9pp) ██████████████████████████████ ACQ target BR (with VM)
     2.  18.1% ( +8.2pp) ███████ ACQ target NS (with VM)
     3.   5.0% ( +0.7pp) █ ACQ target PKP (with VM)

  MCTS Visits (top 3, 6400 total):
     1.  6166 (96.3%) Q=-0.229 ██████████████████████████████████████ ACQ target BR (with VM)
     2.   212 ( 3.3%) Q=-0.394 █ ACQ target NS (with VM)
     3.    22 ( 0.3%) Q=-0.548  ACQ target PKP (with VM)
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 62, vbackups: 6320)

  **Action: ACQ target BR (with VM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$10 stars=13 pres=P0  companies=[NS, PKP, BR]
  VM: $33 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$3 stars=5 pres=P0  companies=[PR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 VM -> BR (price range $17-$45)

### Step 307: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> BR (price range $17-$45)

  NN Values: P0=-0.520, P1=-0.307, P2=+0.840
  NN Priors (top 10 of 17 legal):
     1.  97.6% (-14.6pp) ███████████████████████████████████████ ACQUIRE BR with VM @ $17
     2.   1.2% ( +3.2pp)  ACQUIRE BR with VM @ $18
     3.   0.3% ( +0.2pp)  ACQUIRE BR with VM @ $21
     4.   0.2% ( -0.0pp)  ACQUIRE BR with VM @ $19
     5.   0.2% ( -0.0pp)  ACQUIRE BR with VM @ $20
     6.   0.2% ( +0.5pp)  ACQUIRE BR with VM @ $22
     7.   0.1% ( -0.0pp)  ACQUIRE BR with VM @ $27
     8.   0.1% ( +0.9pp)  ACQUIRE BR with VM @ $26
     9.   0.1% ( +3.9pp)  ACQUIRE BR with VM @ $23
    10.   0.1% ( +1.1pp)  ACQUIRE BR with VM @ $25

  MCTS Visits (top 10, 6400 total):
     1.  6347 (99.2%) Q=-0.228 ███████████████████████████████████████ ACQUIRE BR with VM @ $17
     2.    16 ( 0.2%) Q=-0.564  ACQUIRE BR with VM @ $18
     3.    12 ( 0.2%) Q=-0.647  ACQUIRE BR with VM @ $23
     4.    11 ( 0.2%) Q=-0.658  ACQUIRE BR with VM @ $28
     5.     4 ( 0.1%) Q=-0.606  ACQUIRE BR with VM @ $22
     6.     3 ( 0.0%) Q=-0.686  ACQUIRE BR with VM @ $25
     7.     3 ( 0.0%) Q=-0.697  ACQUIRE BR with VM @ $26
     8.     1 ( 0.0%) Q=-0.625  ACQUIRE BR with VM @ $21
     9.     1 ( 0.0%) Q=-0.664  ACQUIRE BR with VM @ $24
    10.     1 ( 0.0%) Q=-0.707  ACQUIRE BR with VM @ $29
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 61, vbackups: 6206)

  **Action: ACQUIRE BR with VM @ $17**

Phase: ACQ_SELECT_CORP  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$1 stars=9 pres=P0  companies=[NS, PKP]
  VM: $16 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$10 stars=7 pres=P0  companies=[PR, BR*]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with DA($31), VM($16)

### Step 308: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with DA($31), VM($16)

  NN Values: P0=-0.566, P1=-0.344, P2=+0.836
  NN Priors (top 3 of 3 legal):
     1.  97.5% (-10.2pp) ██████████████████████████████████████ ACQ select VM
     2.   1.9% ( +7.0pp)  ACQ select DA
     3.   0.6% ( +3.2pp)  PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 3, 6400 total):
     1.  6348 (99.2%) Q=-0.227 ███████████████████████████████████████ ACQ select VM
     2.    39 ( 0.6%) Q=-0.554  ACQ select DA
     3.    13 ( 0.2%) Q=-0.669  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 60, vbackups: 6346)

  **Action: ACQ select VM**

Phase: ACQ_SELECT_COMPANY  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$1 stars=9 pres=P0  companies=[NS, PKP]
  VM: $16 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$10 stars=7 pres=P0  companies=[PR, BR*]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P0 buying with VM ($16)

### Step 309: P0 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P0 buying with VM ($16)

  NN Values: P0=-0.566, P1=-0.344, P2=+0.840
  NN Priors (top 2 of 2 legal):
     1.  96.2% (-10.2pp) ██████████████████████████████████████ ACQ target NS (with VM)
     2.   3.8% (+10.2pp) █ ACQ target PKP (with VM)

  MCTS Visits (top 2, 6400 total):
     1.  6282 (98.2%) Q=-0.226 ███████████████████████████████████████ ACQ target NS (with VM)
     2.   118 ( 1.8%) Q=-0.381  ACQ target PKP (with VM)
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 59, vbackups: 6304)

  **Action: ACQ target NS (with VM)**

Phase: ACQ_SELECT_PRICE  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$1 stars=9 pres=P0  companies=[NS, PKP]
  VM: $16 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$10 stars=7 pres=P0  companies=[PR, BR*]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 VM -> NS (price range $11-$29)

### Step 310: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 VM -> NS (price range $11-$29)

  NN Values: P0=-0.570, P1=-0.350, P2=+0.840
  NN Priors (top 6 of 6 legal):
     1.  95.1% (-12.1pp) ██████████████████████████████████████ ACQUIRE NS with VM @ $11
     2.   2.5% ( +6.8pp) █ ACQUIRE NS with VM @ $12
     3.   0.9% ( +1.6pp)  ACQUIRE NS with VM @ $13
     4.   0.8% ( +1.6pp)  ACQUIRE NS with VM @ $14
     5.   0.5% ( +1.0pp)  ACQUIRE NS with VM @ $15
     6.   0.2% ( +1.0pp)  ACQUIRE NS with VM @ $16

  MCTS Visits (top 6, 6400 total):
     1.  6277 (98.1%) Q=-0.224 ███████████████████████████████████████ ACQUIRE NS with VM @ $11
     2.    82 ( 1.3%) Q=-0.361  ACQUIRE NS with VM @ $12
     3.    11 ( 0.2%) Q=-0.570  ACQUIRE NS with VM @ $16
     4.    11 ( 0.2%) Q=-0.559  ACQUIRE NS with VM @ $15
     5.    10 ( 0.2%) Q=-0.545  ACQUIRE NS with VM @ $13
     6.     9 ( 0.1%) Q=-0.550  ACQUIRE NS with VM @ $14
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 58, vbackups: 6304)

  **Action: ACQUIRE NS with VM @ $11**

Phase: ACQ_SELECT_CORP  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$3 stars=6 pres=P0  companies=[PKP]
  VM: $5 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$13 stars=9 pres=P0  companies=[PR, NS*, BR*]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P0 may buy with DA($31), VM($5)

### Step 311: P0 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P0 may buy with DA($31), VM($5)

  NN Values: P0=-0.414, P1=-0.453, P2=+0.840
  NN Priors (top 2 of 2 legal):
     1.  92.6% ( -9.0pp) █████████████████████████████████████ ACQ select DA
     2.   7.4% ( +9.0pp) ██ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  6239 (97.5%) Q=-0.223 ██████████████████████████████████████ ACQ select DA
     2.   161 ( 2.5%) Q=-0.347 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 57, vbackups: 6296)

  **Action: ACQ select DA**

  ↳ auto: ACQ target PR (with DA)

Phase: ACQ_SELECT_PRICE  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $31 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$3 stars=6 pres=P0  companies=[PKP]
  VM: $5 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$13 stars=9 pres=P0  companies=[PR, NS*, BR*]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P0 DA -> PR (price range $10-$25)

### Step 312: P0 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P0 DA -> PR (price range $10-$25)

  NN Values: P0=-0.508, P1=-0.385, P2=+0.844
  NN Priors (top 10 of 16 legal):
     1.  94.6% (-13.0pp) █████████████████████████████████████ ACQUIRE PR with DA @ $25
     2.   3.6% ( -0.1pp) █ ACQUIRE PR with DA @ $24
     3.   0.4% ( +0.0pp)  ACQUIRE PR with DA @ $23
     4.   0.2% ( +0.8pp)  ACQUIRE PR with DA @ $20
     5.   0.2% ( +5.2pp)  ACQUIRE PR with DA @ $19
     6.   0.2% ( -0.0pp)  ACQUIRE PR with DA @ $21
     7.   0.2% ( +0.1pp)  ACQUIRE PR with DA @ $14
     8.   0.1% ( +0.1pp)  ACQUIRE PR with DA @ $22
     9.   0.1% ( +0.0pp)  ACQUIRE PR with DA @ $15
    10.   0.1% ( +0.5pp)  ACQUIRE PR with DA @ $18

  MCTS Visits (top 10, 6400 total):
     1.  6292 (98.3%) Q=-0.222 ███████████████████████████████████████ ACQUIRE PR with DA @ $25
     2.    33 ( 0.5%) Q=-0.432  ACQUIRE PR with DA @ $19
     3.    26 ( 0.4%) Q=-0.385  ACQUIRE PR with DA @ $24
     4.    17 ( 0.3%) Q=-0.497  ACQUIRE PR with DA @ $12
     5.     8 ( 0.1%) Q=-0.546  ACQUIRE PR with DA @ $20
     6.     5 ( 0.1%) Q=-0.614  ACQUIRE PR with DA @ $10
     7.     5 ( 0.1%) Q=-0.578  ACQUIRE PR with DA @ $16
     8.     5 ( 0.1%) Q=-0.570  ACQUIRE PR with DA @ $17
     9.     5 ( 0.1%) Q=-0.559  ACQUIRE PR with DA @ $18
    10.     1 ( 0.0%) Q=-0.535  ACQUIRE PR with DA @ $23
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 56, vbackups: 6238)

  **Action: ACQUIRE PR with DA @ $25**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 11  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $4 (NW $203) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $54 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $18 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $70 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=12 pres=P1  companies=[MAD]
  DA: $34 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$1 stars=8 pres=P0  companies=[PR, PKP]
  VM: $30 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $7 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=19 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Closing**: P0 may close PR (DA), NS (VM), PKP (DA), BR (VM)

### Step 313: P0 [CLOSING]

  **Closing**: P0 may close PR (DA), NS (VM), PKP (DA), BR (VM)

  NN Values: P0=-0.527, P1=-0.371, P2=+0.828
  NN Priors (top 4 of 4 legal):
     1.  92.9% ( -6.4pp) █████████████████████████████████████ PASS (CLOSING)
     2.   4.6% ( -0.3pp) █ CLOSE PR
     3.   2.3% ( +0.7pp)  CLOSE PKP
     4.   0.3% ( +6.0pp)  CLOSE NS

  MCTS Visits (top 4, 6400 total):
     1.  6328 (98.9%) Q=-0.220 ███████████████████████████████████████ PASS (CLOSING)
     2.    29 ( 0.5%) Q=-0.401  CLOSE PR
     3.    23 ( 0.4%) Q=-0.377  CLOSE PKP
     4.    20 ( 0.3%) Q=-0.639  CLOSE NS
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 55, vbackups: 6290)

  **Action: PASS (CLOSING)**

### Step 314: P2 [CLOSING]

  **Closing**: P2 may close SX (OS), KK (OS), DR (OS), SZD (SI), SJ (SI), E (SI), HR (SI), FRA

  NN Values: P0=-0.562, P1=-0.334, P2=+0.824
  NN Priors (top 4 of 4 legal):
     1.  99.8% (-10.1pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +0.1pp)  CLOSE SX
     3.   0.1% ( +9.9pp)  CLOSE DR
     4.   0.1% ( +0.1pp)  CLOSE KK

  MCTS Visits (top 4, 6400 total):
     1.  5984 (93.5%) Q=+0.812 █████████████████████████████████████ PASS (CLOSING)
     2.   386 ( 6.0%) Q=+0.812 ██ CLOSE DR
     3.    17 ( 0.3%) Q=+0.815  CLOSE SX
     4.    13 ( 0.2%) Q=+0.814  CLOSE KK
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 54, vbackups: 6009)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 11  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $36 (NW $129) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $20 (NW $135) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $14 (NW $213) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  S: $12 price=$5(idx 1) shares=bank:5/unissued:2/issued:5 income=$-6 stars=2 RECEIVERSHIP  companies=[BSE]
  OS: $13 price=$37(idx 19) shares=bank:5/unissued:0/issued:6 income=$0 stars=9 pres=P2  companies=[SX, KK, DR]
  SM: $80 price=$30(idx 17) shares=bank:2/unissued:2/issued:4 income=$10 stars=13 pres=P1  companies=[MAD]
  DA: $35 price=$24(idx 15) shares=bank:3/unissued:0/issued:5 income=$1 stars=8 pres=P0  companies=[PR, PKP]
  VM: $42 price=$14(idx 10) shares=bank:0/unissued:1/issued:3 income=$12 stars=11 pres=P0  companies=[NS, BR]
  SI: $46 price=$41(idx 20) shares=bank:0/unissued:0/issued:4 income=$39 stars=23 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Dividends**: SI

### Step 315: P2 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.396, P1=-0.406, P2=+0.836
  NN Priors (top 10 of 12 legal):
     1.  99.0% (-14.9pp) ███████████████████████████████████████ DIVIDEND $9
     2.   0.4% ( +0.5pp)  DIVIDEND $6
     3.   0.2% ( +1.3pp)  DIVIDEND $3
     4.   0.2% ( +1.2pp)  DIVIDEND $11
     5.   0.1% ( +0.2pp)  DIVIDEND $8
     6.   0.0% ( -0.0pp)  DIVIDEND $0
     7.   0.0% ( +1.3pp)  DIVIDEND $2
     8.   0.0% ( +0.0pp)  DIVIDEND $10
     9.   0.0% ( +0.0pp)  DIVIDEND $5
    10.   0.0% ( +5.0pp)  DIVIDEND $7

  MCTS Visits (top 10, 6400 total):
     1.  5349 (83.6%) Q=+0.811 █████████████████████████████████ DIVIDEND $9
     2.   334 ( 5.2%) Q=+0.816 ██ DIVIDEND $4
     3.   286 ( 4.5%) Q=+0.814 █ DIVIDEND $7
     4.   118 ( 1.8%) Q=+0.816  DIVIDEND $3
     5.   105 ( 1.6%) Q=+0.815  DIVIDEND $2
     6.   100 ( 1.6%) Q=+0.813  DIVIDEND $11
     7.    72 ( 1.1%) Q=+0.815  DIVIDEND $6
     8.    18 ( 0.3%) Q=+0.818  DIVIDEND $8
     9.     6 ( 0.1%) Q=+0.824  DIVIDEND $0
    10.     6 ( 0.1%) Q=+0.822  DIVIDEND $10
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 53, vbackups: 5457)

  **Action: DIVIDEND $9**

### Step 316: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.340, P1=-0.424, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  99.1% ( -7.7pp) ███████████████████████████████████████ DIVIDEND $2
     2.   0.9% ( +2.1pp)  DIVIDEND $1
     3.   0.0% ( +5.6pp)  DIVIDEND $0

  MCTS Visits (top 3, 6400 total):
     1.  6043 (94.4%) Q=+0.811 █████████████████████████████████████ DIVIDEND $2
     2.   219 ( 3.4%) Q=+0.816 █ DIVIDEND $0
     3.   138 ( 2.2%) Q=+0.813  DIVIDEND $1
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 52, vbackups: 6113)

  **Action: DIVIDEND $2**

### Step 317: P1 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.299, P1=-0.482, P2=+0.828
  NN Priors (top 10 of 11 legal):
     1.  88.7% (-12.5pp) ███████████████████████████████████ DIVIDEND $10
     2.   4.2% ( +3.0pp) █ DIVIDEND $9
     3.   1.7% ( +2.3pp)  DIVIDEND $5
     4.   1.6% ( +0.0pp)  DIVIDEND $4
     5.   0.8% ( +0.2pp)  DIVIDEND $0
     6.   0.8% ( +0.4pp)  DIVIDEND $3
     7.   0.6% ( +1.9pp)  DIVIDEND $8
     8.   0.6% ( +0.0pp)  DIVIDEND $2
     9.   0.4% ( +0.2pp)  DIVIDEND $1
    10.   0.4% ( +3.9pp)  DIVIDEND $6

  MCTS Visits (top 10, 6400 total):
     1.  4255 (66.5%) Q=-0.520 ██████████████████████████ DIVIDEND $10
     2.   953 (14.9%) Q=-0.507 █████ DIVIDEND $9
     3.   209 ( 3.3%) Q=-0.514 █ DIVIDEND $8
     4.   169 ( 2.6%) Q=-0.533 █ DIVIDEND $6
     5.   164 ( 2.6%) Q=-0.502 █ DIVIDEND $2
     6.   155 ( 2.4%) Q=-0.504  DIVIDEND $7
     7.   143 ( 2.2%) Q=-0.539  DIVIDEND $5
     8.   125 ( 2.0%) Q=-0.513  DIVIDEND $0
     9.    97 ( 1.5%) Q=-0.507  DIVIDEND $1
    10.    74 ( 1.2%) Q=-0.521  DIVIDEND $3
  A0GB Value: P0=-0.186, P1=-0.480, P2=+0.793 (depth: 51, vbackups: 6048)

  **Action: DIVIDEND $10**

### Step 318: P0 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=-0.324, P1=-0.424, P2=+0.832
  NN Priors (top 8 of 8 legal):
     1.  93.9% (-13.9pp) █████████████████████████████████████ DIVIDEND $7
     2.   5.1% ( +7.6pp) ██ DIVIDEND $6
     3.   0.3% ( +2.5pp)  DIVIDEND $5
     4.   0.2% ( -0.0pp)  DIVIDEND $0
     5.   0.2% ( +0.1pp)  DIVIDEND $3
     6.   0.1% ( +0.3pp)  DIVIDEND $4
     7.   0.1% ( +0.6pp)  DIVIDEND $1
     8.   0.1% ( +2.9pp)  DIVIDEND $2

  MCTS Visits (top 8, 6400 total):
     1.  6105 (95.4%) Q=-0.202 ██████████████████████████████████████ DIVIDEND $7
     2.   218 ( 3.4%) Q=-0.267 █ DIVIDEND $6
     3.    33 ( 0.5%) Q=-0.307  DIVIDEND $5
     4.    30 ( 0.5%) Q=-0.358  DIVIDEND $2
     5.     6 ( 0.1%) Q=-0.333  DIVIDEND $1
     6.     6 ( 0.1%) Q=-0.299  DIVIDEND $4
     7.     1 ( 0.0%) Q=-0.406  DIVIDEND $3
     8.     1 ( 0.0%) Q=-0.434  DIVIDEND $0
  A0GB Value: P0=-0.063, P1=-0.648, P2=+0.805 (depth: 34, vbackups: 4306)

  **Action: DIVIDEND $7**

### Step 319: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.338, P1=-0.408, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  95.8% (-12.4pp) ██████████████████████████████████████ DIVIDEND $4
     2.   2.2% ( +5.2pp)  DIVIDEND $3
     3.   0.7% ( +2.8pp)  DIVIDEND $2
     4.   0.7% ( +2.2pp)  DIVIDEND $0
     5.   0.6% ( +2.3pp)  DIVIDEND $1

  MCTS Visits (top 5, 6400 total):
     1.  6194 (96.8%) Q=-0.200 ██████████████████████████████████████ DIVIDEND $4
     2.    83 ( 1.3%) Q=-0.324  DIVIDEND $3
     3.    45 ( 0.7%) Q=-0.288  DIVIDEND $1
     4.    39 ( 0.6%) Q=-0.318  DIVIDEND $0
     5.    39 ( 0.6%) Q=-0.311  DIVIDEND $2
  A0GB Value: P0=-0.063, P1=-0.648, P2=+0.805 (depth: 33, vbackups: 6104)

  **Action: DIVIDEND $4**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 11  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $60 (NW $166) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $53 (NW $165) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $41 (NW $244) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $40 price=$22(idx 14) shares=bank:2/unissued:2/issued:4 income=$10 stars=9 pres=P1  companies=[MAD]
  DA: $0 price=$20(idx 13) shares=bank:3/unissued:0/issued:5 income=$1 stars=5 pres=P0  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$50(idx 22) shares=bank:0/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Issue**: SM

### Step 320: P1 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.218, P1=-0.461, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  65.3% ( -8.3pp) ██████████████████████████ ISSUE SM shares
     2.  34.7% ( +8.3pp) █████████████ PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  3648 (57.0%) Q=-0.539 ██████████████████████ ISSUE SM shares
     2.  2752 (43.0%) Q=-0.539 █████████████████ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.063, P1=-0.648, P2=+0.805 (depth: 32, vbackups: 6108)

  **Action: ISSUE SM shares**

### Step 321: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.245, P1=-0.398, P2=+0.812
  NN Priors (top 2 of 2 legal):
     1.  93.7% ( -4.1pp) █████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   6.3% ( +4.1pp) ██ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6283 (98.2%) Q=-0.184 ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   117 ( 1.8%) Q=-0.296  ISSUE VM shares
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 42, vbackups: 3732)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 11  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $60 (NW $166) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $53 (NW $165) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $41 (NW $244) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$22(idx 14) shares=bank:3/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  DA: $0 price=$20(idx 13) shares=bank:3/unissued:0/issued:5 income=$1 stars=5 pres=P0  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$50(idx 22) shares=bank:0/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**IPO**: FRA

### Step 322: P2 [IPO]

  **IPO**: FRA

  NN Values: P0=-0.241, P1=-0.383, P2=+0.801
  NN Priors (top 4 of 4 legal):
     1.  99.9% ( -8.8pp) ███████████████████████████████████████ IPO FRA → float PR
     2.   0.0% ( +6.7pp)  IPO FRA → float S
     3.   0.0% ( +1.2pp)  PASS (IPO)
     4.   0.0% ( +0.9pp)  IPO FRA → float JS

  MCTS Visits (top 4, 6400 total):
     1.  6069 (94.8%) Q=+0.811 █████████████████████████████████████ IPO FRA → float PR
     2.   248 ( 3.9%) Q=+0.814 █ IPO FRA → float S
     3.    42 ( 0.7%) Q=+0.811  IPO FRA → float JS
     4.    41 ( 0.6%) Q=+0.791  PASS (IPO)
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 41, vbackups: 6073)

  **Action: IPO FRA → float PR**

Phase: PAR  |  Turn: 11  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $60 (NW $166) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $53 (NW $165) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $41 (NW $244) order=2 income=$10  companies=[FRA]  shares=[OS=1 (pres), DA=1, SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$22(idx 14) shares=bank:3/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  DA: $0 price=$20(idx 13) shares=bank:3/unissued:0/issued:5 income=$1 stars=5 pres=P0  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$50(idx 22) shares=bank:0/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**PAR**: FRA -> PR

### Step 323: P2 [PAR]

  **PAR**: FRA -> PR

  NN Values: P0=-0.266, P1=-0.361, P2=+0.805
  NN Priors (top 3 of 3 legal):
     1.  99.7% ( -8.6pp) ███████████████████████████████████████ PAR PR @$33 (IPO FRA)
     2.   0.3% ( +5.7pp)  PAR PR @$37 (IPO FRA)
     3.   0.0% ( +2.9pp)  PAR PR @$30 (IPO FRA)

  MCTS Visits (top 3, 6400 total):
     1.  5952 (93.0%) Q=+0.811 █████████████████████████████████████ PAR PR @$33 (IPO FRA)
     2.   332 ( 5.2%) Q=+0.822 ██ PAR PR @$37 (IPO FRA)
     3.   116 ( 1.8%) Q=+0.803  PAR PR @$30 (IPO FRA)
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 40, vbackups: 5993)

  **Action: PAR PR @$33 (IPO FRA)**

--- Turn 12 ---

Phase: INVEST  |  Turn: 12  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $60 (NW $166) order=0 income=$0  shares=[DA=1 (pres), VM=2 (pres), SI=1]
  P1: $53 (NW $165) order=1 income=$0  shares=[SM=2 (pres), VM=1, SI=1]
  P2: $31 (NW $244) order=2 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1, SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$22(idx 14) shares=bank:3/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $76 price=$33(idx 18) shares=bank:2/unissued:1/issued:4 income=$11 stars=12 pres=P2  companies=[FRA]
  DA: $0 price=$20(idx 13) shares=bank:3/unissued:0/issued:5 income=$1 stars=5 pres=P0  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$50(idx 22) shares=bank:0/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining


### Step 324: P0 [INVEST]

  NN Values: P0=-0.316, P1=-0.361, P2=+0.824
  NN Priors (top 9 of 9 legal):
     1.  91.6% (-13.6pp) ████████████████████████████████████ BUY PR share
     2.   7.1% ( -1.1pp) ██ SELL SI share
     3.   1.1% ( +2.0pp)  SELL DA share
     4.   0.1% ( +0.3pp)  PASS (INVEST)
     5.   0.1% ( +5.4pp)  AUCTION slot 0 (CDG, face $60)
     6.   0.0% ( +0.7pp)  BUY SM share
     7.   0.0% ( +2.3pp)  BUY OS share
     8.   0.0% ( +2.6pp)  BUY DA share
     9.   0.0% ( +1.3pp)  SELL VM share

  MCTS Visits (top 9, 6400 total):
     1.  6262 (97.8%) Q=-0.181 ███████████████████████████████████████ BUY PR share
     2.    44 ( 0.7%) Q=-0.348  SELL SI share
     3.    24 ( 0.4%) Q=-0.565  AUCTION slot 0 (CDG, face $60)
     4.    21 ( 0.3%) Q=-0.440  SELL DA share
     5.    15 ( 0.2%) Q=-0.466  BUY OS share
     6.    12 ( 0.2%) Q=-0.528  BUY DA share
     7.     9 ( 0.1%) Q=-0.477  SELL VM share
     8.     9 ( 0.1%) Q=-0.302  BUY SM share
     9.     4 ( 0.1%) Q=-0.285  PASS (INVEST)
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 39, vbackups: 6225)

  **Action: BUY PR share**

### Step 325: P1 [INVEST]

  NN Values: P0=-0.342, P1=-0.414, P2=+0.828
  NN Priors (top 8 of 8 legal):
     1.  92.3% (-13.6pp) ████████████████████████████████████ SELL SI share
     2.   7.6% ( -0.0pp) ███ BUY PR share
     3.   0.1% ( +2.1pp)  BUY SM share
     4.   0.0% ( +3.2pp)  PASS (INVEST)
     5.   0.0% ( +5.9pp)  SELL VM share
     6.   0.0% ( +1.1pp)  SELL SM share
     7.   0.0% ( +0.7pp)  BUY OS share
     8.   0.0% ( +0.5pp)  BUY DA share

  MCTS Visits (top 8, 6400 total):
     1.  5677 (88.7%) Q=-0.561 ███████████████████████████████████ SELL SI share
     2.   344 ( 5.4%) Q=-0.574 ██ BUY PR share
     3.   175 ( 2.7%) Q=-0.568 █ PASS (INVEST)
     4.    81 ( 1.3%) Q=-0.581  BUY SM share
     5.    57 ( 0.9%) Q=-0.696  SELL VM share
     6.    25 ( 0.4%) Q=-0.626  SELL SM share
     7.    24 ( 0.4%) Q=-0.597  BUY OS share
     8.    17 ( 0.3%) Q=-0.632  BUY DA share
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 38, vbackups: 6048)

  **Action: SELL SI share**

### Step 326: P2 [INVEST]

  NN Values: P0=-0.430, P1=-0.424, P2=+0.816
  NN Priors (top 8 of 8 legal):
     1.  97.2% (-14.5pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.9% ( +0.7pp)  SELL DA share
     3.   0.3% ( +1.3pp)  SELL OS share
     4.   0.2% ( +0.7pp)  BUY OS share
     5.   0.2% ( +0.1pp)  BUY SM share
     6.   0.1% ( +3.4pp)  SELL PR share
     7.   0.1% ( +3.4pp)  SELL SI share
     8.   0.1% ( +4.8pp)  BUY DA share

  MCTS Visits (top 8, 6400 total):
     1.  5536 (86.5%) Q=+0.810 ██████████████████████████████████ PASS (INVEST)
     2.   190 ( 3.0%) Q=+0.817 █ SELL DA share
     3.   173 ( 2.7%) Q=+0.797 █ BUY DA share
     4.   168 ( 2.6%) Q=+0.807 █ SELL PR share
     5.   167 ( 2.6%) Q=+0.807 █ SELL SI share
     6.    99 ( 1.5%) Q=+0.810  SELL OS share
     7.    46 ( 0.7%) Q=+0.806  BUY OS share
     8.    21 ( 0.3%) Q=+0.814  BUY SM share
  A0GB Value: P0=-0.019, P1=-0.801, P2=+0.816 (depth: 37, vbackups: 5747)

  **Action: PASS (INVEST)**

### Step 327: P0 [INVEST]

  NN Values: P0=-0.559, P1=-0.357, P2=+0.824
  NN Priors (top 5 of 5 legal):
     1.  43.6% ( -1.8pp) █████████████████ SELL DA share
     2.  42.9% ( -3.5pp) █████████████████ PASS (INVEST)
     3.  12.8% ( +1.8pp) █████ SELL SI share
     4.   0.4% ( +0.3pp)  SELL VM share
     5.   0.3% ( +3.1pp)  SELL PR share

  MCTS Visits (top 5, 6400 total):
     1.  5458 (85.3%) Q=-0.155 ██████████████████████████████████ SELL SI share
     2.   618 ( 9.7%) Q=-0.244 ███ SELL DA share
     3.   299 ( 4.7%) Q=-0.331 █ PASS (INVEST)
     4.    17 ( 0.3%) Q=-0.439  SELL PR share
     5.     8 ( 0.1%) Q=-0.409  SELL VM share
  A0GB Value: P0=-0.039, P1=-0.770, P2=+0.805 (depth: 38, vbackups: 5634)

  **Action: SELL SI share**

### Step 328: P1 [INVEST]

  NN Values: P0=-0.199, P1=-0.508, P2=+0.809
  NN Priors (top 9 of 9 legal):
     1.  45.4% ( -5.2pp) ██████████████████ BUY PR share
     2.  40.7% ( -7.0pp) ████████████████ BUY SM share
     3.  10.6% ( -1.2pp) ████ AUCTION slot 0 (CDG, face $60)
     4.   1.4% ( +0.2pp)  PASS (INVEST)
     5.   1.0% ( +1.1pp)  BUY SI share
     6.   0.5% ( +0.7pp)  BUY OS share
     7.   0.3% ( +9.9pp)  SELL VM share
     8.   0.1% ( +0.9pp)  SELL SM share
     9.   0.0% ( +0.6pp)  BUY DA share

  MCTS Visits (top 9, 6400 total):
     1.  2081 (32.5%) Q=-0.595 █████████████ BUY SM share
     2.  1581 (24.7%) Q=-0.607 █████████ BUY PR share
     3.  1216 (19.0%) Q=-0.574 ███████ PASS (INVEST)
     4.   882 (13.8%) Q=-0.588 █████ AUCTION slot 0 (CDG, face $60)
     5.   247 ( 3.9%) Q=-0.582 █ BUY OS share
     6.   230 ( 3.6%) Q=-0.589 █ BUY SI share
     7.   113 ( 1.8%) Q=-0.695  SELL VM share
     8.    25 ( 0.4%) Q=-0.630  SELL SM share
     9.    25 ( 0.4%) Q=-0.617  BUY DA share
  A0GB Value: P0=-0.055, P1=-0.797, P2=+0.836 (depth: 39, vbackups: 5457)

  **Action: BUY SM share**

### Step 329: P2 [INVEST]

  NN Values: P0=-0.214, P1=-0.486, P2=+0.805
  NN Priors (top 8 of 8 legal):
     1.  92.4% (-10.7pp) ████████████████████████████████████ PASS (INVEST)
     2.   6.7% ( +0.8pp) ██ SELL DA share
     3.   0.2% ( +0.5pp)  SELL OS share
     4.   0.2% ( +0.1pp)  SELL PR share
     5.   0.1% ( +8.7pp)  BUY SM share
     6.   0.1% ( +0.3pp)  BUY OS share
     7.   0.1% ( +0.1pp)  SELL SI share
     8.   0.1% ( +0.1pp)  BUY DA share

  MCTS Visits (top 8, 6400 total):
     1.  5179 (80.9%) Q=+0.815 ████████████████████████████████ PASS (INVEST)
     2.   604 ( 9.4%) Q=+0.819 ███ SELL DA share
     3.   532 ( 8.3%) Q=+0.813 ███ BUY SM share
     4.    35 ( 0.5%) Q=+0.807  SELL OS share
     5.    25 ( 0.4%) Q=+0.811  BUY OS share
     6.    11 ( 0.2%) Q=+0.799  SELL PR share
     7.     9 ( 0.1%) Q=+0.802  SELL SI share
     8.     5 ( 0.1%) Q=+0.800  BUY DA share
  A0GB Value: P0=-0.042, P1=-0.781, P2=+0.797 (depth: 65, vbackups: 2080)

  **Action: PASS (INVEST)**

### Step 330: P0 [INVEST]

  NN Values: P0=-0.254, P1=-0.500, P2=+0.816
  NN Priors (top 10 of 10 legal):
     1.  48.8% ( -5.4pp) ███████████████████ SELL DA share
     2.  38.0% ( -5.1pp) ███████████████ BUY PR share
     3.  10.6% ( -2.0pp) ████ AUCTION slot 0 (CDG, face $60)
     4.   2.2% ( +1.1pp)  PASS (INVEST)
     5.   0.2% ( +1.1pp)  BUY SM share
     6.   0.1% ( +2.5pp)  BUY SI share
     7.   0.0% ( +0.5pp)  BUY OS share
     8.   0.0% ( +6.8pp)  BUY DA share
     9.   0.0% ( +0.1pp)  SELL VM share
    10.   0.0% ( +0.2pp)  SELL PR share

  MCTS Visits (top 10, 6400 total):
     1.  5713 (89.3%) Q=-0.092 ███████████████████████████████████ BUY PR share
     2.   497 ( 7.8%) Q=-0.203 ███ SELL DA share
     3.    82 ( 1.3%) Q=-0.224  AUCTION slot 0 (CDG, face $60)
     4.    43 ( 0.7%) Q=-0.206  PASS (INVEST)
     5.    35 ( 0.5%) Q=-0.352  BUY DA share
     6.    15 ( 0.2%) Q=-0.348  BUY SI share
     7.     9 ( 0.1%) Q=-0.383  BUY OS share
     8.     5 ( 0.1%) Q=-0.613  BUY SM share
     9.     1 ( 0.0%) Q=-0.586  SELL PR share
  A0GB Value: P0=-0.037, P1=-0.816, P2=+0.824 (depth: 45, vbackups: 5134)

  **Action: BUY PR share**

### Step 331: P1 [INVEST]

  NN Values: P0=-0.170, P1=-0.543, P2=+0.824
  NN Priors (top 8 of 8 legal):
     1.  48.2% ( -6.0pp) ███████████████████ AUCTION slot 0 (CDG, face $60)
     2.  47.4% ( -0.9pp) ██████████████████ BUY SM share
     3.   2.8% ( +1.7pp) █ SELL SM share
     4.   0.6% ( -0.0pp)  SELL VM share
     5.   0.5% ( +0.1pp)  PASS (INVEST)
     6.   0.3% ( +1.8pp)  BUY OS share
     7.   0.1% ( +2.8pp)  BUY SI share
     8.   0.1% ( +0.5pp)  BUY DA share

  MCTS Visits (top 8, 6400 total):
     1.  2537 (39.6%) Q=-0.668 ███████████████ BUY SM share
     2.  1377 (21.5%) Q=-0.684 ████████ AUCTION slot 0 (CDG, face $60)
     3.  1074 (16.8%) Q=-0.648 ██████ SELL SM share
     4.   898 (14.0%) Q=-0.646 █████ BUY OS share
     5.   251 ( 3.9%) Q=-0.647 █ PASS (INVEST)
     6.   121 ( 1.9%) Q=-0.681  BUY SI share
     7.   114 ( 1.8%) Q=-0.655  BUY DA share
     8.    28 ( 0.4%) Q=-0.670  SELL VM share
  A0GB Value: P0=-0.051, P1=-0.840, P2=+0.820 (depth: 46, vbackups: 5680)

  **Action: BUY SM share**

### Step 332: P2 [INVEST]

  NN Values: P0=-0.344, P1=-0.449, P2=+0.824
  NN Priors (top 6 of 6 legal):
     1.  98.8% (-13.2pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.8% ( +5.2pp)  SELL DA share
     3.   0.2% ( +1.4pp)  SELL SI share
     4.   0.1% ( +1.5pp)  SELL PR share
     5.   0.1% ( +4.5pp)  SELL OS share
     6.   0.0% ( +0.6pp)  BUY DA share

  MCTS Visits (top 6, 6400 total):
     1.  5466 (85.4%) Q=+0.819 ██████████████████████████████████ PASS (INVEST)
     2.   414 ( 6.5%) Q=+0.820 ██ SELL DA share
     3.   237 ( 3.7%) Q=+0.813 █ SELL OS share
     4.   147 ( 2.3%) Q=+0.824  SELL PR share
     5.   110 ( 1.7%) Q=+0.820  SELL SI share
     6.    26 ( 0.4%) Q=+0.807  BUY DA share
  A0GB Value: P0=-0.037, P1=-0.785, P2=+0.793 (depth: 68, vbackups: 2536)

  **Action: PASS (INVEST)**

### Step 333: P0 [INVEST]

  NN Values: P0=-0.414, P1=-0.414, P2=+0.824
  NN Priors (top 4 of 4 legal):
     1.  93.3% (-11.8pp) █████████████████████████████████████ SELL DA share
     2.   6.7% ( +3.6pp) ██ PASS (INVEST)
     3.   0.0% ( +4.9pp)  SELL VM share
     4.   0.0% ( +3.4pp)  SELL PR share

  MCTS Visits (top 4, 6400 total):
     1.  6294 (98.3%) Q=-0.063 ███████████████████████████████████████ SELL DA share
     2.    80 ( 1.2%) Q=-0.222  PASS (INVEST)
     3.    14 ( 0.2%) Q=-0.501  SELL PR share
     4.    12 ( 0.2%) Q=-0.574  SELL VM share
  A0GB Value: P0=-0.034, P1=-0.781, P2=+0.805 (depth: 72, vbackups: 5465)

  **Action: SELL DA share**

### Step 334: P1 [INVEST]

  NN Values: P0=-0.117, P1=-0.590, P2=+0.832
  NN Priors (top 6 of 6 legal):
     1.  56.5% ( -7.6pp) ██████████████████████ BUY SM share
     2.  23.6% ( +2.1pp) █████████ PASS (INVEST)
     3.  14.6% ( -2.3pp) █████ SELL SM share
     4.   2.5% ( +3.5pp) █ BUY DA share
     5.   2.1% ( +2.7pp)  SELL VM share
     6.   0.7% ( +1.6pp)  BUY OS share

  MCTS Visits (top 6, 6400 total):
     1.  1896 (29.6%) Q=-0.711 ███████████ BUY SM share
     2.  1608 (25.1%) Q=-0.698 ██████████ PASS (INVEST)
     3.  1274 (19.9%) Q=-0.679 ███████ BUY OS share
     4.  1018 (15.9%) Q=-0.684 ██████ BUY DA share
     5.   516 ( 8.1%) Q=-0.709 ███ SELL SM share
     6.    88 ( 1.4%) Q=-0.754  SELL VM share
  A0GB Value: P0=-0.034, P1=-0.781, P2=+0.805 (depth: 71, vbackups: 5969)

  **Action: BUY SM share**

### Step 335: P2 [INVEST]

  NN Values: P0=-0.189, P1=-0.539, P2=+0.812
  NN Priors (top 7 of 7 legal):
     1.  99.2% (-12.4pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.2% ( +1.8pp)  SELL SI share
     3.   0.2% ( +1.6pp)  SELL OS share
     4.   0.1% ( +2.5pp)  SELL PR share
     5.   0.1% ( -0.0pp)  SELL DA share
     6.   0.1% ( +2.0pp)  BUY DA share
     7.   0.1% ( +4.4pp)  BUY OS share

  MCTS Visits (top 7, 6400 total):
     1.  5667 (88.5%) Q=+0.819 ███████████████████████████████████ PASS (INVEST)
     2.   222 ( 3.5%) Q=+0.811 █ BUY OS share
     3.   173 ( 2.7%) Q=+0.817 █ SELL PR share
     4.   140 ( 2.2%) Q=+0.818  SELL SI share
     5.   135 ( 2.1%) Q=+0.817  BUY DA share
     6.    57 ( 0.9%) Q=+0.797  SELL OS share
     7.     6 ( 0.1%) Q=+0.818  SELL DA share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 88, vbackups: 2129)

  **Action: PASS (INVEST)**

### Step 336: P0 [INVEST]

  NN Values: P0=-0.230, P1=-0.566, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  91.7% (-12.3pp) ████████████████████████████████████ PASS (INVEST)
     2.   3.6% ( +3.3pp) █ SELL VM share
     3.   2.4% ( +1.9pp)  BUY DA share
     4.   2.0% ( +2.1pp)  BUY OS share
     5.   0.2% ( +5.1pp)  SELL PR share

  MCTS Visits (top 5, 6400 total):
     1.  6299 (98.4%) Q=-0.039 ███████████████████████████████████████ PASS (INVEST)
     2.    33 ( 0.5%) Q=-0.237  BUY OS share
     3.    29 ( 0.5%) Q=-0.368  SELL VM share
     4.    20 ( 0.3%) Q=-0.391  SELL PR share
     5.    19 ( 0.3%) Q=-0.365  BUY DA share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 87, vbackups: 5666)

  **Action: PASS (INVEST)**

### Step 337: P1 [INVEST]

  NN Values: P0=-0.106, P1=-0.609, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  69.7% (-10.4pp) ███████████████████████████ PASS (INVEST)
     2.  27.1% ( -0.3pp) ██████████ SELL SM share
     3.   3.2% (+10.7pp) █ SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  3673 (57.4%) Q=-0.726 ██████████████████████ SELL SM share
     2.  2454 (38.3%) Q=-0.749 ███████████████ PASS (INVEST)
     3.   273 ( 4.3%) Q=-0.787 █ SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 86, vbackups: 5977)

  **Action: SELL SM share**

### Step 338: P2 [INVEST]

  NN Values: P0=-0.061, P1=-0.609, P2=+0.809
  NN Priors (top 6 of 6 legal):
     1.  99.4% (-12.4pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.2% ( +2.4pp)  SELL SI share
     3.   0.1% ( +3.3pp)  SELL OS share
     4.   0.1% ( +1.8pp)  SELL PR share
     5.   0.1% ( +2.1pp)  BUY DA share
     6.   0.1% ( +2.7pp)  SELL DA share

  MCTS Visits (top 6, 6400 total):
     1.  5631 (88.0%) Q=+0.819 ███████████████████████████████████ PASS (INVEST)
     2.   197 ( 3.1%) Q=+0.821 █ SELL SI share
     3.   176 ( 2.8%) Q=+0.812 █ SELL OS share
     4.   148 ( 2.3%) Q=+0.814  SELL DA share
     5.   140 ( 2.2%) Q=+0.818  BUY DA share
     6.   108 ( 1.7%) Q=+0.815  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 85, vbackups: 3672)

  **Action: PASS (INVEST)**

### Step 339: P0 [INVEST]

  NN Values: P0=-0.097, P1=-0.598, P2=+0.824
  NN Priors (top 6 of 6 legal):
     1.  98.8% ( -4.2pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.5% ( +2.7pp)  BUY DA share
     3.   0.3% ( +0.1pp)  BUY SM share
     4.   0.2% ( +0.3pp)  BUY OS share
     5.   0.1% ( +1.0pp)  SELL VM share
     6.   0.0% ( +0.2pp)  SELL PR share

  MCTS Visits (top 6, 6400 total):
     1.  6361 (99.4%) Q=-0.038 ███████████████████████████████████████ PASS (INVEST)
     2.    20 ( 0.3%) Q=-0.306  BUY DA share
     3.     8 ( 0.1%) Q=-0.244  SELL VM share
     4.     7 ( 0.1%) Q=-0.201  BUY OS share
     5.     3 ( 0.0%) Q=-0.192  BUY SM share
     6.     1 ( 0.0%) Q=-0.420  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 84, vbackups: 5627)

  **Action: PASS (INVEST)**

### Step 340: P1 [INVEST]

  NN Values: P0=-0.068, P1=-0.625, P2=+0.828
  NN Priors (top 6 of 6 legal):
     1.  78.3% (-11.6pp) ███████████████████████████████ BUY SM share
     2.   9.1% ( -0.8pp) ███ PASS (INVEST)
     3.   5.0% ( +5.2pp) ██ SELL SM share
     4.   5.0% ( +4.5pp) █ BUY DA share
     5.   1.5% ( +2.1pp)  SELL VM share
     6.   1.1% ( +0.6pp)  BUY OS share

  MCTS Visits (top 6, 6400 total):
     1.  2396 (37.4%) Q=-0.747 ██████████████ BUY SM share
     2.  2133 (33.3%) Q=-0.711 █████████████ BUY OS share
     3.   783 (12.2%) Q=-0.726 ████ BUY DA share
     4.   779 (12.2%) Q=-0.724 ████ PASS (INVEST)
     5.   234 ( 3.7%) Q=-0.770 █ SELL SM share
     6.    75 ( 1.2%) Q=-0.776  SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 83, vbackups: 5329)

  **Action: BUY SM share**

### Step 341: P2 [INVEST]

  NN Values: P0=-0.114, P1=-0.617, P2=+0.816
  NN Priors (top 7 of 7 legal):
     1.  99.1% (-13.8pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.3% ( +1.9pp)  SELL SI share
     3.   0.2% ( +7.9pp)  SELL OS share
     4.   0.1% ( +0.2pp)  SELL PR share
     5.   0.1% ( +1.0pp)  SELL DA share
     6.   0.1% ( +1.2pp)  BUY DA share
     7.   0.1% ( +1.6pp)  BUY OS share

  MCTS Visits (top 7, 6400 total):
     1.  5730 (89.5%) Q=+0.820 ███████████████████████████████████ PASS (INVEST)
     2.   267 ( 4.2%) Q=+0.798 █ SELL OS share
     3.   152 ( 2.4%) Q=+0.820  SELL SI share
     4.    93 ( 1.5%) Q=+0.815  BUY OS share
     5.    86 ( 1.3%) Q=+0.819  BUY DA share
     6.    50 ( 0.8%) Q=+0.809  SELL DA share
     7.    22 ( 0.3%) Q=+0.818  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 82, vbackups: 3015)

  **Action: PASS (INVEST)**

### Step 342: P0 [INVEST]

  NN Values: P0=-0.153, P1=-0.609, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  91.5% (-12.9pp) ████████████████████████████████████ PASS (INVEST)
     2.   3.3% ( +3.4pp) █ SELL VM share
     3.   2.7% ( +2.5pp) █ BUY DA share
     4.   2.2% ( +1.3pp)  BUY OS share
     5.   0.3% ( +5.7pp)  SELL PR share

  MCTS Visits (top 5, 6400 total):
     1.  6216 (97.1%) Q=-0.025 ██████████████████████████████████████ PASS (INVEST)
     2.    71 ( 1.1%) Q=-0.080  BUY OS share
     3.    54 ( 0.8%) Q=-0.143  BUY DA share
     4.    31 ( 0.5%) Q=-0.323  SELL VM share
     5.    28 ( 0.4%) Q=-0.310  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 81, vbackups: 5729)

  **Action: PASS (INVEST)**

### Step 343: P1 [INVEST]

  NN Values: P0=-0.058, P1=-0.637, P2=+0.828
  NN Priors (top 3 of 3 legal):
     1.  77.1% ( -9.2pp) ██████████████████████████████ PASS (INVEST)
     2.  19.2% ( +6.5pp) ███████ SELL SM share
     3.   3.7% ( +2.6pp) █ SELL VM share

  MCTS Visits (top 3, 6400 total):
     1.  3286 (51.3%) Q=-0.762 ████████████████████ PASS (INVEST)
     2.  2953 (46.1%) Q=-0.746 ██████████████████ SELL SM share
     3.   161 ( 2.5%) Q=-0.790 █ SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 80, vbackups: 5770)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: ACQ_SELECT_CORP  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $76 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$11 stars=12 pres=P2  companies=[FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($1), PR($76), DA($0), SI($10)

### Step 344: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($1), PR($76), DA($0), SI($10)

  NN Values: P0=-0.066, P1=-0.656, P2=+0.809
  NN Priors (top 3 of 3 legal):
     1.  99.7% (-11.4pp) ███████████████████████████████████████ ACQ select PR
     2.   0.3% ( +4.5pp)  PASS (ACQ_SELECT_CORP)
     3.   0.1% ( +6.9pp)  ACQ select SI

  MCTS Visits (top 3, 6400 total):
     1.  5913 (92.4%) Q=+0.821 ████████████████████████████████████ ACQ select PR
     2.   302 ( 4.7%) Q=+0.809 █ ACQ select SI
     3.   185 ( 2.9%) Q=+0.805 █ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 79, vbackups: 3730)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $76 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$11 stars=12 pres=P2  companies=[FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with PR ($76)

### Step 345: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with PR ($76)

  NN Values: P0=-0.024, P1=-0.656, P2=+0.805
  NN Priors (top 9 of 9 legal):
     1.  98.5% (-14.8pp) ███████████████████████████████████████ ACQ target KK (with PR)
     2.   0.3% ( +0.1pp)  ACQ target SX (with PR)
     3.   0.2% ( +5.1pp)  ACQ target SZD (with PR)
     4.   0.2% ( +3.3pp)  ACQ target HR (with PR)
     5.   0.2% ( +3.4pp)  ACQ target SJ (with PR)
     6.   0.2% ( +0.3pp)  ACQ target DR (with PR)
     7.   0.2% ( +1.0pp)  ACQ target PR (with PR)
     8.   0.1% ( +0.8pp)  ACQ target PKP (with PR)
     9.   0.1% ( +0.9pp)  ACQ target E (with PR)

  MCTS Visits (top 9, 6400 total):
     1.  5745 (89.8%) Q=+0.821 ███████████████████████████████████ ACQ target KK (with PR)
     2.   225 ( 3.5%) Q=+0.816 █ ACQ target SZD (with PR)
     3.   153 ( 2.4%) Q=+0.814  ACQ target SJ (with PR)
     4.   104 ( 1.6%) Q=+0.801  ACQ target HR (with PR)
     5.    58 ( 0.9%) Q=+0.820  ACQ target PKP (with PR)
     6.    46 ( 0.7%) Q=+0.807  ACQ target PR (with PR)
     7.    32 ( 0.5%) Q=+0.799  ACQ target E (with PR)
     8.    25 ( 0.4%) Q=+0.815  ACQ target DR (with PR)
     9.    12 ( 0.2%) Q=+0.803  ACQ target SX (with PR)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 78, vbackups: 5795)

  **Action: ACQ target KK (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$0 stars=8 pres=P2  companies=[SX, KK, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $76 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$11 stars=12 pres=P2  companies=[FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 PR -> KK (price range $11-$28)

### Step 346: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 PR -> KK (price range $11-$28)

  NN Values: P0=-0.057, P1=-0.664, P2=+0.809
  NN Priors (top 10 of 18 legal):
     1.  99.8% (-13.5pp) ███████████████████████████████████████ ACQUIRE KK with PR @ $11
     2.   0.2% ( +2.8pp)  ACQUIRE KK with PR @ $12
     3.   0.0% ( +0.0pp)  ACQUIRE KK with PR @ $28
     4.   0.0% ( +0.0pp)  ACQUIRE KK with PR @ $16
     5.   0.0% ( +0.2pp)  ACQUIRE KK with PR @ $21
     6.   0.0% ( +0.1pp)  ACQUIRE KK with PR @ $15
     7.   0.0% ( +0.4pp)  ACQUIRE KK with PR @ $22
     8.   0.0% ( -0.0pp)  ACQUIRE KK with PR @ $13
     9.   0.0% ( -0.0pp)  ACQUIRE KK with PR @ $20
    10.   0.0% ( +0.0pp)  ACQUIRE KK with PR @ $17

  MCTS Visits (top 10, 6400 total):
     1.  5850 (91.4%) Q=+0.821 ████████████████████████████████████ ACQUIRE KK with PR @ $11
     2.   231 ( 3.6%) Q=+0.814 █ ACQUIRE KK with PR @ $26
     3.   137 ( 2.1%) Q=+0.817  ACQUIRE KK with PR @ $12
     4.   118 ( 1.8%) Q=+0.817  ACQUIRE KK with PR @ $14
     5.    18 ( 0.3%) Q=+0.807  ACQUIRE KK with PR @ $25
     6.    16 ( 0.2%) Q=+0.808  ACQUIRE KK with PR @ $22
     7.    12 ( 0.2%) Q=+0.799  ACQUIRE KK with PR @ $27
     8.     8 ( 0.1%) Q=+0.798  ACQUIRE KK with PR @ $23
     9.     6 ( 0.1%) Q=+0.799  ACQUIRE KK with PR @ $21
    10.     3 ( 0.0%) Q=+0.802  ACQUIRE KK with PR @ $15
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 77, vbackups: 5859)

  **Action: ACQUIRE KK with PR @ $11**

Phase: ACQ_SELECT_CORP  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $65 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$14 stars=14 pres=P2  companies=[KK*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($1), PR($65), DA($0), SI($10)

### Step 347: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($1), PR($65), DA($0), SI($10)

  NN Values: P0=-0.094, P1=-0.668, P2=+0.809
  NN Priors (top 3 of 3 legal):
     1.  99.8% ( -9.6pp) ███████████████████████████████████████ ACQ select PR
     2.   0.2% ( +2.3pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +7.3pp)  ACQ select SI

  MCTS Visits (top 3, 6400 total):
     1.  6026 (94.2%) Q=+0.821 █████████████████████████████████████ ACQ select PR
     2.   295 ( 4.6%) Q=+0.810 █ ACQ select SI
     3.    79 ( 1.2%) Q=+0.799  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 76, vbackups: 5849)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $65 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$14 stars=14 pres=P2  companies=[KK*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with PR ($65)

### Step 348: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with PR ($65)

  NN Values: P0=-0.049, P1=-0.668, P2=+0.805
  NN Priors (top 8 of 8 legal):
     1.  98.7% (-14.6pp) ███████████████████████████████████████ ACQ target PKP (with PR)
     2.   0.4% ( +4.0pp)  ACQ target SX (with PR)
     3.   0.2% ( +0.0pp)  ACQ target DR (with PR)
     4.   0.2% ( +1.7pp)  ACQ target PR (with PR)
     5.   0.1% ( +0.1pp)  ACQ target SJ (with PR)
     6.   0.1% ( +4.4pp)  ACQ target SZD (with PR)
     7.   0.1% ( +4.4pp)  ACQ target HR (with PR)
     8.   0.1% ( +0.0pp)  ACQ target E (with PR)

  MCTS Visits (top 8, 6400 total):
     1.  5839 (91.2%) Q=+0.821 ████████████████████████████████████ ACQ target PKP (with PR)
     2.   188 ( 2.9%) Q=+0.816 █ ACQ target SZD (with PR)
     3.   146 ( 2.3%) Q=+0.811  ACQ target SX (with PR)
     4.   135 ( 2.1%) Q=+0.801  ACQ target HR (with PR)
     5.    70 ( 1.1%) Q=+0.808  ACQ target PR (with PR)
     6.    11 ( 0.2%) Q=+0.816  ACQ target DR (with PR)
     7.     8 ( 0.1%) Q=+0.807  ACQ target SJ (with PR)
     8.     3 ( 0.0%) Q=+0.796  ACQ target E (with PR)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 75, vbackups: 5882)

  **Action: ACQ target PKP (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $65 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$14 stars=14 pres=P2  companies=[KK*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$1 stars=5 pres=P2  companies=[PR, PKP]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 PR -> PKP (price range $13-$33)

### Step 349: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 PR -> PKP (price range $13-$33)

  NN Values: P0=-0.084, P1=-0.695, P2=+0.805
  NN Priors (top 10 of 21 legal):
     1.  99.8% (-15.0pp) ███████████████████████████████████████ ACQUIRE PKP with PR @ $13
     2.   0.1% ( +0.2pp)  ACQUIRE PKP with PR @ $14
     3.   0.0% ( +0.1pp)  ACQUIRE PKP with PR @ $23
     4.   0.0% ( +0.0pp)  ACQUIRE PKP with PR @ $24
     5.   0.0% ( +5.6pp)  ACQUIRE PKP with PR @ $22
     6.   0.0% ( +0.3pp)  ACQUIRE PKP with PR @ $30
     7.   0.0% ( +2.7pp)  ACQUIRE PKP with PR @ $31
     8.   0.0% ( +0.0pp)  ACQUIRE PKP with PR @ $15
     9.   0.0% ( +1.0pp)  ACQUIRE PKP with PR @ $18
    10.   0.0% ( +0.8pp)  ACQUIRE PKP with PR @ $17

  MCTS Visits (top 10, 6400 total):
     1.  5779 (90.3%) Q=+0.821 ████████████████████████████████████ ACQUIRE PKP with PR @ $13
     2.   218 ( 3.4%) Q=+0.814 █ ACQUIRE PKP with PR @ $22
     3.   106 ( 1.7%) Q=+0.814  ACQUIRE PKP with PR @ $31
     4.    90 ( 1.4%) Q=+0.811  ACQUIRE PKP with PR @ $21
     5.    61 ( 1.0%) Q=+0.818  ACQUIRE PKP with PR @ $18
     6.    51 ( 0.8%) Q=+0.819  ACQUIRE PKP with PR @ $17
     7.    23 ( 0.4%) Q=+0.808  ACQUIRE PKP with PR @ $25
     8.    17 ( 0.3%) Q=+0.817  ACQUIRE PKP with PR @ $14
     9.    14 ( 0.2%) Q=+0.808  ACQUIRE PKP with PR @ $30
    10.    13 ( 0.2%) Q=+0.812  ACQUIRE PKP with PR @ $16
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 74, vbackups: 5788)

  **Action: ACQUIRE PKP with PR @ $13**

Phase: ACQ_SELECT_CORP  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $52 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$21 stars=16 pres=P2  companies=[KK*, PKP*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($1), PR($52), DA($0), SI($10)

### Step 350: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($1), PR($52), DA($0), SI($10)

  NN Values: P0=-0.082, P1=-0.676, P2=+0.812
  NN Priors (top 3 of 3 legal):
     1.  99.7% ( -3.9pp) ███████████████████████████████████████ ACQ select PR
     2.   0.2% ( +2.2pp)  PASS (ACQ_SELECT_CORP)
     3.   0.1% ( +1.7pp)  ACQ select SI

  MCTS Visits (top 3, 6400 total):
     1.  6178 (96.5%) Q=+0.821 ██████████████████████████████████████ ACQ select PR
     2.   146 ( 2.3%) Q=+0.826  PASS (ACQ_SELECT_CORP)
     3.    76 ( 1.2%) Q=+0.810  ACQ select SI
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 73, vbackups: 5969)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $52 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$21 stars=16 pres=P2  companies=[KK*, PKP*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with PR ($52)

### Step 351: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with PR ($52)

  NN Values: P0=-0.042, P1=-0.672, P2=+0.809
  NN Priors (top 6 of 6 legal):
     1.  99.1% (-13.0pp) ███████████████████████████████████████ ACQ target DR (with PR)
     2.   0.3% ( +3.3pp)  ACQ target SX (with PR)
     3.   0.2% ( +0.7pp)  ACQ target SZD (with PR)
     4.   0.2% ( +0.6pp)  ACQ target SJ (with PR)
     5.   0.1% ( +0.0pp)  ACQ target HR (with PR)
     6.   0.1% ( +8.4pp)  ACQ target E (with PR)

  MCTS Visits (top 6, 6400 total):
     1.  5870 (91.7%) Q=+0.821 ████████████████████████████████████ ACQ target DR (with PR)
     2.   286 ( 4.5%) Q=+0.812 █ ACQ target E (with PR)
     3.   130 ( 2.0%) Q=+0.811  ACQ target SX (with PR)
     4.    60 ( 0.9%) Q=+0.820  ACQ target SZD (with PR)
     5.    48 ( 0.8%) Q=+0.819  ACQ target SJ (with PR)
     6.     6 ( 0.1%) Q=+0.805  ACQ target HR (with PR)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 72, vbackups: 5906)

  **Action: ACQ target DR (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=5 pres=P2  companies=[SX, DR]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $52 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$21 stars=16 pres=P2  companies=[KK*, PKP*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 PR -> DR (price range $15-$38)

### Step 352: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 PR -> DR (price range $15-$38)

  NN Values: P0=-0.068, P1=-0.691, P2=+0.816
  NN Priors (top 10 of 24 legal):
     1.  99.7% (-14.8pp) ███████████████████████████████████████ ACQUIRE DR with PR @ $15
     2.   0.1% ( +4.0pp)  ACQUIRE DR with PR @ $16
     3.   0.0% ( -0.0pp)  ACQUIRE DR with PR @ $25
     4.   0.0% ( +0.0pp)  ACQUIRE DR with PR @ $26
     5.   0.0% ( -0.0pp)  ACQUIRE DR with PR @ $20
     6.   0.0% ( +2.6pp)  ACQUIRE DR with PR @ $32
     7.   0.0% ( +0.0pp)  ACQUIRE DR with PR @ $21
     8.   0.0% ( -0.0pp)  ACQUIRE DR with PR @ $24
     9.   0.0% ( -0.0pp)  ACQUIRE DR with PR @ $33
    10.   0.0% ( +0.0pp)  ACQUIRE DR with PR @ $19

  MCTS Visits (top 10, 6400 total):
     1.  5735 (89.6%) Q=+0.821 ███████████████████████████████████ ACQUIRE DR with PR @ $15
     2.   171 ( 2.7%) Q=+0.818 █ ACQUIRE DR with PR @ $16
     3.   103 ( 1.6%) Q=+0.815  ACQUIRE DR with PR @ $32
     4.    82 ( 1.3%) Q=+0.813  ACQUIRE DR with PR @ $27
     5.    82 ( 1.3%) Q=+0.812  ACQUIRE DR with PR @ $35
     6.    82 ( 1.3%) Q=+0.813  ACQUIRE DR with PR @ $29
     7.    61 ( 1.0%) Q=+0.819  ACQUIRE DR with PR @ $18
     8.    22 ( 0.3%) Q=+0.809  ACQUIRE DR with PR @ $36
     9.    19 ( 0.3%) Q=+0.813  ACQUIRE DR with PR @ $22
    10.    15 ( 0.2%) Q=+0.810  ACQUIRE DR with PR @ $23
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 71, vbackups: 5745)

  **Action: ACQUIRE DR with PR @ $15**

Phase: ACQ_SELECT_CORP  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $37 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$32 stars=17 pres=P2  companies=[KK*, PKP*, DR*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($1), PR($37), DA($0), SI($10)

### Step 353: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($1), PR($37), DA($0), SI($10)

  NN Values: P0=-0.040, P1=-0.660, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  81.3% ( -0.0pp) ████████████████████████████████ ACQ select PR
     2.  18.7% ( +0.0pp) ███████ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  3989 (62.3%) Q=+0.815 ████████████████████████ ACQ select PR
     2.  2411 (37.7%) Q=+0.832 ███████████████ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 70, vbackups: 6126)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $37 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$32 stars=17 pres=P2  companies=[KK*, PKP*, DR*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with PR ($37)

### Step 354: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with PR ($37)

  NN Values: P0=-0.050, P1=-0.652, P2=+0.812
  NN Priors (top 4 of 4 legal):
     1.  97.5% (-11.4pp) ██████████████████████████████████████ ACQ target SZD (with PR)
     2.   2.4% ( -0.1pp)  ACQ target HR (with PR)
     3.   0.1% ( +7.9pp)  ACQ target SJ (with PR)
     4.   0.0% ( +3.7pp)  ACQ target E (with PR)

  MCTS Visits (top 4, 6400 total):
     1.  5584 (87.2%) Q=+0.815 ██████████████████████████████████ ACQ target SZD (with PR)
     2.   515 ( 8.0%) Q=+0.814 ███ ACQ target SJ (with PR)
     3.   212 ( 3.3%) Q=+0.811 █ ACQ target E (with PR)
     4.    89 ( 1.4%) Q=+0.801  ACQ target HR (with PR)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 69, vbackups: 3988)

  **Action: ACQ target SZD (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $37 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$32 stars=17 pres=P2  companies=[KK*, PKP*, DR*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$39 stars=20 pres=P2  companies=[SZD, SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 PR -> SZD (price range $15-$40)

### Step 355: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 PR -> SZD (price range $15-$40)

  NN Values: P0=-0.008, P1=-0.676, P2=+0.820
  NN Priors (top 10 of 23 legal):
     1.  97.6% (-11.9pp) ███████████████████████████████████████ ACQUIRE SZD with PR @ $15
     2.   0.5% ( -0.1pp)  ACQUIRE SZD with PR @ $33
     3.   0.4% ( +2.0pp)  ACQUIRE SZD with PR @ $32
     4.   0.4% ( +0.9pp)  ACQUIRE SZD with PR @ $37
     5.   0.3% ( +0.0pp)  ACQUIRE SZD with PR @ $34
     6.   0.2% ( +0.1pp)  ACQUIRE SZD with PR @ $16
     7.   0.1% ( -0.0pp)  ACQUIRE SZD with PR @ $36
     8.   0.1% ( +0.9pp)  ACQUIRE SZD with PR @ $35
     9.   0.1% ( +1.7pp)  ACQUIRE SZD with PR @ $25
    10.   0.1% ( -0.0pp)  ACQUIRE SZD with PR @ $26

  MCTS Visits (top 10, 6400 total):
     1.  5453 (85.2%) Q=+0.815 ██████████████████████████████████ ACQUIRE SZD with PR @ $15
     2.   139 ( 2.2%) Q=+0.819  ACQUIRE SZD with PR @ $32
     3.   106 ( 1.7%) Q=+0.820  ACQUIRE SZD with PR @ $20
     4.   102 ( 1.6%) Q=+0.820  ACQUIRE SZD with PR @ $18
     5.   101 ( 1.6%) Q=+0.812  ACQUIRE SZD with PR @ $25
     6.    92 ( 1.4%) Q=+0.822  ACQUIRE SZD with PR @ $35
     7.    91 ( 1.4%) Q=+0.820  ACQUIRE SZD with PR @ $19
     8.    87 ( 1.4%) Q=+0.817  ACQUIRE SZD with PR @ $37
     9.    59 ( 0.9%) Q=+0.816  ACQUIRE SZD with PR @ $30
    10.    27 ( 0.4%) Q=+0.822  ACQUIRE SZD with PR @ $16
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 68, vbackups: 5581)

  **Action: ACQUIRE SZD with PR @ $15**

Phase: ACQ_SELECT_CORP  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $22 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$44 stars=20 pres=P2  companies=[KK*, PKP*, DR*, SZD*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with OS($1), PR($22), DA($0), SI($10)

### Step 356: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with OS($1), PR($22), DA($0), SI($10)

  NN Values: P0=-0.004, P1=-0.719, P2=+0.820
  NN Priors (top 2 of 2 legal):
     1.  81.3% ( -9.9pp) ████████████████████████████████ ACQ select PR
     2.  18.7% ( +9.9pp) ███████ PASS (ACQ_SELECT_CORP)

  MCTS Visits (top 2, 6400 total):
     1.  4907 (76.7%) Q=+0.816 ██████████████████████████████ ACQ select PR
     2.  1493 (23.3%) Q=+0.811 █████████ PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 67, vbackups: 5452)

  **Action: ACQ select PR**

Phase: ACQ_SELECT_COMPANY  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $22 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$44 stars=20 pres=P2  companies=[KK*, PKP*, DR*, SZD*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with PR ($22)

### Step 357: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with PR ($22)

  NN Values: P0=-0.016, P1=-0.695, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  92.6% ( -9.4pp) █████████████████████████████████████ ACQ target SJ (with PR)
     2.   7.4% ( +9.4pp) ██ ACQ target E (with PR)

  MCTS Visits (top 2, 6400 total):
     1.  5622 (87.8%) Q=+0.817 ███████████████████████████████████ ACQ target SJ (with PR)
     2.   778 (12.2%) Q=+0.807 ████ ACQ target E (with PR)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 66, vbackups: 4906)

  **Action: ACQ target SJ (with PR)**

Phase: ACQ_SELECT_PRICE  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $1 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $22 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$44 stars=20 pres=P2  companies=[KK*, PKP*, DR*, SZD*, FRA]
  DA: $0 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $10 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$32 stars=16 pres=P2  companies=[SJ, E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 PR -> SJ (price range $16-$41)

### Step 358: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 PR -> SJ (price range $16-$41)

  NN Values: P0=+0.020, P1=-0.723, P2=+0.824
  NN Priors (top 7 of 7 legal):
     1.  99.3% (-11.4pp) ███████████████████████████████████████ ACQUIRE SJ with PR @ $16
     2.   0.5% ( +1.9pp)  ACQUIRE SJ with PR @ $17
     3.   0.1% ( +1.1pp)  ACQUIRE SJ with PR @ $21
     4.   0.1% ( +4.5pp)  ACQUIRE SJ with PR @ $22
     5.   0.1% ( +0.1pp)  ACQUIRE SJ with PR @ $20
     6.   0.0% ( +3.6pp)  ACQUIRE SJ with PR @ $18
     7.   0.0% ( +0.3pp)  ACQUIRE SJ with PR @ $19

  MCTS Visits (top 7, 6400 total):
     1.  5601 (87.5%) Q=+0.817 ███████████████████████████████████ ACQUIRE SJ with PR @ $16
     2.   317 ( 5.0%) Q=+0.824 █ ACQUIRE SJ with PR @ $22
     3.   215 ( 3.4%) Q=+0.818 █ ACQUIRE SJ with PR @ $18
     4.   129 ( 2.0%) Q=+0.816  ACQUIRE SJ with PR @ $17
     5.   102 ( 1.6%) Q=+0.824  ACQUIRE SJ with PR @ $21
     6.    28 ( 0.4%) Q=+0.823  ACQUIRE SJ with PR @ $19
     7.     8 ( 0.1%) Q=+0.819  ACQUIRE SJ with PR @ $20
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 65, vbackups: 5621)

  **Action: ACQUIRE SJ with PR @ $16**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 12  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $59 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $27 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=4 pres=P2  companies=[SX]
  SM: $62 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=11 pres=P1  companies=[MAD]
  PR: $6 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$52 stars=22 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $13 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=3 pres=P2  companies=[PR]
  VM: $30 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $41 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$25 stars=15 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Closing**: P0 may close NS (VM), BR (VM)

### Step 359: P0 [CLOSING]

  **Closing**: P0 may close NS (VM), BR (VM)

  NN Values: P0=-0.113, P1=-0.664, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  99.5% (-10.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.5% (+10.4pp)  CLOSE NS

  MCTS Visits (top 2, 6400 total):
     1.  6102 (95.3%) Q=-0.020 ██████████████████████████████████████ PASS (CLOSING)
     2.   298 ( 4.7%) Q=-0.050 █ CLOSE NS
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 64, vbackups: 5600)

  **Action: PASS (CLOSING)**

### Step 360: P2 [CLOSING]

  **Closing**: P2 may close KK (PR), PKP (PR), DR (PR), SZD (PR), SJ (PR), E (SI), HR (SI), FRA (PR)

  NN Values: P0=-0.082, P1=-0.668, P2=+0.805
  NN Priors (top 4 of 4 legal):
     1.  99.7% (-13.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.1% ( +1.2pp)  CLOSE KK
     3.   0.1% ( +2.3pp)  CLOSE PKP
     4.   0.1% ( +9.9pp)  CLOSE DR

  MCTS Visits (top 4, 6400 total):
     1.  5763 (90.0%) Q=+0.817 ████████████████████████████████████ PASS (CLOSING)
     2.   443 ( 6.9%) Q=+0.816 ██ CLOSE DR
     3.   117 ( 1.8%) Q=+0.817  CLOSE PKP
     4.    77 ( 1.2%) Q=+0.815  CLOSE KK
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 63, vbackups: 5777)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $35 (NW $161) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $8 (NW $191) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $31 (NW $246) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $23 price=$27(idx 16) shares=bank:5/unissued:0/issued:6 income=$-4 stars=4 pres=P2  companies=[SX]
  SM: $72 price=$33(idx 18) shares=bank:0/unissued:1/issued:5 income=$10 stars=12 pres=P1  companies=[MAD]
  PR: $58 price=$45(idx 21) shares=bank:0/unissued:1/issued:4 income=$52 stars=27 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $12 price=$16(idx 11) shares=bank:4/unissued:0/issued:5 income=$-1 stars=3 pres=P2  companies=[PR]
  VM: $42 price=$18(idx 12) shares=bank:0/unissued:1/issued:3 income=$12 stars=11 pres=P0  companies=[NS, BR]
  SI: $66 price=$41(idx 20) shares=bank:2/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Dividends**: PR

### Step 361: P2 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=+0.009, P1=-0.629, P2=+0.828
  NN Priors (top 10 of 15 legal):
     1.  97.6% (-14.6pp) ███████████████████████████████████████ DIVIDEND $12
     2.   1.1% ( +3.6pp)  DIVIDEND $9
     3.   0.5% ( +0.2pp)  DIVIDEND $14
     4.   0.3% ( +0.1pp)  DIVIDEND $3
     5.   0.3% ( -0.0pp)  DIVIDEND $13
     6.   0.1% ( +0.8pp)  DIVIDEND $2
     7.   0.1% ( +0.1pp)  DIVIDEND $0
     8.   0.0% ( +3.1pp)  DIVIDEND $6
     9.   0.0% ( +0.6pp)  DIVIDEND $11
    10.   0.0% ( +1.4pp)  DIVIDEND $8

  MCTS Visits (top 10, 6400 total):
     1.  5230 (81.7%) Q=+0.817 ████████████████████████████████ DIVIDEND $12
     2.   300 ( 4.7%) Q=+0.821 █ DIVIDEND $9
     3.   244 ( 3.8%) Q=+0.822 █ DIVIDEND $5
     4.   208 ( 3.2%) Q=+0.822 █ DIVIDEND $6
     5.   103 ( 1.6%) Q=+0.819  DIVIDEND $8
     6.    60 ( 0.9%) Q=+0.822  DIVIDEND $14
     7.    57 ( 0.9%) Q=+0.818  DIVIDEND $2
     8.    51 ( 0.8%) Q=+0.823  DIVIDEND $10
     9.    50 ( 0.8%) Q=+0.823  DIVIDEND $11
    10.    40 ( 0.6%) Q=+0.820  DIVIDEND $7
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 62, vbackups: 5423)

  **Action: DIVIDEND $12**

### Step 362: P2 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.046, P1=-0.770, P2=+0.828
  NN Priors (top 10 of 14 legal):
     1.  99.6% (-10.8pp) ███████████████████████████████████████ DIVIDEND $0
     2.   0.2% ( +0.1pp)  DIVIDEND $3
     3.   0.1% ( +0.1pp)  DIVIDEND $1
     4.   0.0% ( +0.0pp)  DIVIDEND $2
     5.   0.0% ( +0.5pp)  DIVIDEND $6
     6.   0.0% ( +0.3pp)  DIVIDEND $4
     7.   0.0% ( +0.1pp)  DIVIDEND $5
     8.   0.0% ( +3.3pp)  DIVIDEND $9
     9.   0.0% ( +0.1pp)  DIVIDEND $12
    10.   0.0% ( +0.7pp)  DIVIDEND $10

  MCTS Visits (top 10, 6400 total):
     1.  5934 (92.7%) Q=+0.817 █████████████████████████████████████ DIVIDEND $0
     2.   180 ( 2.8%) Q=+0.804 █ DIVIDEND $8
     3.   123 ( 1.9%) Q=+0.806  DIVIDEND $9
     4.    32 ( 0.5%) Q=+0.808  DIVIDEND $10
     5.    29 ( 0.5%) Q=+0.813  DIVIDEND $6
     6.    22 ( 0.3%) Q=+0.818  DIVIDEND $4
     7.    19 ( 0.3%) Q=+0.810  DIVIDEND $7
     8.    19 ( 0.3%) Q=+0.824  DIVIDEND $1
     9.    18 ( 0.3%) Q=+0.819  DIVIDEND $3
    10.     5 ( 0.1%) Q=+0.820  DIVIDEND $2
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 61, vbackups: 5890)

  **Action: DIVIDEND $0**

### Step 363: P1 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.064, P1=-0.730, P2=+0.832
  NN Priors (top 10 of 12 legal):
     1.  97.7% (-11.2pp) ███████████████████████████████████████ DIVIDEND $11
     2.   1.7% ( +1.9pp)  DIVIDEND $10
     3.   0.1% ( +1.2pp)  DIVIDEND $7
     4.   0.1% ( +0.4pp)  DIVIDEND $8
     5.   0.1% ( +0.3pp)  DIVIDEND $5
     6.   0.1% ( +0.7pp)  DIVIDEND $9
     7.   0.1% ( +0.0pp)  DIVIDEND $4
     8.   0.1% ( +3.6pp)  DIVIDEND $6
     9.   0.0% ( +0.2pp)  DIVIDEND $2
    10.   0.0% ( +0.1pp)  DIVIDEND $1

  MCTS Visits (top 10, 6400 total):
     1.  5748 (89.8%) Q=-0.791 ███████████████████████████████████ DIVIDEND $11
     2.   279 ( 4.4%) Q=-0.789 █ DIVIDEND $10
     3.   133 ( 2.1%) Q=-0.809  DIVIDEND $6
     4.    69 ( 1.1%) Q=-0.825  DIVIDEND $3
     5.    58 ( 0.9%) Q=-0.792  DIVIDEND $9
     6.    56 ( 0.9%) Q=-0.808  DIVIDEND $7
     7.    32 ( 0.5%) Q=-0.801  DIVIDEND $8
     8.    13 ( 0.2%) Q=-0.813  DIVIDEND $5
     9.     5 ( 0.1%) Q=-0.827  DIVIDEND $2
    10.     5 ( 0.1%) Q=-0.827  DIVIDEND $1
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 60, vbackups: 5872)

  **Action: DIVIDEND $11**

### Step 364: P2 [DIVIDENDS]

  **Dividends**: OS

  NN Values: P0=-0.018, P1=-0.750, P2=+0.816
  NN Priors (top 4 of 4 legal):
     1.  99.1% (-11.9pp) ███████████████████████████████████████ DIVIDEND $3
     2.   0.6% ( +3.9pp)  DIVIDEND $2
     3.   0.1% ( +3.4pp)  DIVIDEND $1
     4.   0.1% ( +4.6pp)  DIVIDEND $0

  MCTS Visits (top 4, 6400 total):
     1.  5640 (88.1%) Q=+0.817 ███████████████████████████████████ DIVIDEND $3
     2.   291 ( 4.5%) Q=+0.822 █ DIVIDEND $2
     3.   272 ( 4.2%) Q=+0.822 █ DIVIDEND $0
     4.   197 ( 3.1%) Q=+0.822 █ DIVIDEND $1
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 59, vbackups: 5705)

  **Action: DIVIDEND $3**

### Step 365: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=+0.006, P1=-0.758, P2=+0.828
  NN Priors (top 7 of 7 legal):
     1.  99.1% (-10.7pp) ███████████████████████████████████████ DIVIDEND $6
     2.   0.5% ( +0.3pp)  DIVIDEND $5
     3.   0.1% ( +0.9pp)  DIVIDEND $3
     4.   0.1% ( +4.6pp)  DIVIDEND $0
     5.   0.1% ( +0.0pp)  DIVIDEND $1
     6.   0.1% ( +2.5pp)  DIVIDEND $2
     7.   0.1% ( +2.4pp)  DIVIDEND $4

  MCTS Visits (top 7, 6400 total):
     1.  5928 (92.6%) Q=-0.018 █████████████████████████████████████ DIVIDEND $6
     2.   213 ( 3.3%) Q=-0.029 █ DIVIDEND $0
     3.   101 ( 1.6%) Q=-0.033  DIVIDEND $2
     4.    87 ( 1.4%) Q=-0.038  DIVIDEND $4
     5.    35 ( 0.5%) Q=-0.036  DIVIDEND $3
     6.    29 ( 0.5%) Q=-0.043  DIVIDEND $5
     7.     7 ( 0.1%) Q=-0.016  DIVIDEND $1
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 58, vbackups: 5742)

  **Action: DIVIDEND $6**

### Step 366: P2 [DIVIDENDS]

  **Dividends**: DA

  NN Values: P0=+0.005, P1=-0.770, P2=+0.816
  NN Priors (top 3 of 3 legal):
     1.  98.9% (-11.1pp) ███████████████████████████████████████ DIVIDEND $2
     2.   1.1% ( +6.4pp)  DIVIDEND $1
     3.   0.0% ( +4.7pp)  DIVIDEND $0

  MCTS Visits (top 3, 6400 total):
     1.  5667 (88.5%) Q=+0.817 ███████████████████████████████████ DIVIDEND $2
     2.   458 ( 7.2%) Q=+0.823 ██ DIVIDEND $1
     3.   275 ( 4.3%) Q=+0.822 █ DIVIDEND $0
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 57, vbackups: 5753)

  **Action: DIVIDEND $2**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 12  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $71 (NW $235) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $69 (NW $216) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $60 (NW $295) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $5 price=$22(idx 14) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $17 price=$24(idx 15) shares=bank:0/unissued:1/issued:5 income=$10 stars=6 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$13(idx 9) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$45(idx 21) shares=bank:2/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Issue**: PR

### Step 367: P2 [ISSUE_SHARES]

  **Issue**: PR

  NN Values: P0=+0.015, P1=-0.797, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  99.1% ( -3.3pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   0.9% ( +3.3pp)  ISSUE PR shares

  MCTS Visits (top 2, 6400 total):
     1.  6235 (97.4%) Q=+0.817 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   165 ( 2.6%) Q=+0.802 █ ISSUE PR shares
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 56, vbackups: 5840)

  **Action: PASS (ISSUE_SHARES)**

### Step 368: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.031, P1=-0.777, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  99.4% ( -3.6pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   0.6% ( +3.6pp)  ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  6104 (95.4%) Q=-0.018 ██████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   296 ( 4.6%) Q=-0.016 █ ISSUE VM shares
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 55, vbackups: 6128)

  **Action: PASS (ISSUE_SHARES)**

### Step 369: P1 [ISSUE_SHARES]

  **Issue**: SM

  NN Values: P0=-0.018, P1=-0.809, P2=+0.820
  NN Priors (top 2 of 2 legal):
     1.  99.4% ( -6.3pp) ███████████████████████████████████████ ISSUE SM shares
     2.   0.6% ( +6.3pp)  PASS (ISSUE_SHARES)

  MCTS Visits (top 2, 6400 total):
     1.  6239 (97.5%) Q=-0.791 ██████████████████████████████████████ ISSUE SM shares
     2.   161 ( 2.5%) Q=-0.830 █ PASS (ISSUE_SHARES)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 54, vbackups: 6209)

  **Action: ISSUE SM shares**

--- Turn 13 ---

Phase: INVEST  |  Turn: 13  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $71 (NW $235) order=0 income=$0  shares=[PR=2, VM=2 (pres)]
  P1: $69 (NW $216) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $60 (NW $295) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $5 price=$22(idx 14) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $41 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=9 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$13(idx 9) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$45(idx 21) shares=bank:2/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining


### Step 370: P0 [INVEST]

  NN Values: P0=-0.056, P1=-0.766, P2=+0.816
  NN Priors (top 8 of 8 legal):
     1.  61.3% ( -8.4pp) ████████████████████████ BUY SI share
     2.  38.4% ( -4.0pp) ███████████████ AUCTION slot 0 (CDG, face $60)
     3.   0.2% ( +2.2pp)  PASS (INVEST)
     4.   0.1% ( +2.5pp)  SELL VM share
     5.   0.0% ( +5.7pp)  BUY DA share
     6.   0.0% ( +0.1pp)  SELL PR share
     7.   0.0% ( +1.4pp)  BUY OS share
     8.   0.0% ( +0.6pp)  BUY SM share

  MCTS Visits (top 8, 6400 total):
     1.  4955 (77.4%) Q=-0.012 ██████████████████████████████ BUY SI share
     2.  1082 (16.9%) Q=-0.041 ██████ AUCTION slot 0 (CDG, face $60)
     3.   204 ( 3.2%) Q=-0.037 █ BUY DA share
     4.    60 ( 0.9%) Q=-0.059  SELL VM share
     5.    53 ( 0.8%) Q=-0.057  PASS (INVEST)
     6.    31 ( 0.5%) Q=-0.057  BUY OS share
     7.    12 ( 0.2%) Q=-0.129  BUY SM share
     8.     3 ( 0.0%) Q=-0.036  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 53, vbackups: 6048)

  **Action: BUY SI share**

### Step 371: P2 [INVEST]

  NN Values: P0=-0.036, P1=-0.781, P2=+0.824
  NN Priors (top 9 of 9 legal):
     1.  99.7% ( -6.8pp) ███████████████████████████████████████ AUCTION slot 0 (CDG, face $60)
     2.   0.3% ( +0.0pp)  SELL OS share
     3.   0.0% ( +2.5pp)  BUY OS share
     4.   0.0% ( +0.1pp)  SELL DA share
     5.   0.0% ( +1.4pp)  SELL SI share
     6.   0.0% ( +1.9pp)  BUY DA share
     7.   0.0% ( +0.6pp)  PASS (INVEST)
     8.   0.0% ( +0.0pp)  SELL PR share
     9.   0.0% ( +0.3pp)  BUY SM share

  MCTS Visits (top 9, 6400 total):
     1.  5937 (92.8%) Q=+0.816 █████████████████████████████████████ AUCTION slot 0 (CDG, face $60)
     2.   158 ( 2.5%) Q=+0.815  BUY OS share
     3.   111 ( 1.7%) Q=+0.820  SELL SI share
     4.   107 ( 1.7%) Q=+0.812  BUY DA share
     5.    36 ( 0.6%) Q=+0.814  PASS (INVEST)
     6.    29 ( 0.5%) Q=+0.822  SELL OS share
     7.    14 ( 0.2%) Q=+0.809  BUY SM share
     8.     7 ( 0.1%) Q=+0.822  SELL DA share
     9.     1 ( 0.0%) Q=+0.797  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 52, vbackups: 5077)

  **Action: AUCTION slot 0 (CDG, face $60)**

  ↳ auto: BID $60

Phase: BID_IN_AUCTION  |  Turn: 13  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $21 (NW $235) order=0 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $69 (NW $216) order=2 income=$0  shares=[SM=5 (pres), VM=1]
  P2: $60 (NW $305) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row** [1]: CDG (fv=$60, 5★, inc=$10)

**Corporations**
  OS: $5 price=$22(idx 14) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $41 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=9 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$13(idx 9) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Auction**: CDG current bid=$60 high bidder=P2 starter=P2

### Step 372: P1 [BID_IN_AUCTION]

  **Auction**: CDG current bid=$60 high bidder=P2 starter=P2

  NN Values: P0=-0.054, P1=-0.805, P2=+0.824
  NN Priors (top 10 of 10 legal):
     1.  95.1% (-13.3pp) ██████████████████████████████████████ BID $61
     2.   2.3% ( +7.3pp)  BID $62
     3.   0.6% ( -0.1pp)  BID $68
     4.   0.6% ( +0.8pp)  BID $63
     5.   0.4% ( +0.8pp)  BID $64
     6.   0.4% ( +0.2pp)  BID $65
     7.   0.2% ( -0.0pp)  BID $67
     8.   0.2% ( +2.0pp)  BID $66
     9.   0.2% ( +0.8pp)  PASS (BID_IN_AUCTION)
    10.   0.0% ( +1.6pp)  BID $69

  MCTS Visits (top 10, 6400 total):
     1.  3143 (49.1%) Q=-0.810 ███████████████████ BID $61
     2.  2615 (40.9%) Q=-0.775 ████████████████ PASS (BID_IN_AUCTION)
     3.   421 ( 6.6%) Q=-0.806 ██ BID $62
     4.    59 ( 0.9%) Q=-0.825  BID $66
     5.    44 ( 0.7%) Q=-0.822  BID $63
     6.    44 ( 0.7%) Q=-0.831  BID $69
     7.    38 ( 0.6%) Q=-0.827  BID $64
     8.    21 ( 0.3%) Q=-0.826  BID $65
     9.    12 ( 0.2%) Q=-0.838  BID $68
    10.     3 ( 0.0%) Q=-0.844  BID $67
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 51, vbackups: 5453)

  **Action: BID $61**

  ↳ auto: PASS (BID_IN_AUCTION)
  ↳ auto: PASS (BID_IN_AUCTION)

Phase: INVEST  |  Turn: 13  |  CoO Level: 6  |  Active Player: 1  |  End Card: no

**Players**
  P0: $21 (NW $235) order=0 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $8 (NW $215) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $60 (NW $305) order=1 income=$0  shares=[OS=1 (pres), PR=2 (pres), DA=1 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $5 price=$22(idx 14) shares=bank:5/unissued:0/issued:6 income=$-4 stars=2 pres=P2  companies=[SX]
  SM: $41 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=9 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$13(idx 9) shares=bank:4/unissued:0/issued:5 income=$-1 stars=2 pres=P2  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining


### Step 373: P1 [INVEST]

  NN Values: P0=-0.047, P1=-0.805, P2=+0.836
  NN Priors (top 3 of 3 legal):
     1.  98.8% ( -9.8pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.7% ( +2.9pp)  SELL VM share
     3.   0.5% ( +6.9pp)  SELL SM share

  MCTS Visits (top 3, 6400 total):
     1.  5727 (89.5%) Q=-0.810 ███████████████████████████████████ PASS (INVEST)
     2.   525 ( 8.2%) Q=-0.808 ███ SELL SM share
     3.   148 ( 2.3%) Q=-0.822  SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 50, vbackups: 3622)

  **Action: PASS (INVEST)**

### Step 374: P0 [INVEST]

  NN Values: P0=-0.055, P1=-0.828, P2=+0.828
  NN Priors (top 5 of 5 legal):
     1.  97.4% (-13.4pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.0% ( +0.3pp)  BUY DA share
     3.   0.9% ( +0.9pp)  SELL VM share
     4.   0.4% ( +6.0pp)  SELL SI share
     5.   0.4% ( +6.2pp)  SELL PR share

  MCTS Visits (top 5, 6400 total):
     1.  5904 (92.2%) Q=-0.008 ████████████████████████████████████ PASS (INVEST)
     2.   206 ( 3.2%) Q=-0.032 █ SELL PR share
     3.   201 ( 3.1%) Q=-0.033 █ SELL SI share
     4.    52 ( 0.8%) Q=-0.036  SELL VM share
     5.    37 ( 0.6%) Q=-0.034  BUY DA share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 49, vbackups: 5726)

  **Action: PASS (INVEST)**

### Step 375: P2 [INVEST]

  NN Values: P0=-0.043, P1=-0.816, P2=+0.824
  NN Priors (top 8 of 8 legal):
     1.  99.6% (-14.3pp) ███████████████████████████████████████ SELL OS share
     2.   0.3% ( +1.5pp)  SELL DA share
     3.   0.0% ( +5.9pp)  SELL SI share
     4.   0.0% ( +2.1pp)  BUY OS share
     5.   0.0% ( +1.0pp)  BUY DA share
     6.   0.0% ( +1.9pp)  BUY SM share
     7.   0.0% ( +1.1pp)  SELL PR share
     8.   0.0% ( +0.7pp)  PASS (INVEST)

  MCTS Visits (top 8, 6400 total):
     1.  5832 (91.1%) Q=+0.811 ████████████████████████████████████ SELL OS share
     2.   234 ( 3.7%) Q=+0.807 █ SELL SI share
     3.    82 ( 1.3%) Q=+0.809  SELL DA share
     4.    73 ( 1.1%) Q=+0.800  BUY OS share
     5.    67 ( 1.0%) Q=+0.800  BUY SM share
     6.    64 ( 1.0%) Q=+0.810  BUY DA share
     7.    30 ( 0.5%) Q=+0.800  PASS (INVEST)
     8.    18 ( 0.3%) Q=+0.751  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 48, vbackups: 5871)

  **Action: SELL OS share**

### Step 376: P1 [INVEST]

  NN Values: P0=-0.048, P1=-0.828, P2=+0.820
  NN Priors (top 3 of 3 legal):
     1.  96.4% ( -9.0pp) ██████████████████████████████████████ PASS (INVEST)
     2.   2.3% ( +3.3pp)  SELL VM share
     3.   1.3% ( +5.7pp)  SELL SM share

  MCTS Visits (top 3, 6400 total):
     1.  5722 (89.4%) Q=-0.810 ███████████████████████████████████ PASS (INVEST)
     2.   502 ( 7.8%) Q=-0.808 ███ SELL SM share
     3.   176 ( 2.8%) Q=-0.833 █ SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 47, vbackups: 5838)

  **Action: PASS (INVEST)**

### Step 377: P0 [INVEST]

  NN Values: P0=-0.048, P1=-0.840, P2=+0.816
  NN Priors (top 5 of 5 legal):
     1.  96.0% (-12.5pp) ██████████████████████████████████████ PASS (INVEST)
     2.   2.1% ( +4.4pp)  BUY DA share
     3.   1.1% ( +2.8pp)  SELL VM share
     4.   0.4% ( +2.2pp)  SELL SI share
     5.   0.4% ( +3.1pp)  SELL PR share

  MCTS Visits (top 5, 6400 total):
     1.  5825 (91.0%) Q=-0.007 ████████████████████████████████████ PASS (INVEST)
     2.   268 ( 4.2%) Q=-0.021 █ BUY DA share
     3.   152 ( 2.4%) Q=-0.022  SELL VM share
     4.    87 ( 1.4%) Q=-0.043  SELL PR share
     5.    68 ( 1.1%) Q=-0.040  SELL SI share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 46, vbackups: 5746)

  **Action: PASS (INVEST)**

### Step 378: P2 [INVEST]

  NN Values: P0=-0.035, P1=-0.824, P2=+0.824
  NN Priors (top 8 of 8 legal):
     1.  95.9% (-14.6pp) ██████████████████████████████████████ SELL DA share
     2.   3.9% ( -0.1pp) █ PASS (INVEST)
     3.   0.0% ( +5.9pp)  BUY SI share
     4.   0.0% ( +0.4pp)  SELL SI share
     5.   0.0% ( +0.6pp)  BUY DA share
     6.   0.0% ( +3.7pp)  BUY OS share
     7.   0.0% ( +1.7pp)  BUY SM share
     8.   0.0% ( +2.4pp)  SELL PR share

  MCTS Visits (top 8, 6400 total):
     1.  5677 (88.7%) Q=+0.811 ███████████████████████████████████ SELL DA share
     2.   205 ( 3.2%) Q=+0.798 █ BUY SI share
     3.   182 ( 2.8%) Q=+0.810 █ BUY OS share
     4.   169 ( 2.6%) Q=+0.800 █ PASS (INVEST)
     5.    67 ( 1.0%) Q=+0.796  BUY SM share
     6.    41 ( 0.6%) Q=+0.811  BUY DA share
     7.    32 ( 0.5%) Q=+0.811  SELL SI share
     8.    27 ( 0.4%) Q=+0.712  SELL PR share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 45, vbackups: 5824)

  **Action: SELL DA share**

### Step 379: P1 [INVEST]

  NN Values: P0=-0.040, P1=-0.824, P2=+0.820
  NN Priors (top 3 of 3 legal):
     1.  95.3% (-10.8pp) ██████████████████████████████████████ PASS (INVEST)
     2.   3.3% ( +4.2pp) █ SELL VM share
     3.   1.4% ( +6.5pp)  SELL SM share

  MCTS Visits (top 3, 6400 total):
     1.  5529 (86.4%) Q=-0.810 ██████████████████████████████████ PASS (INVEST)
     2.   561 ( 8.8%) Q=-0.808 ███ SELL SM share
     3.   310 ( 4.8%) Q=-0.822 █ SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 44, vbackups: 5676)

  **Action: PASS (INVEST)**

### Step 380: P0 [INVEST]

  NN Values: P0=-0.042, P1=-0.832, P2=+0.816
  NN Priors (top 5 of 5 legal):
     1.  96.2% (-12.5pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.4% ( +0.0pp)  SELL VM share
     3.   1.2% ( +7.5pp)  BUY DA share
     4.   0.6% ( +4.8pp)  SELL PR share
     5.   0.6% ( +0.2pp)  SELL SI share

  MCTS Visits (top 5, 6400 total):
     1.  5836 (91.2%) Q=-0.006 ████████████████████████████████████ PASS (INVEST)
     2.   386 ( 6.0%) Q=-0.018 ██ BUY DA share
     3.   111 ( 1.7%) Q=-0.053  SELL PR share
     4.    46 ( 0.7%) Q=-0.028  SELL VM share
     5.    21 ( 0.3%) Q=-0.038  SELL SI share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 43, vbackups: 5528)

  **Action: PASS (INVEST)**

### Step 381: P2 [INVEST]

  NN Values: P0=-0.021, P1=-0.816, P2=+0.820
  NN Priors (top 7 of 7 legal):
     1.  99.3% ( -8.6pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.4% ( +0.0pp)  BUY SI share
     3.   0.1% ( +2.4pp)  BUY DA share
     4.   0.1% ( +0.3pp)  SELL SI share
     5.   0.1% ( +1.4pp)  BUY OS share
     6.   0.0% ( +0.3pp)  BUY SM share
     7.   0.0% ( +4.1pp)  SELL PR share

  MCTS Visits (top 7, 6400 total):
     1.  6102 (95.3%) Q=+0.811 ██████████████████████████████████████ PASS (INVEST)
     2.   118 ( 1.8%) Q=+0.809  BUY DA share
     3.    71 ( 1.1%) Q=+0.807  BUY OS share
     4.    47 ( 0.7%) Q=+0.715  SELL PR share
     5.    24 ( 0.4%) Q=+0.810  BUY SI share
     6.    23 ( 0.4%) Q=+0.811  SELL SI share
     7.    15 ( 0.2%) Q=+0.796  BUY SM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 42, vbackups: 5835)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 13  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $235) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $8 (NW $215) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $92 (NW $302) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $5 price=$20(idx 13) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $41 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=9 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$12(idx 8) shares=bank:5/unissued:0/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($10), SI($66)

### Step 382: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($10), SI($66)

  NN Values: P0=-0.032, P1=-0.828, P2=+0.809
  NN Priors (top 2 of 2 legal):
     1.  91.0% ( -1.8pp) ████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   9.0% ( +1.8pp) ███ ACQ select SI

  MCTS Visits (top 2, 6400 total):
     1.  5854 (91.5%) Q=+0.812 ████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   546 ( 8.5%) Q=+0.808 ███ ACQ select SI
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 41, vbackups: 6101)

  **Action: PASS (ACQ_SELECT_CORP)**

  ↳ auto: PASS (ACQ_SELECT_CORP)

### Step 383: P1 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P1 may buy with SM($41)

  NN Values: P0=-0.050, P1=-0.828, P2=+0.824
  NN Priors (top 2 of 2 legal):
     1.  98.3% (-10.6pp) ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   1.7% (+10.6pp)  ACQ select SM

  MCTS Visits (top 2, 6400 total):
     1.  5630 (88.0%) Q=-0.809 ███████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   770 (12.0%) Q=-0.810 ████ ACQ select SM
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 40, vbackups: 5696)

  **Action: PASS (ACQ_SELECT_CORP)**

Phase: CLOSING  |  Turn: 13  |  CoO Level: 6  |  Active Player: 0  |  End Card: no

**Players**
  P0: $21 (NW $235) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $8 (NW $215) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $92 (NW $302) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $64 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $5 price=$20(idx 13) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $41 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=9 pres=P1  companies=[MAD]
  PR: $10 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=23 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $2 price=$12(idx 8) shares=bank:5/unissued:0/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $24 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=9 pres=P0  companies=[NS, BR]
  SI: $66 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$25 stars=17 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Closing**: P0 may close NS (VM), BR (VM)

### Step 384: P0 [CLOSING]

  **Closing**: P0 may close NS (VM), BR (VM)

  NN Values: P0=-0.024, P1=-0.828, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  99.3% (-12.9pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.7% (+12.9pp)  CLOSE NS

  MCTS Visits (top 2, 6400 total):
     1.  5673 (88.6%) Q=-0.005 ███████████████████████████████████ PASS (CLOSING)
     2.   727 (11.4%) Q=-0.010 ████ CLOSE NS
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 39, vbackups: 5693)

  **Action: PASS (CLOSING)**

### Step 385: P2 [CLOSING]

  **Closing**: P2 may close KK (PR), PKP (PR), DR (PR), SZD (PR), SJ (PR), E (SI), HR (SI), FRA (PR)

  NN Values: P0=-0.026, P1=-0.820, P2=+0.812
  NN Priors (top 4 of 4 legal):
     1.  99.5% (-10.0pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.2% ( +1.8pp)  CLOSE KK
     3.   0.1% ( +3.3pp)  CLOSE PKP
     4.   0.1% ( +4.9pp)  CLOSE DR

  MCTS Visits (top 4, 6400 total):
     1.  5998 (93.7%) Q=+0.812 █████████████████████████████████████ PASS (CLOSING)
     2.   193 ( 3.0%) Q=+0.800 █ CLOSE DR
     3.   127 ( 2.0%) Q=+0.798  CLOSE PKP
     4.    82 ( 1.3%) Q=+0.799  CLOSE KK
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 38, vbackups: 5765)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 13  |  CoO Level: 6  |  Active Player: 2  |  End Card: no

**Players**
  P0: $21 (NW $235) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $18 (NW $225) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $92 (NW $302) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $1 price=$20(idx 13) shares=bank:6/unissued:0/issued:6 income=$-4 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $51 price=$24(idx 15) shares=bank:1/unissued:0/issued:6 income=$10 stars=10 pres=P1  companies=[MAD]
  PR: $62 price=$55(idx 23) shares=bank:0/unissued:1/issued:4 income=$52 stars=28 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$12(idx 8) shares=bank:5/unissued:0/issued:5 income=$-1 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $36 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$12 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$25 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Dividends**: PR

### Step 386: P2 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=-0.024, P1=-0.801, P2=+0.816
  NN Priors (top 10 of 16 legal):
     1.  98.3% (-14.7pp) ███████████████████████████████████████ DIVIDEND $9
     2.   0.9% ( +1.7pp)  DIVIDEND $12
     3.   0.4% ( -0.1pp)  DIVIDEND $8
     4.   0.2% ( +0.2pp)  DIVIDEND $10
     5.   0.1% ( +0.0pp)  DIVIDEND $0
     6.   0.1% ( -0.0pp)  DIVIDEND $15
     7.   0.0% ( +3.0pp)  DIVIDEND $6
     8.   0.0% ( +5.8pp)  DIVIDEND $11
     9.   0.0% ( +0.4pp)  DIVIDEND $3
    10.   0.0% ( +0.3pp)  DIVIDEND $2

  MCTS Visits (top 10, 6400 total):
     1.  5601 (87.5%) Q=+0.812 ███████████████████████████████████ DIVIDEND $9
     2.   284 ( 4.4%) Q=+0.812 █ DIVIDEND $11
     3.   130 ( 2.0%) Q=+0.811  DIVIDEND $12
     4.   124 ( 1.9%) Q=+0.805  DIVIDEND $6
     5.    83 ( 1.3%) Q=+0.805  DIVIDEND $4
     6.    52 ( 0.8%) Q=+0.810  DIVIDEND $13
     7.    25 ( 0.4%) Q=+0.803  DIVIDEND $7
     8.    21 ( 0.3%) Q=+0.808  DIVIDEND $3
     9.    21 ( 0.3%) Q=+0.806  DIVIDEND $10
    10.    18 ( 0.3%) Q=+0.808  DIVIDEND $2
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 37, vbackups: 5681)

  **Action: DIVIDEND $9**

### Step 387: P2 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=-0.028, P1=-0.824, P2=+0.789
  NN Priors (top 10 of 17 legal):
     1.  66.8% ( -6.5pp) ██████████████████████████ DIVIDEND $0
     2.  23.5% ( -2.8pp) █████████ DIVIDEND $16
     3.   5.5% ( -0.8pp) ██ DIVIDEND $15
     4.   1.1% ( -0.1pp)  DIVIDEND $3
     5.   0.7% ( -0.1pp)  DIVIDEND $6
     6.   0.7% ( -0.0pp)  DIVIDEND $1
     7.   0.4% ( +0.3pp)  DIVIDEND $2
     8.   0.4% ( -0.1pp)  DIVIDEND $9
     9.   0.2% ( +1.3pp)  DIVIDEND $12
    10.   0.1% ( +0.0pp)  DIVIDEND $5

  MCTS Visits (top 10, 6400 total):
     1.  4779 (74.7%) Q=+0.815 █████████████████████████████ DIVIDEND $0
     2.   790 (12.3%) Q=+0.796 ████ DIVIDEND $16
     3.   188 ( 2.9%) Q=+0.798 █ DIVIDEND $15
     4.   109 ( 1.7%) Q=+0.805  DIVIDEND $10
     5.    88 ( 1.4%) Q=+0.807  DIVIDEND $7
     6.    82 ( 1.3%) Q=+0.801  DIVIDEND $11
     7.    63 ( 1.0%) Q=+0.801  DIVIDEND $12
     8.    60 ( 0.9%) Q=+0.799  DIVIDEND $13
     9.    57 ( 0.9%) Q=+0.807  DIVIDEND $4
    10.    55 ( 0.9%) Q=+0.808  DIVIDEND $3
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 36, vbackups: 5822)

  **Action: DIVIDEND $0**

### Step 388: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.026, P1=-0.828, P2=+0.812
  NN Priors (top 10 of 10 legal):
     1.  98.5% (-13.3pp) ███████████████████████████████████████ DIVIDEND $2
     2.   0.7% ( +0.5pp)  DIVIDEND $5
     3.   0.4% ( +0.1pp)  DIVIDEND $8
     4.   0.2% ( -0.0pp)  DIVIDEND $3
     5.   0.2% ( -0.0pp)  DIVIDEND $1
     6.   0.0% ( +0.1pp)  DIVIDEND $9
     7.   0.0% ( +0.2pp)  DIVIDEND $6
     8.   0.0% ( +0.2pp)  DIVIDEND $0
     9.   0.0% ( +5.2pp)  DIVIDEND $4
    10.   0.0% ( +7.0pp)  DIVIDEND $7

  MCTS Visits (top 10, 6400 total):
     1.  5832 (91.1%) Q=-0.005 ████████████████████████████████████ DIVIDEND $2
     2.   283 ( 4.4%) Q=-0.019 █ DIVIDEND $7
     3.   207 ( 3.2%) Q=-0.019 █ DIVIDEND $4
     4.    37 ( 0.6%) Q=-0.028  DIVIDEND $5
     5.    17 ( 0.3%) Q=-0.025  DIVIDEND $8
     6.     7 ( 0.1%) Q=-0.026  DIVIDEND $0
     7.     5 ( 0.1%) Q=-0.032  DIVIDEND $3
     8.     4 ( 0.1%) Q=-0.030  DIVIDEND $1
     9.     4 ( 0.1%) Q=-0.037  DIVIDEND $6
    10.     4 ( 0.1%) Q=-0.027  DIVIDEND $9
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 35, vbackups: 4778)

  **Action: DIVIDEND $2**

### Step 389: P1 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.009, P1=-0.816, P2=+0.816
  NN Priors (top 9 of 9 legal):
     1.  98.7% (-14.6pp) ███████████████████████████████████████ DIVIDEND $8
     2.   1.1% ( +2.8pp)  DIVIDEND $7
     3.   0.0% ( +0.8pp)  DIVIDEND $4
     4.   0.0% ( +2.4pp)  DIVIDEND $6
     5.   0.0% ( +0.3pp)  DIVIDEND $5
     6.   0.0% ( +7.2pp)  DIVIDEND $3
     7.   0.0% ( +0.1pp)  DIVIDEND $2
     8.   0.0% ( +0.3pp)  DIVIDEND $1
     9.   0.0% ( +0.7pp)  DIVIDEND $0

  MCTS Visits (top 9, 6400 total):
     1.  5380 (84.1%) Q=-0.808 █████████████████████████████████ DIVIDEND $8
     2.   477 ( 7.5%) Q=-0.798 ██ DIVIDEND $7
     3.   343 ( 5.4%) Q=-0.816 ██ DIVIDEND $3
     4.   107 ( 1.7%) Q=-0.820  DIVIDEND $6
     5.    31 ( 0.5%) Q=-0.825  DIVIDEND $4
     6.    31 ( 0.5%) Q=-0.820  DIVIDEND $0
     7.    15 ( 0.2%) Q=-0.825  DIVIDEND $1
     8.    13 ( 0.2%) Q=-0.831  DIVIDEND $5
     9.     3 ( 0.0%) Q=-0.832  DIVIDEND $2
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 34, vbackups: 5603)

  **Action: DIVIDEND $8**

  ↳ auto: AUTO:END_CARD

Phase: ISSUE_SHARES  |  Turn: 13  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $295) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $60 (NW $243) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $110 (NW $346) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Issue**: PR

### Step 390: P2 [ISSUE_SHARES]

  **Issue**: PR

  NN Values: P0=-0.032, P1=-0.840, P2=+0.805
  NN Priors (top 2 of 2 legal):
     1.  99.6% ( -6.9pp) ███████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   0.4% ( +6.9pp)  ISSUE PR shares

  MCTS Visits (top 2, 6400 total):
     1.  6020 (94.1%) Q=+0.816 █████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   380 ( 5.9%) Q=+0.814 ██ ISSUE PR shares
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 33, vbackups: 5607)

  **Action: PASS (ISSUE_SHARES)**

### Step 391: P0 [ISSUE_SHARES]

  **Issue**: VM

  NN Values: P0=-0.034, P1=-0.832, P2=+0.797
  NN Priors (top 2 of 2 legal):
     1.  92.0% ( -8.1pp) ████████████████████████████████████ PASS (ISSUE_SHARES)
     2.   8.0% ( +8.1pp) ███ ISSUE VM shares

  MCTS Visits (top 2, 6400 total):
     1.  5114 (79.9%) Q=-0.005 ███████████████████████████████ PASS (ISSUE_SHARES)
     2.  1286 (20.1%) Q=-0.000 ████████ ISSUE VM shares
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 32, vbackups: 5669)

  **Action: PASS (ISSUE_SHARES)**

Phase: IPO  |  Turn: 13  |  CoO Level: 7  |  Active Player: 1  |  End Card: YES

**Players**
  P0: $43 (NW $295) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $60 (NW $243) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $110 (NW $346) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**IPO**: CDG

### Step 392: P1 [IPO]

  **IPO**: CDG

  NN Values: P0=-0.033, P1=-0.836, P2=+0.805
  NN Priors (top 3 of 3 legal):
     1.  88.7% ( -3.2pp) ███████████████████████████████████ IPO CDG → float S
     2.   9.9% ( +1.3pp) ███ IPO CDG → float JS
     3.   1.5% ( +1.9pp)  PASS (IPO)

  MCTS Visits (top 3, 6400 total):
     1.  5144 (80.4%) Q=-0.808 ████████████████████████████████ IPO CDG → float S
     2.   975 (15.2%) Q=-0.801 ██████ IPO CDG → float JS
     3.   281 ( 4.4%) Q=-0.802 █ PASS (IPO)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 31, vbackups: 5463)

  **Action: IPO CDG → float S**

Phase: PAR  |  Turn: 13  |  CoO Level: 7  |  Active Player: 1  |  End Card: YES

**Players**
  P0: $43 (NW $295) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $60 (NW $243) order=2 income=$10  companies=[CDG]  shares=[SM=5 (pres), VM=1]
  P2: $110 (NW $346) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**PAR**: CDG -> S

### Step 393: P1 [PAR]

  **PAR**: CDG -> S

  NN Values: P0=-0.029, P1=-0.836, P2=+0.805
  NN Priors (top 2 of 2 legal):
     1.  98.0% ( -6.5pp) ███████████████████████████████████████ PAR S @$30 (IPO CDG)
     2.   2.0% ( +6.5pp)  PAR S @$37 (IPO CDG)

  MCTS Visits (top 2, 6400 total):
     1.  6137 (95.9%) Q=-0.808 ██████████████████████████████████████ PAR S @$30 (IPO CDG)
     2.   263 ( 4.1%) Q=-0.831 █ PAR S @$37 (IPO CDG)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 30, vbackups: 5143)

  **Action: PAR S @$30 (IPO CDG)**

--- Turn 14 ---

Phase: INVEST  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $295) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $60 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1]
  P2: $110 (NW $346) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$50(idx 22) shares=bank:1/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining


### Step 394: P2 [INVEST]

  NN Values: P0=-0.015, P1=-0.828, P2=+0.820
  NN Priors (top 8 of 8 legal):
     1.  97.0% (-12.5pp) ██████████████████████████████████████ PASS (INVEST)
     2.   2.5% ( +0.0pp)  BUY SI share
     3.   0.1% ( +4.0pp)  SELL SI share
     4.   0.1% ( +0.3pp)  BUY DA share
     5.   0.1% ( +0.7pp)  BUY S share
     6.   0.1% ( -0.0pp)  BUY OS share
     7.   0.1% ( +6.0pp)  BUY SM share
     8.   0.0% ( +1.6pp)  SELL PR share

  MCTS Visits (top 8, 6400 total):
     1.  5672 (88.6%) Q=+0.815 ███████████████████████████████████ PASS (INVEST)
     2.   345 ( 5.4%) Q=+0.828 ██ BUY SI share
     3.   168 ( 2.6%) Q=+0.798 █ BUY SM share
     4.   120 ( 1.9%) Q=+0.798  SELL SI share
     5.    39 ( 0.6%) Q=+0.778  SELL PR share
     6.    37 ( 0.6%) Q=+0.808  BUY S share
     7.    16 ( 0.2%) Q=+0.803  BUY DA share
     8.     3 ( 0.0%) Q=+0.798  BUY OS share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 29, vbackups: 5993)

  **Action: PASS (INVEST)**

### Step 395: P0 [INVEST]

  NN Values: P0=-0.011, P1=-0.836, P2=+0.805
  NN Priors (top 8 of 8 legal):
     1.  85.9% (-12.0pp) ██████████████████████████████████ PASS (INVEST)
     2.   5.3% ( -0.1pp) ██ SELL PR share
     3.   2.6% ( -0.4pp) █ SELL VM share
     4.   2.2% ( +8.3pp)  BUY S share
     5.   1.5% ( +0.0pp)  SELL SI share
     6.   1.2% ( +1.4pp)  BUY OS share
     7.   1.0% ( +2.6pp)  BUY DA share
     8.   0.4% ( +0.1pp)  BUY SM share

  MCTS Visits (top 8, 6400 total):
     1.  5044 (78.8%) Q=-0.003 ███████████████████████████████ PASS (INVEST)
     2.   655 (10.2%) Q=-0.005 ████ BUY S share
     3.   241 ( 3.8%) Q=-0.013 █ SELL PR share
     4.   202 ( 3.2%) Q=-0.007 █ BUY DA share
     5.   135 ( 2.1%) Q=-0.010  BUY OS share
     6.    64 ( 1.0%) Q=-0.029  SELL VM share
     7.    44 ( 0.7%) Q=-0.030  SELL SI share
     8.    15 ( 0.2%) Q=-0.032  BUY SM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 28, vbackups: 5517)

  **Action: PASS (INVEST)**

### Step 396: P1 [INVEST]

  NN Values: P0=-0.024, P1=-0.828, P2=+0.824
  NN Priors (top 9 of 9 legal):
     1.  95.3% (-11.6pp) ██████████████████████████████████████ BUY SI share
     2.   4.5% ( +0.7pp) █ BUY S share
     3.   0.1% ( +0.5pp)  BUY OS share
     4.   0.0% ( +7.8pp)  SELL VM share
     5.   0.0% ( +0.1pp)  BUY DA share
     6.   0.0% ( +0.1pp)  PASS (INVEST)
     7.   0.0% ( +1.0pp)  BUY SM share
     8.   0.0% ( +0.5pp)  SELL SM share
     9.   0.0% ( +1.0pp)  SELL S share

  MCTS Visits (top 9, 6400 total):
     1.  5607 (87.6%) Q=-0.809 ███████████████████████████████████ BUY SI share
     2.   471 ( 7.4%) Q=-0.811 ██ SELL VM share
     3.   177 ( 2.8%) Q=-0.830 █ BUY S share
     4.    49 ( 0.8%) Q=-0.818  SELL S share
     5.    40 ( 0.6%) Q=-0.828  BUY SM share
     6.    26 ( 0.4%) Q=-0.825  BUY OS share
     7.    17 ( 0.3%) Q=-0.827  SELL SM share
     8.     9 ( 0.1%) Q=-0.823  BUY DA share
     9.     4 ( 0.1%) Q=-0.839  PASS (INVEST)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 27, vbackups: 5335)

  **Action: BUY SI share**

### Step 397: P2 [INVEST]

  NN Values: P0=-0.032, P1=-0.738, P2=+0.812
  NN Priors (top 7 of 7 legal):
     1.  99.5% ( -9.3pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.1% ( +2.9pp)  SELL SI share
     3.   0.1% ( -0.0pp)  BUY DA share
     4.   0.1% ( +0.6pp)  BUY S share
     5.   0.1% ( +1.7pp)  BUY OS share
     6.   0.0% ( +3.0pp)  BUY SM share
     7.   0.0% ( +1.1pp)  SELL PR share

  MCTS Visits (top 7, 6400 total):
     1.  6023 (94.1%) Q=+0.815 █████████████████████████████████████ PASS (INVEST)
     2.   125 ( 2.0%) Q=+0.807  SELL SI share
     3.    97 ( 1.5%) Q=+0.800  BUY SM share
     4.    83 ( 1.3%) Q=+0.807  BUY OS share
     5.    34 ( 0.5%) Q=+0.808  BUY S share
     6.    34 ( 0.5%) Q=+0.790  SELL PR share
     7.     4 ( 0.1%) Q=+0.798  BUY DA share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 26, vbackups: 5606)

  **Action: PASS (INVEST)**

### Step 398: P0 [INVEST]

  NN Values: P0=-0.026, P1=-0.742, P2=+0.805
  NN Priors (top 8 of 8 legal):
     1.  96.3% (-14.2pp) ██████████████████████████████████████ PASS (INVEST)
     2.   1.4% ( +3.1pp)  SELL SI share
     3.   0.6% ( +2.0pp)  BUY S share
     4.   0.4% ( +0.0pp)  SELL VM share
     5.   0.4% ( +0.3pp)  BUY OS share
     6.   0.3% ( +0.1pp)  SELL PR share
     7.   0.3% ( +3.0pp)  BUY DA share
     8.   0.1% ( +5.5pp)  BUY SM share

  MCTS Visits (top 8, 6400 total):
     1.  5662 (88.5%) Q=-0.002 ███████████████████████████████████ PASS (INVEST)
     2.   226 ( 3.5%) Q=-0.010 █ SELL SI share
     3.   164 ( 2.6%) Q=-0.030 █ BUY SM share
     4.   154 ( 2.4%) Q=-0.012  BUY DA share
     5.   151 ( 2.4%) Q=-0.007  BUY S share
     6.    20 ( 0.3%) Q=-0.030  BUY OS share
     7.    12 ( 0.2%) Q=-0.036  SELL PR share
     8.    11 ( 0.2%) Q=-0.037  SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 25, vbackups: 5771)

  **Action: PASS (INVEST)**

### Step 399: P1 [INVEST]

  NN Values: P0=-0.043, P1=-0.762, P2=+0.812
  NN Priors (top 5 of 5 legal):
     1.  98.6% (-11.5pp) ███████████████████████████████████████ PASS (INVEST)
     2.   0.5% ( +1.5pp)  SELL S share
     3.   0.4% ( +4.1pp)  SELL SI share
     4.   0.3% ( +0.4pp)  SELL VM share
     5.   0.3% ( +5.5pp)  SELL SM share

  MCTS Visits (top 5, 6400 total):
     1.  5173 (80.8%) Q=-0.811 ████████████████████████████████ PASS (INVEST)
     2.   634 ( 9.9%) Q=-0.801 ███ SELL SM share
     3.   435 ( 6.8%) Q=-0.802 ██ SELL SI share
     4.   122 ( 1.9%) Q=-0.810  SELL S share
     5.    36 ( 0.6%) Q=-0.814  SELL VM share
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 24, vbackups: 5506)

  **Action: PASS (INVEST)**

  ↳ auto: AUTO:WRAP_UP

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($91)

### Step 400: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($91)

  NN Values: P0=-0.034, P1=-0.781, P2=+0.797
  NN Priors (top 3 of 3 legal):
     1.  99.7% ( -3.6pp) ███████████████████████████████████████ ACQ select SI
     2.   0.3% ( +0.8pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +2.8pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  6333 (99.0%) Q=+0.815 ███████████████████████████████████████ ACQ select SI
     2.    52 ( 0.8%) Q=+0.805  PASS (ACQ_SELECT_CORP)
     3.    15 ( 0.2%) Q=+0.804  ACQ select PR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 23, vbackups: 5578)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($91)

### Step 401: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($91)

  NN Values: P0=-0.037, P1=-0.785, P2=+0.797
  NN Priors (top 6 of 6 legal):
     1.  86.6% ( -9.5pp) ██████████████████████████████████ ACQ target FRA (with SI)
     2.  13.3% ( +1.6pp) █████ ACQ target DR (with SI)
     3.   0.0% ( +1.0pp)  ACQ target KK (with SI)
     4.   0.0% ( +0.0pp)  ACQ target PKP (with SI)
     5.   0.0% ( +5.5pp)  ACQ target SJ (with SI)
     6.   0.0% ( +1.4pp)  ACQ target SZD (with SI)

  MCTS Visits (top 6, 6400 total):
     1.  5136 (80.2%) Q=+0.815 ████████████████████████████████ ACQ target FRA (with SI)
     2.   914 (14.3%) Q=+0.815 █████ ACQ target DR (with SI)
     3.   240 ( 3.8%) Q=+0.811 █ ACQ target SJ (with SI)
     4.    82 ( 1.3%) Q=+0.813  ACQ target SZD (with SI)
     5.    26 ( 0.4%) Q=+0.810  ACQ target KK (with SI)
     6.     2 ( 0.0%) Q=+0.799  ACQ target PKP (with SI)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 22, vbackups: 5977)

  **Action: ACQ target FRA (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$23 stars=24 pres=P2  companies=[KK, PKP, DR, SZD, SJ, FRA]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $91 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$15 stars=20 pres=P2  companies=[E, HR]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> FRA (price range $28-$74)

### Step 402: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> FRA (price range $28-$74)

  NN Values: P0=-0.038, P1=-0.793, P2=+0.801
  NN Priors (top 10 of 47 legal):
     1.  99.5% (-14.9pp) ███████████████████████████████████████ ACQUIRE FRA with SI @ $28
     2.   0.3% ( -0.0pp)  ACQUIRE FRA with SI @ $29
     3.   0.0% ( -0.0pp)  ACQUIRE FRA with SI @ $51
     4.   0.0% ( +0.0pp)  ACQUIRE FRA with SI @ $46
     5.   0.0% ( +0.0pp)  ACQUIRE FRA with SI @ $33
     6.   0.0% ( +0.1pp)  ACQUIRE FRA with SI @ $39
     7.   0.0% ( -0.0pp)  ACQUIRE FRA with SI @ $32
     8.   0.0% ( +7.1pp)  ACQUIRE FRA with SI @ $50
     9.   0.0% ( +0.0pp)  ACQUIRE FRA with SI @ $47
    10.   0.0% ( -0.0pp)  ACQUIRE FRA with SI @ $64

  MCTS Visits (top 10, 6400 total):
     1.  5596 (87.4%) Q=+0.815 ██████████████████████████████████ ACQUIRE FRA with SI @ $28
     2.   421 ( 6.6%) Q=+0.813 ██ ACQUIRE FRA with SI @ $50
     3.   142 ( 2.2%) Q=+0.811  ACQUIRE FRA with SI @ $56
     4.   106 ( 1.7%) Q=+0.810  ACQUIRE FRA with SI @ $59
     5.    86 ( 1.3%) Q=+0.808  ACQUIRE FRA with SI @ $68
     6.    12 ( 0.2%) Q=+0.806  ACQUIRE FRA with SI @ $29
     7.    10 ( 0.2%) Q=+0.803  ACQUIRE FRA with SI @ $37
     8.     9 ( 0.1%) Q=+0.802  ACQUIRE FRA with SI @ $36
     9.     4 ( 0.1%) Q=+0.793  ACQUIRE FRA with SI @ $48
    10.     4 ( 0.1%) Q=+0.795  ACQUIRE FRA with SI @ $45
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 21, vbackups: 5490)

  **Action: ACQUIRE FRA with SI @ $28**

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$0 stars=19 pres=P2  companies=[KK, PKP, DR, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $63 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$25 stars=22 pres=P2  companies=[E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($63)

### Step 403: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($63)

  NN Values: P0=-0.038, P1=-0.770, P2=+0.797
  NN Priors (top 3 of 3 legal):
     1.  99.7% ( -7.1pp) ███████████████████████████████████████ ACQ select SI
     2.   0.3% ( +3.9pp)  PASS (ACQ_SELECT_CORP)
     3.   0.1% ( +3.3pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  6164 (96.3%) Q=+0.815 ██████████████████████████████████████ ACQ select SI
     2.   213 ( 3.3%) Q=+0.809 █ PASS (ACQ_SELECT_CORP)
     3.    23 ( 0.4%) Q=+0.792  ACQ select PR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 20, vbackups: 5595)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$0 stars=19 pres=P2  companies=[KK, PKP, DR, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $63 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$25 stars=22 pres=P2  companies=[E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($63)

### Step 404: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($63)

  NN Values: P0=-0.021, P1=-0.785, P2=+0.797
  NN Priors (top 5 of 5 legal):
     1.  99.7% (-11.0pp) ███████████████████████████████████████ ACQ target DR (with SI)
     2.   0.1% ( -0.0pp)  ACQ target KK (with SI)
     3.   0.1% ( +8.4pp)  ACQ target PKP (with SI)
     4.   0.0% ( +0.8pp)  ACQ target SZD (with SI)
     5.   0.0% ( +1.7pp)  ACQ target SJ (with SI)

  MCTS Visits (top 5, 6400 total):
     1.  5811 (90.8%) Q=+0.815 ████████████████████████████████████ ACQ target DR (with SI)
     2.   465 ( 7.3%) Q=+0.815 ██ ACQ target PKP (with SI)
     3.    69 ( 1.1%) Q=+0.813  ACQ target SJ (with SI)
     4.    52 ( 0.8%) Q=+0.814  ACQ target SZD (with SI)
     5.     3 ( 0.0%) Q=+0.802  ACQ target KK (with SI)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 19, vbackups: 5818)

  **Action: ACQ target DR (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$0 stars=19 pres=P2  companies=[KK, PKP, DR, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $63 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$25 stars=22 pres=P2  companies=[E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> DR (price range $15-$38)

### Step 405: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> DR (price range $15-$38)

  NN Values: P0=-0.040, P1=-0.781, P2=+0.797
  NN Priors (top 10 of 24 legal):
     1.  99.7% (-14.9pp) ███████████████████████████████████████ ACQUIRE DR with SI @ $15
     2.   0.2% ( -0.0pp)  ACQUIRE DR with SI @ $16
     3.   0.0% ( +3.9pp)  ACQUIRE DR with SI @ $20
     4.   0.0% ( -0.0pp)  ACQUIRE DR with SI @ $26
     5.   0.0% ( +0.9pp)  ACQUIRE DR with SI @ $21
     6.   0.0% ( +0.0pp)  ACQUIRE DR with SI @ $38
     7.   0.0% ( +4.0pp)  ACQUIRE DR with SI @ $25
     8.   0.0% ( +1.3pp)  ACQUIRE DR with SI @ $19
     9.   0.0% ( +1.5pp)  ACQUIRE DR with SI @ $17
    10.   0.0% ( +0.1pp)  ACQUIRE DR with SI @ $33

  MCTS Visits (top 10, 6400 total):
     1.  5603 (87.5%) Q=+0.815 ███████████████████████████████████ ACQUIRE DR with SI @ $15
     2.   230 ( 3.6%) Q=+0.814 █ ACQUIRE DR with SI @ $20
     3.   217 ( 3.4%) Q=+0.815 █ ACQUIRE DR with SI @ $25
     4.    93 ( 1.5%) Q=+0.814  ACQUIRE DR with SI @ $17
     5.    80 ( 1.2%) Q=+0.814  ACQUIRE DR with SI @ $19
     6.    72 ( 1.1%) Q=+0.813  ACQUIRE DR with SI @ $30
     7.    44 ( 0.7%) Q=+0.814  ACQUIRE DR with SI @ $27
     8.    17 ( 0.3%) Q=+0.812  ACQUIRE DR with SI @ $21
     9.    13 ( 0.2%) Q=+0.808  ACQUIRE DR with SI @ $31
    10.    11 ( 0.2%) Q=+0.806  ACQUIRE DR with SI @ $34
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 18, vbackups: 5610)

  **Action: ACQUIRE DR with SI @ $15**

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=16 pres=P2  companies=[KK, PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $48 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$28 stars=23 pres=P2  companies=[DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($48)

### Step 406: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($48)

  NN Values: P0=-0.031, P1=-0.770, P2=+0.801
  NN Priors (top 3 of 3 legal):
     1.  99.6% ( -8.0pp) ███████████████████████████████████████ ACQ select SI
     2.   0.3% ( +1.4pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +6.6pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  6260 (97.8%) Q=+0.815 ███████████████████████████████████████ ACQ select SI
     2.    74 ( 1.2%) Q=+0.800  ACQ select PR
     3.    66 ( 1.0%) Q=+0.811  PASS (ACQ_SELECT_CORP)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 17, vbackups: 6147)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=16 pres=P2  companies=[KK, PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $48 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$28 stars=23 pres=P2  companies=[DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($48)

### Step 407: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($48)

  NN Values: P0=-0.013, P1=-0.785, P2=+0.805
  NN Priors (top 4 of 4 legal):
     1.  99.8% ( -6.0pp) ███████████████████████████████████████ ACQ target KK (with SI)
     2.   0.1% ( +0.8pp)  ACQ target PKP (with SI)
     3.   0.1% ( +3.0pp)  ACQ target SZD (with SI)
     4.   0.1% ( +2.2pp)  ACQ target SJ (with SI)

  MCTS Visits (top 4, 6400 total):
     1.  6248 (97.6%) Q=+0.815 ███████████████████████████████████████ ACQ target KK (with SI)
     2.    57 ( 0.9%) Q=+0.815  ACQ target PKP (with SI)
     3.    54 ( 0.8%) Q=+0.814  ACQ target SZD (with SI)
     4.    41 ( 0.6%) Q=+0.813  ACQ target SJ (with SI)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 16, vbackups: 6255)

  **Action: ACQ target KK (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=16 pres=P2  companies=[KK, PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $48 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$28 stars=23 pres=P2  companies=[DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> KK (price range $11-$28)

### Step 408: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> KK (price range $11-$28)

  NN Values: P0=-0.042, P1=-0.770, P2=+0.805
  NN Priors (top 10 of 18 legal):
     1.  99.6% (-14.8pp) ███████████████████████████████████████ ACQUIRE KK with SI @ $11
     2.   0.2% ( +0.9pp)  ACQUIRE KK with SI @ $12
     3.   0.0% ( +1.4pp)  ACQUIRE KK with SI @ $16
     4.   0.0% ( +0.1pp)  ACQUIRE KK with SI @ $28
     5.   0.0% ( +0.5pp)  ACQUIRE KK with SI @ $22
     6.   0.0% ( +0.0pp)  ACQUIRE KK with SI @ $17
     7.   0.0% ( +2.9pp)  ACQUIRE KK with SI @ $21
     8.   0.0% ( +0.7pp)  ACQUIRE KK with SI @ $15
     9.   0.0% ( +0.8pp)  ACQUIRE KK with SI @ $20
    10.   0.0% ( +1.9pp)  ACQUIRE KK with SI @ $18

  MCTS Visits (top 10, 6400 total):
     1.  5476 (85.6%) Q=+0.815 ██████████████████████████████████ ACQUIRE KK with SI @ $11
     2.   189 ( 3.0%) Q=+0.815 █ ACQUIRE KK with SI @ $21
     3.   172 ( 2.7%) Q=+0.815 █ ACQUIRE KK with SI @ $23
     4.   133 ( 2.1%) Q=+0.814  ACQUIRE KK with SI @ $27
     5.   119 ( 1.9%) Q=+0.814  ACQUIRE KK with SI @ $18
     6.    90 ( 1.4%) Q=+0.814  ACQUIRE KK with SI @ $16
     7.    70 ( 1.1%) Q=+0.814  ACQUIRE KK with SI @ $12
     8.    47 ( 0.7%) Q=+0.814  ACQUIRE KK with SI @ $20
     9.    42 ( 0.7%) Q=+0.814  ACQUIRE KK with SI @ $15
    10.    35 ( 0.5%) Q=+0.815  ACQUIRE KK with SI @ $22
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 15, vbackups: 5492)

  **Action: ACQUIRE KK with SI @ $11**

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=13 pres=P2  companies=[PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $37 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$31 stars=25 pres=P2  companies=[KK*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($37)

### Step 409: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($37)

  NN Values: P0=-0.033, P1=-0.770, P2=+0.805
  NN Priors (top 3 of 3 legal):
     1.  99.6% (-11.7pp) ███████████████████████████████████████ ACQ select SI
     2.   0.4% ( +7.3pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +4.4pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  6122 (95.7%) Q=+0.815 ██████████████████████████████████████ ACQ select SI
     2.   277 ( 4.3%) Q=+0.814 █ PASS (ACQ_SELECT_CORP)
     3.     1 ( 0.0%) Q=+0.801  ACQ select PR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 14, vbackups: 6145)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=13 pres=P2  companies=[PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $37 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$31 stars=25 pres=P2  companies=[KK*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($37)

### Step 410: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($37)

  NN Values: P0=-0.011, P1=-0.781, P2=+0.805
  NN Priors (top 3 of 3 legal):
     1.  99.6% (-11.4pp) ███████████████████████████████████████ ACQ target PKP (with SI)
     2.   0.2% ( +9.0pp)  ACQ target SJ (with SI)
     3.   0.2% ( +2.4pp)  ACQ target SZD (with SI)

  MCTS Visits (top 3, 6400 total):
     1.  5871 (91.7%) Q=+0.815 ████████████████████████████████████ ACQ target PKP (with SI)
     2.   391 ( 6.1%) Q=+0.815 ██ ACQ target SJ (with SI)
     3.   138 ( 2.2%) Q=+0.815  ACQ target SZD (with SI)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 13, vbackups: 5897)

  **Action: ACQ target PKP (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=13 pres=P2  companies=[PKP, SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $37 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$31 stars=25 pres=P2  companies=[KK*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> PKP (price range $13-$33)

### Step 411: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> PKP (price range $13-$33)

  NN Values: P0=-0.032, P1=-0.762, P2=+0.812
  NN Priors (top 10 of 21 legal):
     1.  99.5% (-14.9pp) ███████████████████████████████████████ ACQUIRE PKP with SI @ $13
     2.   0.3% ( +1.0pp)  ACQUIRE PKP with SI @ $14
     3.   0.0% ( +0.0pp)  ACQUIRE PKP with SI @ $24
     4.   0.0% ( +0.2pp)  ACQUIRE PKP with SI @ $23
     5.   0.0% ( +0.3pp)  ACQUIRE PKP with SI @ $18
     6.   0.0% ( +0.1pp)  ACQUIRE PKP with SI @ $17
     7.   0.0% ( +0.1pp)  ACQUIRE PKP with SI @ $19
     8.   0.0% ( +0.0pp)  ACQUIRE PKP with SI @ $15
     9.   0.0% ( +0.0pp)  ACQUIRE PKP with SI @ $25
    10.   0.0% ( +3.8pp)  ACQUIRE PKP with SI @ $31

  MCTS Visits (top 10, 6400 total):
     1.  5521 (86.3%) Q=+0.815 ██████████████████████████████████ ACQUIRE PKP with SI @ $13
     2.   287 ( 4.5%) Q=+0.814 █ ACQUIRE PKP with SI @ $32
     3.   212 ( 3.3%) Q=+0.814 █ ACQUIRE PKP with SI @ $31
     4.   208 ( 3.2%) Q=+0.814 █ ACQUIRE PKP with SI @ $33
     5.    84 ( 1.3%) Q=+0.815  ACQUIRE PKP with SI @ $14
     6.    38 ( 0.6%) Q=+0.815  ACQUIRE PKP with SI @ $21
     7.    17 ( 0.3%) Q=+0.815  ACQUIRE PKP with SI @ $23
     8.     8 ( 0.1%) Q=+0.816  ACQUIRE PKP with SI @ $18
     9.     7 ( 0.1%) Q=+0.815  ACQUIRE PKP with SI @ $17
    10.     5 ( 0.1%) Q=+0.812  ACQUIRE PKP with SI @ $19
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 12, vbackups: 5544)

  **Action: ACQUIRE PKP with SI @ $13**

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=10 pres=P2  companies=[SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $24 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$38 stars=27 pres=P2  companies=[KK*, PKP*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($24)

### Step 412: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($24)

  NN Values: P0=-0.029, P1=-0.758, P2=+0.812
  NN Priors (top 3 of 3 legal):
     1.  99.6% ( -9.0pp) ███████████████████████████████████████ ACQ select SI
     2.   0.4% ( +6.5pp)  PASS (ACQ_SELECT_CORP)
     3.   0.0% ( +2.5pp)  ACQ select PR

  MCTS Visits (top 3, 6400 total):
     1.  6167 (96.4%) Q=+0.815 ██████████████████████████████████████ ACQ select SI
     2.   232 ( 3.6%) Q=+0.816 █ PASS (ACQ_SELECT_CORP)
     3.     1 ( 0.0%) Q=+0.805  ACQ select PR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 11, vbackups: 6159)

  **Action: ACQ select SI**

Phase: ACQ_SELECT_COMPANY  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=10 pres=P2  companies=[SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $24 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$38 stars=27 pres=P2  companies=[KK*, PKP*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Company**: P2 buying with SI ($24)

### Step 413: P2 [ACQ_SELECT_COMPANY]

  **Acquisition — Select Company**: P2 buying with SI ($24)

  NN Values: P0=-0.014, P1=-0.770, P2=+0.812
  NN Priors (top 2 of 2 legal):
     1.  99.9% (-11.2pp) ███████████████████████████████████████ ACQ target SZD (with SI)
     2.   0.1% (+11.2pp)  ACQ target SJ (with SI)

  MCTS Visits (top 2, 6400 total):
     1.  5837 (91.2%) Q=+0.815 ████████████████████████████████████ ACQ target SZD (with SI)
     2.   563 ( 8.8%) Q=+0.815 ███ ACQ target SJ (with SI)
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 10, vbackups: 5841)

  **Action: ACQ target SZD (with SI)**

Phase: ACQ_SELECT_PRICE  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-4 stars=10 pres=P2  companies=[SZD, SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $24 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$38 stars=27 pres=P2  companies=[KK*, PKP*, DR*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Price**: P2 SI -> SZD (price range $15-$40)

### Step 414: P2 [ACQ_SELECT_PRICE]

  **Acquisition — Select Price**: P2 SI -> SZD (price range $15-$40)

  NN Values: P0=-0.030, P1=-0.750, P2=+0.812
  NN Priors (top 10 of 10 legal):
     1.  99.3% (-14.9pp) ███████████████████████████████████████ ACQUIRE SZD with SI @ $15
     2.   0.2% ( +2.5pp)  ACQUIRE SZD with SI @ $16
     3.   0.1% ( +1.3pp)  ACQUIRE SZD with SI @ $24
     4.   0.1% ( +5.6pp)  ACQUIRE SZD with SI @ $21
     5.   0.1% ( +0.1pp)  ACQUIRE SZD with SI @ $23
     6.   0.1% ( +1.8pp)  ACQUIRE SZD with SI @ $20
     7.   0.1% ( +0.1pp)  ACQUIRE SZD with SI @ $22
     8.   0.0% ( -0.0pp)  ACQUIRE SZD with SI @ $19
     9.   0.0% ( +2.7pp)  ACQUIRE SZD with SI @ $17
    10.   0.0% ( +0.7pp)  ACQUIRE SZD with SI @ $18

  MCTS Visits (top 10, 6400 total):
     1.  5518 (86.2%) Q=+0.815 ██████████████████████████████████ ACQUIRE SZD with SI @ $15
     2.   325 ( 5.1%) Q=+0.815 ██ ACQUIRE SZD with SI @ $21
     3.   175 ( 2.7%) Q=+0.815 █ ACQUIRE SZD with SI @ $16
     4.   122 ( 1.9%) Q=+0.815  ACQUIRE SZD with SI @ $17
     5.   106 ( 1.7%) Q=+0.815  ACQUIRE SZD with SI @ $20
     6.    91 ( 1.4%) Q=+0.815  ACQUIRE SZD with SI @ $24
     7.    47 ( 0.7%) Q=+0.815  ACQUIRE SZD with SI @ $18
     8.     8 ( 0.1%) Q=+0.816  ACQUIRE SZD with SI @ $23
     9.     7 ( 0.1%) Q=+0.815  ACQUIRE SZD with SI @ $22
    10.     1 ( 0.0%) Q=+0.816  ACQUIRE SZD with SI @ $19
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 9, vbackups: 5557)

  **Action: ACQUIRE SZD with SI @ $15**

Phase: ACQ_SELECT_CORP  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $26 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-2 stars=6 pres=P2  companies=[SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $9 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$39 stars=29 pres=P2  companies=[KK*, PKP*, DR*, SZD*, E, HR, FRA*]

**Deck**: 0 remaining

**Acquisition — Select Corp**: P2 may buy with PR($26), SI($9)

### Step 415: P2 [ACQ_SELECT_CORP]

  **Acquisition — Select Corp**: P2 may buy with PR($26), SI($9)

  NN Values: P0=-0.025, P1=-0.758, P2=+0.816
  NN Priors (top 2 of 2 legal):
     1.  99.9% ( -6.4pp) ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.   0.1% ( +6.4pp)  ACQ select PR

  MCTS Visits (top 2, 6400 total):
     1.  6391 (99.9%) Q=+0.815 ███████████████████████████████████████ PASS (ACQ_SELECT_CORP)
     2.     9 ( 0.1%) Q=+0.812  ACQ select PR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 8, vbackups: 6121)

  **Action: PASS (ACQ_SELECT_CORP)**

  ↳ auto: PASS (ACQ_SELECT_CORP)
  ↳ auto: PASS (ACQ_SELECT_CORP)

Phase: CLOSING  |  Turn: 14  |  CoO Level: 7  |  Active Player: 0  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $69 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $60 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=11 pres=P1  companies=[CDG]
  OS: $1 price=$16(idx 11) shares=bank:6/unissued:0/issued:6 income=$-7 stars=2 RECEIVERSHIP  companies=[SX]
  SM: $3 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $108 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-2 stars=14 pres=P2  companies=[SJ]
  DA: $1 price=$10(idx 6) shares=bank:5/unissued:0/issued:5 income=$-4 stars=2 RECEIVERSHIP  companies=[PR]
  VM: $30 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $9 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$39 stars=29 pres=P2  companies=[KK, PKP, DR, SZD, E, HR, FRA]

**Deck**: 0 remaining

**Closing**: P0 may close NS (VM), BR (VM)

### Step 416: P0 [CLOSING]

  **Closing**: P0 may close NS (VM), BR (VM)

  NN Values: P0=-0.020, P1=-0.770, P2=+0.816
  NN Priors (top 3 of 3 legal):
     1.  99.2% ( -9.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.5% ( +5.1pp)  CLOSE NS
     3.   0.3% ( +4.3pp)  CLOSE BR

  MCTS Visits (top 3, 6400 total):
     1.  5721 (89.4%) Q=-0.001 ███████████████████████████████████ PASS (CLOSING)
     2.   371 ( 5.8%) Q=-0.000 ██ CLOSE NS
     3.   308 ( 4.8%) Q=-0.000 █ CLOSE BR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 7, vbackups: 5768)

  **Action: PASS (CLOSING)**

### Step 417: P2 [CLOSING]

  **Closing**: P2 may close KK (SI), PKP (SI), DR (SI), SZD (SI), E (SI), HR (SI), FRA (SI)

  NN Values: P0=-0.024, P1=-0.766, P2=+0.820
  NN Priors (top 6 of 6 legal):
     1.  99.3% ( -8.4pp) ███████████████████████████████████████ PASS (CLOSING)
     2.   0.2% ( +2.1pp)  CLOSE KK
     3.   0.1% ( +0.3pp)  CLOSE PKP
     4.   0.1% ( +0.7pp)  CLOSE SZD
     5.   0.1% ( -0.0pp)  CLOSE DR
     6.   0.1% ( +5.3pp)  CLOSE E

  MCTS Visits (top 6, 6400 total):
     1.  5989 (93.6%) Q=+0.815 █████████████████████████████████████ PASS (CLOSING)
     2.   221 ( 3.5%) Q=+0.814 █ CLOSE E
     3.   106 ( 1.7%) Q=+0.814  CLOSE KK
     4.    51 ( 0.8%) Q=+0.814  CLOSE SZD
     5.    28 ( 0.4%) Q=+0.814  CLOSE PKP
     6.     5 ( 0.1%) Q=+0.812  CLOSE DR
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 6, vbackups: 6022)

  **Action: PASS (CLOSING)**

  ↳ auto: AUTO:INCOME

Phase: DIVIDENDS  |  Turn: 14  |  CoO Level: 7  |  Active Player: 2  |  End Card: YES

**Players**
  P0: $43 (NW $300) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $5 (NW $243) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $110 (NW $356) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $74 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $70 price=$30(idx 17) shares=bank:2/unissued:3/issued:4 income=$10 stars=12 pres=P1  companies=[CDG]
  SM: $13 price=$18(idx 12) shares=bank:1/unissued:0/issued:6 income=$10 stars=6 pres=P1  companies=[MAD]
  PR: $106 price=$68(idx 25) shares=bank:0/unissued:1/issued:4 income=$-2 stars=14 pres=P2  companies=[SJ]
  VM: $32 price=$33(idx 18) shares=bank:0/unissued:1/issued:3 income=$2 stars=10 pres=P0  companies=[NS, BR]
  SI: $48 price=$55(idx 23) shares=bank:0/unissued:0/issued:4 income=$39 stars=33 pres=P2  companies=[KK, PKP, DR, SZD, E, HR, FRA]

**Deck**: 0 remaining

**Dividends**: PR

### Step 418: P2 [DIVIDENDS]

  **Dividends**: PR

  NN Values: P0=+0.020, P1=-0.758, P2=+0.824
  NN Priors (top 10 of 23 legal):
     1.  97.8% (-14.1pp) ███████████████████████████████████████ DIVIDEND $21
     2.   1.0% ( +0.2pp)  DIVIDEND $22
     3.   0.5% ( -0.0pp)  DIVIDEND $20
     4.   0.3% ( +0.4pp)  DIVIDEND $9
     5.   0.2% ( +2.6pp)  DIVIDEND $19
     6.   0.2% ( +2.1pp)  DIVIDEND $18
     7.   0.1% ( +0.9pp)  DIVIDEND $15
     8.   0.0% ( +0.1pp)  DIVIDEND $8
     9.   0.0% ( +0.0pp)  DIVIDEND $12
    10.   0.0% ( +0.0pp)  DIVIDEND $0

  MCTS Visits (top 10, 6400 total):
     1.  5469 (85.5%) Q=+0.815 ██████████████████████████████████ DIVIDEND $21
     2.   321 ( 5.0%) Q=+0.814 ██ DIVIDEND $16
     3.   142 ( 2.2%) Q=+0.815  DIVIDEND $19
     4.   141 ( 2.2%) Q=+0.814  DIVIDEND $18
     5.    77 ( 1.2%) Q=+0.816  DIVIDEND $22
     6.    57 ( 0.9%) Q=+0.813  DIVIDEND $15
     7.    41 ( 0.6%) Q=+0.808  DIVIDEND $1
     8.    37 ( 0.6%) Q=+0.811  DIVIDEND $9
     9.    33 ( 0.5%) Q=+0.810  DIVIDEND $6
    10.    28 ( 0.4%) Q=+0.815  DIVIDEND $20
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 5, vbackups: 5596)

  **Action: DIVIDEND $21**

### Step 419: P2 [DIVIDENDS]

  **Dividends**: SI

  NN Values: P0=+0.022, P1=-0.664, P2=+0.824
  NN Priors (top 10 of 13 legal):
     1.  99.2% (-14.9pp) ███████████████████████████████████████ DIVIDEND $12
     2.   0.8% ( +3.1pp)  DIVIDEND $9
     3.   0.0% ( +0.6pp)  DIVIDEND $6
     4.   0.0% ( +0.1pp)  DIVIDEND $3
     5.   0.0% ( +0.0pp)  DIVIDEND $11
     6.   0.0% ( +0.0pp)  DIVIDEND $8
     7.   0.0% ( +5.9pp)  DIVIDEND $0
     8.   0.0% ( +0.1pp)  DIVIDEND $2
     9.   0.0% ( +0.6pp)  DIVIDEND $10
    10.   0.0% ( +0.1pp)  DIVIDEND $5

  MCTS Visits (top 10, 6400 total):
     1.  5551 (86.7%) Q=+0.815 ██████████████████████████████████ DIVIDEND $12
     2.   331 ( 5.2%) Q=+0.812 ██ DIVIDEND $0
     3.   193 ( 3.0%) Q=+0.814 █ DIVIDEND $7
     4.   173 ( 2.7%) Q=+0.815 █ DIVIDEND $9
     5.    38 ( 0.6%) Q=+0.813  DIVIDEND $6
     6.    37 ( 0.6%) Q=+0.815  DIVIDEND $10
     7.    34 ( 0.5%) Q=+0.813  DIVIDEND $4
     8.    34 ( 0.5%) Q=+0.812  DIVIDEND $1
     9.     3 ( 0.0%) Q=+0.812  DIVIDEND $3
    10.     3 ( 0.0%) Q=+0.814  DIVIDEND $5
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 4, vbackups: 5599)

  **Action: DIVIDEND $12**

### Step 420: P0 [DIVIDENDS]

  **Dividends**: VM

  NN Values: P0=-0.014, P1=-0.723, P2=+0.805
  NN Priors (top 10 of 11 legal):
     1.  98.0% (-11.3pp) ███████████████████████████████████████ DIVIDEND $10
     2.   1.8% ( +4.0pp)  DIVIDEND $9
     3.   0.3% ( +0.5pp)  DIVIDEND $0
     4.   0.0% ( +0.7pp)  DIVIDEND $8
     5.   0.0% ( +0.5pp)  DIVIDEND $6
     6.   0.0% ( +0.2pp)  DIVIDEND $1
     7.   0.0% ( +1.1pp)  DIVIDEND $3
     8.   0.0% ( +1.2pp)  DIVIDEND $5
     9.   0.0% ( +0.4pp)  DIVIDEND $7
    10.   0.0% ( +2.6pp)  DIVIDEND $2

  MCTS Visits (top 10, 6400 total):
     1.  5668 (88.6%) Q=-0.001 ███████████████████████████████████ DIVIDEND $10
     2.   361 ( 5.6%) Q=-0.002 ██ DIVIDEND $9
     3.   120 ( 1.9%) Q=-0.009  DIVIDEND $2
     4.    65 ( 1.0%) Q=-0.006  DIVIDEND $5
     5.    56 ( 0.9%) Q=-0.008  DIVIDEND $3
     6.    40 ( 0.6%) Q=-0.006  DIVIDEND $0
     7.    37 ( 0.6%) Q=-0.005  DIVIDEND $8
     8.    23 ( 0.4%) Q=-0.007  DIVIDEND $6
     9.    18 ( 0.3%) Q=-0.006  DIVIDEND $7
    10.     6 ( 0.1%) Q=-0.019  DIVIDEND $1
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 3, vbackups: 5794)

  **Action: DIVIDEND $10**

### Step 421: P1 [DIVIDENDS]

  **Dividends**: S

  NN Values: P0=-0.018, P1=-0.730, P2=+0.816
  NN Priors (top 10 of 11 legal):
     1.  99.9% (-15.0pp) ███████████████████████████████████████ DIVIDEND $10
     2.   0.1% ( +4.9pp)  DIVIDEND $9
     3.   0.0% ( +3.2pp)  DIVIDEND $7
     4.   0.0% ( +0.5pp)  DIVIDEND $4
     5.   0.0% ( -0.0pp)  DIVIDEND $8
     6.   0.0% ( +1.4pp)  DIVIDEND $5
     7.   0.0% ( +1.4pp)  DIVIDEND $6
     8.   0.0% ( +2.2pp)  DIVIDEND $0
     9.   0.0% ( +0.0pp)  DIVIDEND $1
    10.   0.0% ( +0.0pp)  DIVIDEND $3

  MCTS Visits (top 10, 6400 total):
     1.  5547 (86.7%) Q=-0.814 ██████████████████████████████████ DIVIDEND $10
     2.   312 ( 4.9%) Q=-0.815 █ DIVIDEND $9
     3.   182 ( 2.8%) Q=-0.817 █ DIVIDEND $7
     4.   119 ( 1.9%) Q=-0.818  DIVIDEND $0
     5.    75 ( 1.2%) Q=-0.818  DIVIDEND $6
     6.    71 ( 1.1%) Q=-0.819  DIVIDEND $5
     7.    58 ( 0.9%) Q=-0.823  DIVIDEND $2
     8.    27 ( 0.4%) Q=-0.819  DIVIDEND $4
     9.     3 ( 0.0%) Q=-0.806  DIVIDEND $1
    10.     3 ( 0.0%) Q=-0.805  DIVIDEND $3
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 2, vbackups: 5578)

  **Action: DIVIDEND $10**

### Step 422: P1 [DIVIDENDS]

  **Dividends**: SM

  NN Values: P0=-0.006, P1=-0.734, P2=+0.824
  NN Priors (top 3 of 3 legal):
     1.  99.6% (-13.2pp) ███████████████████████████████████████ DIVIDEND $2
     2.   0.4% ( +3.4pp)  DIVIDEND $1
     3.   0.0% ( +9.7pp)  DIVIDEND $0

  MCTS Visits (top 3, 6400 total):
     1.  5689 (88.9%) Q=-0.814 ███████████████████████████████████ DIVIDEND $2
     2.   495 ( 7.7%) Q=-0.820 ███ DIVIDEND $0
     3.   216 ( 3.4%) Q=-0.817 █ DIVIDEND $1
  A0GB Value: P0=-0.001, P1=-0.814, P2=+0.815 (depth: 1, vbackups: 5708)

  **Action: DIVIDEND $2**

  ↳ auto: AUTO:END_CARD

Phase: GAME_OVER  |  Turn: 14  |  CoO Level: 7  |  Active Player: 1  |  End Card: YES

**Players**
  P0: $117 (NW $339) order=1 income=$0  shares=[PR=2, VM=2 (pres), SI=1]
  P1: $57 (NW $270) order=2 income=$0  shares=[S=2 (pres), SM=5 (pres), VM=1, SI=1]
  P2: $176 (NW $412) order=0 income=$0  shares=[PR=2 (pres), SI=2 (pres)]

**FI**: $74 income=$5

**Auction Row**: (empty)

**Corporations**
  S: $30 price=$24(idx 15) shares=bank:2/unissued:3/issued:4 income=$10 stars=8 pres=P1  companies=[CDG]
  SM: $1 price=$14(idx 10) shares=bank:1/unissued:0/issued:6 income=$10 stars=5 pres=P1  companies=[MAD]
  PR: $22 price=$50(idx 22) shares=bank:0/unissued:1/issued:4 income=$-2 stars=6 pres=P2  companies=[SJ]
  VM: $2 price=$27(idx 16) shares=bank:0/unissued:1/issued:3 income=$2 stars=7 pres=P0  companies=[NS, BR]
  SI: $0 price=$68(idx 25) shares=bank:0/unissued:0/issued:4 income=$39 stars=29 pres=P2  companies=[KK, PKP, DR, SZD, E, HR, FRA]

**Deck**: 0 remaining


---

## Game Over

Completed in 423 decision points (2367855 virtual backups from subtree reuse)

  P0: net worth $339
  P1: net worth $270
  P2: net worth $412

**Winner: P2 ($412)**
Terminal values (blend=0.75): P0=-0.001, P1=-0.814, P2=+0.815

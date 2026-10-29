# Shikoku 1889 - Rules Specification

This document restates the rules of Shikoku 1889 for use as the basis of a
game engine. It follows the Grand Trunk Games rulebook (2022 edition). Section
references such as (R§8.7.6) point to that rulebook.

Scope and conventions:

- This covers the **standard game**. The beginner variant appears only briefly
  (Section 15) and is out of scope for the engine.
- **Static data is summarized, not transcribed.** The exact stock market grid,
  the hex-by-hex map, and tile geometry and counts will be extracted when the
  engine is built. This document gives the parts of that data needed to
  understand the rules.
- **Hex IDs** follow the 18xx.games convention, because the printed board has
  no coordinates. Hexes are flat-topped. Columns run A-L from west to east,
  rows run 1-14 from north to south, and each column uses only odd or only even
  row numbers.
- Amounts are in yen (¥). The rulebook uses ¥1 as its smallest unit.
- "Share" means a 10% unit of ownership. "Certificate" means a physical card:
  a President's Certificate is one certificate worth two shares.
- **Where the rulebook is silent, ambiguous, or contradicts itself, this
  document follows the 18xx.games implementation**, since the bot will play
  on that platform. Such rulings are marked **Interpretation:** and collected
  in Section 12. Section 13 compares the rulebook with 18xx.games, and
  Section 14 records decisions and open questions.

## Table of Contents

1. [Game Overview](#1-game-overview)
2. [Components and Definitions](#2-components-and-definitions)
3. [Setup](#3-setup)
4. [Game Sequence](#4-game-sequence)
5. [Private Company Auction](#5-private-company-auction)
6. [Stock Round](#6-stock-round)
7. [Connectivity](#7-connectivity)
8. [Operating Round](#8-operating-round)
9. [Phases](#9-phases)
10. [Private Company Abilities](#10-private-company-abilities)
11. [Game End and Scoring](#11-game-end-and-scoring)
12. [Clarifications and Interpretations](#12-clarifications-and-interpretations)
13. [Differences from the 18xx.games Implementation](#13-differences-from-the-18xxgames-implementation)
14. [Open Questions](#14-open-questions)
15. [Beginner Game (Out of Scope)](#15-beginner-game-out-of-scope)
16. [Glossary](#16-glossary)

---

## 1. Game Overview

Shikoku 1889 is an 18xx game for 2-6 players set on the island of Shikoku,
Japan. Every player starts with the same cash and plays an investor. Players:

- buy private companies in a one-time auction at the start;
- start (par) corporations and buy and sell their shares in Stock Rounds;
- as president of a corporation, run it in Operating Rounds: build track,
  place station tokens, buy trains, run trains for revenue, and either pay
  that revenue out as dividends or keep it in the treasury.

The player with the highest **net worth** (cash + market value of shares +
face value of player-owned private companies) at game end wins.

The game ends immediately when a player goes bankrupt, or after a final set of
Operating Rounds once the bank runs out of money.

---

## 2. Components and Definitions

### 2.1 Money and the Bank

- The bank starts with **¥7,000**. Starting capital is paid out of it.
- All cash holdings, share holdings, and corporation treasuries are public
  information (R§11).
- When the bank runs out, it becomes unlimited and the game end is triggered
  (Section 11.2).

### 2.2 Corporations

There are 7 corporations. Each has a charter that holds its treasury, trains,
station tokens, and any private companies it owns. A corporation's assets
belong to the corporation, not to its president.

| Abbr | Name | Home city (hex) | Station tokens |
|------|------|-----------------|----------------|
| AR | Awa Railway | Tokushima (K8) | 2 |
| IR | Iyo Railroad | Matsuyama (E2) | 2 |
| SR | Sanuki Railway | Marugame (I2) | 2 |
| KO | Takamatsu-Kotohira Electric Rail | Takamatsu (K4) | 2 |
| TR | Tosa Electric Rail | Kôchi (F9) | 3 |
| KU | Tosa Kuroshio Railroad | Kubokawa (C10) | 1 |
| UR | Uwajima Railroad | Uwajima (B7) | 3 |

- The first station token (the home token) is free and goes on the home city.
  Each further token costs ¥40 to place.
- Each home city has a city spot reserved for that corporation's home token
  until the token is placed (Section 8.5).
- The rulebook spells some names inconsistently ("Iyo Railroad" and "Iyo
  Railway"); the charter spellings are used above.

### 2.3 Certificates and Share Locations

Each corporation is divided into 10 shares, issued as 9 certificates:

- one **President's Certificate** (20%, two shares, one certificate);
- eight ordinary **10% shares**.

Shares can be in three places:

- **Initial Offering (IO):** where all of a corporation's certificates start,
  with the President's Certificate on top. IO shares pay no dividends.
- **Players' hands.**
- **Open Market:** where sold shares go. Dividends on Open Market shares go to
  the corporation's own treasury. At most 50% (5 shares) of any corporation may
  be in the Open Market, and the President's Certificate can never go there.

Corporations never own shares, either their own or another corporation's.

The **president** of a corporation is the player holding its President's
Certificate. The president makes every decision for the corporation.

### 2.4 Private Companies

Private companies are single certificates that pay fixed revenue to their
owner (a player or a corporation) at the start of every Operating Round, and
during the auction in one special case (Section 5.4). Most have a special
ability; see Section 10.

| Letter | Name (18xx.games ID) | Face value | Revenue | In play with |
|--------|----------------------|------------|---------|--------------|
| A | Takamatsu Electric Track (TR) | ¥20 | ¥5 | all player counts |
| B | Mitsubishi Ferry (MF) | ¥30 | ¥5 | all player counts |
| C | Ehime Railroad (ER) | ¥40 | ¥10 | all player counts |
| D | Sumitomo Besshi Mine Railroad (SMR) | ¥50 | ¥15 | all player counts |
| E | Dôgo Railway (DR) | ¥60 | ¥15 | all player counts |
| F | Pilgrimage Railway (SIR) | ¥80 | ¥20 | 3+ players |
| G | Uno-Takamatsu Ferry (UTF) | ¥150 | ¥30 (¥50 from phase 5) | 4+ players |

Privates are referred to by letter in this document. The 18xx.games ID "TR"
for private A collides with the corporation TR.

All privates close when phase 5 begins, except G while owned by a player
(Section 9.3). Closed privates are removed from play and are worth nothing.

### 2.5 Trains

Trains are grouped into **ranks** by number or letter. A train's number is the
maximum number of revenue centers its route may count; a D-train (Diesel) has
no limit.

| Rank | Count | Price | Rusts when | Notes |
|------|-------|-------|------------|-------|
| 2 | 6 | ¥80 | phase 4 begins (first 4-train bought) | |
| 3 | 5 | ¥180 | phase 6 begins (first 6-train bought) | |
| 4 | 4 | ¥300 | phase D begins (first D-train bought) | |
| 5 | 3 | ¥450 | never | first 5-train closes the private companies |
| 6 | 2 | ¥630 | never | first 6-train makes D-trains available |
| D | unlimited | ¥1,100, or ¥800 plus a 4-, 5-, or 6-train traded in | never | available from phase 6 onward |

Ranks 5, 6, and D are permanent. Unsold trains wait in the **Train Yard**.
Trains discarded or traded in go to the **Open Market**, where any corporation
may buy them at list price (Section 8.8).

### 2.6 The Map

The map has 52 hexes. Summary of the features that rules refer to:

- **Revenue centers** are cities, towns, and off-board locations. Every hex
  holds at most one revenue center.
  - **Cities** (white circles) have one or more **spots** for station tokens.
    A printed city with no tile counts as one spot and ¥0 revenue until a tile
    is laid.
  - **Towns** (black dots) have no spots and can never be blocked.
  - **Off-board locations** (red hexes) have printed revenue that rises with
    the phase. A route may only start or end at one.
- **Ports:** four coastal towns with an anchor icon: Nakamura (B11),
  Nankoku (G10), Mugi (I12), and Komatsujima (J9). Private B can place the
  special port tile on them.
- **Terrain** (each costs ¥80 to build on; Section 8.4.6):
  - 14 mountain hexes;
  - 3 river hexes (H5, I6, K6);
  - dense urban areas: Kotohira (I4) and Kôchi (F9).
- **Labels:** H on Kotohira (I4), T on Takamatsu (K4), K on Kôchi (F9). Only
  tiles with the matching label may be placed on a labeled hex, and labeled
  tiles go only on their own hex.
- **Pre-printed track:**
  - Yellow: Ôzu (C4, a ¥20 city with one exit, toward Yawatahama) and
    Takamatsu (K4, a ¥30 T city with three exits). Both can be upgraded.
  - Green: Kôchi (F9, a ¥30 K city with two spots and four exits). It can be
    upgraded.
  - Gray (never upgraded): Yawatahama (B3, ¥20 town), Uwajima (B7, ¥40 city
    with two spots), Muroto (G14, ¥20 town), and J7 (plain track over the
    river).
- **Off-board revenue** (phases 2-4 / 5+ / D-trains in phase D):
  - Hiroshima (F1): ¥30 / ¥60 / ¥100. The kanji is 今治 (Imabari), the name
    18xx.games uses.
  - Okayama (J1): ¥20 / ¥40 / ¥80.
  - Awaji (L7): ¥20 / ¥40 / ¥80.
- **Private company icons** mark Takamatsu (K4, private A) and Ôzu (C4,
  private C); these hexes cannot be upgraded while the private is
  player-owned.
- The sea around the island is off the map.

### 2.7 Track Tiles

- Tiles come in three colors, ranked **yellow < green < brown**. There are no
  gray tiles in the supply.
- Every tile is **limited** to its printed count. A tile removed by an upgrade
  returns to the supply and can be used again. Pre-printed hexes are not tiles
  and never return to the supply.
- Tile families (revenue / city spots by color):

| Family | Yellow | Green | Brown |
|--------|--------|-------|-------|
| Plain track | curves and straight (#7, #8, #9) | junctions and crossings (#16, #19, #20, #23-#29) | three- and four-way junctions (#39-#42, #45-#47) |
| Town | ¥10 (#3, #58); port ¥30 (#437) | none | none |
| Unlabeled city | ¥20, 1 spot (#5, #6, #57) | ¥30, 1 spot (#12, #13, #205, #206) or 2 spots (#14, #15) | ¥40, 2 spots (#448, #611) |
| H (Kotohira) | ¥40, 1 spot (#438) | ¥60, 2 spots (#439) | ¥80, 3 spots (#492) |
| T (Takamatsu) | printed: ¥30, 1 spot | ¥40, 2 spots (#440) | ¥60, 2 spots (#466) |
| K (Kôchi) | none | printed: ¥30, 2 spots | ¥60, 3 spots (#465) |

- Towns never upgrade: no green or brown town tiles exist.
- The port tile #437 can only be placed using private B's ability.
- The rulebook's tile manifest (R§16) lists each tile's count and upgrades.
  It is a player aid; legality comes from the preservation rule (Section 8.4.3).
  The manifest omits some legal upgrades (Section 12.3).

### 2.8 Stock Market

The stock market is a 2D grid of 11 rows. Row length shrinks toward the
bottom, so the right edge forms a staircase. Prices range from ¥350 at the top
right to ¥10 at the bottom left.

- **Par values:** ¥65, ¥70, ¥75, ¥80, ¥90, ¥100. Each has a pink space in the
  fourth column of the top six rows (¥100 in the top row down to ¥65 in the
  sixth).
- **Yellow zone** (the ¥40-¥50 spaces at the left of the lower rows): shares of
  a corporation whose token is here do not count toward the certificate limit.
- **Orange zone** (the ¥10-¥30 spaces in the bottom-left corner): shares of a
  corporation whose token is here do not count toward the certificate limit,
  and are exempt from the 60% holding limit.
- These effects last only while the token is in the zone.

The full grid will be transcribed with the static data. The 18xx.games
`MARKET` constant matches the printed board, including zones and par spaces.

---

## 3. Setup

1. **Private companies by player count** (R§2.I):
   - 2 players: A-E.
   - 3 players: A-F.
   - 4-6 players: A-G.

   Arrange them in a column in letter order (which is also ascending face
   value), with A on top.
2. **Bank:** ¥7,000.
3. **Starting capital** from the bank:

| Players | 2 | 3 | 4 | 5 | 6 |
|---------|---|---|---|---|---|
| Capital | ¥420 | ¥420 | ¥420 | ¥390 | ¥390 |
| Certificate limit | 25 | 19 | 14 | 12 | 11 |

4. **Priority Deal and seating:** seating order is random and the Priority Deal
   goes to a random player. (The rulebook deals numbered order markers and
   reseats players; that is equivalent to a random order.)
5. All corporation certificates start in the Initial Offering; all trains start
   in the Train Yard; all tiles start in the supply.

---

## 4. Game Sequence

```
Private Company Auction (once)
repeat:
    Stock Round
    k Operating Rounds, with k set by the phase at the end of the Stock Round:
        phase 2 -> 1 OR; phases 3-4 -> 2 ORs; phases 5, 6, D -> 3 ORs
until the game ends (bankruptcy: immediately; bank broken: see Section 11.2)
```

- The number of ORs in a set is fixed when the Stock Round ends. A phase change
  during the set does not change how many ORs remain; the new count applies
  from the next Stock Round onward.
- The first Stock Round begins with the player who holds the Priority Deal
  after the auction.

---

## 5. Private Company Auction

The auction is a reserve-bid ("waterfall") auction held once, before the first
Stock Round (R§5).

### 5.1 Structure

- The privates are sold strictly top-down. Only the **topmost** unsold private
  can be bought outright. Any other unsold private can only be **bid on**.
- The Priority Deal holder acts first. Turns go clockwise. On each turn the
  active player does exactly one of the following:
  - **Purchase** the topmost private at its current price (face value, unless
    reduced; Section 5.4), paid to the bank.
  - **Bid** on any unsold private other than the topmost, by placing a new bid
    or raising their own existing bid. A bid must be at least ¥5 above the
    current highest bid on that private, or at least ¥5 above face value if it
    has no bids.
  - **Pass.** A player who passes may still act on later turns.
- **Cash limit:** a player may never spend or bid more than they have. A
  player's outstanding bids across all privates count against their cash, so
  the sum of their bids (plus any purchase) may not exceed their cash.
  Bids are not paid until won.

### 5.2 Resolving Bids When a Private Becomes Topmost

Whenever the topmost private is sold, the next private down becomes topmost.
Before the next player's turn, check it:

- **No bids:** normal turns resume with the next player clockwise from the
  player whose turn it was.
- **One bid:** that bidder buys it for their bid.
- **Several bids:** a closed auction among the bidders only.
  - The first to act is the next bidder clockwise from the current high
    bidder.
  - Going clockwise among the remaining bidders, each must raise the high bid
    by at least ¥5 or drop out. The high bidder does not act until someone
    outbids them.
  - A player who drops out loses their bid (they pay nothing) and cannot
    re-enter for this private.
  - The last bidder left buys the private for their high bid.

After each such sale, check the new topmost private the same way. Several
privates can be sold in a chain before the next normal turn.

### 5.3 End of the Auction

- The auction ends when every private has been sold.
- The Priority Deal goes to the player after the one who took the last normal
  turn: the player clockwise from whoever made the final purchase at face
  value (or reduced price), counting any chain of bid resolutions it triggered.
- Play continues with the first Stock Round.

### 5.4 Everyone Passes

If every player passes in succession:

- **No private has been bought yet** (so private A is still topmost): reduce
  A's purchase price by ¥5. Play continues with the player who had the first
  chance to buy A (the first player of the auction). If passes continue until
  A's price reaches ¥0, that player receives A for free. This counts as their
  purchase, and play continues with the next player clockwise.
- **At least one private is owned:** every owned private pays its revenue to
  its owner, as if an Operating Round had occurred. Play continues with the
  player who began this run of passes. The price of the topmost private does
  not change.

---

## 6. Stock Round

### 6.1 Turn Structure

- The Priority Deal holder starts; turns go clockwise.
- On a turn, a player may **sell** any number of shares and **buy** one
  certificate, in either order (sell then buy, or buy then sell). They may not
  sell, buy, and then sell again.
- A player who neither sells nor buys passes. Passing does not remove a player
  from the round: they may act again on their next turn if the round has not
  ended.
- The round ends when all players pass in succession.

### 6.2 Buying

A player may buy one of these certificates per turn:

1. **A President's Certificate** of a corporation not yet started. The buyer:
   - chooses a par value from ¥65, ¥70, ¥75, ¥80, ¥90, ¥100;
   - pays **2 × par** to the bank;
   - places the corporation's stock token on the matching par space of the
     stock market (stacked beneath any tokens already there; Section 6.7).

   The par value is recorded and never changes.
2. **A 10% share from the Initial Offering** of a started corporation, for
   its **par value**, paid to the bank.
3. **A 10% share from the Open Market**, for the corporation's **current
   market price**, paid to the bank.

All payments for shares go to the bank, never to the corporation.

**Purchase restrictions** (R§7.2.4):

- **60% holding limit:** a player may not buy a share of a corporation in which
  they already hold 60% or more, unless its token is in the orange zone.
  Normally, then, a player can reach at most 60%.
- **Certificate limit:** a player may not buy a certificate that counts toward
  the limit if they are already at or above the limit. This holds even if the
  purchase would trigger a presidency change that would bring them back under
  it. Shares of corporations in the yellow or orange zone don't count, so a
  player may buy them at the limit (Section 12.1).
- **No buying back:** a player may not buy shares of a corporation they have
  sold earlier in the same Stock Round.
- A player can never spend more cash than they have.

### 6.3 Selling

- **No selling at all in the first Stock Round.**
- A player may sell shares of one or more corporations in their sell step.
  For each corporation:
  - all its shares sold in that step form one block;
  - the player receives the **current market price** for every share in the
    block;
  - the shares go to the Open Market;
  - the stock token then moves **down one space per share sold**, stopping at
    the bottom of its column.

  Splitting a block cannot help the seller, since a later part would sell at a
  lower price (Section 12.2).

**Sale restrictions** (R§7.3.1):

- A sale may not put more than 50% (5 shares) of a corporation in the Open
  Market.
- The President's Certificate cannot be sold. A president can sell shares
  below another player's holding only through a presidency transfer
  (Section 6.5).
- Private companies cannot be sold in a Stock Round. Their only sale is to a
  corporation during an Operating Round (Section 8.10).

### 6.4 Mandatory Sales at the Start of a Turn

At the start of their Stock Round turn, a player who is over the certificate
limit, or above 60% of a corporation outside the orange zone, must sell until
compliant. Rules:

- These sales are part of the turn's sell step and follow all sale
  restrictions.
- If a sale moves a corporation's token into the yellow or orange zone, its
  shares stop counting (and, in the orange zone, stop being limited to 60%);
  re-check compliance after each sale.

How a player can end up over a limit:

- a corporation's token leaves the yellow or orange zone;
- they receive two 10% shares as the outgoing president in a presidency
  transfer.

### 6.5 Presidency Changes

The President's Certificate must change hands as soon as another player holds
**strictly more** shares of the corporation than the president, whether
because of a purchase or a sale (R§7.5). Ties never change the presidency.

When it changes hands:

- The new president gives the old president two 10% shares and receives the
  President's Certificate and the charter, with all its assets.
- If several players are tied for most shares, the one closest clockwise from
  the current president becomes president.
- **On a purchase:** a buyer whose holding now exceeds the president's becomes
  president.
- **On a sale by the president:** the president may sell down below another
  player only if, after the sale, some other player:
  - holds at least two 10% shares, and
  - holds strictly more shares than the selling president.

  That player (ties broken clockwise from the president) takes the
  presidency. The exchange happens during the sale, and the two shares the
  seller receives may themselves be sold as part of it.
- If no player qualifies, the president cannot sell below two shares. They
  can still sell ordinary shares as long as they keep the most shares, ties
  included.

Example (R§7.5): the UR president holds 6 shares, the player to their left
holds 2, the IO holds 2, and the Open Market is empty. The president sells 5
shares. They swap the President's Certificate for the neighbor's two 10%
shares and sell five 10% shares, keeping one. The neighbor (2 shares) now
holds more than the old president (1) and becomes president.

### 6.6 Floating

- A started corporation **floats** when only 5 shares (50%) remain in its
  Initial Offering. Shares in the Open Market do not affect this.
- On floating it receives **10 × par** from the bank into its treasury (full
  capitalization), and its station tokens go on its charter.
- Its home token is placed at the start of the next Operating Round
  (Section 8.1).
- A corporation that has started but not floated does not operate. In
  particular it does not move left for failing to pay dividends.

### 6.7 Stock Market Movement and Order

| Event | Move | At the edge |
|-------|------|-------------|
| Dividends paid | 1 right | at the right end of its row: 1 up instead; at ¥350 (top right): no move |
| Withheld, or no earnings | 1 left | at the left end of its row: 1 down instead; at ¥10 (bottom left): no move |
| Shares sold | 1 down per share | at the bottom of its column: stop |
| Sold out at end of Stock Round | 1 up | top row: no move |

- **Stacking:** a token entering a space already holding tokens goes
  **beneath** them. This applies to placement and every move.
- **Stock price order** (R§3.2), used for operating order and other
  "descending price" sequences:
  1. Higher price first.
  2. Equal prices in different spaces: the token farther right first.
  3. Same space: top of the stack first, then downward.

### 6.8 Certificate Limit

- Limits by player count: 2p 25, 3p 19, 4p 14, 5p 12, 6p 11.
- Counted certificates: each private company, each 10% share, and each
  President's Certificate count as one. Shares of corporations in the yellow
  or orange zone are not counted.
- The limit is enforced only in Stock Rounds: on purchases (Section 6.2) and
  by mandatory sales at the start of a turn (Section 6.4). Being over the
  limit during an Operating Round has no effect.

### 6.9 End of the Stock Round

When all players have passed in succession:

1. **Priority Deal** goes to the player who would have acted after the final
   pass. Equivalently, the player clockwise from the last player to buy or
   sell. If nobody bought or sold all round, it stays where it is.
2. **Sold-out adjustment:** in descending stock price order, each corporation
   whose shares are all held by players (none in the IO or the Open Market)
   moves up one space.
3. The number of ORs in the coming set is fixed from the current phase
   (Section 4).

---

## 7. Connectivity

Connectivity is shared by track laying, station token placement, the
must-own-a-train rule, and (with extra conditions) routes (R§4.2).

A corporation **has connectivity** to a point on the map if it can trace a
path of any length along track from a city containing one of its station
tokens to that point, such that:

- no track segment is used twice;
- no city or town is visited twice;
- the path never reverses direction at a **junction** (a hex edge where two
  track segments of the same tile meet, as on #23-#25). Entering such an edge
  on one segment and leaving on the other is a reversal, and counts as reusing
  track;
- the path does not continue through a **blocked city**. A blocked city has
  every spot filled by other corporations' tokens. The path can reach such a
  city but not pass through it. A city holding any of the corporation's own
  tokens, or with a free spot, is never blocked for it;
- the path does not continue through an **off-board location**. It can reach
  it but not pass through.

Towns never block.

---

## 8. Operating Round

### 8.1 Start of Each Operating Round

1. **Private revenue:** every private company pays its revenue from the bank
   to its owner (a player, or a corporation's treasury).
2. **Home tokens:** every corporation that floated since the previous OR
   places its free home token on its reserved spot in its home city.
3. Corporations then operate one at a time in stock price order
   (Section 8.2).

### 8.2 Operating Order

- Each floated corporation operates once per OR. Unfloated corporations are
  skipped.
- The next corporation to operate is always the highest in stock price order
  (Section 6.7) among those that have not yet operated this OR. (The physical
  game flips a corporation's stock token once it has operated, and flips them
  all back at the end of the OR.)
- Re-evaluating the order after each turn matters only when a corporation that
  has not yet operated changes price during the OR. In practice that happens
  when a president sells shares to pay for a forced train purchase.
- **Interpretation:** a corporation that floats during an OR (possible through
  private E's exchange; Section 10.5) does not operate until the next OR,
  because its home token is placed only at the start of an OR.

### 8.3 Corporation Turn Structure

The president performs these steps in order. Each is optional unless noted:

- A. Lay or upgrade one track tile (Section 8.4).
- B. Place one station token (Section 8.5).
- C. Run trains: mandatory if the corporation owns trains (Section 8.6).
- D. Distribute earnings and adjust the stock price: mandatory
  (Section 8.7).
- E. Buy trains: mandatory in one case (Sections 8.8-8.9).
- F. Buy private companies: phases 3-4 only, at **any point** in the turn
  (Section 8.10).

Steps A-E must be done in this order; a corporation cannot, for example, buy a
train before laying track. Step F may happen before, between, or after the
other steps.

### 8.4 Step A: Lay or Upgrade Track

A corporation gets **one track action** per OR: either lay one tile or upgrade
one tile.

#### 8.4.1 Laying and Upgrading

- **Lay:** place a yellow tile on a hex that has no track yet. Track
  pre-printed on the map counts as existing track.
- **Upgrade:** replace a tile, or pre-printed track, with a tile exactly one
  color higher: yellow to green or green to brown.
  - The removed tile returns to the supply.
  - Pre-printed yellow (Ôzu, Takamatsu) and green (Kôchi) track can be
    upgraded; gray and red hexes never can.
- The new tile's color must be available in the current phase:

| Phase | Tile colors |
|-------|-------------|
| 2 | yellow |
| 3-4 | yellow, green |
| 5+ | yellow, green, brown |

- The tile must be in the supply. It is possible to have no legal lay.

#### 8.4.2 Which Tile Fits Which Hex

- **Empty plain hex:** yellow plain track (#7, #8, #9).
- **Empty city hex:** yellow city (#5, #6, #57). Kotohira takes only #438.
- **Empty town hex, including ports:** yellow town (#3, #58). On a port, private
  B may instead place #437.
- **Kotohira (H) and Takamatsu (T)** take only tiles with their label. Kôchi
  (K) can only be upgraded to #465.
- Unlabeled tiles never go on labeled hexes, and labeled tiles go only on
  their own hex.

#### 8.4.3 Preservation

The new tile must keep everything already on the hex (R§8.3):

- every existing track segment, in the same position. The new tile may be
  rotated, but each old connection must still exist;
- the same revenue center: a city stays a city, a town stays a town, and plain
  track never gains a revenue center;
- all station tokens. They move onto the new tile's city, so it needs enough
  spots;
- the hex's label.

Printed terrain costs and revenue values are not preserved: the new tile
brings its own revenue.

#### 8.4.4 Connectivity Requirement

The lay or upgrade must do at least one of the following (R§8.3):

1. be placed in the corporation's **home hex**;
2. add **new track** that has connectivity (Section 7) to one of the
   corporation's station tokens, counting the new tile as already placed; or
3. **increase the revenue** of a city that has connectivity to one of the
   corporation's station tokens. This allows a value-only upgrade, such as
   #15 to #448, which adds no track.

For an upgrade that adds track, the new segment itself must be reachable;
being reachable only through an existing segment of the same tile isn't
enough. Because of the junction rule, a branch added at an edge the
corporation reaches only from inside the hex cannot be reached that way
(Section 12.4).

#### 8.4.5 Placement Restrictions

- **Map edges and gray hexes:** a tile may not have track running off the map,
  or into a gray hex side with no track. (R§8.3.1 "Connections".)
  - **Interpretation:** a red off-board hex side with no track is treated the
    same as a gray one (Section 12.5).
- **Other hexes:** track may point into any other hex side, even one with no
  track or where no matching track can ever be built (such as against a brown
  tile with no track on that side).
- **Private icons:** Takamatsu (K4) may not be upgraded while private A is
  owned by a player. Ôzu (C4) may not be upgraded while private C is owned by a
  player. Both restrictions end once the private is owned by a corporation or
  closed.

#### 8.4.6 Track Cost

- Laying or upgrading is free, except for terrain. A tile costs ¥80 if there is
  a terrain icon:
  - **on the map hex**, when laying the first tile there; or
  - **on the tile being replaced**, when upgrading.
- The corporation pays from its treasury and must be able to afford it.
- Terrain types (each ¥80): mountain, river, dense urban area.
- Consequences:
  - Mountain and river hexes cost ¥80 only for the first (yellow) tile.
  - Kotohira costs ¥80 for every tile: the hex, #438, and #439 all carry the
    icon.
  - Kôchi costs ¥80 for its brown upgrade, since the pre-printed green track
    carries the icon.
- A corporation that owns private D pays nothing for mountains. River and
  dense urban costs still apply (Section 10.4).

### 8.5 Step B: Place a Station Token

- **Once per OR**, a corporation with a token left on its charter may place it
  for **¥40**:
  - in an open spot of a city it has connectivity to (Section 7);
  - not in a hex where it already has a token (at most one token per
    corporation per hex);
  - not in a spot reserved for another corporation's home token.
    **Reserved spots:** a home city keeps one spot reserved until its owner
    places its home token (Section 8.1). A token may be placed only if the
    city would still have room for every home token not yet placed there.
    For example, nobody else can token single-spot Matsuyama before IR places
    its home token. Once Matsuyama is upgraded to a two-spot tile, one other
    corporation can.
- The free home token does not count as the OR's token placement.
- Tokens are permanent. They block other corporations' routes through a city
  once every spot is filled (Section 7).

### 8.6 Step C: Run Trains

A corporation that owns trains must run them. Each train runs one route, and
earns the sum of the revenue values counted on it.

#### 8.6.1 Route Rules (R§8.5.1)

A route is a continuous path along track, following the connectivity rules
for track and junctions (Section 7), that:

- **counts at least 2 revenue centers;**
- **counts at most N revenue centers** for an N-train. D-trains have no limit;
- **counts every revenue center it visits.** A train cannot pass through a
  city, town, or off-board location without counting it, so an N-train's
  route can visit at most N revenue centers;
- **visits each revenue center at most once;**
- **includes a revenue center holding one of the corporation's station
  tokens.** The token need not be at either end;
- **passes through no blocked city.** A blocked city (Section 7) may be the
  first or last stop, and counts there;
- **passes through no off-board location.** An off-board location can only be
  the first or last stop.

Across all of a corporation's trains in one OR:

- **No track segment or hex-edge crossing may be used by more than one
  train.** Each route also uses each segment at most once.
- Different trains **may** count the same revenue center, as long as they
  share no track.

#### 8.6.2 Revenue

- A train earns the total of its counted revenue centers.
  - Cities and towns pay their current tile value. A printed city with no
    tile pays ¥0.
  - Off-board locations pay by phase:

| Phase | Off-board value |
|-------|-----------------|
| 2-4 | yellow value |
| 5, 6 | brown value |
| D | brown value; **D-trains only** count the gray (D) value |

- The corporation's **earnings** are the sum over its trains.
- All revenue values are multiples of ¥10, so earnings always divide evenly
  into per-share dividends.
- **Interpretation:** the rulebook requires trains to run but not to find the
  best routes; the president picks them. The engine may assume the
  maximum-revenue set of routes (Section 12.7).

### 8.7 Step D: Distribute Earnings and Adjust the Stock Price

The president chooses one of two options. There is no partial payout.

| Choice | Money | Stock token |
|--------|-------|-------------|
| **Pay dividends** | Each share pays earnings ÷ 10: players get it per share held (President's Certificate = 2 shares); Open Market shares pay the corporation's treasury; IO shares pay nobody (that money stays in the bank) | 1 right (or up at row end; none at ¥350) |
| **Withhold** | All earnings go to the treasury | 1 left (or down at row start; none at ¥10) |

- A corporation with **zero earnings** (no trains, or no legal route) cannot
  pay. It moves left as if it had withheld.
  - A newly floated corporation therefore always moves left in its first OR,
    since it has no train yet.
- Only this OR's earnings can be paid out; a corporation can never pay
  dividends from its treasury.

Example (R§8.6): IR earns ¥200. Five shares are held by Player 1, one by each
of Players 2 and 3, one is in the IO, and two are in the Open Market.

- Paying out gives ¥20 per share: Player 1 gets ¥100, Players 2 and 3 get
  ¥20 each, the IO share gets nothing, and IR's treasury gets ¥40 for the two
  Open Market shares.
- Withholding puts ¥200 in IR's treasury.

### 8.8 Step E: Buy Trains

A corporation may buy any number of trains, **one at a time**, resolving each
purchase fully (including any phase change) before the next. Sources:

- **Train Yard:** only the **lowest remaining rank**, at list price. For
  example, every 2-train must be sold before any 3-train can be bought.
  - **Exception:** from phase 6 on, D-trains are available alongside any
    6-trains still in the Yard.
- **Diesel exchange:** a corporation may trade in one of its own 4-, 5-, or
  6-trains and pay ¥800 for a D-train.
  - A traded-in 5- or 6-train goes to the Open Market.
  - A traded-in 4-train is removed. A 4-train can only be traded in for the
    first D-train, and buying that D-train starts phase D, which rusts all
    4-trains.
  - An exchange can trigger a phase change.
- **Open Market:** any train there, at list price.
- **Another corporation** (whoever its president is): any of its trains, at a
  price both presidents agree on, from ¥1 up to the buyer's treasury. When the
  same player runs both corporations, they set the price alone.

General rules:

- Payment comes from the buying corporation's treasury. A player's cash is
  never used, except in a forced purchase (Section 8.9).
- **Train limit:** a corporation may not hold more trains than the phase's
  limit (Section 9).
  - A corporation **at** its limit may not buy any train. This holds even if
    the purchase would rust some of its own trains and bring it back under the
    limit.
  - It may still make a Diesel exchange, which doesn't change its train count.
- **Phase change:** buying the first train of a new rank from the Yard
  (including by exchange) starts the next phase **immediately**, with all its
  effects (Section 9). The buying corporation then continues its turn.
- **Rusting:** when a phase rusts a rank, every train of that rank is removed
  from play with no compensation: on corporations and in the Open Market
  alike.
- **Excess trains after a phase change:** if a corporation now holds more
  trains than the new limit, its president must discard the excess (choosing
  which) to the Open Market, without compensation.
  - Discards are made in this order: first the corporation that caused the
    phase change, then the others in descending stock price order.
  - Trains are never discarded voluntarily.

### 8.9 Forced Train Purchase and Bankruptcy

#### 8.9.1 When a Train Is Required

At the end of its train step, a corporation **must own at least one train**
if it has a station token in a city that has connectivity to another revenue
center (that is, it could run a route). If it has no train, it must buy one.

#### 8.9.2 Procedure

1. **From another corporation (optional):** the corporation may buy a train
   from another corporation at an agreed price, from ¥1 up to its own
   **treasury**. The president may not add money for this, and the
   corporation is never obliged to accept an offer.
2. **Otherwise, from the Yard or Open Market:**
   - If the treasury can afford any available train, the corporation buys one.
   - If the treasury cannot afford any, the president must make up the
     difference from personal cash for the **cheapest available train** in
     the Yard or Open Market. The president may contribute no more than
     that train needs, so the cheapest train is the one bought.
3. **Raising money:** if the president's cash is not enough, they must sell
   shares. The rules for these sales:
   - The president chooses which shares and in what order.
   - The usual sale restrictions apply (Section 6.3): the Open Market cap, and
     the President's Certificate is never sold.
   - **No presidency may change hands** as a result of these sales, for any
     corporation. A president can therefore only sell down to the point where
     they still hold at least as many shares as every other player.
   - The president may not sell more than is needed to afford the cheapest
     train.
   - Exception: if the president holds more than 60% of a corporation that is
     not in the orange zone, they must also sell enough of it to get down to
     60%, even if that raises more than needed.
   - Each sale moves stock tokens down as usual. This can change the order of
     corporations still to operate this OR.
4. **Bankruptcy:** if, after every legal sale, the president still cannot
   cover the shortfall:
   - the president is **bankrupt**;
   - their remaining cash goes to the corporation's treasury;
   - the **game ends immediately** (Section 11.1).

Example (R§8.7.6): SR must buy a train, and the only one available is a
¥1,100 D-train. SR has ¥300, so its president owes ¥800. The president has
¥200 and must raise ¥600 more, but every legal sale together raises only ¥500.
The president is bankrupt: their cash goes to SR, and the game ends.

### 8.10 Step F: Buy Private Companies

- Only in **phases 3 and 4**, at **any point** during its operating turn, a
  corporation may buy any number of private companies from **players**:
  - at a price agreed with the owner, from ½ to 2 × face value inclusive;
  - paid from its treasury to the selling player.
- A private owned by a corporation can never be sold again.
- A corporation-owned private pays its revenue to that corporation's treasury.
- Some abilities work only for a player owner, some only for a corporation
  owner (Section 10).
- Example (R§8.8): private C (face ¥40) can be sold for ¥20-¥80. Selling it
  lets the seller immediately upgrade Ôzu (Section 10.3).

### 8.11 End of the Operating Round

After every floated corporation has operated:

- if the OR set has ORs remaining, start the next OR (Section 8.1);
- otherwise start a Stock Round, unless the game is ending (Section 11).

---

## 9. Phases

The game starts in phase 2. Buying the first train of a new rank from the
Train Yard starts the matching phase **immediately**. This includes buying the
first D-train by exchange.

| Phase | Begins when | Tile colors | Train limit | Rusts | Off-board value | ORs per set | Other effects |
|-------|-------------|-------------|-------------|-------|-----------------|-------------|---------------|
| 2 | game start | yellow | 4 | - | yellow | 1 | corporations may **not** buy privates |
| 3 | first 3-train | yellow, green | 4 | - | yellow | 2 | corporations may buy privates |
| 4 | first 4-train | yellow, green | 3 | 2-trains | yellow | 2 | corporations may buy privates |
| 5 | first 5-train | yellow, green, brown | 2 | - | brown | 3 | privates close (except G if player-owned) |
| 6 | first 6-train | yellow, green, brown | 2 | 3-trains | brown | 3 | D-trains available |
| D | first D-train | yellow, green, brown | 2 | 4-trains | brown; gray for D-trains | 3 | D-trains count gray off-board values |

### 9.1 Order of Effects

When a phase begins, all of its effects apply at once:

- rusting;
- new tile colors and off-board values;
- private company closing and the private-purchase window;
- the new train limit, followed by excess-train discards (Section 8.8).

The new "ORs per set" value takes effect only at the end of the next Stock
Round (Section 4).

### 9.2 Train Yard Availability

- The Yard offers only its lowest remaining rank.
- When the last train of a rank is sold, the next rank becomes available.
  Buying its first train starts the next phase. For example, all six 2-trains
  are sold during phase 2, and the 3-train bought after them starts phase 3.
- **Exception:** D-trains become available when phase 6 begins, alongside any
  6-trains still in the Yard. The first D-train starts phase D.
- The Open Market can hold trains of any rank still in play.

### 9.3 Phase 5: Privates Close

When the first 5-train is bought, every private company closes, except G
(Uno-Takamatsu Ferry) if a player owns it. Closing ends both revenue and
abilities.

- The blocks on Takamatsu and Ôzu (privates A and C) end.
- Corporation-owned privates close too, including G.
- If player-owned, G stays open for the rest of the game:
  - its revenue rises to ¥50;
  - it can no longer be sold to a corporation;
  - it counts ¥150 toward its owner's net worth.

---

## 10. Private Company Abilities

### 10.1 A - Takamatsu Electric Track (¥20, revenue ¥5)

- **Blocker:** while A is owned by a player, Takamatsu (K4) cannot be
  upgraded.
- The block ends once A is owned by a corporation or closed.
- Takamatsu is KO's home city. It is pre-printed yellow, so in practice A
  blocks its green upgrade.

### 10.2 B - Mitsubishi Ferry (¥30, revenue ¥5)

While B is owned by a player, the owner may, **once**, place the port tile
**#437** (yellow, port town worth ¥30) on one of the four port hexes:
Nakamura (B11), Nankoku (G10), Mugi (I12), or Komatsujima (J9).

Details:

- **Timing:** at any time except during the operating turn of a corporation
  controlled by another player.
- **No corporation needed:** the owner need not be a president.
- **No connectivity needed:** the tile need not connect to any track.
- **Free:** it is not any corporation's track action.
- **Empty hex only:** the hex must have no tile yet. #437 is yellow, so it
  cannot replace a #3 or #58 already there.
- **Final:** #437 can never be upgraded.
- Using the ability does not close B.
- The ability is lost if B is sold to a corporation.

**Interpretation (18xx.games):**

- The tile must still obey the placement restrictions in Section 8.4.5.
- The ability can be used only at these points (Section 12.8):
  - during a Stock Round;
  - during the track step of a corporation the owner is president of;
  - between corporation turns in an Operating Round.

### 10.3 C - Ehime Railroad (¥40, revenue ¥10)

- **Blocker:** while C is owned by a player, Ôzu (C4) cannot be upgraded.
- **Sale bonus:** when C is sold to a corporation, the **selling player** may
  immediately upgrade Ôzu. This upgrade comes on top of any track action the
  buying corporation takes that turn.
  - Sales happen only in phases 3-4, so the upgrade is always from the
    pre-printed yellow track to a green city tile (#12, #13, #14, #15, #205, or
    #206).
  - The usual tile legality applies: phase color, supply, preservation, and
    placement restrictions.
- **Interpretation:** the bonus upgrade needs no connectivity and costs
  nothing, since Ôzu has no terrain.

### 10.4 D - Sumitomo Besshi Mine Railroad (¥50, revenue ¥15)

- While D is owned by a corporation, that corporation ignores the ¥80 cost of
  **mountain** hexes.
- River hexes and dense urban areas still cost ¥80.
- No effect while owned by a player.

### 10.5 E - Dôgo Railway (¥60, revenue ¥15)

- While E is owned by a player, the owner may close E and take one 10% IR
  share from the Initial Offering in exchange.
- The share is free: E is surrendered instead of cash.
- The exchange cannot push a player over the certificate limit: one
  certificate goes and one comes.
- The exchanged share leaves the IO, so it counts toward IR floating
  (Section 6.6).

**Interpretation (18xx.games):** the rulebook does not settle these points.

- **Timing:** the exchange is allowed at any time.
- **Before IR starts:** the exchange is allowed even before anyone has bought
  IR's President's Certificate.
  - Until IR is started, the share has no market price and counts ¥0 toward
    net worth (Section 11.3).
  - When a player later buys IR's President's Certificate, IR floats once
    two more IO shares are sold, instead of the usual three. The exchanged
    share already counts as one of the five shares that must leave the IO.
- **60% limit:** the exchange is blocked if it would take the player above
  60% of IR, unless IR's token is in the orange zone.

### 10.6 F - Pilgrimage Railway (¥80, revenue ¥20)

- No special ability.
- Not used in 2-player games.
- 18xx.games calls it "South Iyo Railway".

### 10.7 G - Uno-Takamatsu Ferry (¥150, revenue ¥30, later ¥50)

- Used only with 4 or more players.
- If owned by a player when the first 5-train is bought, it does **not** close
  (Section 9.3). From then on:
  - its revenue is ¥50;
  - it can no longer be sold to a corporation;
  - it counts ¥150 toward its owner's net worth at game end.
- If owned by a corporation at that moment, it closes like the others.

---

## 11. Game End and Scoring

### 11.1 Bankruptcy

A player can go bankrupt only by failing to fund a forced train purchase
(Section 8.9). The game ends at once, before anything else happens.

### 11.2 Breaking the Bank

The bank **breaks** when it runs out of money, that is, when players and
corporation treasuries together hold ¥7,000 or more. From then on:

- The bank becomes unlimited, and every payment continues normally.
- If it broke during a **Stock Round**: finish that Stock Round, play one more
  full set of ORs (sized by the phase as usual), then the game ends.
- If it broke during an **Operating Round**: finish the current set of ORs,
  then the game ends.

### 11.3 Determining the Winner

Each player's **net worth** is:

- cash;
- plus, for each corporation, shares held × the current market price (the
  President's Certificate counts as 2 shares).
  - Shares of a started but unfloated corporation are valued at its current
    market price.
  - Shares of a corporation that has not been started are worth ¥0. This can
    only happen through private E's exchange (Section 10.5), and it is how
    18xx.games scores it;
- plus the face value of each private company the player owns that is still
  open (in practice, usually only G at ¥150).

Corporation treasuries and corporation-owned assets count for nobody. The
highest net worth wins. A bankrupt player is scored normally and can still
win.

Example (R§10.3): 6 IR shares at ¥110, 3 UR shares at ¥90, ¥840 cash, and
private G give 660 + 270 + 840 + 150 = ¥1,920.

---

## 12. Clarifications and Interpretations

Cases the rulebook leaves open or states loosely, and how this document reads
them.

1. **Certificate limit and zone shares.** R§7.2.4 says a player "already at
   the Certificate Limit" may not buy. Shares of yellow- or orange-zone
   corporations do not count toward the limit, so buying one does not change
   the count. This document allows it, as 18xx.games does.
2. **Selling in blocks.** All shares of one corporation sold in a turn's sell
   step form a single block at the pre-sale price, followed by one drop per
   share (R§7.3). An engine can model each corporation's sale as one action
   with a share count; splitting would only lower the price of later shares.
3. **Tile manifest vs. preservation.** R§8.3 defines upgrade legality by
   preservation; the manifest (R§16) is a "tip for first play". The manifest
   omits these upgrades, which preservation allows:
   - green #28 → brown #39;
   - green #29 → brown #39;
   - yellow #57 → green #205 and #206.

   Without the first two, #39 could never be placed. This document follows
   preservation, which is also what 18xx.games checks, and treats the gaps as
   printing errors. Separately, the manifest prints the second "#23" row where
   it means #24.
4. **New track must itself be reachable.** "Adds new track that has
   connectivity" (R§8.3) means at least one newly added segment must be
   reachable from a station token under Section 7, including the no-reversal
   rule at junctions. A branch added at an edge the corporation enters only
   from inside the hex is not reachable through that edge.
5. **Red off-board sides.** R§8.3.1 forbids pointing track off the map or into
   a gray hex side with no track. Off-board hexes are fixed like gray hexes,
   so the same rule is applied to their sides without track. 18xx.games treats
   gray, red, and blue hexes this way.
6. **Forced purchases from other corporations.** R§8.7.6 allows a corporation
   that must buy a train to buy one from another corporation, with treasury
   money only. The quick-start summary (R§14.2) says forced purchases may not
   be made between corporations. This document reconciles the two: the
   president's money can only go toward a Yard or Open Market train.
7. **Route choice.** The rulebook requires trains to run but leaves routes to
   the president. No situation is known where running less than the maximum
   helps (withholding already covers holding back money), so the engine may
   compute maximum-revenue routes. A replay interface should still accept
   explicit routes.
8. **Timing windows for player abilities.** The rulebook lets private B's tile
   be placed "at any time" outside other players' corporation turns, and
   gives private E no timing at all. Following 18xx.games:
   - B may be used during a Stock Round, during the track step of a
     corporation the owner is president of, or between corporation turns in
     an Operating Round;
   - E may be exchanged at any time.

   When building the engine, check two more things against 18xx.games:
   - which players may use these abilities on another player's Stock Round
     turn;
   - whether using one counts as a Stock Round action (and so resets the run
     of passes).
9. **Auction pass loop.** Once a private is owned, a full round of passes pays
   private revenue and play continues (R§5.2 note). Nothing forces the
   auction to end; in practice players buy. A self-play engine may need a
   safeguard.
10. **Corporations floating mid-OR.** Only private E's exchange can float a
    corporation outside a Stock Round. The home token waits for the next OR
    (R§7.4, R§14.2), so the corporation does not operate in the current one.
11. **Agreed prices.** Private sales (R§8.8) and train sales between
    corporations (R§8.7) use mutually agreed prices. When the two sides are
    different players, both must consent. Self-play needs an explicit
    offer/accept mechanism or a restriction (see `1889.md` §4.4).
12. **Ties for the win** are not addressed by the rulebook.
13. **Bank breaking during the auction** (through pass-round revenue) is not
    addressed. It is practically unreachable; treat it like a Stock Round
    break.

---

## 13. Differences from the 18xx.games Implementation

The bot will play on 18xx.games, so its implementation
(`submodules/18xx/lib/engine/game/g_1889/` and the base engine) settles
ambiguities and contradictions in the rulebook. Points checked against it:

| Topic | Rulebook | 18xx.games |
|-------|----------|------------|
| Market, phases, trains, certificate limits, starting cash | as above | matches |
| Station token counts | charters: KU 1; AR, IR, SR, KO 2; TR, UR 3 | matches |
| River hexes H5, I6 | drawn as river (¥80) | encoded as river + mountain, with private D discounting only pure-mountain hexes, so the cost outcome is the same |
| Tile upgrades | preservation rule; the manifest omits some upgrades | checks preservation (Section 12.3) |
| Forced purchase from other corporations | treasury money only (Section 12.6) | `EBUY_FROM_OTHERS = :never`: purchases from other corporations use treasury money only, and are not offered at all once the president has sold shares or the treasury is empty |
| Presidency change during forced sales | forbidden | `EBUY_PRES_SWAP = false` |
| Operating order | dynamic, by stock price | fixed at the start of the OR, re-sorted after emergency share sales; same result |
| Private B timing | any time outside other players' corporation turns | Stock Round, the owner's own corporation's track step, and between OR turns |
| Private C bonus | "upgrade Ôzu" | lay a green city tile (#12-#15, #205, #206) on C4, paid by the seller |
| Private E exchange | timing, IR started, and 60% limit unspecified | allowed at any time, even before IR is started (`available_share` takes any non-President IO share); blocked above 60% (`holding_ok?`); shares of unstarted corporations count ¥0 in `Player#value` |
| Consent for private and train sales across players | mutual agreement | not enforced by the engine; players coordinate outside it |

---

## 14. Decisions and Open Questions

Decided (2026-10-29):

1. Ambiguities and contradictions in the rulebook are resolved the way
   18xx.games resolves them.
2. Private E may be exchanged before IR is started. Shares of an unstarted
   corporation count ¥0 toward net worth (Sections 10.5 and 11.3).
3. Private E's exchange respects the 60% holding limit.
4. Tile upgrade legality comes from the preservation rule. The manifest's
   gaps are printing errors (Section 12.3).

Still open:

1. **Consent model** for private sales and train sales between different
   players' corporations (Section 12.11; `1889.md` §4.4).
2. **Selling an unstarted corporation's share** (reachable only through
   private E): it has no market price. 18xx.games's sell check does not
   obviously exclude it. Check its behavior before implementing.
3. **Ability timing details** listed in Section 12.8.

---

## 15. Beginner Game (Out of Scope)

For reference only (R§14.1):

- **No auction.** Each player receives one random private from a reduced set
  and pays its face value:

| Players | Privates used |
|---------|---------------|
| 2 | E, F |
| 3 | C, E, F |
| 4 | C, D, E, F |
| 5 | B, C, D, E, F |
| 6 | A, B, C, D, E, F |

- The Priority Deal is random.
- **Extra tiles** join the supply: 2× #6, 1× #7, 1× #8, 1× #9, 1× #23,
  1× #24, and 1× #57.
- 18xx.games's beginner mode also removes private abilities and blockers and
  changes private values and revenues. Those details are not in this
  rulebook.

---

## 16. Glossary

- **Blocked city:** a city with every spot filled by other corporations'
  tokens. Routes and connectivity can reach it but not pass through.
- **Breaking the bank:** the bank running out of money, which triggers the
  game end.
- **Certificate:** a card for a private company, a 10% share, or a President's
  Certificate. Each counts as one toward the certificate limit.
- **Connectivity:** a traceable path from a corporation's station token
  (Section 7).
- **Exchange (Diesel):** trading in a 4-, 5-, or 6-train plus ¥800 for a
  D-train.
- **Float:** having only 50% of shares left in the Initial Offering. The
  corporation receives 10 × par and starts operating from the next OR.
- **Initial Offering (IO):** unsold shares of a corporation, sold at par.
- **Junction:** a hex edge where two track segments of one tile meet.
  Reversing there is not allowed.
- **Lay track:** placing a yellow tile on a hex with no track.
- **Open Market:** sold shares (bought at market price) and discarded or
  traded-in trains (bought at list price).
- **Par value:** the price chosen when starting a corporation. It sets the IO
  share price and the starting market space.
- **Phase:** game stage, set by the highest train rank bought from the Yard.
- **Preservation:** an upgrade keeps all existing track, revenue centers,
  tokens, and labels.
- **President:** the holder of the President's Certificate, who makes all of
  the corporation's decisions.
- **Priority Deal:** the player who acts first in the next Stock Round.
- **Revenue center:** a city, town, or off-board location.
- **Rust:** removal of every train of a rank when a phase begins (phases 4, 6,
  and D).
- **Sold out:** all 10 shares held by players. The token moves up at the end
  of the Stock Round.
- **Stock price order:** highest price first; ties go to the token farther
  right, then to the token higher in a stack.
- **Train limit:** the maximum number of trains a corporation may own, set by
  the phase.
- **Upgrade:** replacing a tile with one exactly one color higher, keeping
  everything on it.
- **Withhold:** putting all earnings into the treasury instead of paying
  dividends.

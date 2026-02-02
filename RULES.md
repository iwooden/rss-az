# Rolling Stock Stars - Complete Rules Specification

This document contains the complete rules for Rolling Stock Stars, extracted from the official rulebook for use in game engine implementation.

## Table of Contents
1. [Game Overview](#game-overview)
2. [Game Components](#game-components)
3. [Setup](#setup)
4. [Game Sequence](#game-sequence)
5. [Glossary](#glossary)
6. [Procedures](#procedures)
7. [Corporation Special Abilities](#corporation-special-abilities)
8. [Company Data](#company-data)
9. [Share Price Card Data](#share-price-card-data)
10. [Cost of Ownership](#cost-of-ownership)
11. [Clarifications and Edge Cases](#clarifications-and-edge-cases)
12. [Variants](#variants)

---

## Game Overview

Rolling Stock Stars is a card game for 2-6 players. Players take the role of investors who:
- Buy companies at auction
- Convert companies into corporations (IPO)
- Buy and sell shares of corporations
- As president of a corporation, manage its acquisitions and dividends

The player with the most wealth (cash + face value of private companies + share value of owned shares) at game end wins.

---

## Game Components

### Money
- Denominations: 1, 2, 5, 10, 20 (coins, denoted as ●)
- Bank is unlimited

### Companies (36 total)
Five colors with star ratings:
- **Red (1★)**: 6 companies, face values 1, 2, 5, 6, 7, 8
- **Orange (2★)**: 8 companies, face values 11, 12, 13, 14, 15, 16, 17, 19
- **Yellow (3★)**: 8 companies, face values 20, 21, 22, 23, 24, 25, 26, 29
- **Green (4★)**: 7 companies, face values 30, 31, 32, 33, 34, 36, 43
- **Blue (5★)**: 7 companies, face values 45, 46, 47, 50, 56, 58, 60

Each company card has:
- Face value (upper left)
- Price span for acquisition (in parentheses)
- Star rating (1-5 stars based on color)
- Income (upper right circle)
- Synergy indicators (circles and diamonds with company codes)
- Cost of ownership on back (0-4●)

### Corporations (8 total)
Each corporation has:
- Charter card with special ability
- 4-7 shares (varies by corporation)
- One President's Share per corporation

| Corporation | Abbrev | Shares |
|-------------|--------|--------|
| Junkyard Scrappers | JS | 7 |
| Synergistic | S | 7 |
| Overseas Trading | OS | 6 |
| Stock Masters | SM | 6 |
| Prussian Railway | PR | 5 |
| Doppler AG | DA | 5 |
| Vintage Machinery | VM | 4 |
| Stars, Inc. | SI | 4 |

### Share Price Cards (27 total)
Values: 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37, 41, 45, 50, 55, 61, 68, 75

Each card shows:
- Share price (center)
- IPO box (left) - colors eligible to IPO at this price
- Maximum dividend per share (right)
- Adjacent prices for +★, +★★, -★, -★★ adjustments (corners)
- Star requirements table by number of issued shares

### Other Components
- 6 player order cards
- 135 synergy markers
- Game end card (7● on one side, 10● on other)
- Foreign investor card

---

## Setup

1. Each player receives **30●** from the Bank (25● in 6-player game)
2. Each player receives a random **player order card** (determining initial player order)
3. Set aside 8 **corporation charter cards**, each with their sorted share stack:
   - President's Share on top
   - Then shares numbered 1, 2, 3... in ascending order
4. Lay out **share price cards** in a row, ascending order (0● to 75●)
5. Place **foreign investor card**; foreign investor receives **4●** from bank
6. Build the **company deck**:
   a. Place game end card at bottom (7● side up)
   b. Find highest face value company of each color (abbreviations printed in red/yellow)
   c. Start one face-down stack per color with those companies
   d. Shuffle remaining companies separately by color
   e. Add companies to each stack equal to player count:
      - 4 players: add 5 orange (not 4)
      - 5 players: add 7 orange (not 5)
      - 6 players: use ALL companies of all colors
   f. Remove remaining companies from game (without revealing)
   g. Shuffle each color stack
   h. Stack on game end card: blue, then green, then yellow, then orange, then red (red on top)
7. Draw and reveal companies equal to player count - these are **available for auction**
8. Begin first turn

---

## Game Sequence

The game is played in turns. Each turn has 9 phases:

### Phase 1: Investment

In **Player Order**, players choose ONE action:
- **Buy One Share**
- **Sell One Share**
- **Start An Auction**
- **Pass**

After passing, turn player order card vertical. After any non-pass action, turn it horizontal.

**Phase ends when all player order cards are vertical** (all players passed consecutively).

### Phase 2: Wrap-up

1. Determine new **Player Order** by descending remaining money (ties broken by old player order)
2. Redistribute player order cards
3. In **ascending Face Value order**, Foreign Investor buys as many available companies as possible at Face Value:
   - For each purchase, draw and reveal new company, mark as **unavailable** (turn vertical)
4. After Foreign Investor done, all unavailable companies become **available** (turn horizontal)

### Phase 3: Acquisition

In **Any Order**, corporations buy companies from:
- Other corporations
- Players
- Foreign Investor

**Transaction rules:**
- Each transaction involves exactly ONE company
- Price must be within allowed price span on company card
- Both parties must agree on price
- Transferred company AND money are turned **vertical** until end of phase (cannot be reused)
- A company turned vertical may be the only company a corporation owns

**Foreign Investor rules:**
- Only sells at **maximum allowed price**
- Intention to buy must be **announced** before executing
- Any corporation with **higher share price** and enough money may **intervene** and buy instead
- If multiple intervene, highest share price has priority
- If no intervention, announcing corporation **must** buy

**Corporations in Receivership:**
- Never sell companies
- Only buy from Foreign Investor
- Handled at **beginning** of phase:
  - Highest share price corporation in receivership tries to buy most expensive affordable company
  - Non-receivership corporations may intervene as usual
  - Repeat until cannot afford more
  - Then next highest receivership corporation, etc.

**Overseas Trading special:** Always considered highest share price; pays only Face Value to Foreign Investor.

### Phase 4: Closing

In **Any Order**, players and corporations may **close companies** (remove from game).

**Mandatory closing:**
- If a player will have negative total income in Phase 5 and cannot pay, they **must** close enough negative-income companies

**Automatic closing:**
- Foreign Investor closes companies whose Cost of Ownership exceeds Income
- Corporations in Receivership:
  - Close red companies if Cost of Ownership ≥ 4●
  - Close orange companies if Cost of Ownership ≥ 7●
  - Always keep company with highest Face Value

**Junkyard Scrappers special:** Receives 2× printed income as scrapping bonus when closing.

### Phase 5: Income

Players, corporations, and Foreign Investor **Collect Income** (see Procedures).

If a corporation cannot pay negative income, it **Goes Bankrupt**.

### Phase 6: Dividends

In **Share Price Order** (descending), each corporation:
1. **Pays Dividends** (see Procedures)
2. **Adjusts Share Price** (see Procedures)
3. Turns share price card **vertical**

Corporations in Receivership pay dividend of **0●**.

### Phase 7: End Card

Check for game end:
1. If **75● share price card** is owned by a corporation → **game ends**
2. If no unowned companies left → **flip game end card**
3. If game end card already flipped → **game ends**

### Phase 8: Issue Share

In **Share Price Order** (descending), each corporation may **Issue One Share** (see Procedures).

Then turn share price card **horizontal** (whether issued or not).

Corporations in Receivership **must** issue if they have shares left.

### Phase 9: IPO

In **descending Face Value order**, private companies may **Form Corporations** (see Procedures).

---

## Glossary

### Any Order
Players act (for themselves or as presidents) whenever and as often as they wish, even concurrently. Phase ends when nobody has acted for a reasonable time.

### Bank
Entity with unlimited money. Owns all issued shares not owned by players.

### Company
- Start face-down in deck
- When drawn: initially **unavailable**, become **available** as instructed
- **Unowned** until auctioned to player or sold to Foreign Investor → becomes **private company**
- When bought by corporation or used to form corporation → becomes **subsidiary company**
- Subsidiary companies can trade between corporations but never become private again

### Corporation
- Owns: 1+ companies (never zero), 0+ money, charter card, share price card (usually)
- Unissued shares stacked on charter card
- 4-7 shares per corporation (varies)
- Maximum 8 corporations simultaneously

### Cost of Ownership
- Defined by **back of top card** of deck (or game end card if deck exhausted)
- Deducted from income of each company matching colors in central rectangle
- Game end card: matches ANY displayed color

### Deck
- Contains unrevealed companies and game end card (at bottom)
- Game end card is never drawn (but flipped eventually)
- Whichever face is up defines Cost of Ownership

### Entity
Players, corporations, Foreign Investor, and Bank. Assets may only transfer between entities per rules.

### Face Value
Unique value printed on each company card.

### Face Value Order
Descending: highest first. Ascending: lowest first.

### Foreign Investor
Entity with rules-determined actions. Owns 0+ money and 0+ companies.

### Money
Measured in ● (coins). Only integers.

### Player
Owns 0+ money, 0+ companies, 0+ shares. May be president of any number of corporations.

### Player Order
Marked by player order cards 1, 2, 3... **Cyclic**: after highest, returns to 1.

### President
Player owning corporation's President's Share. Acts on behalf of corporation. If Bank owns President's Share → corporation in **Receivership**.

### Receivership
All issued shares owned by Bank. Corporation's actions determined by rules until a player buys a share (must be President's Share), becoming new president.

### Row
The row of share price cards (ascending, left to right). When card taken, leave gap. Return cards to original spot.

### Share
- **Issued shares**: owned by players or Bank
- **Unissued shares**: face-down on charter card
- Numbered on back (only significant for unissued)
- Always take top share when issuing
- One **President's Share** per corporation

### Share Price
Current value of corporation's shares. Marked by share price card. Corporation without share price card has price of **75●**.

### Share Price Order
Descending (highest share price first).

### Synergy
Bonus income for corporations only. For each pair of companies with each other's code in synergy indicators, add marked ● amount. Count each pair **once only**. Place synergy marker on circle (not diamond).

---

## Procedures

**Procedures are atomic.** If any part cannot execute, the whole procedure cannot execute.

### Adjust Share Price

1. On share price card, find **required stars** for number of issued shares
2. Calculate **owned stars**:
   - Sum stars on all owned companies
   - Add 1 star per 10● owned
   - Stars, Inc.: add 2 additional stars
3. Compare:
   - Equal → no change
   - Owned ≥ 2 lower → target is **-★★** price
   - Owned 1 lower → target is **-★** price
   - Owned 1 higher → target is **+★** price
   - Owned ≥ 2 higher → target is **+★★** price
4. Take target share price card (skip if in use, continue in same direction)
5. Return old card to row
6. If taking 0● card → **Go Bankrupt**
7. If no higher card available when rising → take no card (share price = 75●)

### Auction

1. Starting player chooses available company, bids ≥ Face Value
2. In **Player Order** (from starter), players either:
   - Raise bid (must have enough money)
   - Leave auction (skipped for remainder)
3. When one player remains: pay bid to Bank, receive company
4. If deck has cards: draw and reveal top card, turn **vertical** (unavailable this phase)
5. Next action goes to player after auction **starter** (not winner)

### Buy One Share

1. Take one share from Bank
2. If corporation in Receivership: **must** take President's Share
3. Corporation returns share price card, takes **next higher available**
4. Player pays **new** share price to Bank
5. Check for **Change of Presidency**
6. If new price = 75● → **game ends immediately** after payment

### Change of Presidency

If any player owns more shares of a corporation than current president:
- Next player in Player Order (after current president) with more shares becomes president

*Note: In the physical game, players exchange share cards to transfer the "President's Share" marker. This is purely a tracking mechanism—share counts do not change. In our digital model, shares are fungible and presidency is tracked separately.*

### Collect Income

1. Sum income of all owned companies
2. Subtract Cost of Ownership for each applicable company
3. Foreign Investor: add **5●**
4. Corporations add **Synergy** income
5. Apply corporation special abilities:
   - **Prussian Railway**: +1● per company owned
   - **Doppler AG**: double printed income of highest Face Value company
   - **Synergistic**: +1● per 2 synergy markers (rounded down)
   - **Vintage Machinery**: reduce total Cost of Ownership by up to 10● (minimum 0●)
6. If positive: receive from Bank. If negative: pay to Bank
7. If corporation cannot pay → **Go Bankrupt**

### Form Corporation

1. Separate company from player's assets
2. Select unused charter card, place above company
3. Place sorted share stack on charter card
4. Select available share price card with company's color in IPO box, place left of charter
5. Player receives **President's Share** (turn face up)
6. Bank receives **2nd share** (turn face up)
7. If Face Value > Share Price:
   - Player receives **one additional share**
   - Bank receives **one additional share**
8. Player pays to corporation: (total share price of player's shares) - Face Value
9. Bank pays to corporation: total share price of Bank's shares

### Go Bankrupt

1. Remove all corporation's companies from game
2. Collect **all** shares (from all owners), place face-down on charter card (sorted as setup)
3. Charter + shares available for future IPO
4. Return corporation's money to Bank
5. Return share price card to row

### Issue One Share

Same as **Sell One Share** except:
- Acting entity is corporation (not player)
- Share comes from **unissued stack** (turn face up)
- **Stock Masters special**: share price does NOT change; receives current share price

### Pay Dividends

1. Choose dividend per share: integer, ≥ 0●, ≤ maximum on share price card
2. Corporation must have enough money: (dividend × issued shares)
3. Pay dividend to owner of each issued share (players or Bank)

### Sell One Share

1. Player gives one share to Bank
2. Corporation returns share price card, takes **next lower available**
3. Bank pays **new** share price to player
4. If new price = 0● → **Go Bankrupt**
5. If sold share was last player-owned share → corporation enters **Receivership**
6. Otherwise: check **Change of Presidency**

*Note: Players may sell shares freely. If selling causes the player to no longer have the most shares, presidency transfers per "Change of Presidency" rules. The physical game's "President's Share" card is just a marker—in our model, shares are fungible.*

---

## Corporation Special Abilities

| Corporation | Abbrev | Shares | Special Ability |
|-------------|--------|--------|-----------------|
| **Junkyard Scrappers** | JS | 7 | When closing a company, immediately receives 2× printed income as scrapping bonus. |
| **Synergistic** | S | 7 | Receives +1● income for every 2 synergy markers it owns (rounded down). |
| **Overseas Trading** | OS | 6 | Always has first priority buying from Foreign Investor (as if highest share price). Pays only Face Value (not max price) to Foreign Investor. |
| **Stock Masters** | SM | 6 | When issuing a share, share price does not change. Receives current share price. |
| **Prussian Railway** | PR | 5 | Receives +1● income for each company it owns. |
| **Doppler AG** | DA | 5 | Doubles the printed income of its company with highest Face Value. |
| **Vintage Machinery** | VM | 4 | Reduces its total Cost of Ownership by up to 10● (but not below 0●). |
| **Stars, Inc.** | SI | 4 | Adds 2 additional stars to its star count when adjusting share price. |

---

## Company Data

### Red Companies (1★)

| Abbrev | Face Value | Price Span | Income | Synergies (this company receives bonus from) |
|--------|------------|------------|--------|----------------------------------------------|
| BME | 1 | (1-2) | 1 | — |
| BSE | 2 | (1-3) | 1 | — |
| KME | 5 | (3-7) | 2 | BME +1 |
| AKE | 6 | (3-8) | 2 | — |
| BPM | 7 | (4-9) | 2 | BSE +1, AKE +1 |
| MHE | 8 | (4-10) | 2 | KME +1, AKE +1, BPM +1 |

**Last in group:** MHE (triggers CoO increase when revealed)

### Orange Companies (2★)

| Abbrev | Face Value | Price Span | Income | Synergies (this company receives bonus from) |
|--------|------------|------------|--------|----------------------------------------------|
| WT | 11 | (6-14) | 3 | — |
| BY | 12 | (6-16) | 3 | WT +2 |
| BD | 13 | (7-17) | 3 | BME +1, WT +2 |
| HE | 14 | (7-18) | 3 | BME +1, KME +1, BY +2, BD +2 |
| OL | 15 | (8-20) | 3 | KME +1, AKE +1, MHE +1 |
| SX | 16 | (8-21) | 3 | BSE +1, BPM +1, MHE +1, BY +2 |
| MS | 17 | (9-22) | 3 | BSE +1, AKE +1, BPM +1, MHE +1, OL +2, SX +2 |
| PR | 19 | (10-25) | 3 | BME +1, BSE +1, KME +1, AKE +1, BPM +1, MHE +1, HE +2, OL +2, SX +2, MS +2 |

**Last in group:** PR (triggers CoO increase when revealed)

### Yellow Companies (3★)

| Abbrev | Face Value | Price Span | Income | Synergies (this company receives bonus from) |
|--------|------------|------------|--------|----------------------------------------------|
| DSB | 20 | (10-26) | 5 | OL +2, MS +2, PR +2 |
| KK | 21 | (11-28) | 5 | BY +2, SX +2 |
| NS | 22 | (11-29) | 5 | OL +2, PR +2 |
| SBB | 23 | (12-30) | 5 | WT +2, BD +2, KK +4 |
| B | 24 | (12-32) | 5 | PR +2, NS +4 |
| PKP | 25 | (13-33) | 5 | SX +2, MS +2, PR +2, KK +4 |
| SNCF | 26 | (13-34) | 5 | BD +2, SBB +4, B +4 |
| DR | 29 | (15-38) | 5 | WT +2, BY +2, BD +2, HE +2, OL +2, SX +2, MS +2, PR +2, DSB +4, KK +4, NS +4, SBB +4, B +4, PKP +4, SNCF +4 |

**Last in group:** DR (triggers CoO increase when revealed)

### Green Companies (4★)

| Abbrev | Face Value | Price Span | Income | Synergies (this company receives bonus from) |
|--------|------------|------------|--------|----------------------------------------------|
| SZD | 30 | (15-40) | 7 | PKP +4 |
| SJ | 31 | (16-41) | 7 | — |
| FS | 32 | (16-42) | 7 | KK +4, SBB +4, SNCF +4 |
| RENFE | 33 | (17-44) | 7 | SNCF +4 |
| BR | 34 | (17-45) | 7 | — |
| BSR | 36 | (18-48) | 7 | DSB +4, PKP +4, DR +4, SJ +8 |
| E | 43 | (22-57) | 7 | NS +4, B +4, SNCF +4, BR +8 |

**Last in group:** E (triggers CoO increase when revealed)

### Blue Companies (5★)

| Abbrev | Face Value | Price Span | Income | Synergies (this company receives bonus from) |
|--------|------------|------------|--------|----------------------------------------------|
| HH | 45 | (23-60) | 10 | DSB +4, PKP +4, DR +4, BSR +8 |
| HA | 46 | (23-61) | 10 | NS +4, B +4, SNCF +4, E +8 |
| HR | 47 | (24-62) | 10 | NS +4, B +4, DR +4, E +8 |
| MAD | 50 | (25-66) | 10 | RENFE +8 |
| FRA | 56 | (28-74) | 10 | KK +4, SBB +4, PKP +4, DR +4 |
| LHR | 58 | (29-77) | 10 | BR +8, E +8, FRA +16 |
| CDG | 60 | (30-80) | 10 | SBB +4, SNCF +4, E +8, MAD +16, FRA +16, LHR +16 |

**Last in group:** CDG (triggers CoO increase when revealed)

### Companies by Face Value (Reference Chart)

```
Red (1★):    BME(1), BSE(2), KME(5), AKE(6), BPM(7), MHE(8)
Orange (2★): WT(11), BY(12), BD(13), HE(14), OL(15), SX(16), MS(17), PR(19)
Yellow (3★): DSB(20), KK(21), NS(22), SBB(23), B(24), PKP(25), SNCF(26), DR(29)
Green (4★):  SZD(30), SJ(31), FS(32), RENFE(33), BR(34), BSR(36), E(43)
Blue (5★):   HH(45), HA(46), HR(47), MAD(50), FRA(56), LHR(58), CDG(60)
```

**Highest face value per color** (set aside during setup, marked with red/yellow abbreviation):
- Red: MHE (8)
- Orange: PR (19)
- Yellow: DR (29)
- Green: E (43)
- Blue: CDG (60)

---

## Share Price Card Data

### Share Prices (27 total)
0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37, 41, 45, 50, 55, 61, 68, 75

### IPO Eligibility (Par Prices by Company Color)
| Star Tier | Color | Valid Par Prices |
|-----------|-------|------------------|
| 1★ | Red | 10, 11, 12, 13, 14 |
| 2★ | Orange | 10, 11, 12, 13, 14, 16, 18, 20 |
| 3★ | Yellow | 16, 18, 20, 22, 24, 27 |
| 4★ | Green | 22, 24, 27, 30, 33, 37 |
| 5★ | Blue | 30, 33, 37 |

### All Unique Par Prices
10, 11, 12, 13, 14, 16, 18, 20, 22, 24, 27, 30, 33, 37

### Target Stars (for Share Price Adjustment)
Formula: `round(issued_shares × price / 10)`

Used in Phase 6 (Dividends) to determine share price movement. A corporation's actual star count is compared against target stars to determine if price moves up, down, or stays.

Implementation: `core/data.pyx::get_required_stars(price_index, issued_shares)`

### Maximum Dividend Per Share
Formula: `price // 3` (integer division)

The maximum dividend a corporation can pay per share, based on its current share price.

Implementation: `core/data.pyx::get_max_dividend(price_index)`

---

## Cost of Ownership

Cost of Ownership is determined by the back of the top card of the company deck (or the game end card if deck is exhausted).

### Cost of Ownership Table

| CoO Level | Trigger | Red (1★) | Orange (2★) | Yellow (3★) | Green (4★) | Blue (5★) |
|-----------|---------|----------|-------------|-------------|------------|-----------|
| 1-3 | — | 0 | 0 | 0 | 0 | 0 |
| 4 | Green cards on top of deck | 2 | 0 | 0 | 0 | 0 |
| 5 | Blue cards on top of deck | 4 | 4 | 0 | 0 | 0 |
| 6 | Game end card (7● side) | 7 | 7 | 7 | 0 | 0 |
| 7 | Game end card flipped (10● side) | 10 | 10 | 10 | 10 | 0 |

**Notes:**
- CoO level increases as higher-tier cards reach the top of the deck
- When the back of the top deck card shows a cost rectangle with certain colors, those colored companies suffer the indicated cost
- Game end card (7● side) affects red, orange, and yellow companies
- Game end card flipped (10● side) affects all companies except blue

### Adjusted Income Formula
```
Adjusted Income = Base Income - Cost of Ownership
```
A company's effective income can become negative if Cost of Ownership exceeds its base income.

---

## Clarifications and Edge Cases

### Synergies
- Only possible **within a corporation** (never for players or Foreign Investor)
- Count each pair **once only** (if A synergizes with B, B synergizes with A - still one bonus)
- Place synergy marker on the circle indicator only (not the diamond)

### Pass vs. Leave Auction
- **Pass**: An action during Phase 1. Does not prevent future actions. Phase ends when ALL pass consecutively.
- **Leave Auction**: During an auction, permanently exit that auction. Not an action itself.

### After Auction
- Next action goes to player after **starter** (not winner) in player order

### Asset Transfers
- Never transfer assets except as explicitly allowed by rules
- Cannot sponsor corporations, steal from treasury, gift to other players

### Share Price When Buying/Selling
- Displayed price is "last known price"
- When buying: pay **next higher** available price
- When selling: receive **next lower** available price
- Skip cards in use by other corporations

### Newly Drawn Companies
- Not available for auction in same turn (wait until next turn)
- Foreign Investor cannot buy them in Phase 2 of same turn

### Phase 3 Transactions
- Never use same ● or company twice in phase
- Turn both money and company **vertical** after transaction
- Execute each transaction separately (no "swaps")

### Cost of Ownership
- Defined solely by **back of top deck card** (or game end card)
- Once a card is drawn, its back is irrelevant

### Receivership Automatic Actions
- **Phase 3**: Corporation tries to buy most expensive company it can afford from Foreign Investor
- **Phase 4**: Close red companies if CoO ≥ 4●, orange if CoO ≥ 7●; always keep highest face value company
- **Phase 6**: Dividend is always 0●
- **Phase 8**: Must issue a share if possible (even if it causes bankruptcy)

---

## Variants

### Open Companies Variant
Build deck as usual, then declare open for inspection. All cards may be turned face-up. Use unused green/blue company cards face-down to mark current cost of ownership.

### Two-Player, Two-Handed Variant
Set up as 4-player game. Each player controls two simulated players:
- Player A: positions 1 and 4
- Player B: positions 2 and 3

To win: your **lower-ranked** simulated player must beat opponent's lower-ranked player.

---

## Game End Conditions

Game ends when ANY of:
1. A corporation takes 75● share price card during **Buy One Share** → ends immediately after payment
2. 75● share price card in use during **Phase 7**
3. **Phase 7** starts with game end card already flipped

### Final Scoring
Sum for each player:
- Cash
- Face Value of each private company
- Share Price of each owned share

**Ties**: broken by Player Order (lower number wins)

*Note: Corporation cash and companies do NOT count toward player scores.*

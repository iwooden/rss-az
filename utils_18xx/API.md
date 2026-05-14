# 18xx.games API Notes for Rolling Stock Stars

These notes are based on the local `submodules/18xx` checkout. They are meant
as a quick reference for maintaining `utils_18xx/api_client.py`,
`utils_18xx/live.py`, and `utils_18xx/action_mapper.py`.

## Server Entry Points

- `submodules/18xx/api.rb` defines the Roda `Api` app. Authentication comes
  from the `auth_token` cookie (`api.rb:260`). Login sets the same cookie in
  `routes/user.rb:124`.
- `submodules/18xx/assets/app/lib/connection.rb:32` prefixes browser API calls
  with `/api` and posts JSON with `Content-Type: application/json`.
- `GET /api/game/:id` is handled in `submodules/18xx/routes/game.rb:10`. It
  returns `game.to_h(include_actions: true, logged_in_user_id: user&.id)`.
- `POST /api/game/:id/action` is handled in `submodules/18xx/routes/game.rb:58`.
  Production/non-pin games load the Ruby engine from stored actions, set
  `r.params['user'] = user.id`, call `engine.process_action(r.params,
  validate_auto_actions: true)`, store `engine.raw_actions.last`, update game
  state, then publishes the accepted action to `/game/:id`.
- The action route rejects users who are neither players nor the game owner
  (`routes/game.rb:56`) and rejects archived games (`routes/game.rb:60`).
- Pin games take a different path (`routes/game.rb:65`): the client must submit
  the next action `id` and a `meta` object describing active players, round,
  turn, game status/result, etc. This is not the production/live path.

## Game JSON Shape

`submodules/18xx/models/game.rb:148` serializes games. Important fields:

- `id`: numeric for server games; hotseat fixtures can use strings.
- `title`: for RSS this is `"Rolling Stock Stars"`.
- `players`: ordered list of `{id, name}`. Trust this order for player index
  mapping; `Game#ordered_players` applies the saved `player_order` or a seed
  shuffle (`models/game.rb:125`).
- `acting`: user IDs whose turn it is.
- `round`, `turn`, `status`, `result`, `settings`, `min_players`,
  `max_players`.
- `actions`: full action history when `include_actions` is true.

Webhook turn notifications are produced from the `/turn` MessageBus channel in
`submodules/18xx/queue.rb:99`. The message text is roughly:

```text
Your Turn in Rolling Stock Stars "<description>" (<round> <turn>)
<base_url>/game/<id>
```

`submodules/18xx/lib/hooks.rb` sends custom webhooks as JSON. For Slack/Google
style destinations the payload is `{ "text": "<@webhook_user_id> ..." }`; for
Discord it is `{ "content": "...", "allowed_mentions": ... }`.

## Action Hash Conventions

All game moves are serialized through `Engine::Action::Base#to_h`
(`submodules/18xx/lib/engine/action/base.rb:69`):

```json
{
  "type": "bid",
  "entity": 1234,
  "entity_type": "player",
  "id": 42,
  "user": 1234,
  "created_at": 1700000000
}
```

Action-specific fields are added by each action class. On inbound hashes,
`Engine::Action::Base.action_from_h` maps `"type": "buy_shares"` to
`Engine::Action::BuyShares` using the type helper (`action/base.rb:32`).
`entity_type` and `entity` are resolved with `game.get(entity_type, entity)`
(`action/base.rb:19`).

When posting a new action, the client normally does not need to send `id`,
`created_at`, or `user`; the route/engine fill them. The DB model strips
`id`, `user`, `created_at`, `meta`, and `_client_id` before storing the raw
action JSON (`submodules/18xx/models/action.rb:11`), then re-adds server-side
`id`, `user`, and `created_at` in `Action#to_h`.
`auto_actions` is omitted when empty and present only when the engine generated
automatic follow-up actions.

### Auto Actions Matter

The browser runs the action locally with `add_auto_actions: true` before
posting (`assets/app/view/game/actionable.rb:104`), then posts `action.to_h`
(`actionable.rb:128`). The server recomputes auto actions and requires them to
match exactly except for `created_at` (`lib/engine/game/base.rb:857` and
`base.rb:948`).

Implication for the Python client: a production post that causes Ruby
`round.auto_actions` must include the same `auto_actions` array the browser
would submit. Posting the main action only will fail with `"Auto actions do not
match"`. Fixtures show many RSS passes with generated auto passes. Current
`utils_18xx/live.py` posts only the main action, so this is a likely future fix
area for live play.

## Rolling Stock Stars Engine Files

RSS is a variant class, not a standalone rules engine:

- `submodules/18xx/lib/engine/game/g_rolling_stock_stars/game.rb:3` inherits
  `GRollingStock::Game`.
- RSS entities live in
  `submodules/18xx/lib/engine/game/g_rolling_stock_stars/entities.rb`.
- Shared Rolling Stock/RSS phase and step logic is under
  `submodules/18xx/lib/engine/game/g_rolling_stock/`.
- Acquisition uses `Step::ReceiverProposeAndPurchase` and
  `Step::ProposeAndPurchase` (`g_rolling_stock/game.rb:330`).
- Closing uses `Step::CloseCompanies` (`g_rolling_stock/game.rb:351`).
- Acquisition and closing rounds are unordered; `GRollingStock::Game#pass_entity`
  maps the logged-in user to the player entity in unordered rounds
  (`g_rolling_stock/game.rb:884`).

## RSS Action Formats We Need

Use Ruby engine IDs:

- Player entities use numeric user IDs from `game_data["players"]`.
- Corporations use string IDs such as `"PR"`, `"OS"`, `"SI"`, `"DA"`,
  `"VM"`, `"SM"`, `"JS"`, `"S"`.
- Companies use string IDs such as `"MHE"`, `"AKE"`, `"BPM"`.
- Shares are serialized as strings like `"SI_1"` or `"VM_0"`. Do not assume
  index `0` is unpostable; RSS fixtures include sales of `_0` shares.

### Pass

Generic format:

```json
{"type": "pass", "entity": 1234, "entity_type": "player"}
```

The correct entity depends on the current round:

- Investment, bidding, acquisition, closing: player entity.
- IPO: active company entity, e.g. `{"entity": "AKE", "entity_type": "company"}`.
- Dividends and issue shares: active corporation entity.

### Bid / Auction

`Engine::Action::Bid` fields are in
`submodules/18xx/lib/engine/action/bid.rb`.

```json
{
  "type": "bid",
  "entity": 1234,
  "entity_type": "player",
  "company": "AKE",
  "price": 8
}
```

RSS stock/investment bidding is implemented in
`g_rolling_stock/step/buy_sell_shares_bid_companies.rb`. Legal stock actions
are `buy_shares`, `sell_shares`, `bid`, and `pass`; during an auction only
`bid` and `pass` are available (`buy_sell_shares_bid_companies.rb:13`).

### Buy Shares

`Engine::Action::BuyShares` fields are in
`submodules/18xx/lib/engine/action/buy_shares.rb`.

```json
{
  "type": "buy_shares",
  "entity": 1234,
  "entity_type": "player",
  "shares": ["SI_1"],
  "percent": 10,
  "share_price": 13
}
```

`share_price` here is a numeric transaction/share price, not the market-cell
ID string used by `par`. For RSS buy/sell/issue actions this is the price
after the share-price movement: next higher for buys, next lower for player
sells and normal issues, and current price for Stock Masters issues. The
browser may omit nullable fields such as
`share_price`, `swap`, `purchase_for`, and `borrow_from` when not needed.

### Sell Shares / Issue Shares

`Engine::Action::SellShares` fields are in
`submodules/18xx/lib/engine/action/sell_shares.rb`.

Player selling:

```json
{
  "type": "sell_shares",
  "entity": 1234,
  "entity_type": "player",
  "shares": ["SI_1"],
  "percent": 10,
  "share_price": 14
}
```

Corporation issuing shares is also `sell_shares`, but with a corporation
entity:

```json
{
  "type": "sell_shares",
  "entity": "SI",
  "entity_type": "corporation",
  "shares": ["SI_2"],
  "percent": 10,
  "share_price": 14
}
```

RSS issue shares is handled by
`g_rolling_stock/step/issue_shares.rb`; `process_sell_shares` sells one bundle
and then passes the step (`issue_shares.rb:37`).

### IPO / Par

RSS IPO is `par` on the active company (`g_rolling_stock/step/ipo_company.rb:10`).

```json
{
  "type": "par",
  "entity": "BPM",
  "entity_type": "company",
  "corporation": "VM",
  "share_price": "10,0,6"
}
```

`Engine::Action::Par` requires `corporation` and `share_price`
(`submodules/18xx/lib/engine/action/par.rb:10`). `share_price` is the
stock-market cell ID string returned by `SharePrice#id`, observed as
`"<price>,0,<column>"` in RSS fixtures. IPO can also be passed with the active
company as the entity.

### Dividends

RSS dividends are variable-only (`g_rolling_stock/step/dividend.rb:21`).

```json
{
  "type": "dividend",
  "entity": "SI",
  "entity_type": "corporation",
  "kind": "variable",
  "amount": 4
}
```

`Engine::Action::Dividend` requires `kind`; `amount` is optional at the action
class level but RSS uses it for the variable dividend amount.

### Acquisition Offers

Acquisition is an unordered round with two steps:

- `ReceiverProposeAndPurchase` can generate forced receiver offers and only
  exposes `respond` to active responders (`receiver_propose_and_purchase.rb:10`).
- `ProposeAndPurchase` exposes `offer`, `respond`, and `pass`
  (`propose_and_purchase.rb:10`).

Offer format:

```json
{
  "type": "offer",
  "entity": 1234,
  "entity_type": "player",
  "corporation": "SI",
  "company": "KME",
  "price": 7
}
```

`Engine::Action::Offer` stores `corporation`, `company`, and `price`
(`submodules/18xx/lib/engine/action/offer.rb:7`). The entity is the proposing
player, not the corporation.

For Foreign Investor companies, the posted `price` must be the fixed
`foreign_price`: face value for corps with the `:overseas` ability, otherwise
the company max price (`propose_and_purchase.rb:126` and
`propose_and_purchase.rb:141`).

Respond format:

```json
{
  "type": "respond",
  "entity": 1234,
  "entity_type": "player",
  "corporation": "PR",
  "company": "AKE",
  "accept": "true"
}
```

`Engine::Action::Respond` parses `accept` with `h["accept"] == "true"`
(`submodules/18xx/lib/engine/action/respond.rb:19`) and serializes it back as
the strings `"true"` or `"false"`, not JSON booleans. The `corporation` field
identifies the outstanding offer via `(corporation, company)`. In FI
right-of-refusal responses, that may be the original purchasing corporation,
while the actual intervening corporation is derived from the round's responder
list.

### Closing Companies

Closing uses action type `sell_company`, not `close`.

```json
{
  "type": "sell_company",
  "entity": 1234,
  "entity_type": "player",
  "company": "SX",
  "price": 0
}
```

`Engine::Action::SellCompany` requires both `company` and `price`
(`submodules/18xx/lib/engine/action/sell_company.rb:10`). RSS closing validates
that the player can close the selected company and then calls
`close_company` (`g_rolling_stock/step/close_companies.rb:63`). Fixtures use
`price: 0`.

## Fixture Examples

Useful RSS fixtures:

- `submodules/18xx/public/fixtures/RollingStockStars/dividend.json`
- `submodules/18xx/public/fixtures/RollingStockStars/83092.json`

These contain concrete examples of all main live-client action types:
`bid`, `pass`, `par`, `buy_shares`, `sell_shares`, `offer`, `respond`,
`dividend`, and `sell_company`, including `auto_actions`.

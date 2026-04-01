#!/usr/bin/env ruby
# frozen_string_literal: true

# extract_states.rb
#
# Replays an 18xx.games JSON game file (Rolling Stock Stars) and extracts a
# state snapshot after every action.  The snapshot array is written to stdout
# as valid JSON.
#
# Usage:
#   ruby extract_states.rb path/to/game.json > states.json
#
# The first element of the output array is the initial-state record (before any
# action is applied, action_id: 0).  Every subsequent element corresponds to
# the game state immediately after that action has been fully processed,
# including any automated follow-up actions that the engine performs without
# player input.
#
# The 18xx engine gem lives in the submodule at:
#   submodules/18xx/lib/
# and is loaded via require_relative so no gem installation is needed.

# ---------------------------------------------------------------------------
# 0.  Bootstrap
# ---------------------------------------------------------------------------

# Silence the DEBUG-level engine logger before anything is loaded.  The logger
# is written to $stdout by default, which would corrupt our JSON output.  We
# redirect it to /dev/null and then re-open $stdout for our own use.
require 'logger'

# The engine creates LOGGER = Logger.new($stdout) at require time, so we need
# to suppress it before requiring the engine.  We do this by temporarily
# redirecting $stdout to /dev/null while loading, then restoring it.
real_stdout = $stdout.dup
$stdout.reopen('/dev/null', 'w')

# Resolve the path to the 18xx engine lib directory relative to this script,
# regardless of the working directory when the script is invoked.
SCRIPT_DIR = File.dirname(File.expand_path(__FILE__))
REPO_ROOT = File.expand_path('..', SCRIPT_DIR)
$LOAD_PATH.unshift(File.join(REPO_ROOT, 'submodules', '18xx', 'lib'))

require_relative '../submodules/18xx/lib/engine'

# Restore real stdout for our JSON output.
$stdout.reopen(real_stdout)
real_stdout.close

# Now set the engine LOGGER to FATAL so any lazy-evaluated debug blocks are
# also suppressed at runtime (some debug calls happen lazily).
LOGGER.level = ::Logger::FATAL

require 'json'

# ---------------------------------------------------------------------------
# 1.  Helpers
# ---------------------------------------------------------------------------

# Build the players array for a snapshot from the current game state.
#
# Each player entry contains:
#   name        - player name string
#   id          - player id (integer, as stored in the JSON)
#   cash        - cash on hand
#   value       - net worth (cash + shares + companies - debts)
#   companies   - array of company symbols owned directly
#   shares      - hash of { corp_sym => share_count } for all held shares
def snapshot_players(game)
  game.players.map do |player|
    shares_by_corp = {}
    player.shares_by_corporation.each do |corp, share_list|
      next if share_list.empty?

      shares_by_corp[corp.name] = share_list.size
    end

    {
      name:      player.name,
      id:        player.id,
      cash:      player.cash,
      value:     game.player_value(player),
      companies: player.companies.map(&:sym),
      shares:    shares_by_corp,
    }
  end
end

# Build the corporations array for a snapshot.
#
# Each corp entry contains:
#   name             - corporation symbol
#   price            - current share price (integer) or nil if not IPO'd/closed
#   cash             - treasury cash
#   floated          - boolean
#   companies        - array of company symbols owned by this corp
#   shares_in_market - number of shares sitting in the share pool (market)
#   president        - name of the president player (or nil if in receivership)
def snapshot_corporations(game)
  game.corporations.map do |corp|
    market_shares = game.share_pool.shares_by_corporation[corp]&.size || 0

    {
      name:             corp.name,
      price:            corp.share_price&.price,
      cash:             corp.cash,
      floated:          corp.floated?,
      companies:        corp.companies.map(&:sym),
      shares_in_market: market_shares,
      president:        corp.owner&.name,
    }
  end
end

# Build the foreign_investor hash for a snapshot.
#
# Contains:
#   cash      - cash held by FI
#   companies - array of company symbols owned by FI
def snapshot_foreign_investor(game)
  fi = game.foreign_investor
  {
    cash:      fi.cash,
    companies: fi.companies.map(&:sym),
  }
end

# Assemble a full state snapshot object.  The action_id and action_type fields
# are set by the caller since this helper only reads the current game state.
def build_snapshot(game, action_id:, action_type:, round_override: nil)
  entity = game.round.current_entity
  active_player_id = nil
  active_corp_name = nil

  if entity&.player?
    active_player_id = entity.id
  elsif entity&.corporation?
    active_corp_name = entity.name
  elsif entity&.company?
    # IPO round: entity is a Company, the acting player is the owner
    active_player_id = entity.owner.id if entity.owner&.player?
  end

  {
    action_id:        action_id,
    action_type:      action_type,
    round:            round_override || game.round.class.short_name,
    turn:             game.turn,
    active_player:    active_player_id,
    active_corp:      active_corp_name,
    players:          snapshot_players(game),
    corporations:     snapshot_corporations(game),
    foreign_investor: snapshot_foreign_investor(game),
    offering:         game.offering.map(&:sym),
    deck_size:        game.company_deck.size,
    # Normalize cost_level to our engine's contiguous numbering:
    # Ruby 7 (END_CARD_FRONT) → 6, Ruby 8 (END_CARD_BACK) → 7
    cost_level:       case game.cost_level
                      when 7 then 6
                      when 8 then 7
                      else game.cost_level
                      end,
  }
end

# Post-process snapshots to annotate each ACQ round's first snapshot with
# the actual acquisition outcomes (which companies transferred and at what
# price), including seller info and a cross_president flag.  This lets the
# Python replay harness skip the expensive state-diffing and seller-detection
# loops it previously had to do at replay time.
def annotate_acq_outcomes(snapshots, raw_actions, committed_ids)
  # Group ACQ-round snapshots by game turn
  acq_by_turn = {}
  snapshots.each do |s|
    next unless s[:round] == 'ACQ'
    (acq_by_turn[s[:turn]] ||= []) << s
  end
  return if acq_by_turn.empty?

  # Index committed offer actions by ID for fast lookup per ACQ round.
  offer_actions_by_id = {}
  raw_actions.each do |a|
    next unless a['type'] == 'offer' && committed_ids.include?(a['id'])
    offer_actions_by_id[a['id']] = a
  end

  acq_by_turn.each do |_turn, acq_snaps|
    first_acq = acq_snaps.first
    last_acq  = acq_snaps.last

    # Find last non-ACQ snapshot before first ACQ snapshot
    first_idx = snapshots.index(first_acq)
    before = nil
    (first_idx - 1).downto(0) do |i|
      unless snapshots[i][:round] == 'ACQ'
        before = snapshots[i]
        break
      end
    end
    next unless before

    # Scope offer prices to THIS ACQ round's action ID range
    acq_min_id = first_acq[:action_id]
    acq_max_id = last_acq[:action_id]
    offer_prices = {}
    offer_actions_by_id.each do |aid, a|
      next unless aid >= acq_min_id && aid <= acq_max_id
      offer_prices[[a['company'], a['corporation']]] = a['price'].to_i
    end

    # Build company -> corp_name maps (before and after)
    before_corps = {}
    before[:corporations].each do |c|
      c[:companies].each { |comp| before_corps[comp.to_s] = c[:name] }
    end

    after_corps = {}
    last_acq[:corporations].each do |c|
      c[:companies].each { |comp| after_corps[comp.to_s] = c[:name] }
    end

    # Corp president map: corp_name -> president player name
    presidents = {}
    before[:corporations].each { |c| presidents[c[:name]] = c[:president] }

    # Player company map: company -> {name:, id:}
    player_companies = {}
    before[:players].each do |p|
      p[:companies].each { |comp| player_companies[comp.to_s] = { name: p[:name], id: p[:id] } }
    end

    # FI company set — include companies from BOTH the before (INV) snapshot
    # and the first ACQ snapshot, because WRAP_UP (Phase 2) runs between INV
    # and ACQ: FI buys from the offering during WRAP_UP, so those companies
    # only appear in FI at the first ACQ snapshot, not the INV snapshot.
    fi_companies = Set.new
    (before[:foreign_investor][:companies] || []).each { |comp| fi_companies << comp.to_s }
    (first_acq[:foreign_investor][:companies] || []).each { |comp| fi_companies << comp.to_s }

    # Diff: companies that moved to a (different) corp
    outcomes = []
    after_corps.each do |company, new_owner|
      old_corp = before_corps[company]
      next if old_corp == new_owner

      price = offer_prices[[company, new_owner]] || 0

      if old_corp
        # Corp-to-corp transfer
        outcomes << {
          company: company, buyer: new_owner, price: price,
          seller_type: 'corp', seller: old_corp,
          cross_president: presidents[old_corp] != presidents[new_owner],
        }
      elsif player_companies[company]
        # Player-to-corp transfer
        pi = player_companies[company]
        outcomes << {
          company: company, buyer: new_owner, price: price,
          seller_type: 'player', seller: pi[:name], seller_id: pi[:id],
          cross_president: pi[:name] != presidents[new_owner],
        }
      elsif fi_companies.include?(company)
        # FI-to-corp transfer (engine handles these — never cross-president)
        outcomes << {
          company: company, buyer: new_owner, price: price,
          seller_type: 'fi', cross_president: false,
        }
      end
    end

    first_acq[:acq_outcomes] = outcomes unless outcomes.empty?
  end
end

# ---------------------------------------------------------------------------
# 2.  Find games needing extraction
# ---------------------------------------------------------------------------
# If a directory is given, scan it for game JSON files missing extracts.
# If a single file is given, extract just that file (legacy mode).

def process_game(json_path)
  data = JSON.parse(File.read(json_path))

  game = Engine::Game.load(data, at_action: 0)

  initial_deck_order    = game.company_deck.map(&:sym)
  initial_offering      = game.offering.map(&:sym)
  initial_player_order  = game.players.map { |p| p.id }

  initial_record = build_snapshot(game, action_id: 0, action_type: 'initial').merge(
    deck_order:     initial_deck_order,
    initial_offering: initial_offering,
    player_order:   initial_player_order,
  )

  snapshots = [initial_record]
  undo_groups = []  # stack of {engine: [...], snaps: [...]} hashes

  # Track ALL processed action IDs+types (including skip_types like program_*)
  # so that undo can determine what the engine actually undoes.  Snapshots only
  # exist for non-skip actions, so we need this parallel stack to avoid popping
  # a snapshot when the engine undoes a skip_type action.
  engine_action_stack = []

  actions = data['actions'] || []

  # Action types that carry no game-state meaning for our engine.
  # program_* are auto-pass convenience features; message is chat.
  # These are skipped entirely (though their auto_actions are preserved
  # by flattening on the Python side).
  skip_types = Set.new(%w[program_share_pass program_close_pass program_disable message])

  actions.each do |raw_action|
    action_id   = raw_action['id']
    action_type = raw_action['type']

    if action_type == 'undo'
      # Let the engine handle the undo — it correctly processes all prior
      # actions (including program_*) and reverts the undone action(s).
      game = Engine::Game.load(data, at_action: action_id)

      # Pop from engine_action_stack to find what the engine actually undid,
      # and only pop a snapshot if the undone action produced one.
      engine_group = []
      snap_group = []

      if raw_action['action_id']
        target_id = raw_action['action_id']
        while engine_action_stack.size > 0 && engine_action_stack.last[:id] > target_id
          engine_group.push(engine_action_stack.pop)
        end
        while snapshots.size > 1 && snapshots.last[:action_id] > target_id
          snap_group.push(snapshots.pop)
        end
      else
        if engine_action_stack.size > 0
          undone = engine_action_stack.pop
          engine_group.push(undone)
          # Only pop a snapshot if the undone action was a real (non-skip) action
          if !skip_types.include?(undone[:type]) && snapshots.size > 1
            snap_group.push(snapshots.pop)
          end
        end
      end

      # Always push a group when action_id is specified — even if nothing was
      # popped.  The 18xx engine pushes an (empty) redo entry in this case, and
      # subsequent redo must consume it instead of a stale earlier group.
      # For simple undo (no action_id), only push if something was undone.
      if !engine_group.empty? || raw_action['action_id']
        undo_groups.push({ engine: engine_group, snaps: snap_group })
      end
      next
    end

    if action_type == 'redo'
      # Restore the most recently undone group (engine actions + snapshots).
      game = Engine::Game.load(data, at_action: action_id)
      unless undo_groups.empty?
        group = undo_groups.pop
        # Groups were pushed in reverse order (highest action_id first),
        # so reverse to restore chronological order.
        group[:engine].reverse.each { |a| engine_action_stack.push(a) }
        group[:snaps].reverse.each { |s| snapshots.push(s) }
      end
      next
    end

    # New (non-undo/redo) actions clear the redo stack in the 18xx engine
    # (base.rb line 804: active_undos.clear).  Mirror this so that undone
    # actions stored in undo_groups don't leak back via a later redo.
    undo_groups.clear unless undo_groups.empty?

    # Capture the round BEFORE processing: the snapshot records state AFTER
    # the action, but the round label should reflect WHEN the action was taken
    # (the last action in a round would otherwise show the next round's name).
    round_before = game.round.class.short_name

    # Annotate metadata BEFORE processing — captures state at the time the
    # action was taken, which the Python replay harness uses for decisions.
    extra = {}

    # Tag sell_company (CLO) with adjusted_income so Python can detect
    # non-negative-income closes without querying engine state.
    if action_type == 'sell_company'
      company = game.companies.find { |c| c.sym.to_s == raw_action['company'] }
      extra[:adjusted_income] = game.company_income(company) if company
    end

    # Detect forced actions — our Cython engine auto-applies these.
    forced = false
    if action_type == 'dividend'
      corp = game.corporations.find { |c| c.name == raw_action['entity'] }
      if corp && corp.floated?
        forced = corp.receivership? || game.max_dividend_per_share(corp) == 0
      end
    elsif action_type == 'pass' && round_before == 'IPO'
      entity = game.round.current_entity
      if entity&.company? && entity.owner&.player?
        # Check affordability using our Cython engine's cost formula:
        #   par >= face → 1 player share, cost = par - face
        #   par <  face → 2 player shares, cost = 2*par - face
        can_afford = false
        unless game.corporations.all?(&:ipoed)
          face_value = entity.value
          player_cash = entity.owner.cash
          game.available_par_prices(entity).each do |pp|
            player_shares = pp.price >= face_value ? 1 : 2
            cost = (player_shares * pp.price) - face_value
            if cost <= player_cash
              can_afford = true
              break
            end
          end
        end
        forced = true unless can_afford
      end
    end

    result = game.process_action(raw_action)
    game = result if result.is_a?(Engine::Game::Base)

    # Track on engine action stack (ALL actions including skip_types)
    # so that undo can determine what was undone.
    engine_action_stack.push({ id: action_id, type: action_type })

    # Skip snapshots for meta actions (program_*, message) — they don't
    # change game state meaningfully, but they must still be processed above
    # so the engine's auto-pass flags stay correct.
    next if skip_types.include?(action_type)

    snap = build_snapshot(game, action_id: action_id, action_type: action_type,
                          round_override: round_before)
    snap[:forced] = true if forced
    snap.merge!(extra) unless extra.empty?
    snapshots << snap
  end

  # Collect ALL committed action IDs: snapshot IDs (for real actions) plus
  # engine_action_stack IDs for skip_type actions (program_*, message) that
  # remain committed but never produce snapshots.  The Python side needs
  # both to correctly filter auto_actions from undone program_* actions.
  committed_ids = snapshots.drop(1).map { |s| s[:action_id] }
  engine_action_stack.each do |entry|
    committed_ids << entry[:id] if skip_types.include?(entry[:type])
  end
  snapshots[0][:committed_action_ids] = committed_ids

  # Post-process: annotate ACQ outcomes with seller info and cross-president
  # flags so the Python side can skip expensive seller-detection loops.
  annotate_acq_outcomes(snapshots, actions, committed_ids.to_set)

  snapshots
end

abort "Usage: ruby #{$PROGRAM_NAME} <data_dir|game.json>" unless ARGV.size >= 1

target = ARGV[0]

if File.directory?(target)
  # Batch mode: scan directory for games missing extracts
  game_files = Dir.glob(File.join(target, '*.json'))
                  .reject { |f| f.end_with?('_extract.json') }
                  .sort

  pending = game_files.select do |f|
    extract_path = f.sub(/\.json$/, '_extract.json')
    !File.exist?(extract_path)
  end

  if pending.empty?
    $stderr.puts "All #{game_files.size} games already extracted."
  else
    $stderr.puts "Extracting #{pending.size} of #{game_files.size} games..."
    pending.each_with_index do |json_path, idx|
      game_id = File.basename(json_path, '.json')
      $stderr.print "  [#{idx + 1}/#{pending.size}] #{game_id}..."
      $stderr.flush

      begin
        snapshots = process_game(json_path)
        extract_path = json_path.sub(/\.json$/, '_extract.json')
        File.write(extract_path, JSON.generate(snapshots))
        $stderr.puts " OK (#{snapshots.size} snapshots)"
      rescue => e
        $stderr.puts " FAILED: #{e.message}"
      end
    end
  end
else
  # Single-file mode (legacy): output to stdout
  abort "File not found: #{target}" unless File.exist?(target)
  snapshots = process_game(target)
  $stdout.puts JSON.generate(snapshots)
end

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
#   tests/18xx_games/18xx/lib/
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
$LOAD_PATH.unshift(File.join(SCRIPT_DIR, '18xx', 'lib'))

require_relative '18xx/lib/engine'

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
def build_snapshot(game, action_id:, action_type:)
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
    round:            game.round.class.short_name,
    turn:             game.turn,
    active_player:    active_player_id,
    active_corp:      active_corp_name,
    players:          snapshot_players(game),
    corporations:     snapshot_corporations(game),
    foreign_investor: snapshot_foreign_investor(game),
    offering:         game.offering.map(&:sym),
    deck_size:        game.company_deck.size,
    cost_level:       game.cost_level,
  }
end

# ---------------------------------------------------------------------------
# 2.  Load the game JSON
# ---------------------------------------------------------------------------

abort "Usage: ruby #{$PROGRAM_NAME} <game.json>" unless ARGV.size >= 1

json_path = ARGV[0]
abort "File not found: #{json_path}" unless File.exist?(json_path)

data = JSON.parse(File.read(json_path))

# ---------------------------------------------------------------------------
# 3.  Capture the initial state (action_id 0)
# ---------------------------------------------------------------------------
# Load the game up to action 0 (i.e. just after setup, before any player
# action has been processed).

game = Engine::Game.load(data, at_action: 0)

# The company_deck and offering are populated by setup(), which runs when
# at_action: 0.  We capture both for the initial record.
initial_deck_order    = game.company_deck.map(&:sym)
initial_offering      = game.offering.map(&:sym)
initial_player_order  = game.players.map { |p| p.id }

initial_record = build_snapshot(game, action_id: 0, action_type: 'initial').merge(
  deck_order:     initial_deck_order,
  initial_offering: initial_offering,
  player_order:   initial_player_order,
)

snapshots = [initial_record]

# ---------------------------------------------------------------------------
# 4.  Step through all actions one at a time
# ---------------------------------------------------------------------------
# We replay each action from the raw actions array in order.
#
# For normal actions, process_action mutates the game in place and returns self.
# For undo/redo, the engine's incremental clone mechanism can produce stale
# state (the cloned game doesn't always match a clean reload).  To guarantee
# correct snapshots, we detect undo/redo actions and reload the game from
# scratch with Engine::Game.load(data, at_action: id).  This is slower but
# ensures the snapshot always reflects the true game state.

actions = data['actions'] || []

actions.each do |raw_action|
  action_id   = raw_action['id']
  action_type = raw_action['type']

  if action_type == 'undo' || action_type == 'redo'
    # Reload from scratch to get the correct post-undo/redo state.
    game = Engine::Game.load(data, at_action: action_id)
  else
    result = game.process_action(raw_action)
    game = result if result.is_a?(Engine::Game::Base)
  end

  snapshots << build_snapshot(game, action_id: action_id, action_type: action_type)
end

# ---------------------------------------------------------------------------
# 5.  Output
# ---------------------------------------------------------------------------

$stdout.puts JSON.generate(snapshots)

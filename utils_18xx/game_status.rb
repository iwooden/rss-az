#!/usr/bin/env ruby
# frozen_string_literal: true

# Recompute the game fields 18xx.games derives from its engine after each
# action (routes/game.rb#set_game_state): round, turn, acting, finished, result.
#
# Usage: ruby game_status.rb GAME_JSON [FROM_COUNT]
#
# Prints a JSON list with one state per action-prefix length FROM_COUNT..N,
# where N is the number of actions in GAME_JSON. FROM_COUNT defaults to N, so
# the default output is just the current state.
#
# States after FROM_COUNT also carry 'user': the user id 18xx.games stores for
# that prefix's last action. MessageBus copies of actions omit it unless the
# poster was not the entity's player (Action::Base.from_h), so it is the
# entity's player just before the action.

require 'logger'

real_stdout = $stdout.dup
$stdout.reopen('/dev/null', 'w')

SCRIPT_DIR = File.dirname(File.expand_path(__FILE__))
REPO_ROOT = File.expand_path('..', SCRIPT_DIR)
$LOAD_PATH.unshift(File.join(REPO_ROOT, 'submodules', '18xx', 'lib'))

require_relative '../submodules/18xx/lib/engine'

$stdout.reopen(real_stdout)
real_stdout.close

LOGGER.level = ::Logger::FATAL

require 'json'

game_path, from_arg = ARGV
unless game_path
  warn 'Usage: ruby game_status.rb GAME_JSON [FROM_COUNT]'
  exit 64
end

game_data = JSON.parse(File.read(game_path))
actions = game_data['actions'] || []
from_count = (from_arg ? Integer(from_arg) : actions.size).clamp(0, actions.size)

previous = nil
states = (from_count..actions.size).map do |count|
  engine = Engine::Game.load(game_data, actions: actions.take(count))
  if engine.exception
    warn engine.exception.to_s
    exit 2
  end

  finished = engine.finished
  state = {
    'action_count' => count,
    'round' => engine.round.name,
    'turn' => engine.turn,
    'acting' => engine.active_players_id,
    'finished' => finished,
    'result' => finished ? engine.result : {},
  }
  if previous
    action = actions[count - 1]
    state['user'] = action['user'] ||
                    previous.get(action['entity_type'], action['entity'])&.player&.id
  end
  previous = engine
  state
end

puts JSON.generate(states)

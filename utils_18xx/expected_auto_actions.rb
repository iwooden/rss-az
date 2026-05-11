#!/usr/bin/env ruby
# frozen_string_literal: true

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

game_path, action_path = ARGV
unless game_path && action_path
  warn 'Usage: ruby expected_auto_actions.rb GAME_JSON ACTION_JSON'
  exit 64
end

game_data = JSON.parse(File.read(game_path))
action = JSON.parse(File.read(action_path))

engine = Engine::Game.load(game_data, actions: game_data['actions'] || [])
engine = engine.process_action(action, add_auto_actions: true)

if engine.exception
  warn engine.exception.to_s
  exit 2
end

puts JSON.generate(engine.raw_actions.last.to_h)

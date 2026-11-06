#!/usr/bin/env ruby
# frozen_string_literal: true

# Composes an 18xx.games hotseat log of Shikoku 1889 from a game's semantic
# intents. utils_18xx/g1889_selfplay_log.py makes the intents from the engine's
# decisions and runs this; it reads one JSON document on stdin and writes
# {game, notes} on stdout, or an error's context on stderr and exits 1.
#
# It walks a live Engine::Game in normal (not load) mode, so Ruby validates every
# action as it would a human's in the UI. Ruby's objects give the concrete ids
# the engine does not track (share, tile copy, train copy, city, par price), and
# Ruby's steps say which of the engine's forced decisions need a logged action:
#
# - A pass goes out only if Ruby's step at the intent offers one. Otherwise it is
#   dropped, with a note unless the engine forced it or Ruby is known to skip it.
# - Before each intent, the OR steps it has moved past are closed with Ruby's
#   pass: Track kept open after a lay for MF's port tile, or a step the engine
#   skipped with nothing to offer.
# - run_routes goes out if Ruby's Route step asks, even with no route. Each
#   route's revenue is Ruby's own computation, checked against the solver's.
# - A forced dividend goes out only if Ruby's Dividend step asks (zero revenue
#   skips it).
# - The engine offers DR's exchange on an SR turn Ruby skips for having nothing
#   else to offer. Ruby takes it at any time, so it goes out before the action
#   that made Ruby skip the turn, which then is a real one (apply_exchange). MF's
#   port tile, which changes no one's options, goes out where Ruby rests next.

require 'json'
require 'logger'

module G1889Compose
  ROOT = File.expand_path('..', __dir__)
  # Ruby's OR steps in the order a turn runs them, for "has Ruby reached this step?".
  STEP_RANK = { 'Track' => 0, 'Token' => 1, 'Route' => 2, 'Dividend' => 3, 'DiscardTrain' => 4, 'BuyTrain' => 5,
                'BuyCompany' => 6 }.freeze
  PASSABLE = %w[Track Token BuyTrain BuyCompany].freeze

  # An intent Ruby refused, or that cannot be made into an action.
  class Rejected < StandardError; end

  module_function

  # Loads the engine with its stdout logger silenced, as tests/g1889/replay/extract_states.rb does.
  def load_engine
    real_stdout = $stdout.dup
    $stdout.reopen(File::NULL, 'w')
    $LOAD_PATH.unshift(File.join(ROOT, 'submodules', '18xx', 'lib'))
    require File.join(ROOT, 'submodules', '18xx', 'lib', 'engine')
  ensure
    $stdout.reopen(real_stdout)
    real_stdout.close
    LOGGER.level = Logger::FATAL
  end

  class Composer
    attr_reader :notes

    def initialize(doc)
      @doc = doc
      @game = Engine::Game.load(game_data([]))
      @actions = []
      @action_rounds = []  # for each action, the round and turn it went out in
      @notes = []
      @index = nil
      @intent = nil
    end

    def game_data(actions)
      n = @doc['players'].size
      {
        'id' => @doc['id'], 'title' => '1889', 'description' => @doc['description'], 'status' => 'active',
        'players' => @doc['players'], 'min_players' => n, 'max_players' => n,
        'settings' => { 'seed' => @doc['seed'], 'unlisted' => false, 'optional_rules' => [] }, 'user_settings' => nil,
        'mode' => 'hotseat', 'user' => { 'id' => 0, 'name' => 'You' },
        'created_at' => @doc['created_date'], 'updated_at' => @doc['created_at'], 'actions' => actions
      }
    end

    def run
      @doc['intents'].each_with_index do |intent, index|
        @index = index
        @intent = intent
        apply(intent)
      end
      @index = @intent = nil
      close_steps_until_finished
      reject('the game is not finished after the last intent') unless @game.finished
      data = reload
      data.merge('status' => 'finished', 'result' => @game.result.to_h { |id, value| [id.to_s, value] },
                 'turn' => @game.turn, 'round' => @game.round.name, 'acting' => [],
                 'game_end_reason' => @game.game_end_reason&.to_s)
    rescue Rejected
      raise
    rescue Engine::GameError => e
      reject("#{e.class}: #{e.message}")
    end

    # Loads the finished log the way 18xx.games does, which must take it whole.
    def reload
      data = game_data(@actions)
      loaded = Engine::Game.load(data)
      raise Rejected, "reloading the log: #{loaded.exception.class}: #{loaded.exception.message}" if loaded.exception
      raise Rejected, 'the reloaded log did not finish the game' unless loaded.finished
      raise Rejected, 'the reloaded log has another result' unless loaded.result == @game.result

      data
    end

    def apply(intent)
      return submit(intent.slice('type', 'entity', 'entity_type', 'message')) if intent['type'] == 'message'

      settle(intent)
      case intent['type']
      when 'pass' then apply_pass(intent)
      when 'bid' then submit(intent.slice('type', 'entity', 'entity_type', 'company', 'price'))
      when 'par' then apply_par(intent)
      when 'buy_shares' then intent['entity_type'] == 'company' ? apply_exchange(intent) : apply_buy_shares(intent)
      when 'sell_shares' then apply_sell_shares(intent)
      when 'lay_tile' then apply_lay_tile(intent)
      when 'place_token' then apply_place_token(intent)
      when 'run_routes' then apply_run_routes(intent)
      when 'dividend' then apply_dividend(intent)
      when 'buy_train' then apply_buy_train(intent)
      when 'discard_train' then apply_discard_train(intent)
      when 'buy_company' then submit(intent.slice('type', 'entity', 'entity_type', 'company', 'price'))
      when 'bankrupt' then submit(intent.slice('type', 'entity', 'entity_type'))
      else reject("unknown intent #{intent['type']}")
      end
    end

    def round_kind
      @game.round.class.name.split('::').last
    end

    def submit(action)
      where = [round_kind, @game.turn]
      @game.process_action(action.merge('created_at' => @doc['created_at']))
      reject("Ruby rejected #{action}: #{@game.exception.class}: #{@game.exception.message}") if @game.exception
      @actions << @game.raw_actions.last
      @action_rounds << where
    end

    def reject(message)
      entity = current_entity
      raise Rejected, JSON.generate({
        'message' => message, 'intent_index' => @index, 'intent' => @intent,
        'round' => round_kind, 'turn' => @game.turn, 'round_num' => @game.round.round_num,
        'step' => step_name(active_step), 'entity' => entity&.id, 'actions' => entity ? @game.round.actions_for(entity) : [],
        'actions_so_far' => @actions.size, 'last_action' => @actions.last
      })
    end

    def note(kind, message)
      @notes << { 'kind' => kind, 'message' => message, 'intent_index' => @index, 'intent' => @intent,
                  'turn' => @game.turn, 'round_num' => @game.round.round_num, 'step' => step_name(active_step),
                  'entity' => current_entity&.id }
    end

    def active_step
      @game.round.active_step
    end

    def current_entity
      active_step&.current_entity
    end

    def step_name(step)
      step&.class&.name&.split('::')&.last
    end

    # ---------------------------------------------------------- closing steps

    # Closes what the engine has moved past. An intent of a corp's turn closes
    # earlier turns and this turn's earlier steps. Any other, in an SR or the
    # auction, closes the OR that came before it. A discard closes nothing: Ruby's
    # DiscardTrain interrupts the step it comes in.
    def settle(intent)
      return if intent['type'] == 'discard_train'

      close_steps(intent['operator'], intent['order'], intent['reach'])
    end

    def here?(operator, order, reach)
      return false unless operator && current_entity&.id == operator
      return false unless order == [@game.turn, @game.round.round_num]

      reach.nil? || (STEP_RANK[step_name(active_step)] || -1) >= STEP_RANK.fetch(reach)
    end

    # Whether Ruby has already left that corp's turn, or that SR (order [turn, 0]),
    # having nothing for it to do there.
    def past?(operator, order)
      return false unless order

      now = [@game.turn, @game.round.round_num]
      return (now <=> order).positive? unless now == order

      operator && @game.round.entity_index > (@game.round.entities.index(corporation(operator)) || -1)
    end

    def close_steps(operator, order, reach)
      while !@game.finished && @game.round.operating? && !here?(operator, order, reach) && !past?(operator, order) &&
            pass_open_step
      end
    end

    def close_steps_until_finished
      close_steps(nil, nil, nil)
    end

    # Passes Ruby's active step if it is one a pass closes.
    def pass_open_step
      entity = current_entity
      return false unless PASSABLE.include?(step_name(active_step)) && @game.round.actions_for(entity).include?('pass')

      submit('type' => 'pass', 'entity' => entity.id, 'entity_type' => type_of(entity))
      true
    end

    def type_of(entity)
      return 'company' if entity.company?

      entity.player? ? 'player' : 'corporation'
    end

    # ---------------------------------------------------------------- intents

    def entity_of(intent)
      @game.get(intent['entity_type'], intent['entity'])
    end

    def corporation(sym)
      @game.corporation_by_id(sym)
    end

    # Whether Ruby is at the step or turn the pass is for. It may be past it,
    # having skipped a turn the engine forced or a step with nothing to do. In an
    # SR it may have ended the turn and come back to the player: the engine's "done"
    # ends a turn that has acted, its pass one that has not.
    def at_pass?(intent)
      now = [@game.turn, @game.round.round_num]
      if intent['reach']
        return now == intent['order'] && step_name(active_step) == intent['reach'] && current_entity&.id == intent['entity']
      end
      return true unless intent['round']
      return false unless round_kind == intent['round'] && now[0] == intent['order'][0]

      intent['round'] != 'Stock' || @game.round.current_actions.any? == intent['done']
    end

    def apply_pass(intent)
      if at_pass?(intent) && @game.round.actions_for(entity_of(intent)).include?('pass')
        submit(intent.slice('type', 'entity', 'entity_type'))
      elsif !intent['optional']
        note('dropped_pass', 'a pass the engine offered as a choice, where Ruby offers no such step')
      end
    end

    def apply_par(intent)
      price = @game.stock_market.par_prices.find { |p| p.price == intent['price'] }
      reject("no par price #{intent['price']}") unless price
      submit(intent.slice('type', 'entity', 'entity_type', 'corporation').merge('share_price' => price.id))
    end

    def apply_buy_shares(intent)
      corp = corporation(intent['corporation'])
      shares = intent['source'] == 'pool' ? @game.share_pool.shares_of(corp) : corp.ipo_shares
      share = shares.reject(&:president).first
      reject("no #{intent['source']} share of #{corp.id}") unless share
      submit(intent.slice('type', 'entity', 'entity_type').merge('shares' => [share.id], 'percent' => share.percent))
    end

    # DR's exchange. On an SR turn Ruby did not stop at, it goes out before the
    # action after which Ruby skipped the turn, so the exchange falls in the same
    # SR with the same private revenue paid, and its owner gets a real turn if the
    # share gives them something to do. Without such an action (the auction, or an
    # OR with no corps that paid out in between), or if that action needs the share
    # still to be there, it goes out where Ruby rests.
    def apply_exchange(intent)
      return apply_buy_shares(intent) if intent['operator'] || owners_sr_turn?(intent)

      turn = intent['order'][0]
      last = @actions.rindex { |a| !(a['type'] == 'message' || port_tile?(a)) }
      hoistable = [['Stock', turn], ['Operating', turn - 1]].include?(@action_rounds[last])
      return if hoistable && exchange_before(last, intent)

      note('late_exchange', "DR's exchange in an SR Ruby has left, with no action of that SR to go before") unless
        round_kind == 'Stock' && @game.turn == turn
      apply_buy_shares(intent)
    end

    def port_tile?(action)
      action['type'] == 'lay_tile' && action['entity'] == 'MF'
    end

    def exchange_before(index, intent)
      action, *later = @actions[index..]
      messages, tiles = later.partition { |a| a['type'] == 'message' }
      rewind_to(index)
      messages.each { |m| submit(m) }
      apply_buy_shares(intent)
      submit(action)
      tiles.each { |t| submit(t) }
      note('moved_exchange', "DR's exchange went out before the action after which Ruby skipped its owner's turn")
      true
    rescue Rejected
      rewind_to(index)
      [action, *later].each { |a| submit(a) }
      false
    end

    def owners_sr_turn?(intent)
      round_kind == 'Stock' && @game.turn == intent['order'][0] && current_entity&.id == intent['player']
    end

    # Takes the game back to before action ``index``, which goes on to be resubmitted.
    def rewind_to(index)
      @actions = @actions.take(index)
      @action_rounds = @action_rounds.take(index)
      @game = Engine::Game.load(game_data(@actions.dup))
      reject("rewinding: #{@game.exception.class}: #{@game.exception.message}") if @game.exception
    end

    def apply_sell_shares(intent)
      percent = intent['count'] * 10
      bundle = @game.sellable_bundles(entity_of(intent), corporation(intent['corporation']))
                 .find { |b| b.percent == percent }
      reject("Ruby offers no #{percent}% bundle of #{intent['corporation']}") unless bundle
      submit(intent.slice('type', 'entity', 'entity_type').merge('shares' => bundle.shares.map(&:id),
                                                                 'percent' => bundle.percent))
    end

    def apply_lay_tile(intent)
      tile = @game.tiles.find { |t| t.name == intent['tile'] }
      reject("no tile #{intent['tile']} in the supply") unless tile
      submit(intent.slice('type', 'entity', 'entity_type', 'hex', 'rotation').merge('tile' => tile.id))
    end

    def apply_place_token(intent)
      corp = corporation(intent['entity'])
      city = @game.hex_by_id(intent['hex']).tile.cities.first
      reject("no city on #{intent['hex']}") unless city
      slot = city.get_slot(corp)
      reject("no slot for #{corp.id} on #{intent['hex']}") unless slot
      submit(intent.slice('type', 'entity', 'entity_type').merge('city' => city.id, 'slot' => slot,
                                                                 'tokener' => corp.id))
    end

    def apply_run_routes(intent)
      corp = corporation(intent['entity'])
      unless @game.round.actions_for(corp).include?('run_routes')
        reject("the solver runs #{intent['routes']}, where Ruby asks for no routes") unless intent['routes'].empty?
        return
      end

      routes = build_routes(corp, intent['routes'])
      submit(intent.slice('type', 'entity', 'entity_type')
                   .merge(Engine::Action::RunRoutes.new(corp, routes: routes).args_to_h))
    end

    # Ruby's Route objects for the solver's routes, each checked by Ruby's own
    # Route#revenue, as it would a human's clicks: Ruby pays a logged revenue
    # unchecked, so the log's revenues must be proved here.
    def build_routes(corp, specs)
      free = @game.route_trains(corp).dup
      routes = []
      specs.each do |spec|
        train = free.find { |t| t.name == spec['train'] }
        reject("#{corp.id} has no #{spec['train']} train to run") unless train
        free.delete(train)
        routes << Engine::Route.new(@game, @game.phase, train, routes: routes, connection_hexes: spec['connections'])
      end
      routes.zip(specs).each do |route, spec|
        revenue = route.revenue
        hexes = route.hexes.map(&:id)
        reject("Ruby reads #{spec['connections']} as #{hexes} for #{revenue}") unless hexes == spec['stops']
        reject("Ruby's #{route.train.name} on #{hexes} earns #{revenue}, the solver's #{spec['revenue']}") \
          unless revenue == spec['revenue']
      end
      routes
    end

    def apply_dividend(intent)
      corp = corporation(intent['entity'])
      if @game.round.actions_for(corp).include?('dividend')
        submit(intent.slice('type', 'entity', 'entity_type', 'kind'))
      elsif !intent['optional']
        reject("the engine chooses a dividend (#{intent['kind']}), where Ruby offers none")
      end
    end

    def apply_buy_train(intent)
      corp = corporation(intent['entity'])
      name = intent['train']
      train = case intent['source']
              when 'depot' then @game.depot.depot_trains.find { |t| t.name == name }
              when 'pool' then @game.depot.discarded.find { |t| t.name == name }
              else corporation(intent['source']).trains.find { |t| t.name == name }
              end
      reject("no #{name} train at #{intent['source']}") unless train
      action = intent.slice('type', 'entity', 'entity_type', 'price').merge('train' => train.id, 'variant' => name)
      if intent['exchange']
        traded = corp.trains.find { |t| t.name == intent['exchange'] }
        reject("#{corp.id} has no #{intent['exchange']} to trade") unless traded
        action['exchange'] = traded.id
      end
      submit(action)
    end

    def apply_discard_train(intent)
      train = corporation(intent['entity']).trains.find { |t| t.name == intent['train'] }
      reject("#{intent['entity']} has no #{intent['train']} to discard") unless train
      submit(intent.slice('type', 'entity', 'entity_type').merge('train' => train.id))
    end
  end

  def main
    doc = JSON.parse($stdin.read)
    load_engine
    composer = Composer.new(doc)
    $stdout.write(JSON.generate({ 'game' => composer.run, 'notes' => composer.notes }))
  rescue Rejected => e
    warn(e.message)
    exit(1)
  end
end

G1889Compose.main if $PROGRAM_NAME == __FILE__

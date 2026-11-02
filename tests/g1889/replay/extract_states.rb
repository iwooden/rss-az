#!/usr/bin/env ruby
# frozen_string_literal: true

# Ruby state extractor for differential replay of Shikoku 1889
# (1889/engine-design.md section 10.2).
#
# Replays an 18xx.games game through the Ruby engine in submodules/18xx and
# prints JSON on stdout: an initial snapshot, then one record per action that
# survives undo and redo, holding the action, pre-action annotations, and the
# state after it. tests/g1889/replay/extracts.py runs this, caches the output,
# and documents the schema.
#
#   ruby tests/g1889/replay/extract_states.rb path/to/game.json
#   ruby tests/g1889/replay/extract_states.rb --new-game N [--seed S] [--players ID,ID,...]
#
# Ruby is the oracle, so the extractor records what the engine does and never
# corrects it. Actions are processed the way Engine::Game.load processes them:
# Game::Base.filtered_actions resolves undo and redo, every surviving action
# goes through Game::Base#process_to_action in load mode (which applies its
# auto_actions), and an error raises instead of silently stopping the load.
# The output repeats the final state of an independent full Game.load, so a
# test can check that stepping one action at a time changed nothing.

require 'json'
require 'logger'

module G1889Extract
  ROOT = File.expand_path('../../..', __dir__)
  SCHEMA_VERSION = 1
  TITLE = '1889'

  module_function

  # Loads the engine with its stdout logger silenced, as utils_18xx/extract_states.rb does.
  def load_engine
    real_stdout = $stdout.dup
    $stdout.reopen(File::NULL, 'w')
    $LOAD_PATH.unshift(File.join(ROOT, 'submodules', '18xx', 'lib'))
    require File.join(ROOT, 'submodules', '18xx', 'lib', 'engine')
    # auto_router.rb embeds its JavaScript in a %x{} literal at class level,
    # which MRI would hand to a shell; this stub swallows it.
    Engine.const_set(:AutoRouter, Class.new).define_singleton_method(:`) { |_| nil } unless defined?(Engine::AutoRouter)
    require File.join(ROOT, 'submodules', '18xx', 'lib', 'engine', 'auto_router')
  ensure
    $stdout.reopen(real_stdout)
    real_stdout.close
    LOGGER.level = Logger::FATAL
  end

  # Raised for a failure in the extractor's own annotation code, which
  # Game::Base#process_action would otherwise record as a rejected action.
  class AnnotationError < StandardError; end

  # Calls a block before every Game::Base#process_single_action, so a host
  # action and each of its auto_actions is annotated in the state it meets.
  module SingleActionHook
    attr_accessor :before_single_action

    def process_single_action(action)
      begin
        @before_single_action&.call(action)
      rescue StandardError => e
        raise AnnotationError, "annotating #{action.type}: #{e.class}: #{e.message}\n#{e.backtrace.first(8).join("\n")}"
      end
      super
    end
  end

  # Builds snapshots and annotations of one game. Every query here is
  # read-only: in particular it never indexes the engine's default-proc
  # hashes (shares_by_corporation, players_sold), which would insert keys.
  class Reader
    def initialize(game, player_ids)
      @game = game
      @player_ids = player_ids
      @corp_index = game.corporations.each_with_index.to_h
      @company_index = game.companies.each_with_index.to_h
      @supply_names = game.all_tiles.map(&:name).uniq
    end

    # ----------------------------------------------------------------- refs

    def ref(entity)
      return nil if entity.nil?
      return { 'type' => 'player', 'id' => entity.id } if entity.player?
      return { 'type' => 'corporation', 'id' => entity.name } if entity.corporation?
      return { 'type' => 'company', 'id' => entity.sym } if entity.company?
      return { 'type' => 'pool' } if entity == @game.share_pool
      return { 'type' => 'bank' } if entity == @game.bank
      return { 'type' => 'depot' } if entity == @game.depot

      { 'type' => entity.class.name }
    end

    def held(holder, corporation)
      holder.shares_by_corporation.fetch(corporation, [])
    end

    def percent(holder, corporation)
      held(holder, corporation).sum(&:percent)
    end

    def step_name(step)
      step&.class&.name&.split('::')&.last
    end

    def count_by_name(trains)
      trains.map(&:name).tally
    end

    def company_order(companies)
      companies.sort_by { |c| @company_index.fetch(c, @company_index.size) }.map(&:sym)
    end

    def auction_step
      @game.round.steps.find { |s| s.is_a?(Engine::Step::WaterfallAuction) }
    end

    # Ruby round_state, made JSON-safe, for diagnostics only.
    def plain(value)
      case value
      when nil, true, false, Integer, Float, String then value
      when Symbol then value.to_s
      when Array, Set then value.map { |v| plain(v) }
      when Hash then value.to_h { |k, v| [plain_key(k), plain(v)] }
      when Engine::Player then value.id
      when Engine::Corporation then value.name
      when Engine::Company then value.sym
      when Engine::Hex then value.id
      when Engine::Tile then value.name
      when Engine::Train then value.id
      when Engine::Action::Base then value.type
      when Engine::Route then { 'train' => value.train.id, 'hexes' => value.hexes.map(&:id) }
      else value.class.name
      end
    end

    def plain_key(key)
      key = plain(key)
      key.is_a?(String) ? key : key.to_s
    end

    # ------------------------------------------------------------- snapshot

    def snapshot
      {
        'game' => game_snapshot,
        'players' => @player_ids.map { |id| player_snapshot(@game.player_by_id(id)) },
        'corporations' => @game.corporations.to_h { |c| [c.name, corporation_snapshot(c)] },
        'privates' => @game.companies.to_h { |c| [c.sym, private_snapshot(c)] },
        'hexes' => @game.hexes.to_h { |h| [h.id, hex_snapshot(h)] },
        'tiles' => @supply_names.to_h { |name| [name, 0] }.merge(@game.tiles.map(&:name).tally),
        'depot' => count_by_name(@game.depot.upcoming),
        'pool' => count_by_name(@game.depot.discarded),
        'auction' => auction_snapshot,
        'stock' => stock_snapshot,
        'operating' => operating_snapshot,
        'round_state' => round_state_snapshot,
      }
    end

    def game_snapshot
      round = @game.round
      step = round.active_step
      phase = @game.phase
      {
        'phase' => phase.name,
        'train_limit' => phase.current[:train_limit],
        'tile_colors' => phase.tiles.map(&:to_s),
        'round' => round.class.short_name,
        'round_num' => round.round_num,
        'turn' => @game.turn,
        'or_count' => @game.operating_rounds,
        'step' => step_name(step),
        'current_entity' => ref(step&.current_entity),
        'active_players' => @game.active_players_id,
        'round_entities' => round.entities.map { |e| ref(e)['id'] },
        'entity_index' => round.entity_index,
        'priority' => @game.send(:priority_deal_player)&.id,
        'player_order' => @game.players.map(&:id),
        'bank_cash' => @game.bank.cash,
        'bank_broken' => @game.bank.broken?,
        'finished' => @game.finished,
        'end_reason' => @game.game_end_reason&.to_s,
        'result' => @game.finished ? @game.result.transform_keys(&:to_s) : nil,
      }
    end

    def player_snapshot(player)
      shares = player.shares_by_corporation
                     .reject { |_, list| list.empty? }
                     .sort_by { |corp, _| @corp_index[corp] }
                     .to_h do |corp, list|
        [corp.name, { 'percent' => list.sum(&:percent), 'president' => list.any?(&:president) }]
      end
      step = auction_step
      bids = step ? step.bids.values.flatten.select { |b| b.entity == player }.to_h { |b| [b.company.sym, b.price] } : {}
      {
        'id' => player.id,
        'name' => player.name,
        'cash' => player.cash,
        'value' => @game.player_value(player),
        'shares' => shares,
        'privates' => company_order(player.companies),
        'bids' => bids,
        'passed' => !!player.passed?,
        'bankrupt' => !!player.bankrupt,
      }
    end

    def corporation_snapshot(corp)
      price = corp.share_price
      history = corp.operating_history
      last = history.empty? ? nil : history[history.keys.max]
      {
        'ipoed' => corp.ipoed,
        'par' => corp.par_price&.price,
        'price' => price&.price,
        'market' => price && [*price.coordinates, price.corporations.index(corp)],
        'floated' => corp.floated?,
        'cash' => corp.cash,
        'ipo_percent' => percent(corp.ipo_owner, corp),
        'pool_percent' => percent(@game.share_pool, corp),
        'president' => corp.owner&.player? ? corp.owner.id : nil,
        'tokens_left' => corp.tokens.count { |t| !t.used },
        'token_hexes' => corp.tokens.select(&:used).map { |t| t.city&.hex&.id },
        'trains' => count_by_name(corp.trains),
        'train_ids' => corp.trains.map(&:id),
        'privates' => company_order(corp.companies),
        'operated' => !history.empty?,
        'last_revenue' => last&.revenue,
        'last_dividend' => last&.dividend_kind&.to_s,
        'closed' => corp.closed?,
      }
    end

    def private_snapshot(company)
      {
        'owner' => ref(company.owner),
        'closed' => company.closed?,
        'value' => company.value,
        'revenue' => company.revenue,
        'discount' => company.discount,
        'abilities' => company.all_abilities.map { |a| a.type.to_s },
      }
    end

    def hex_snapshot(hex)
      tile = hex.tile
      {
        'tile' => tile.name,
        'rotation' => tile.rotation,
        'tokens' => tile.cities.flat_map(&:tokens).compact.map { |t| t.corporation.name },
        'reserved' => tile.cities.flat_map(&:reservations).compact.map(&:name),
      }
    end

    def auction_snapshot
      return nil unless (step = auction_step)

      auctioning = step.send(:auctioning)
      {
        'companies' => step.companies.map(&:sym),
        'auctioning' => auctioning&.sym,
        'min_bids' => step.companies.to_h { |c| [c.sym, step.min_bid(c)] },
      }
    end

    def stock_snapshot
      round = @game.round
      return nil unless round.is_a?(Engine::Round::Stock)

      sold = round.players_sold
      {
        'sold' => @player_ids.to_h do |id|
          corps = sold.fetch(@game.player_by_id(id), {})
          [id.to_s, corps.sort_by { |c, _| @corp_index[c] }.to_h { |c, when_| [c.name, when_.to_s] }]
        end,
        'current_actions' => round.current_actions.map(&:type),
        'bought_from_ipo' => round.bought_from_ipo,
        'last_to_act' => round.last_to_act&.id,
        'pass_order' => round.pass_order.map(&:id),
      }
    end

    def operating_snapshot
      round = @game.round
      return nil unless round.is_a?(Engine::Round::Operating)

      buy_train = round.steps.find { |s| s.is_a?(Engine::Step::BuyTrain) }
      {
        'operator' => round.current_operator&.name,
        'routes_revenue' => round.routes.empty? ? nil : @game.routes_revenue(round.routes),
        'ebuy_sold' => buy_train.instance_variable_get(:@corporations_sold)&.map(&:name) || [],
        'ebuy_last_share_price' => buy_train.last_share_sold_price,
      }
    end

    def round_state_snapshot
      round = @game.round
      keys = round.instance_variable_get(:@round_state_keys)&.keys || []
      keys.to_h { |k| [k.to_s, plain(round.instance_variable_get("@#{k}"))] }
    end

    # ----------------------------------------------------------- annotation

    def annotate(action)
      round = @game.round
      step = round.active_step
      pre = {
        'type' => action.type,
        'round' => round.class.short_name,
        'round_num' => round.round_num,
        'turn' => @game.turn,
        'step' => step_name(step),
        'handler' => step_name(handler_step(action)),
        'current_entity' => ref(step&.current_entity),
        'active_players' => @game.active_players_id,
      }
      extra = case action
              when Engine::Action::Bid then bid_info(action)
              when Engine::Action::Par then par_info(action)
              when Engine::Action::BuyShares then buy_shares_info(action)
              when Engine::Action::SellShares then sell_shares_info(action)
              when Engine::Action::LayTile then lay_tile_info(action)
              when Engine::Action::PlaceToken then place_token_info(action)
              when Engine::Action::RunRoutes then run_routes_info(action)
              when Engine::Action::Dividend then dividend_info
              when Engine::Action::BuyTrain then buy_train_info(action)
              when Engine::Action::DiscardTrain then { 'train_type' => action.train.name }
              when Engine::Action::BuyCompany then buy_company_info(action)
              when Engine::Action::Bankrupt then { 'president' => action.entity.owner&.id }
              else {}
              end
      pre.merge(extra)
    end

    # The step Round::Base#process_action will hand the action to.
    def handler_step(action)
      @game.round.steps.find do |s|
        next unless s.active?

        s.blocking? || s.actions(action.entity).include?(action.type)
      end
    end

    def bid_info(action)
      step = auction_step
      company = action.company
      {
        'company' => company.sym,
        'price' => action.price,
        'topmost' => step.companies.first&.sym,
        'auctioning' => step.send(:auctioning)&.sym,
        'min_bid' => step.min_bid(company),
        'purchase' => step.may_purchase?(company),
      }
    end

    def par_info(action)
      { 'corporation' => action.corporation.name, 'par_price' => action.share_price.price }
    end

    def share_source(owner)
      return 'pool' if owner == @game.share_pool
      return 'ipo' if owner.corporation? && owner == owner.ipo_owner

      ref(owner)['type']
    end

    def buy_shares_info(action)
      bundle = action.bundle
      {
        'source' => share_source(bundle.owner),
        'corporation' => bundle.corporation.name,
        'percent' => bundle.percent,
        'price' => bundle.price,
      }
    end

    def sell_shares_info(action)
      bundle = action.bundle
      {
        'corporation' => bundle.corporation.name,
        'percent' => bundle.percent,
        'count' => bundle.shares.size,
        'price' => bundle.price,
      }
    end

    def lay_tile_info(action)
      { 'hex' => action.hex.id, 'tile' => action.tile.name, 'rotation' => action.rotation,
        'old_tile' => action.hex.tile.name, 'old_rotation' => action.hex.tile.rotation }
    end

    def place_token_info(action)
      city = action.city
      { 'hex' => city.hex.id, 'city' => city.tile.cities.index(city) }
    end

    def dividend_info
      routes = @game.round.routes
      { 'revenue' => routes.empty? ? 0 : @game.routes_revenue(routes) }
    end

    def buy_train_info(action)
      train = action.train
      owner = train.owner
      source = if owner == @game.depot
                 @game.depot.discarded.include?(train) ? 'pool' : 'depot'
               else
                 ref(owner)&.fetch('type')
               end
      info = { 'train_type' => train.name, 'source' => source, 'price' => action.price }
      info['seller'] = owner.name if owner&.corporation?
      info['exchange_type'] = action.exchange.name if action.exchange
      info
    end

    def buy_company_info(action)
      { 'company' => action.company.sym, 'seller' => ref(action.company.owner), 'price' => action.price }
    end

    # revenue is what Ruby pays. RunRoutes hands a route's logged revenue, if
    # any, to Route.new, which caches it, so Ruby checks and computes only
    # routes logged without one. recomputed_revenue rebuilds the routes
    # without logged revenues; max_revenue is AutoRouter's search, ported below.
    def run_routes_info(action)
      corp = action.entity
      info = { 'revenue' => @game.routes_revenue(action.routes), 'trains' => action.routes.map { |r| r.train.id } }
      info.merge!(recompute_routes(action))
      info.merge(max_revenue(corp))
    end

    def recompute_routes(action)
      h = action.to_h
      if h['routes'].any? { |r| r['connections'].nil? || r['connections'].empty? }
        return { 'recomputed_revenue' => nil, 'recomputed_error' => 'a logged route has no connections' }
      end

      h['routes'] = h['routes'].map { |r| r.except('revenue', 'revenue_str') }
      routes = Engine::Action::Base.action_from_h(h, @game).routes
      routes.each { |r| r.routes = routes }
      { 'recomputed_revenue' => @game.routes_revenue(routes),
        'recomputed_routes' => routes.map { |r| r.revenue } }
    rescue Engine::GameError => e
      { 'recomputed_revenue' => nil, 'recomputed_error' => e.message }
    end

    # Engine::AutoRouter#compute without the browser: its path walk is Ruby,
    # but its route combination (Autorouter#find_best_combo) is JavaScript run
    # through Opal. This ports that search: trains are tried last first, a
    # route may not share a hexside bit with an earlier choice, a prefix is cut
    # when its estimate plus the best single routes of the remaining trains
    # cannot beat the best total, and a full combination is scored by
    # real_revenue. 1889 has no autoroute groups, so there is one bitfield.
    def max_revenue(corp)
      router = Engine::AutoRouter.new(@game)
      trains = @game.route_trains(corp).sort_by(&:price)
      train_routes, timed_out = router.path(trains, corp)
      lists = train_routes.values.reverse.map { |routes| routes.map { |r| [r, r.revenue] } + [nil] }
      rest = Array.new(lists.size + 1, 0)
      (lists.size - 1).downto(0) { |i| rest[i] = rest[i + 1] + lists[i].first[1] }
      best = { revenue: 0, routes: [] }
      combo = []
      search = lambda do |i, estimate, bits|
        lists[i].each do |entry|
          route, revenue = entry
          next if route && conflicts?(route.bitfield, bits)

          if route
            combo.push(route)
            next_estimate = estimate + revenue
            next_bits = merge_bits(route.bitfield, bits)
          else
            next_estimate = estimate
            next_bits = bits
          end
          if i == lists.size - 1
            if next_estimate > best[:revenue]
              real = router.real_revenue(combo)
              if real > best[:revenue]
                best = { revenue: real, routes: combo.map do |r|
                  { 'train' => r.train.id, 'revenue' => r.revenue, 'hexes' => r.hexes.map(&:id) }
                end }
              end
            end
          elsif next_estimate + rest[i + 1] > best[:revenue]
            search.call(i + 1, next_estimate, next_bits)
          end
          combo.pop if route
        end
      end
      search.call(0, 0, []) unless lists.empty?
      { 'max_revenue' => best[:revenue], 'max_routes' => best[:routes], 'max_timed_out' => timed_out }
    end

    def conflicts?(a, b)
      [a.size, b.size].min.times.any? { |i| (a[i] & b[i]) != 0 }
    end

    def merge_bits(a, b)
      Array.new([a.size, b.size].max) { |i| (a[i] || 0) | (b[i] || 0) }
    end
  end

  def player_ids(game)
    game.instance_variable_get(:@names).keys
  end

  def check!(game, what)
    return unless (e = game.exception)
    raise "extractor failed at #{what}: #{e.message}" if e.is_a?(AnnotationError)

    raise "Ruby rejected #{what}: #{e.class}: #{e.message}"
  end

  def extract_fixture(path)
    data = JSON.parse(File.read(path))
    raise "#{path}: not a #{TITLE} game (#{data['title']})" unless data['title'] == TITLE

    raw = data['actions'] || []
    filtered, = Engine::Game::Base.filtered_actions(raw)
    game = Engine::Game.load(data, at_action: 0)
    ids = player_ids(game)
    reader = Reader.new(game, ids)
    records = [{ 'index' => 0, 'action' => nil, 'state' => reader.snapshot }]

    # Game.load processes its actions with @loading set, which skips some
    # validation (step/tracker.rb, step/token.rb) and is how 18xx.games loads
    # a stored game.
    game.instance_variable_set(:@loading, true)
    pending = []
    game.extend(SingleActionHook)
    game.before_single_action = ->(action) { pending << reader.annotate(action) }
    raw.each_with_index do |action, i|
      next unless filtered[i]

      pending.clear
      game.process_to_action(action['id'])
      check!(game, "action #{action['id']} (#{action['type']})")
      expected = 1 + (action['auto_actions']&.size || 0)
      raise "action #{action['id']}: #{pending.size} single actions, expected #{expected}" unless pending.size == expected

      record = { 'index' => records.size, 'action' => action, 'pre' => pending.first }
      record['auto_pre'] = pending.drop(1) if pending.size > 1
      record['state'] = reader.snapshot
      records << record
    end
    game.before_single_action = nil
    game.instance_variable_set(:@loading, false)

    full = Engine::Game.load(data)
    check!(full, 'the full load')
    {
      'schema' => SCHEMA_VERSION,
      'source' => {
        'kind' => 'fixture', 'file' => File.basename(path), 'id' => data['id'],
        'seed' => game.seed, 'optional_rules' => game.optional_rules.map(&:to_s),
      },
      'players' => ids.map { |id| { 'id' => id, 'name' => game.player_by_id(id).name } },
      'raw_action_count' => raw.size,
      'surviving_action_count' => filtered.compact.size,
      'records' => records,
      'full_load' => {
        'processed_action_ids' => full.actions.map { |a| a.to_h['id'] },
        'state' => Reader.new(full, player_ids(full)).snapshot,
      },
    }
  end

  def extract_new_game(num_players, seed, ids)
    ids ||= (1..num_players).to_a
    raise "--players lists #{ids.size} ids for #{num_players} players" unless ids.size == num_players

    names = ids.each_with_index.to_h { |id, i| [id, "Player #{i + 1}"] }
    game = Engine::Game::G1889::Game.new(names, id: "new-game-#{seed}", seed: seed)
    {
      'schema' => SCHEMA_VERSION,
      'source' => { 'kind' => 'new_game', 'seed' => game.seed, 'optional_rules' => [] },
      'players' => ids.map { |id| { 'id' => id, 'name' => names[id] } },
      'raw_action_count' => 0,
      'surviving_action_count' => 0,
      'records' => [{ 'index' => 0, 'action' => nil, 'state' => Reader.new(game, ids).snapshot }],
    }
  end

  def parse_id(text)
    text.match?(/\A-?\d+\z/) ? text.to_i : text
  end

  def main(argv)
    if argv.first == '--new-game'
      num_players = Integer(argv[1])
      opts = argv.drop(2).each_slice(2).to_h
      unknown = opts.keys - %w[--seed --players]
      abort "unknown options: #{unknown.join(' ')}" unless unknown.empty?
      seed = Integer(opts.fetch('--seed', '1'))
      ids = opts['--players']&.split(',')&.map { |s| parse_id(s) }
      load_engine
      extract = extract_new_game(num_players, seed, ids)
    else
      abort "usage: ruby #{$PROGRAM_NAME} <game.json> | --new-game N [--seed S] [--players ID,...]" unless argv.size == 1
      abort "file not found: #{argv[0]}" unless File.exist?(argv[0])
      load_engine
      extract = extract_fixture(argv[0])
    end
    $stdout.write(JSON.generate(extract))
  end
end

G1889Extract.main(ARGV) if $PROGRAM_NAME == __FILE__

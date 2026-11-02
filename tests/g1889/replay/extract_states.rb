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
#   ruby tests/g1889/replay/extract_states.rb --graphs specs.json
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
  SCHEMA_VERSION = 5
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
        'legal' => stock_legal(round),
      }
    end

    # What the turn player may do, from BuySellParShares' own checks: its
    # action types, and the corporations each kind of move can take. It calls
    # only what step.actions already evaluates for that player, so it inserts
    # no hash keys Ruby has not.
    def stock_legal(round)
      step = round.active_step
      return nil unless step.is_a?(Engine::Step::BuySellParShares) && (entity = step.current_entity)&.player?

      types = step.actions(entity)
      corps = @game.corporations
      buys = types.include?('buy_shares')
      {
        'actions' => types,
        'buy_ipo' => buys ? names(corps.select { |c| c.ipoed && buyable?(step, entity, c.ipo_shares) }) : [],
        'buy_pool' => buys ? names(corps.select { |c| buyable?(step, entity, held(@game.share_pool, c)) }) : [],
        'par' => types.include?('par') ? names(corps.select { |c| parable?(step, entity, c) }) : [],
        'sell' => types.include?('sell_shares') ? corps.to_h do |c|
          [c.name, @game.bundles_for_corporation(entity, c).select { |b| step.can_sell?(entity, b) }.map(&:percent)]
        end.reject { |_, percents| percents.empty? } : {},
      }
    end

    def parable?(step, entity, corporation)
      @game.can_par?(corporation, entity) && step.can_buy?(entity, corporation.shares.first&.to_bundle)
    end

    def buyable?(step, entity, shares)
      share = shares.select(&:buyable).min_by(&:percent)
      share && step.can_buy?(entity, share.to_bundle)
    end

    def names(corporations)
      corporations.map(&:name)
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
        'must_buy' => must_buy_info(round),
        'auto_router' => round.active_step.is_a?(Engine::Step::Route) ? auto_router_info(round.current_operator) : nil,
      }.merge(network_snapshot(round))
    end

    # At a BuyTrain snapshot, Game#must_buy_train? for the operator: 'fresh'
    # on a fresh Graph, as the other oracles, and 'game' as the game's own
    # graph answers it from the route it has cached, whose
    # route_train_purchase survives Graph#clear; nil when nothing is cached
    # (Ruby would compute it fresh). Reading the cache leaves the game as it was.
    def must_buy_info(round)
      corp = round.current_operator
      return nil unless corp && round.active_step.is_a?(Engine::Step::BuyTrain)

      log_size = @game.log.size
      fresh = with_graph(Engine::Graph.new(@game)) { @game.must_buy_train?(corp) }
      cached = @game.instance_variable_get(:@graph).instance_variable_get(:@routes)[corp]
      game = cached && (corp.trains.empty? && !@game.depot.depot_trains.empty? && !!cached[:route_train_purchase])
      { 'fresh' => !!fresh, 'game' => game }
    ensure
      @game.log.slice!(log_size..) if log_size
    end

    # ------------------------------------------------------- track, tokens

    # The connectivity, token, and lay oracles for every corporation with a
    # placed token, answered by Ruby's own Graph, Tokener, and Tracker code.
    # They run on a fresh Graph swapped in for the game's, whose
    # route_train_purchase cache outlives Graph#clear and would otherwise
    # change what the replay does later; log lines Ruby writes while
    # answering are dropped.
    def network_snapshot(round)
      corps = @game.corporations.select { |c| c.tokens.any?(&:used) }
      # Ruby keeps Track open after the turn's lay while the president owns
      # MF (tile_lay_abilities_should_block?), but no corporation may lay.
      track = round.active_step
      track = nil unless track.is_a?(Engine::Step::Track) && round.num_laid_track.zero?
      log_size = @game.log.size
      with_graph(Engine::Graph.new(@game)) do |graph|
        {
          'graph' => corps.to_h { |c| [c.name, graph_info(graph, c)] },
          'tokens' => corps.to_h { |c| [c.name, tokenable_hexes(graph, c)] },
          'lays' => track && corps.to_h { |c| [c.name, legal_lays(track, c)] },
        }
      end
    ensure
      @game.log.slice!(log_size..) if log_size
    end

    def with_graph(graph)
      saved = @game.instance_variable_get(:@graph)
      @game.instance_variable_set(:@graph, graph)
      yield graph
    ensure
      @game.instance_variable_set(:@graph, saved)
    end

    # connected_hexes (hex => edges reached), connected_nodes (by hex), and
    # route_info's route_train_purchase.
    def graph_info(graph, corp)
      hexes = graph.connected_hexes(corp).select { |hex, _| hex }
      {
        'hexes' => hexes.sort_by { |hex, _| hex.id }.to_h { |hex, edges| [hex.id, edges.sort] },
        'nodes' => graph.connected_nodes(corp).keys.map { |node| node.hex.id }.uniq.sort,
        'route' => !!graph.route_info(corp)&.dig(:route_train_purchase),
      }
    end

    # Cities the corporation's Token step would accept now, setting aside
    # whose turn it is: City#tokenable? (a token left and affordable, a free
    # or reserved slot, no token of its own on the hex), reachable_hexes
    # (Token#available_hex), and connected_nodes (Tokener#check_connected).
    def tokenable_hexes(graph, corp)
      reachable = graph.reachable_hexes(corp)
      nodes = graph.connected_nodes(corp)
      @game.hexes.select do |hex|
        reachable[hex] && hex.tile.cities.any? { |city| nodes[city] && city.tokenable?(corp) }
      end.map(&:id).sort
    end

    # Every lay the Track step offers the corporation, setting aside whose
    # turn it is: [hex, tile, rotation, cost, ok] for each hex available_hex
    # accepts and each legal rotation of upgradeable_tiles there. cost is
    # what lay_tile charges (upgrade_cost, then tile_cost_with_discount); the
    # lay needs cost <= cash. ok is check_track_restrictions!, which
    # Game.load skips, run by laying the tile in place and taking it back up.
    def legal_lays(step, corp)
      @game.hexes.flat_map do |hex|
        next [] unless step.available_hex(corp, hex)

        cost = @game.upgrade_cost(hex.tile, hex, corp, corp)
        step.upgradeable_tiles(corp, hex).flat_map do |tile|
          tile.legal_rotations.map do |rotation|
            tile.rotate!(rotation)
            [hex.id, tile.name, rotation, @game.tile_cost_with_discount(tile, hex, corp, corp, cost),
             track_restrictions_ok?(step, corp, hex, tile)]
          end
        end
      end
    end

    # Step::Tracker#check_track_restrictions! for the rotated tile on hex: it
    # reads the corporation's graph with the tile laid, so the tile goes down
    # with Hex#lay (which moves tokens and reservations across) on a fresh
    # graph, with loading off so the check runs, and comes back up.
    def track_restrictions_ok?(step, corp, hex, tile)
      old_tile = hex.tile
      before = hex_snapshot(hex)
      loading = @game.loading
      with_graph(Engine::Graph.new(@game)) do
        @game.instance_variable_set(:@loading, false)
        hex.lay(tile)
        step.check_track_restrictions!(corp, old_tile, tile)
        true
      rescue Engine::GameError
        false
      ensure
        hex.lay(old_tile) if hex.tile == tile
        @game.instance_variable_set(:@loading, loading)
      end
    ensure
      raise AnnotationError, "probing #{tile.name} changed #{hex.id}" unless hex_snapshot(hex) == before
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
    # without logged revenues. AutoRouter's best is in the Route snapshot
    # before (operating.auto_router).
    def run_routes_info(action)
      info = { 'revenue' => @game.routes_revenue(action.routes), 'trains' => action.routes.map { |r| r.train.id } }
      info.merge(recompute_routes(action))
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

    # Engine::AutoRouter#compute for the operator at a Route step, without
    # the browser: its path walk is Ruby, but its route combination
    # (Autorouter#find_best_combo) is JavaScript run through Opal. This ports
    # that search: trains are tried last first, a route may not share a
    # hexside bit with an earlier choice, a prefix is cut when its estimate
    # plus the best single routes of the remaining trains cannot beat the best
    # total, and a full combination is scored by real_revenue. 1889 has no
    # autoroute groups, so there is one bitfield. It runs on a fresh Graph, as
    # the network oracles do, with AutoRouter's own limits: route_limit keeps
    # each train's 10,000 best routes (candidates counts them), and the path
    # walk stops after 30 s (timed_out).
    def auto_router_info(corp)
      log_size = @game.log.size
      with_graph(Engine::Graph.new(@game)) do
        router = Engine::AutoRouter.new(@game)
        trains = @game.route_trains(corp).sort_by(&:price)
        train_routes, timed_out = router.path(trains, corp)
        best = best_combination(router, train_routes)
        {
          'revenue' => best[:revenue],
          'routes' => best[:routes].map do |r|
            { 'train' => r.train.id, 'revenue' => r.revenue, 'hexes' => r.hexes.map(&:id),
              'ordered_hexes' => r.ordered_hexes.map(&:id) }
          end,
          'timed_out' => timed_out,
          'candidates' => train_routes.to_h { |train, routes| [train.id, routes.size] },
        }
      end
    ensure
      @game.log.slice!(log_size..) if log_size
    end

    def best_combination(router, train_routes)
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
              best = { revenue: real, routes: combo.map(&:clone) } if real > best[:revenue]
            end
          elsif next_estimate + rest[i + 1] > best[:revenue]
            search.call(i + 1, next_estimate, next_bits)
          end
          combo.pop if route
        end
      end
      search.call(0, 0, []) unless lists.empty?
      best[:routes].each { |r| r.routes = best[:routes] }
      best
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

  # Ruby's Graph on hand-built maps. Each spec's ops run in order on a new
  # 4-player game: ['lay', hex, tile, rotation] lays a supply tile with
  # Hex#lay and the supply bookkeeping, and ['token', corporation, hex]
  # places the corporation's next token free. Records graph_info for every
  # corporation with a token, and for each lay in the spec's probe
  # ({corporation: [[hex, tile, rotation], ...]}) whether
  # check_track_restrictions! accepts it. An optional phase (its name) is
  # entered with Phase#next!, and trains ({corporation: [train name, ...]})
  # are new Train objects the corporation owns; routes then holds
  # auto_router_info for each corporation with trains.
  def extract_graphs(path)
    # Step::Tracker's checks outside a round; the engine loads after this file.
    checker_class = Class.new do
      include Engine::Step::Tracker

      def initialize(game)
        @game = game
      end
    end
    specs = JSON.parse(File.read(path))
    graphs = specs.map do |spec|
      game = Engine::Game::G1889::Game.new((1..4).to_h { |i| [i, "Player #{i}"] }, id: 'graphs', seed: 1)
      corps = []
      spec['ops'].each do |kind, *args|
        if kind == 'lay'
          hex_id, name, rotation = args
          hex = game.hex_by_id(hex_id)
          tile = game.tiles.find { |t| t.name == name } || raise("#{name} is not in supply")
          tile.rotate!(rotation)
          game.update_tile_lists(tile, hex.tile)
          hex.lay(tile)
        else
          corp = game.corporation_by_id(args[0])
          game.hex_by_id(args[1]).tile.cities.first.place_token(corp, corp.next_token, free: true,
                                                                  check_tokenable: false)
          corps |= [corp]
        end
      end
      game.phase.next! while spec['phase'] && game.phase.name != spec['phase']
      (spec['trains'] || {}).each do |sym, names|
        corp = game.corporation_by_id(sym)
        names.each do |name|
          train = Engine::Train.new(**game.class::TRAINS.find { |t| t[:name] == name })
          train.owner = corp
          corp.trains << train
        end
      end
      reader = Reader.new(game, player_ids(game))
      graph = Engine::Graph.new(game)
      checker = checker_class.new(game)
      {
        'routes' => (spec['trains'] || {}).to_h do |sym, _|
          [sym, reader.auto_router_info(game.corporation_by_id(sym))]
        end,
        'graph' => corps.to_h { |c| [c.name, reader.graph_info(graph, c)] },
        'restrictions' => (spec['probe'] || {}).to_h do |sym, lays|
          corp = game.corporation_by_id(sym)
          [sym, lays.map do |hex_id, name, rotation|
            tile = game.tiles.find { |t| t.name == name }
            tile.rotate!(rotation)
            reader.track_restrictions_ok?(checker, corp, game.hex_by_id(hex_id), tile)
          end]
        end,
      }
    end
    { 'schema' => SCHEMA_VERSION, 'source' => { 'kind' => 'graphs' }, 'graphs' => graphs }
  end

  def parse_id(text)
    text.match?(/\A-?\d+\z/) ? text.to_i : text
  end

  def main(argv)
    if argv.first == '--graphs'
      abort "usage: ruby #{$PROGRAM_NAME} --graphs specs.json" unless argv.size == 2
      load_engine
      extract = extract_graphs(argv[1])
    elsif argv.first == '--new-game'
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

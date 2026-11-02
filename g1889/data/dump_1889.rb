# frozen_string_literal: true

# Dumps Shikoku 1889 static data from the 18xx.games engine in the submodule as
# JSON on stdout. It reads instantiated engine objects (hexes, tiles, parts,
# corporations, the stock market) and the game's own constants; it never parses
# Ruby source. g1889/data/generate.py runs it and derives the checked-in tables.
#
#   ruby g1889/data/dump_1889.rb > dump.json
#
# tests/g1889/upgrade_oracle.rb requires this file for load_engine and
# new_game.

require 'json'
require 'logger'

module G1889Dump
  ROOT = File.expand_path('../..', __dir__)
  PLAYER_COUNTS = (2..6).freeze
  # Base-engine and 1889 rule constants worth carrying into the engine step.
  RULE_CONSTANTS = %i[
    BANK_CASH CAPITALIZATION MUST_SELL_IN_BLOCKS SELL_AFTER SELL_MOVEMENT SELL_BUY_ORDER
    POOL_SHARE_DROP SOLD_OUT_INCREASE MARKET_SHARE_LIMIT PRESIDENT_SALES_TO_MARKET
    CERT_LIMIT_INCLUDES_PRIVATES CERT_LIMIT_TYPES NEXT_SR_PLAYER_ORDER MIN_BID_INCREMENT
    HOME_TOKEN_TIMING TRACK_RESTRICTION TILE_LAYS IMPASSABLE_HEX_COLORS MUST_BUY_TRAIN
    EBUY_FROM_OTHERS EBUY_PRES_SWAP EBUY_DEPOT_TRAIN_MUST_BE_CHEAPEST DISCARDED_TRAINS
    GAME_END_CHECK BANKRUPTCY_ENDS_GAME_AFTER
  ].freeze
  # Stand-ins for the phase and train Part::RevenueCenter#route_base_revenue reads.
  PhaseStub = Struct.new(:tiles)
  TrainStub = Struct.new(:name)

  module_function

  # Loads the engine with its stdout logger silenced.
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

  def new_game(num_players)
    Engine::Game::G1889::Game.new((1..num_players).map { |i| "p#{i}" })
  end

  def revenue_centers(tile)
    tile.cities + tile.towns + tile.offboards
  end

  # Path end as an endpoint: sides 0-5, or 6 + index into revenue_centers.
  def path_end(tile, part)
    return part.num if part.edge?

    idx = revenue_centers(tile).index(part)
    raise "#{tile.name}: unsupported path end #{part.class}" unless idx

    6 + idx
  end

  def dump_tile(game, tile)
    tile.rotate!(0)
    phases = game.phase.phases
    {
      name: tile.name,
      color: tile.color,
      label: tile.label&.to_s,
      blocks_lay: !!tile.blocks_lay,
      paths: tile.paths.map do |p|
        raise "#{tile.name}: unsupported path" if p.lanes != Engine::Part::Path::LANES || p.track != :broad

        [path_end(tile, p.a), path_end(tile, p.b)]
      end,
      exits: tile.exits.sort,
      revenue_centers: revenue_centers(tile).map do |rc|
        {
          type: rc.type,
          slots: rc.city? ? rc.slots : 0,
          # Base revenue in each phase for a non-D train, then for a D train in
          # the last phase.
          phase_revenue: phases.map { |ph| rc.route_base_revenue(PhaseStub.new(ph[:tiles]), TrainStub.new('2')) },
          d_revenue: rc.route_base_revenue(PhaseStub.new(phases.last[:tiles]), TrainStub.new('D')),
        }
      end,
      upgrades: tile.upgrades.map { |u| { cost: u.cost, terrains: u.terrains } },
      icons: tile.icons.map { |i| { image: i.image, blocks_lay: i.blocks_lay? } },
      # Parts the generator does not model; it fails if any appear.
      unsupported: {
        junction: !tile.junction.nil?, borders: tile.borders.size, stubs: tile.stubs.size,
        partitions: tile.partitions.size, future_label: !tile.future_label.nil?, halts: tile.halts.size,
      },
    }
  end

  def dump_hex(game, hex)
    tile = hex.tile
    {
      id: hex.id,
      x: hex.x,
      y: hex.y,
      location_name: hex.location_name,
      tile: dump_tile(game, tile),
      neighbors: hex.all_neighbors.transform_values(&:id),
      # Sides track may leave through: Hex#neighbors drops impassable neighbors.
      open_sides: hex.neighbors.keys.sort,
      blockers: tile.blockers.map(&:sym),
      reservations: tile.cities.flat_map { |c| c.reservations.compact.map(&:name) } + tile.reservations.map(&:name),
    }
  end

  def dump_market(game)
    game.stock_market.market.map do |row|
      row.map { |sp| sp && { price: sp.price, types: sp.types } }
    end
  end

  def dump(game)
    by_count = PLAYER_COUNTS.to_h do |n|
      g = new_game(n)
      [n, {
        privates: g.companies.map(&:sym),
        cert_limit: g.cert_limit,
        starting_cash: g.players.first.cash,
      }]
    end
    {
      layout: game.layout,
      hexes: game.hexes.map { |h| dump_hex(game, h) },
      tiles: game.tiles.group_by(&:name).map { |_, ts| dump_tile(game, ts.first).merge(count: ts.size) },
      corporations: game.corporations.map do |c|
        {
          sym: c.name, name: c.full_name, home: c.coordinates, city: c.city, token_prices: c.tokens.map(&:price),
          float_percent: c.float_percent, max_ownership_percent: c.max_ownership_percent, capitalization: c.capitalization,
        }
      end,
      companies: game.class::COMPANIES,
      trains: game.class::TRAINS,
      phases: game.class::PHASES,
      market: dump_market(game),
      players: by_count,
      rules: RULE_CONSTANTS.to_h { |name| [name, game.class.const_get(name)] },
    }
  end
end

if $PROGRAM_NAME == __FILE__
  G1889Dump.load_engine
  puts JSON.pretty_generate(G1889Dump.dump(G1889Dump.new_game(4)))
end

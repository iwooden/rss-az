# frozen_string_literal: true

# The 18xx engine's own tile upgrade check, for tests/g1889/test_data_vs_ruby.py.
#
# A pair (old tile, rotation) -> (new tile, rotation) on a hex is legal when
# Game::Base#upgrades_to? and Step::Tracker#legal_tile_rotation? both accept it.
# Old tiles are each hex's preprinted tile and every supply tile at every
# rotation; new tiles are every supply tile at every rotation. Reachability is
# game state, so hex_neighbors reports every side as reached.
# legal_tile_rotation? reads only the hex's tile and Hex#neighbors, so supply
# tiles run once per distinct neighbor set. Prints JSON:
#
#   preprinted: [[hex, new, rotation], ...]
#   supply:     {"<open sides>": [[old, rotation, new, rotation], ...]}
#   abilities:  [[private, hex, new, rotation], ...]  (tile_lay abilities, as SpecialTrack checks them)

require_relative '../../g1889/data/dump_1889'

G1889Dump.load_engine

class UpgradeChecker
  include Engine::Step::Tracker

  def initialize(game)
    @game = game
  end

  def hex_neighbors(_entity, _hex)
    Engine::Tile::ALL_EDGES
  end

  # [new name, rotation] for each tile in tiles that may replace hex.tile.
  def legal(entity, hex, tiles, special: false)
    tiles.flat_map do |tile|
      next [] unless @game.upgrades_to?(hex.tile, tile, special)

      Engine::Tile::ALL_EDGES
        .select { |rotation| legal_tile_rotation?(entity, hex, tile.rotate!(rotation)) }
        .map { |rotation| [tile.name, rotation] }
    end
  end
end

game = G1889Dump.new_game(4)
checker = UpgradeChecker.new(game)
corporation = game.corporations.first
supply = game.tiles.uniq(&:name)
olds = supply.map(&:dup)

preprinted = game.hexes.flat_map do |hex|
  checker.legal(corporation, hex, supply).map { |name, rotation| [hex.id, name, rotation] }
end

by_sides = {}
game.hexes.each do |hex|
  key = hex.neighbors.keys.sort.join
  next if by_sides.key?(key)

  printed = hex.tile
  by_sides[key] = olds.flat_map do |old|
    Engine::Tile::ALL_EDGES.flat_map do |rotation|
      hex.tile = old.rotate!(rotation)
      checker.legal(corporation, hex, supply).map { |name, new_rotation| [old.name, rotation, name, new_rotation] }
    end
  end
  hex.tile = printed
end

abilities = game.companies.flat_map do |company|
  company.all_abilities.select { |a| a.type == :tile_lay }.flat_map do |ability|
    tiles = supply.select { |t| ability.tiles.include?(t.name) }
    ability.hexes.flat_map do |id|
      checker.legal(company, game.hex_by_id(id), tiles, special: ability.special)
        .map { |name, rotation| [company.sym, id, name, rotation] }
    end
  end
end

puts JSON.generate(preprinted: preprinted, supply: by_sides, abilities: abilities)

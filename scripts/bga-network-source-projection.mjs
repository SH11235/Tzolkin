const copy = structuredClone;

export function sourceCheckpoint(ledger, sourceToNative) {
  const players = Array.from({ length: sourceToNative.size });
  for (const [id, p] of Object.entries(ledger.players))
    players[sourceToNative.get(id)] = {
      id: sourceToNative.get(id),
      resources: copy(p.resources),
      score: p.score,
      workers: p.workers,
      temples: copy(p.temples),
      technologies: copy(p.technologies),
      buildings: copy(p.buildings),
      monuments: copy(p.monuments),
      wealth: [...p.wealth].sort(),
      feedWorkers: p.feedWorkers,
      feedDiscount: p.feedDiscount,
      cornTiles: p.cornTiles,
      woodTiles: p.woodTiles,
      skullsPlaced: p.skullsPlaced,
    };
  const gears = Object.fromEntries(
    Object.entries(ledger.initial.gears).map(([gear, def]) => [
      gear,
      Array.from({ length: Number(def.teethnbr) }, () => null),
    ]),
  );
  for (const w of ledger.workers.values())
    if (w.place && w.place !== 'startingPlayer')
      gears[w.place][Number(w.position)] =
        w.player === null
          ? { playerId: -1, dummy: true }
          : { playerId: sourceToNative.get(String(w.player)) };
  const claimed = [...ledger.workers.values()].find((w) => w.place === 'startingPlayer');
  return {
    players,
    gears,
    jungle: copy(ledger.jungle),
    skullSupply: ledger.skullSupply,
    skullSpaces: ledger.skullSpaces.map((id) => (id === null ? null : sourceToNative.get(id))),
    buildings: copy(ledger.buildings),
    monuments: copy(ledger.monuments),
    firstPlayer: sourceToNative.get(ledger.firstPlayer),
    firstPlayerClaimed: claimed ? sourceToNative.get(String(claimed.player)) : null,
    accumulatedCorn: ledger.accumulatedCorn,
  };
}

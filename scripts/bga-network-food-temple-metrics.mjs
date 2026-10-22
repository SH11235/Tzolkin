export function foodCost(player, corn) {
  const free = player.feedAll ? player.workers : Math.min(player.workers, player.feedWorkers);
  const price = Math.max(0, 2 - player.feedDiscount);
  const fed =
    price === 0 ? player.workers - free : Math.min(player.workers - free, Math.floor(corn / price));
  return { free, price, fed, cost: fed * price, unfed: player.workers - free - fed };
}

export function templePositionPoints(player, tracks) {
  return Object.entries(player.temples).reduce(
    (sum, [temple, level]) => sum + tracks[temple].points[level + 1],
    0,
  );
}

export function templeLeaderAwards(players, tracks, age) {
  const result = [];
  for (const [temple, track] of Object.entries(tracks)) {
    const highest = Math.max(...Object.values(players).map((player) => player.temples[temple]));
    const winners = Object.entries(players).filter(
      ([, player]) => player.temples[temple] === highest,
    );
    const points = track[age === 1 ? 'age1Bonus' : 'age2Bonus'] / (winners.length > 1 ? 2 : 1);
    for (const [sourceActor] of winners)
      result.push({ sourceActor, temple, points, winnerCount: winners.length });
  }
  return result;
}

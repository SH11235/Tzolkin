import { isDeepStrictEqual } from 'node:util';
import { originalFinalScoringCaption } from './bga-network.mjs';

const fail = (message) => {
  throw new Error(`Network final scoring: ${message}`);
};
const suffixes = {
  graveyards: 'Tombs',
  municipals: 'Civic buildings',
  shrines: 'Shrines',
  constructions: 'buildings and monuments',
  cornTiles: 'Corn Harvest Tile',
  woodTiles: 'Wood Harvest Tile',
  technologyLevels: 'technology level',
  workers: 'workers in play',
  highestTemple: 'steps above the starting space',
  allSkulls: 'Skulls on Chichen Itza',
};
const clockTypes = new Set(['updateReflexionTime', 'timeJokerUsed']);
const integer = (value) => {
  if (
    !['number', 'string'].includes(typeof value) ||
    !/^(0|[1-9]\d*)$/.test(String(value)) ||
    !Number.isSafeInteger(Number(value))
  )
    fail('noncanonical receipt integer');
  return Number(value);
};
const exactKeys = (value, keys) =>
  value && isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort());
const reference = (n) => ({
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
  uid: n.uid,
});

export function finalMonumentMetric(state, player, card, catalog) {
  const key = card.scoreKey;
  if (!Object.hasOwn(suffixes, key)) fail('unreviewed monument metric');
  let nbr, points;
  if (['graveyards', 'municipals', 'shrines'].includes(key)) {
    const category = { graveyards: 'graveyard', municipals: 'municipal', shrines: 'shrine' }[key];
    nbr = [...player.buildings, ...player.monuments].filter((id) => {
      const definitions = [...catalog.BUILDINGS, ...catalog.MONUMENTS].filter((c) => c.id === id);
      if (definitions.length !== 1) fail('unknown construction identity');
      return definitions[0].category === category;
    }).length;
    points = nbr * 4;
  } else if (key === 'constructions') {
    nbr = player.buildings.length + player.monuments.length;
    points = nbr * 2;
  } else if (key === 'cornTiles' || key === 'woodTiles') {
    nbr = player[key];
    points = nbr * 4;
  } else if (key === 'technologyLevels') {
    nbr = Object.values(player.technologies).reduce((sum, level) => sum + level, 0);
    points = nbr * 3;
  } else if (key === 'workers') {
    nbr = player.workers;
    points = (nbr - 3) * 6;
  } else if (key === 'highestTemple') {
    nbr = Math.max(0, ...Object.values(player.temples));
    points = nbr * 3;
  } else {
    nbr = state.skullSpaces.filter((s) => s !== null).length;
    points = nbr * 3;
  }
  integer(nbr);
  integer(points);
  return { nbr, points };
}

export function finalResourceComponents(resources) {
  return {
    resourcePoints:
      (resources.corn + resources.wood * 2 + resources.stone * 3 + resources.gold * 4) / 4,
    skullPoints: resources.skull * 3,
  };
}

/** Bind complete original receipts before probing any legal calendar outcome. */
export function finalScoringWitness({
  group,
  before,
  state,
  observation,
  sourceToNative,
  scoresBeforeGroup,
  sourceBoard,
  differences,
  unresolvedTiming,
  catalog,
}) {
  const entries = group.entries;
  const states = entries.filter((n) => n.type === 'gameStateChange');
  const headers = entries.filter((n) => n.type === 'finalScoring' && Array.isArray(n.args));
  const claimed = state.firstPlayerClaimed;
  if (
    unresolvedTiming ||
    before.phase !== 'playing' ||
    before.round !== 27 ||
    before.pending !== null ||
    before.turnIndex !== before.turnOrder.length - 1 ||
    !isDeepStrictEqual(before.foodDays, [8, 14, 21]) ||
    state.phase !== 'playing' ||
    state.additionalBuildings !== false ||
    state.expansion != null ||
    state.players.some((p) => p.buildingSkulls !== 0) ||
    !isDeepStrictEqual(state.foodDays, [8, 14, 21, 27]) ||
    state.pending?.task.type !== 'rotation' ||
    !isDeepStrictEqual(state.pending.after, []) ||
    claimed === null ||
    observation?.actor !== claimed ||
    !observation.legalActions.length ||
    observation.legalActions.some((row) => row.action.type !== 'rotate') ||
    !isDeepStrictEqual(
      states.map((n) => Number(n.args.id)),
      [90, 91, 92, 94, 95],
    ) ||
    sourceToNative.get(String(states[0]?.args.active_player)) !== before.currentPlayer ||
    sourceToNative.get(String(states[1]?.args.active_player)) !== before.firstPlayer ||
    states.slice(2).some((n) => sourceToNative.get(String(n.args.active_player)) !== claimed) ||
    headers.length !== 1 ||
    !differences.some((d) => /^player\.\d+\.score$/.test(d.field)) ||
    differences.some(
      (d) =>
        !/^player\.\d+\.score$/.test(d.field) &&
        !(d.field === 'firstPlayer' && d.native === before.firstPlayer),
    )
  )
    fail('boundary is outside the reviewed final-food-day profile');
  const header = headers[0],
    end = states.at(-1);
  if (
    originalFinalScoringCaption(header)?.caption !== 'Final scoring!' ||
    header.args.length !== 0 ||
    entries.indexOf(header) <= entries.indexOf(states[3]) ||
    entries.indexOf(end) <= entries.indexOf(header) ||
    entries.some(
      (n, i) =>
        n.packetIndex !== header.packetIndex ||
        n.moveId !== group.moveId ||
        n.notificationIndex !== entries[0].notificationIndex + i,
    )
  )
    fail('final header is not an original closed packet');
  const prefix = entries.slice(0, entries.indexOf(header));
  const prefixTypes = new Set([
    'gameStateChange',
    'takeGoods',
    'payGoods',
    'firstPlayerCorn',
    'firstPlayer',
    'foodDayCost',
    'updateScore',
    'templeLeader',
    'templeScore',
  ]);
  if (
    prefix.some((n) => !prefixTypes.has(n.type) && !clockTypes.has(n.type)) ||
    entries.slice(entries.indexOf(end) + 1).some((n) => !clockTypes.has(n.type))
  )
    fail('foreign effects surround final receipts');
  const gamePrefix = prefix.filter((n) => !clockTypes.has(n.type));
  const payout = gamePrefix[3];
  if (
    !isDeepStrictEqual(
      gamePrefix.slice(0, 6).map((n) => n.type),
      [
        'gameStateChange',
        'gameStateChange',
        'takeGoods',
        'firstPlayerCorn',
        'firstPlayer',
        'gameStateChange',
      ],
    ) ||
    gamePrefix[2].args.type !== 'corn' ||
    sourceToNative.get(String(gamePrefix[2].args.player_id)) !== claimed ||
    sourceToNative.get(String(payout?.args.player_id)) !== claimed ||
    sourceToNative.get(String(gamePrefix[4].args.player_id)) !== claimed ||
    integer(gamePrefix[2].args.quantity) !== integer(payout.args.corn)
  )
    fail('final food day lacks its closed original corn payout');
  const foodEnd = gamePrefix.indexOf(states[3]);
  const feeding = gamePrefix.slice(6, foodEnd);
  const temple = gamePrefix.slice(foodEnd + 1);
  for (const [section, phase] of [
    [feeding, 'feeding'],
    [temple, 'temple'],
  ]) {
    if (section.length % 2) fail('unpaired food-day effect');
    for (let i = 0; i < section.length; i += 2) {
      const effect = section[i],
        caption = section[i + 1];
      if (
        !sourceToNative.has(String(effect.args.player_id)) ||
        String(effect.args.player_id) !== String(caption.args.player_id) ||
        effect.notificationIndex + 1 !== caption.notificationIndex
      )
        fail('food-day effect and caption actors differ');
      if (phase === 'feeding') {
        if (caption.type !== 'foodDayCost') fail('unknown feeding receipt');
        if (effect.type === 'payGoods') {
          if (
            effect.args.type !== 'corn' ||
            integer(effect.args.quantity) !== integer(caption.args.cost)
          )
            fail('foreign resource effect in final feeding');
        } else if (
          effect.type !== 'updateScore' ||
          !Number.isSafeInteger(effect.args.points) ||
          effect.args.points !== -integer(caption.args.points)
        )
          fail('feeding penalty differs from original receipt');
      } else if (
        effect.type !== 'updateScore' ||
        !['templeLeader', 'templeScore'].includes(caption.type) ||
        integer(effect.args.points) !== integer(caption.args.points)
      )
        fail('temple effect differs from original receipt');
    }
  }
  const scores = structuredClone(scoresBeforeGroup);
  for (const n of prefix.filter((n) => n.type === 'updateScore')) {
    const id = String(n.args.player_id);
    if (!Object.hasOwn(scores, id) || !Number.isSafeInteger(n.args.points))
      fail('invalid pre-header score increment');
    scores[id] += n.args.points;
    if (!Number.isSafeInteger(scores[id])) fail('unsafe pre-header score');
  }
  return finalScoreReceiptComponents({
    entries,
    header,
    end,
    state,
    sourceToNative,
    scores,
    sourceBoard,
    catalog,
  });
}

export function finalScoreReceiptComponents({
  entries,
  header,
  end,
  state,
  sourceToNative,
  scores,
  sourceBoard,
  catalog,
}) {
  const players = new Map();
  for (const [id, actor] of sourceToNative) {
    const p = state.players[actor],
      source = sourceBoard.players[actor];
    for (const field of [
      'resources',
      'technologies',
      'temples',
      'buildings',
      'monuments',
      'workers',
      'cornTiles',
      'woodTiles',
    ])
      if (!isDeepStrictEqual(p[field], source[field])) fail('metric inputs differ from source');
    if (
      p.score !== scores[id] ||
      new Set([...p.buildings, ...p.monuments]).size !== p.buildings.length + p.monuments.length
    )
      fail('independent pre-header score or owned identities differ');
    players.set(id, {
      actor,
      before: scores[id],
      monuments: [],
      resource: null,
      remaining: [...p.monuments],
    });
  }
  if (!isDeepStrictEqual(state.skullSpaces, sourceBoard.skullSpaces))
    fail('source skull positions differ');
  const receipts = entries.slice(entries.indexOf(header) + 1, entries.indexOf(end));
  let resourcesStarted = false,
    resourceActor = 0;
  if (receipts.length % 2) fail('unpaired final receipt');
  for (let i = 0; i < receipts.length; i += 2) {
    const update = receipts[i],
      receipt = receipts[i + 1];
    const bound = originalFinalScoringCaption(receipt);
    const id = String(receipt.args.player_id),
      player = players.get(id);
    const hasMetric = Object.hasOwn(receipt.args, 'nbr');
    if (
      update.type !== 'updateScore' ||
      receipt.type !== 'finalScoring' ||
      !bound ||
      !player ||
      update.packetIndex !== receipt.packetIndex ||
      update.notificationIndex + 1 !== receipt.notificationIndex ||
      !exactKeys(update.args, ['player_id', 'player_name', 'points']) ||
      !exactKeys(receipt.args, [
        'player_id',
        'player_name',
        'points',
        ...(hasMetric ? ['nbr'] : []),
      ]) ||
      String(update.args.player_id) !== id ||
      update.args.player_name !== null ||
      receipt.args.player_name !== bound.playerName ||
      integer(update.args.points) !== integer(receipt.args.points)
    )
      fail('receipt actor, name, shape or adjacent score increment differs');
    const points = integer(receipt.args.points);
    if (hasMetric) {
      if (resourcesStarted) fail('monument receipt follows resource receipts');
      const nbr = integer(receipt.args.nbr),
        p = state.players[player.actor];
      const matches = player.remaining.filter((id) => {
        const cards = catalog.MONUMENTS.filter((c) => c.id === id);
        if (cards.length !== 1) fail('unknown monument identity');
        const card = cards[0],
          expected = finalMonumentMetric(state, p, card, catalog);
        return (
          bound.caption ===
            'Monument: ${player_name} scores ${points} for ${nbr} ' + suffixes[card.scoreKey] &&
          nbr === expected.nbr &&
          points === expected.points
        );
      });
      if (matches.length !== 1) fail('missing, duplicate, ambiguous or incorrect monument receipt');
      player.remaining.splice(player.remaining.indexOf(matches[0]), 1);
      player.monuments.push({ monument: matches[0], nbr, points, source: reference(receipt) });
    } else {
      resourcesStarted = true;
      if (
        bound.caption !== 'Remaining resources: ${player_name} scores ${points} points' ||
        player.resource !== null ||
        player.actor !== resourceActor++
      )
        fail('resource receipt identity, order or multiplicity differs');
      player.resource = { points, source: reference(receipt) };
    }
  }
  for (const [id, player] of players) {
    const source = sourceBoard.players[player.actor];
    if (player.remaining.length || player.resource === null) fail('incomplete final receipt set');
    const total =
      player.before +
      player.resource.points +
      player.monuments.reduce((sum, r) => sum + r.points, 0);
    if (!Number.isSafeInteger(total) || source.score !== total)
      fail('source receipt totals differ');
    player.total = total;
    delete player.remaining;
    player.sourceActor = id;
  }
  return { header: reference(header), players: [...players.values()] };
}

export function bindFinalScoringProjections(witness, state, responses) {
  if (!responses.length) fail('missing legal rotation projections');
  for (const response of responses) {
    const after = response.snapshot?.state;
    if (
      response.error ||
      after?.phase !== 'finished' ||
      after.finalScores.length !== state.players.length
    )
      fail('legal rotation does not reach complete final scoring');
    for (const player of witness.players) {
      const rows = after.finalScores.filter((r) => r.playerId === player.actor);
      const score = rows[0];
      if (
        rows.length !== 1 ||
        score.pointsBeforeFinal !== player.before ||
        score.monumentPoints !== player.monuments.reduce((sum, r) => sum + r.points, 0) ||
        Math.floor(score.resourcePoints + score.skullPoints) !== player.resource.points ||
        Math.floor(score.total) !== player.total ||
        !isDeepStrictEqual(
          after.players[player.actor].resources,
          state.players[player.actor].resources,
        ) ||
        !isDeepStrictEqual(
          after.players[player.actor].monuments,
          state.players[player.actor].monuments,
        )
      )
        fail('legal projection disagrees with individual original receipts');
    }
  }
  return witness;
}

export function assertFinalScoringResolutionGroup(group) {
  const entries = group.entries;
  const rotations = entries.filter((n) => n.type === 'rotateWheel');
  const terminals = entries.filter((n) => n.type === 'gameStateChange');
  const allowed = new Set([
    'flipBoard',
    'rotateWheel',
    'pickUpWorker',
    'gameStateChange',
    'simpleNode',
    ...clockTypes,
  ]);
  if (
    rotations.length !== 1 ||
    terminals.length !== 1 ||
    Number(terminals[0].args.id) !== 99 ||
    entries.indexOf(terminals[0]) <= entries.indexOf(rotations[0]) ||
    entries.some((n) => !allowed.has(n.type)) ||
    entries.some(
      (n) =>
        n.type === 'pickUpWorker' && (n.args.city !== undefined || n.args.position !== undefined),
    )
  )
    fail('final receipt fence requires the next original rotation and terminal only');
}

import { readFileSync } from 'node:fs';
import { isDeepStrictEqual } from 'node:util';
import { assertNetworkClockContract, inspectCancellationClusters } from './bga-network.mjs';
import { sourceFeedingWitnesses } from './bga-network-feeding.mjs';
import {
  appendRewardFence,
  bindNewFixedReward,
  proveFixedBuildingPoints,
  reconcileRewardFences,
  rewardFields,
  trackRewardQueues,
} from './bga-network-rewards.mjs';
import { catalogFingerprint } from './bga-replay.mjs';

const catalogBytes = readFileSync(
  new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url),
);
const catalog = JSON.parse(catalogBytes);
const R = ['corn', 'wood', 'stone', 'gold', 'skull'];
const TECH = ['agriculture', 'extraction', 'architecture', 'theology'];
const TEMPLE = ['chaac', 'quetzalcoatl', 'kukulcan'];
const temple = (x) => (x === 'kukulcan' ? 'kukulkan' : x);
const category = (x) => ({ tomb: 'graveyard', civic: 'municipal' })[x] ?? x;
const copy = structuredClone;
const sourcePosition = (value, teeth) =>
  ['number', 'string'].includes(typeof value) &&
  /^(0|[1-9]\d*)$/.test(String(value)) &&
  Number.isSafeInteger(Number(value)) &&
  Number.isSafeInteger(teeth) &&
  Number(value) < teeth;
const fail = (message) => {
  throw new Error(`Network native: ${message}`);
};
const amounts = (x) => R.map((k) => Number(x?.[k] ?? 0));
const canonical = (x) =>
  Array.isArray(x)
    ? x.map(canonical)
    : x && typeof x === 'object'
      ? Object.fromEntries(
          Object.entries(x)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => [k, canonical(v)]),
        )
      : x;
const key = (x) => JSON.stringify(canonical(x));

function effectsSignature(effects) {
  const counts = {};
  const add = (k, n = 1) => {
    counts[k] = (counts[k] ?? 0) + n;
  };
  for (const e of effects) {
    if (e.type === 'technology') add(`technology:${e.technology}`, e.steps);
    else if (e.type === 'temple') add(`temple:${e.temple}`, e.steps);
    else if (e.type === 'resources')
      for (const [r, n] of Object.entries(e.resources)) add(`resource:${r}`, n);
    else if (e.type === 'points') add('points', e.amount);
    else if (e.type === 'feed') add('feed', e.workers);
    else if (e.type === 'feedDiscount') add('feedDiscount', e.amount);
    else if (['worker', 'build', 'trade', 'action'].includes(e.type)) add(e.type);
    else fail(`unsupported native effect ${e.type}`);
  }
  return key(counts);
}

function sourceEffects(def) {
  const effects = [];
  const known = new Set([
    'cost',
    'quantity',
    'kind',
    'description',
    'dummyWorkers',
    'actions',
    'farm',
    'bigFarm',
    'points',
    'worker',
    ...R,
    ...TECH,
    ...TEMPLE,
  ]);
  if (Object.keys(def).some((k) => !known.has(k))) fail('unsupported source definition effect');
  for (const k of ['quantity', 'farm', 'bigFarm', 'points', 'worker', ...R, ...TECH, ...TEMPLE])
    if (def[k] !== undefined && (!Number.isSafeInteger(def[k]) || def[k] < 0 || def[k] > 32))
      fail('invalid source effect amount');
  if (def.actions !== undefined && (!Array.isArray(def.actions) || def.actions.length > 4))
    fail('unsupported source action list');
  for (const k of TECH)
    if (def[k]) effects.push({ type: 'technology', technology: k, steps: Number(def[k]) });
  for (const k of TEMPLE)
    if (def[k]) effects.push({ type: 'temple', temple: temple(k), steps: Number(def[k]) });
  const resources = Object.fromEntries(R.filter((k) => def[k]).map((k) => [k, Number(def[k])]));
  if (Object.keys(resources).length) effects.push({ type: 'resources', resources });
  if (def.farm) effects.push({ type: 'feed', workers: Number(def.farm) });
  if (def.bigFarm) effects.push({ type: 'feedDiscount', amount: Number(def.bigFarm) });
  if (def.points) effects.push({ type: 'points', amount: Number(def.points) });
  if (def.worker)
    effects.push(...Array.from({ length: Number(def.worker) }, () => ({ type: 'worker' })));
  for (const a of def.actions ?? []) {
    if (a === 'freeTechnology') effects.push({ type: 'technology', technology: 'any', steps: 1 });
    else if (a === 'templeReward') effects.push({ type: 'temple', temple: 'any', steps: 1 });
    else if (a === 'construct') effects.push({ type: 'build' });
    else if (a === 'anyAction') effects.push({ type: 'action' });
    else if (a === 'trade') effects.push({ type: 'trade' });
    else fail(`unsupported source action ${a}`);
  }
  return effectsSignature(effects);
}

function monumentKey(def) {
  if (typeof def.description !== 'string') fail('missing monument scoring description');
  const s = def.description.toLowerCase();
  const rules = [
    ['cornTiles', s.includes('corn harvest tile')],
    ['woodTiles', s.includes('wood harvest tile')],
    ['allMonuments', s.includes('regardless of who')],
    ['graveyards', s.includes('gray-bordered')],
    ['municipals', s.includes('green-bordered')],
    ['shrines', s.includes('blue-bordered')],
    ['workers', s.includes('number of workers')],
    ['technologyLevels', s.includes('each level of technology')],
    ['maxTechnologies', s.includes('third level')],
    ['highestTemple', s.includes('choose one temple')],
    ['templePoints', s.includes('in each temple')],
    ['allSkulls', s.includes('each skull placed')],
    ['constructions', s.includes('each building you constructed')],
  ].filter(([, v]) => v);
  if (rules.length !== 1) fail('unknown/ambiguous monument definition');
  return rules[0][0];
}

/** Preserve physical instances while matching full cost/category/effect semantics. */
export function networkCardMapper(initial, seed = null) {
  const assigned = new Map();
  const used = new Set();
  function candidates(card) {
    const def = initial.constructions?.[card.type]?.[card.type_arg];
    if (!def) fail('unknown card definition');
    if (card.type === 'monument')
      return catalog.MONUMENTS.filter(
        (n) =>
          category(def.kind) === n.category &&
          key(amounts(def.cost)) === key(amounts(n.cost)) &&
          monumentKey(def) === n.scoreKey,
      );
    const age = { building_age_1: 1, building_age_2: 2 }[card.type];
    if (!age) fail('unknown card type');
    return catalog.BUILDINGS.filter(
      (n) =>
        n.age === age &&
        category(def.kind) === n.category &&
        key(amounts(def.cost)) === key(amounts(n.cost)) &&
        sourceEffects(def) === effectsSignature(n.effects),
    );
  }
  function bind(card, preferred = null) {
    const id = String(card.id);
    const signature = key({ type: card.type, typeArg: String(card.type_arg) });
    if (assigned.has(id)) {
      const old = assigned.get(id);
      if (old.signature !== signature || (preferred && old.nativeId !== preferred))
        fail('card instance rebound');
      return old.nativeId;
    }
    const matches = candidates(card).filter((n) => !used.has(n.id));
    const selected = preferred
      ? matches.find((n) => n.id === preferred)
      : matches.sort((a, b) => a.id.localeCompare(b.id))[0];
    if (!selected) fail('card effect/instance mapping mismatch');
    // Multiple candidates are accepted only if identical complete native definitions except ID/name.
    const semantics = (n) =>
      key({
        age: n.age,
        category: n.category,
        cost: amounts(n.cost),
        effects: n.effects && effectsSignature(n.effects),
        scoreKey: n.scoreKey,
      });
    if (matches.some((n) => semantics(n) !== semantics(selected)))
      fail('ambiguous non-equivalent card mapping');
    assigned.set(id, {
      nativeId: selected.id,
      signature,
      sourceType: card.type,
      typeArg: String(card.type_arg),
    });
    used.add(selected.id);
    return selected.id;
  }
  for (const [cards, ids] of [
    [initial.buildings, seed?.buildings],
    [initial.monuments, seed?.monuments],
  ]) {
    if (!Array.isArray(cards) || (ids && cards.length !== ids.length))
      fail('initial market length mismatch');
    for (let i = 0; i < cards.length; i++) bind(cards[i], ids?.[i] ?? null);
  }
  return { bind, entries: () => Object.fromEntries(assigned) };
}

/** Direct first-Playing projection; source options and teacher qualification remain separate. */
export function deriveNetworkInitial(index) {
  const source = index.initial;
  const playerCount = Object.keys(source.players).length;
  if (
    ![3, 4].includes(playerCount) ||
    source.workers.filter((w) => w.player === null).length !== (4 - playerCount) * 6
  )
    fail('source neutral setup count does not match the original roster');
  const distribution = index.groups.find((g) => g.entries.some((n) => n.type === 'wealthTiles'));
  if (!distribution) fail('missing original wealth distribution');
  const mapper = networkCardMapper(source);
  const ledger = initialLedger(index);
  ledger.buildings = source.buildings.map((c) => mapper.bind(c));
  ledger.monuments = source.monuments.map((c) => mapper.bind(c));
  for (const group of index.groups.filter((g) => g.moveId <= distribution.moveId))
    for (const entry of group.entries) updateLedger(ledger, entry, mapper);
  if (ledger.stateId !== 4 || ledger.day !== 0 || ledger.activePlayer !== ledger.firstPlayer)
    fail('initial source is not the first untouched Playing boundary');
  const start = source.playerorder.findIndex((id) => String(id) === ledger.firstPlayer);
  if (start < 0) fail('starting player absent from source seat ring');
  const ring = [...source.playerorder.slice(start), ...source.playerorder.slice(0, start)].map(
    String,
  );
  const identities = new Map(ring.map((id, i) => [id, i]));
  const geometry = sourceCheckpoint(ledger, identities);
  const players = ring.map((id, i) => {
    const original = source.players[id],
      p = ledger.players[id];
    if (
      !/^[a-f\d]{6}$/i.test(original.color) ||
      Number(original.flipped) !== 0 ||
      p.workers < 3 ||
      p.workers > 4
    )
      fail('unsupported initial player counters');
    return {
      id: i,
      name: original.name,
      color: `#${original.color}`,
      resources: copy(p.resources),
      score: p.score,
      workers: p.workers,
      temples: copy(p.temples),
      technologies: copy(p.technologies),
      buildings: copy(p.buildings),
      monuments: copy(p.monuments),
      wealth: [...p.wealth].sort(),
      feedWorkers: p.feedWorkers,
      feedAll: false,
      feedDiscount: p.feedDiscount,
      cornTiles: p.cornTiles,
      woodTiles: p.woodTiles,
      skullsPlaced: p.skullsPlaced,
      buildingSkulls: 0,
      doubleAdvanceAvailable: true,
      templePoints: 0,
    };
  });
  for (const slots of Object.values(geometry.gears))
    for (let i = 0; i < slots.length; i++)
      if (slots[i] && slots[i].playerId !== -1) slots[i].dummy = false;
  const total = (age) => {
    const values = Object.values(source.constructions[`building_age_${age}`]);
    for (const def of values) {
      sourceEffects(def);
      if (def.quantity !== undefined && def.quantity < 1) fail('invalid card multiplicity');
    }
    const count = values.reduce((n, d) => n + Number(d.quantity ?? 1), 0);
    if (count !== catalog.BUILDINGS.filter((c) => c.age === age).length)
      fail('source/native deck multiplicities differ');
    return count;
  };
  const initial = {
    version: 1,
    additionalBuildings: false,
    phase: 'playing',
    round: 1,
    age: 1,
    players,
    currentPlayer: 0,
    firstPlayer: 0,
    turnOrder: ring.map((_, i) => i),
    turnIndex: 0,
    turn: { mode: 'none', count: 0, begged: false },
    gears: geometry.gears,
    jungle: geometry.jungle,
    skullSupply: ledger.skullSupply,
    skullSpaces: geometry.skullSpaces,
    firstPlayerClaimed: geometry.firstPlayerClaimed,
    accumulatedCorn: ledger.accumulatedCorn,
    buildings: geometry.buildings,
    buildingDeckCount: total(1) - geometry.buildings.length,
    age2DeckCount: total(2),
    monuments: geometry.monuments,
    pending: null,
    log: [],
    foodDays: [],
    finalScores: [],
    hidden: { seed: 'unknown', setupOffers: 'unknown', deckOrder: 'unknown' },
  };
  return {
    schema: 'tzolkin-bga-network-initial-preparation-v1',
    tableId: index.tableId,
    sourceRawSha256: index.rawSha256,
    sourceActorIds: ring,
    sourceAnchor: {
      moveId: distribution.moveId,
      notifications: distribution.entries.map((n) => ({
        packetIndex: n.packetIndex,
        notificationIndex: n.notificationIndex,
        uid: n.uid,
      })),
    },
    cardMappings: mapper.entries(),
    sourceCountersModified: false,
    qualificationRaised: false,
    witnesses: {
      schema: 'tzolkin-bga-public-witnesses-v1',
      tableId: index.tableId,
      rawSha256: index.rawSha256,
      catalogHash: catalogFingerprint(catalogBytes),
      initialActionId: distribution.moveId,
      initial,
      witnesses: [],
    },
  };
}

function initialLedger(index) {
  const s = index.initial;
  return {
    players: Object.fromEntries(
      Object.entries(s.players).map(([id, p]) => [
        id,
        {
          resources: Object.fromEntries(R.map((k) => [k, Number(p[k])])),
          score: Number(p.score),
          technologies: Object.fromEntries(TECH.map((k) => [k, Number(p[k])])),
          temples: Object.fromEntries(TEMPLE.map((k) => [temple(k), Number(p[k]) - 1])),
          workers: s.workers.filter((w) => String(w.player) === id).length,
          cornTiles: Number(p.corn_tile),
          woodTiles: Number(p.wood_tile),
          buildings: [],
          monuments: [],
          wealth: [],
          feedWorkers: 0,
          feedDiscount: 0,
          skullsPlaced: 0,
          doubleAdvanceAvailable: p.flipped === undefined ? null : Number(p.flipped) === 0,
        },
      ]),
    ),
    workers: new Map(s.workers.map((w) => [String(w.id), copy(w)])),
    day: Number(s.day),
    activePlayer: String(s.gamestate.active_player),
    stateId: Number(s.gamestate.id),
    age: 1,
    foodDays: [],
    feedingWitnesses: [],
    pendingFeeding: null,
    lastRotation: null,
    ageSource: { field: 'initialState.buildings[].type', value: 'building_age_1' },
    firstPlayer: String(s.firstPlayer),
    accumulatedCorn: Number(s.cornOnWheel),
    jungle: Object.fromEntries(
      Object.entries(s.jungle).map(([k, v]) => [
        k,
        { corn: Number(v.corn) + Number(v.wood), wood: Number(v.wood) },
      ]),
    ),
    skullSupply: Number(s.remainingSkulls),
    skullSpaces: Array.from({ length: 10 }, () => null),
    buildings: [],
    monuments: [],
    initial: s,
  };
}

export function validateNeutralNetworkRotation(workers, gears, args) {
  for (const old of workers.values()) {
    if (old.player !== null) continue;
    const observed = args.workers.filter((w) => String(w.id) === String(old.id));
    const teeth = Number(gears[old.place]?.teethnbr);
    if (
      observed.length !== 1 ||
      observed[0].player !== null ||
      observed[0].place !== old.place ||
      !sourcePosition(observed[0].position, teeth) ||
      Number(observed[0].position) !== (Number(old.position) + 1 + args.twice) % teeth
    )
      fail('neutral rotation identity/position differs from the original gear advance');
  }
}

function updateLedger(ledger, entry, mapper) {
  const a = entry.args;
  const playerId = a.player_id === undefined ? null : String(a.player_id);
  const p = ledger.players[playerId];
  if (entry.type === 'takeGoods' || entry.type === 'payGoods') {
    if (
      !p ||
      !R.includes(a.type) ||
      !Number.isSafeInteger(Number(a.quantity)) ||
      Number(a.quantity) < 0
    )
      fail('invalid resource effect');
    p.resources[a.type] += Number(a.quantity) * (entry.type === 'takeGoods' ? 1 : -1);
    if (entry.type === 'takeGoods' && a.type === 'skull') ledger.skullSupply -= Number(a.quantity);
  } else if (entry.type === 'updateScore') {
    if (!p || !Number.isFinite(Number(a.points))) fail('invalid score effect');
    p.score += Number(a.points);
  } else if (entry.type === 'increaseTechnology' || entry.type === 'decreaseTechnology') {
    if (!p || !TECH.includes(a.technology_id)) fail('invalid technology effect');
    p.technologies[a.technology_id] = Number(a.level);
  } else if (entry.type === 'alterTemple') {
    if (!p || !TEMPLE.includes(a.temple_id)) fail('invalid temple effect');
    p.temples[temple(a.temple_id)] = Number(a.step) - 1;
  } else if (entry.type === 'placeWorker') {
    const id = String(a.worker_id);
    const old = ledger.workers.get(id);
    if (!p || (old && (String(old.player) !== playerId || old.place !== null)))
      fail('worker placement identity');
    const teeth =
      a.place === 'startingPlayer' ? 1 : Number(ledger.initial.gears[a.place]?.teethnbr);
    if (
      !Number.isSafeInteger(Number(a.position)) ||
      Number(a.position) < 0 ||
      Number(a.position) >= teeth ||
      !Number.isFinite(teeth)
    )
      fail('worker placement position');
    ledger.workers.set(id, { id, player: playerId, place: a.place, position: Number(a.position) });
  } else if (entry.type === 'pickUpWorker') {
    const old = ledger.workers.get(String(a.worker_id));
    if (
      !p ||
      !old ||
      String(old.player) !== playerId ||
      old.place === null ||
      (a.position === undefined
        ? old.place !== 'startingPlayer'
        : Number(old.position) !== Number(a.position))
    )
      fail('worker removal identity');
    old.place = null;
    old.position = null;
  } else if (entry.type === 'newWorker') {
    if (!p) fail('new worker player');
    p.workers++;
  } else if (entry.type === 'rotateWheel') {
    if (
      ![0, 1].includes(a.twice) ||
      !Array.isArray(a.workers) ||
      Number(a.day) !== ledger.day + 1 + a.twice
    )
      fail('unsupported calendar advance');
    validateNeutralNetworkRotation(ledger.workers, ledger.initial.gears, a);
    ledger.day = Number(a.day);
    for (const w of a.workers) {
      const old = ledger.workers.get(String(w.id));
      if (!old || String(old.player) !== String(w.player)) fail('rotation worker identity');
      if (
        !Object.hasOwn(ledger.initial.gears, w.place) ||
        !sourcePosition(w.position, Number(ledger.initial.gears[w.place].teethnbr))
      )
        fail('rotation worker position');
      ledger.workers.set(String(w.id), copy(w));
    }
    ledger.lastRotation = {
      moveId: entry.moveId,
      packetIndex: entry.packetIndex,
      notificationIndex: entry.notificationIndex,
      uid: entry.uid,
      day: Number(a.day),
      twice: a.twice,
    };
  } else if (entry.type === 'flipBoard') {
    if (!p || ![0, 1].includes(a.side)) fail('invalid speed-up rights');
    p.doubleAdvanceAvailable = a.side === 0;
  } else if (entry.type === 'firstPlayer') {
    if (!p) fail('first player identity');
    ledger.firstPlayer = playerId;
    ledger.accumulatedCorn = 0;
  } else if (entry.type === 'cornOnWheel') ledger.accumulatedCorn++;
  else if (entry.type === 'gameStateChange') {
    ledger.activePlayer = String(a.active_player);
    ledger.stateId = Number(a.id);
    if (ledger.stateId === 92) {
      const event = ledger.feedingWitnesses.find((e) => e.start.uid === entry.uid);
      if (event) {
        ledger.foodDays.push(event.scheduledDay);
        ledger.pendingFeeding = event;
      }
    }
  } else if (entry.type === 'takeTile') {
    if (p && a.type === 'corn_tile') p.cornTiles++;
    if (p && a.type === 'wood_tile') p.woodTiles++;
    const box = ledger.jungle[a.action];
    if (!box || !['corn_tile', 'wood_tile'].includes(a.type)) fail('invalid jungle tile effect');
    box[a.type === 'corn_tile' ? 'corn' : 'wood']--;
  } else if (entry.type === 'skullplace') {
    const position = Number(a.place);
    if (
      !p ||
      !Number.isInteger(position) ||
      position < 1 ||
      position >= 10 ||
      ledger.skullSpaces[position] !== null
    )
      fail('invalid skull placement');
    ledger.skullSpaces[position] = playerId;
    p.skullsPlaced++;
  } else if (entry.type === 'construct') {
    if (!p) fail('construction owner');
    const c = a.building,
      id = mapper.bind(c),
      field = c.type === 'monument' ? 'monuments' : 'buildings';
    if (!ledger[field].includes(id)) fail('construction absent from public market');
    ledger[field] = ledger[field].filter((x) => x !== id);
    p[field].push(id);
    const def = ledger.initial.constructions[c.type][c.type_arg];
    p.feedWorkers += Number(def.farm ?? 0);
    p.feedDiscount += Number(def.bigFarm ?? 0);
  } else if (entry.type === 'pickBuilding') ledger.buildings.push(mapper.bind(a.building));
  else if (entry.type === 'resetAllBuildings') {
    ledger.buildings = a.buildings.map((c) => mapper.bind(c));
    if (!a.buildings.length || a.buildings.some((c) => c.type !== 'building_age_2'))
      fail('source age switch lacks an explicit age-II market');
    ledger.age = 2;
    ledger.ageSource = {
      moveId: entry.moveId,
      packetIndex: entry.packetIndex,
      notificationIndex: entry.notificationIndex,
      uid: entry.uid,
      field: 'args.buildings[].type',
      value: 'building_age_2',
    };
  } else if (entry.type === 'wealthTiles') {
    for (const tile of Object.values(a)) {
      const owner = ledger.players[String(tile.location_arg)],
        def = ledger.initial.wealthTiles[tile.type_arg];
      if (!owner || !def) fail('initial wealth identity');
      const matches = catalog.STARTING_WEALTH.filter(
        (w) =>
          w.gear === def.dummyWorkers.gear &&
          w.position === Number(def.dummyWorkers.position) &&
          sourceEffects(def) ===
            effectsSignature([...w.effects, { type: 'resources', resources: w.resources }]),
      );
      if (matches.length !== 1) fail('initial wealth semantic mapping');
      owner.wealth.push(matches[0].id);
      owner.feedWorkers += Number(def.farm ?? 0);
    }
  }
}

function differences(ledger, state, sourceToNative, terminalCalendar = null) {
  const out = [];
  const check = (field, observed, native) => {
    if (!isDeepStrictEqual(observed, native)) out.push({ field, observed, native });
  };
  for (const [id, p] of Object.entries(ledger.players)) {
    const q = state.players[sourceToNative.get(id)];
    for (const k of R) check(`player.${id}.resources.${k}`, p.resources[k], q.resources[k]);
    for (const k of TECH)
      check(`player.${id}.technology.${k}`, p.technologies[k], q.technologies[k]);
    for (const k of Object.keys(p.temples))
      check(`player.${id}.temple.${k}`, p.temples[k], q.temples[k]);
    check(`player.${id}.workers`, p.workers, q.workers);
    check(`player.${id}.score`, p.score, Math.floor(q.score));
    const occupied = [...ledger.workers.values()].filter(
      (w) => String(w.player) === id && w.place !== null,
    ).length;
    const placed =
      Object.values(state.gears)
        .flat()
        .filter((w) => w?.playerId === q.id).length + (state.firstPlayerClaimed === q.id ? 1 : 0);
    check(`player.${id}.available`, p.workers - occupied, q.workers - placed);
    check(`player.${id}.cornTiles`, p.cornTiles, q.cornTiles);
    check(`player.${id}.woodTiles`, p.woodTiles, q.woodTiles);
    check(`player.${id}.buildings`, p.buildings, q.buildings);
    check(`player.${id}.monuments`, p.monuments, q.monuments);
    check(`player.${id}.wealth`, [...p.wealth].sort(), [...q.wealth].sort());
    check(`player.${id}.feedWorkers`, p.feedWorkers, q.feedWorkers);
    check(`player.${id}.feedDiscount`, p.feedDiscount, q.feedDiscount);
    check(`player.${id}.skullsPlaced`, p.skullsPlaced, q.skullsPlaced);
    if (p.doubleAdvanceAvailable !== null)
      check(
        `player.${id}.doubleAdvanceAvailable`,
        p.doubleAdvanceAvailable,
        q.doubleAdvanceAvailable,
      );
  }
  for (const [gear, slots] of Object.entries(state.gears)) {
    const observed = Array.from({ length: slots.length }, () => null);
    for (const w of ledger.workers.values())
      if (w.place === gear)
        observed[Number(w.position)] =
          w.player === null ? -1 : sourceToNative.get(String(w.player));
    check(
      `gear.${gear}`,
      observed,
      slots.map((w) => w?.playerId ?? null),
    );
  }
  check('firstPlayer', sourceToNative.get(ledger.firstPlayer), state.firstPlayer);
  check('accumulatedCorn', ledger.accumulatedCorn, state.accumulatedCorn);
  check('jungle', ledger.jungle, state.jungle);
  check('skullSupply', ledger.skullSupply, state.skullSupply);
  check(
    'skullSpaces',
    ledger.skullSpaces.map((id) => (id === null ? null : sourceToNative.get(id))),
    state.skullSpaces,
  );
  check('buildings', ledger.buildings, state.buildings);
  check('monuments', ledger.monuments, state.monuments);
  if (state.phase === 'playing') {
    check(
      'round',
      terminalCalendar && ledger.day === terminalCalendar.sourcePlayingDay
        ? terminalCalendar.nativePlayingRound
        : ledger.day + 1,
      state.round,
    );
    if (ledger.initial.gamestates[ledger.stateId]?.type === 'activeplayer')
      check('activePlayer', sourceToNative.get(ledger.activePlayer), state.currentPlayer);
  }
  return out;
}

/** Post-hoc source chronology only; no future notification becomes an observation or label. */
export function terminalPlayingCalendarWitness(index) {
  const timeline = [];
  let day = Number(index.initial.day);
  for (const group of index.groups)
    for (const entry of group.entries) {
      const before = day;
      if (entry.type === 'rotateWheel') {
        const a = entry.args;
        day = Number(a.day);
      }
      timeline.push({
        ...entry,
        moveId: group.moveId,
        sourceDayBefore: before,
        sourceDayAfter: day,
      });
    }
  const candidates = timeline.filter(
    (n) =>
      n.type === 'rotateWheel' &&
      n.sourceDayBefore === 25 &&
      Number(n.args.day) === 27 &&
      n.args.twice === 1,
  );
  if (!candidates.length) return null;
  if (
    timeline.some(
      (n) =>
        n.type === 'rotateWheel' &&
        (![0, 1].includes(n.args.twice) ||
          !sourcePosition(n.args.day, 31) ||
          Number(n.args.day) !== n.sourceDayBefore + 1 + n.args.twice),
    )
  )
    fail('terminal calendar source day/twice chronology is contradictory');
  if (candidates.length !== 1) fail('terminal calendar has multiple overshoot boundaries');
  const start = candidates[0],
    startIndex = timeline.indexOf(start),
    tail = timeline.slice(startIndex + 1);
  const isState = (n, id) => n.type === 'gameStateChange' && Number(n.args.id) === id;
  const terminal = tail.filter((n) => isState(n, 99));
  if (terminal.length !== 1) fail('terminal calendar requires one original terminal boundary');
  const end = terminal[0],
    segment = timeline.slice(startIndex, timeline.indexOf(end) + 1);
  const starts = segment.filter((n) => isState(n, 4)),
    ends = segment.filter((n) => isState(n, 90));
  const meals = segment.filter((n) => isState(n, 92)),
    rotations = segment.filter((n) => n.type === 'rotateWheel');
  const priorMeals = timeline.slice(0, startIndex).filter((n) => isState(n, 92));
  const order = index.initial.playerorder.map(String),
    first = String(starts[0]?.args.active_player);
  const firstIndex = order.indexOf(first);
  const expectedOrder = [...order.slice(firstIndex), ...order.slice(0, firstIndex)];
  const actorOrder = (ns) => ns.map((n) => String(n.args.active_player));
  const lastRotation = rotations[1],
    meal = meals[0];
  const designation = segment.filter((n) => isState(n, 91));
  if (
    priorMeals.length !== 3 ||
    starts.length !== order.length ||
    ends.length !== order.length ||
    firstIndex < 0 ||
    starts[0].moveId !== start.moveId ||
    !isDeepStrictEqual(actorOrder(starts), expectedOrder) ||
    !isDeepStrictEqual(actorOrder(ends), expectedOrder) ||
    meals.length !== 1 ||
    rotations.length !== 2 ||
    !meal ||
    meal.sourceDayBefore !== 27 ||
    designation.length !== 1 ||
    String(designation[0].args.active_player) !== first ||
    String(meal.args.active_player) !== first ||
    String(end.args.active_player) !== first ||
    segment.indexOf(designation[0]) <= segment.indexOf(ends.at(-1)) ||
    segment.indexOf(designation[0]) >= segment.indexOf(meal) ||
    !lastRotation ||
    Number(lastRotation.args.day) !== 28 ||
    lastRotation.args.twice !== 0 ||
    lastRotation.moveId !== meal.moveId ||
    end.moveId !== meal.moveId ||
    segment.indexOf(meal) <= segment.indexOf(ends.at(-1)) ||
    segment.indexOf(lastRotation) <= segment.indexOf(meal) ||
    !segment.some(
      (n) =>
        n.type === 'finalScoring' &&
        segment.indexOf(n) > segment.indexOf(meal) &&
        segment.indexOf(n) < segment.indexOf(lastRotation),
    ) ||
    starts.some(
      (n, i) =>
        segment.indexOf(n) >= segment.indexOf(ends[i]) ||
        (i + 1 < starts.length && segment.indexOf(ends[i]) >= segment.indexOf(starts[i + 1])),
    ) ||
    timeline
      .slice(timeline.indexOf(end) + 1)
      .some((n) => !['simpleNode', 'updateReflexionTime'].includes(n.type))
  )
    fail(
      'terminal calendar lacks one final actor cycle, unique feeding and final one-day rotation',
    );
  const ref = (n) => ({
    moveId: n.moveId,
    packetIndex: n.packetIndex,
    notificationIndex: n.notificationIndex,
    uid: n.uid,
    type: n.type,
    sourceDayBefore: n.sourceDayBefore,
    sourceDayAfter: n.sourceDayAfter,
  });
  return {
    schema: 'tzolkin-bga-terminal-calendar-witness-v1',
    sourceRawSha256: index.rawSha256,
    sourcePriorDay: 25,
    sourcePlayingDay: 27,
    sourcePlayingActualRound: 28,
    sourceFinalDay: 28,
    nativePlayingRound: 27,
    sourceActorOrder: expectedOrder,
    start: ref(start),
    priorFeeding: priorMeals.map(ref),
    turnStarts: starts.map(ref),
    turnEnds: ends.map(ref),
    finalFeeding: ref(meal),
    finalRotation: ref(lastRotation),
    terminal: ref(end),
    scope:
      'Original physical calendar and bounded native last-Playing stage are separate coordinates. Future chronology verifies reconstruction only; it supplies no features, rewards or human choices.',
  };
}

export function bindTerminalCalendarTransition(witness, sourceDay, before, after, sourceToNative) {
  const order = witness.sourceActorOrder.map((id) => sourceToNative.get(id)),
    actor = order[0];
  if (
    sourceDay !== witness.sourcePriorDay ||
    before.round !== 26 ||
    after.round !== 27 ||
    before.phase !== 'playing' ||
    after.phase !== 'playing' ||
    before.pending?.task.type !== 'rotation' ||
    after.pending !== null ||
    !isDeepStrictEqual(before.foodDays, [8, 14, 21]) ||
    !isDeepStrictEqual(after.foodDays, [8, 14, 21]) ||
    before.firstPlayerClaimed !== actor ||
    after.firstPlayerClaimed !== null ||
    after.firstPlayer !== actor ||
    after.currentPlayer !== actor ||
    after.turnIndex !== 0 ||
    !isDeepStrictEqual(after.turnOrder, order) ||
    before.players[actor]?.doubleAdvanceAvailable !== true ||
    after.players[actor]?.doubleAdvanceAvailable !== false ||
    before.players.some(
      (p, i) =>
        i !== actor && p.doubleAdvanceAvailable !== after.players[i]?.doubleAdvanceAvailable,
    )
  )
    fail(
      'terminal calendar does not bind the original source boundary to the native two-day transition',
    );
  return {
    ...witness,
    nativePriorRound: before.round,
    nativePlayingRound: after.round,
    nativeFoodDays: copy(after.foodDays),
    nativeActorOrder: order,
    decisionTimeFeatureEquivalence: 'notEstablished',
  };
}

function sourceCheckpoint(ledger, sourceToNative) {
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

/** Only the first source Playing boundary after the feeding's own calendar movement. */
export function feedingSourceBoardCheckpoint(ledger, sourceToNative, moveId, boundary) {
  const event = ledger.pendingFeeding;
  if (
    !event ||
    boundary?.unresolvedTimingFence !== false ||
    !['endTurn', 'rotate'].includes(boundary?.operationType) ||
    !event.eligible ||
    event.scheduledDay === 27 ||
    ledger.stateId !== 4 ||
    ledger.activePlayer !== ledger.firstPlayer ||
    ledger.lastRotation?.moveId !== moveId ||
    ledger.lastRotation.moveId < event.moveId
  )
    return null;
  const expected = sourceCheckpoint(ledger, sourceToNative);
  const start = ledger.initial.playerorder.findIndex((id) => String(id) === ledger.firstPlayer);
  if (start < 0) fail('source post-feeding starter is not in the original seat order');
  const order = [
    ...ledger.initial.playerorder.slice(start),
    ...ledger.initial.playerorder.slice(0, start),
  ].map((id) => sourceToNative.get(String(id)));
  if (order.some((id) => id === undefined)) fail('source post-feeding seat binding is incomplete');
  for (const slots of Object.values(expected.gears))
    for (const worker of slots) if (worker && worker.playerId !== -1) worker.dummy = false;
  Object.assign(expected, {
    round: ledger.day + 1,
    age: ledger.age,
    foodDays: copy(ledger.foodDays),
    currentPlayer: sourceToNative.get(ledger.activePlayer),
    turnOrder: order,
    turnIndex: order.indexOf(sourceToNative.get(ledger.activePlayer)),
  });
  return {
    expected,
    evidence: {
      scheduledDay: event.scheduledDay,
      sourceStart: event.start,
      sourceEnd: event.end,
      sourcePhysicalDay: event.sourcePhysicalDay,
      sourceActualRound: event.sourceActualRound,
      sourceRotation: copy(ledger.lastRotation),
      comparisonSourceDay: ledger.day,
      comparisonRound: ledger.day + 1,
      ageSource: copy(ledger.ageSource),
      foodHistory: ledger.feedingWitnesses
        .filter((e) => e.moveId <= event.moveId)
        .map((e) => ({ scheduledDay: e.scheduledDay, start: e.start })),
      fields: Object.keys(expected),
      derivations: {
        round: 'Source calendar day + 1 at post-rotation Playing, not the scheduled feeding day.',
        age: 'Original age-I market, then explicit resetAllBuildings age-II types.',
        foodDays:
          'Ordinal explicit original state92 feeding markers under the declared basic profile.',
        turnOrder:
          'Original playerorder rotated to the observed source firstPlayer; current actor from state4.',
        gears: 'Original worker owner: null is neutral/dummy; a roster ID is a real worker.',
      },
      nativeIndividualReceiptMatch: 'notConnected',
      nativeOperationType: boundary.operationType,
      scope:
        'Complete source-derived board after feeding and source calendar rotation, before the next actor operation. This is not an intermediate native feeding receipt.',
    },
  };
}

/** Complete only the untouched base-game boundary from independent source facts.
 * Later checkpoints have different calendar/turn timing and deliberately stay partial.
 */
export function initialSourceBoardCheckpoint(index, ledger, sourceToNative) {
  const source = index.initial;
  const distribution = index.groups.find((g) => g.entries.some((n) => n.type === 'wealthTiles'));
  const boundary = distribution?.entries.findLast(
    (n) => n.type === 'gameStateChange' && Number(n.args.id) === 4,
  );
  if (
    ![3, 4].includes(sourceToNative.size) ||
    Number(source.day) !== 0 ||
    Number(source.nextFoodDay) !== 7 ||
    Number(source.gamestate.id) !== 2 ||
    ledger.day !== 0 ||
    ledger.stateId !== 4 ||
    !boundary ||
    String(boundary.args.active_player) !== ledger.firstPlayer ||
    ledger.activePlayer !== ledger.firstPlayer ||
    !Array.isArray(source.skulls) ||
    source.skulls.length !== 0 ||
    source.buildings.length === 0 ||
    source.buildings.some((c) => c.type !== 'building_age_1') ||
    new Set(source.workers.map((w) => String(w.id))).size !== source.workers.length ||
    source.workers.filter((w) => w.player === null).length !== (4 - sourceToNative.size) * 6 ||
    [...sourceToNative.keys()].some(
      (id) => source.workers.filter((w) => String(w.player) === id).length !== 3,
    ) ||
    [...ledger.workers.values()].some((w) =>
      w.player === null
        ? sourceToNative.size !== 3 || !Object.hasOwn(source.gears, w.place)
        : !sourceToNative.has(String(w.player)) || w.place !== null || w.position !== null,
    ) ||
    Object.values(ledger.players).some((p) => p.wealth.length !== 2) ||
    index.groups
      .filter((g) => g.moveId <= distribution.moveId)
      .some((g) =>
        g.entries.some((n) => ['rotateWheel', 'foodDayCost', 'placeWorker'].includes(n.type)),
      )
  )
    fail('initial source coverage requires untouched post-wealth Playing boundary');
  const expected = sourceCheckpoint(ledger, sourceToNative);
  const first = source.playerorder.findIndex((id) => String(id) === ledger.firstPlayer);
  const order = [...source.playerorder.slice(first), ...source.playerorder.slice(0, first)].map(
    (id) => sourceToNative.get(String(id)),
  );
  if (first < 0 || order.some((id) => id === undefined)) fail('initial source turn order');
  Object.assign(expected, {
    round: Number(source.day) + 1,
    age: 1,
    foodDays: [],
    currentPlayer: sourceToNative.get(String(boundary.args.active_player)),
    turnOrder: order,
    turnIndex: order.indexOf(sourceToNative.get(String(boundary.args.active_player))),
  });
  // Original null-owned workers retain their own source gear positions and neutral marker.
  return {
    expected,
    evidence: {
      state: 'connected',
      sourceRawSha256: index.rawSha256,
      moveId: distribution.moveId,
      boundary: {
        packetIndex: boundary.packetIndex,
        notificationIndex: boundary.notificationIndex,
        uid: boundary.uid,
        type: boundary.type,
        field: 'args.id/active_player',
      },
      sourceFields: [
        'initialState.day',
        'initialState.nextFoodDay',
        'initialState.gamestate.id',
        'initialState.playerorder',
        'initialState.firstPlayer',
        'initialState.players',
        'initialState.workers',
        'initialState.gears',
        'initialState.jungle',
        'initialState.skulls',
        'initialState.remainingSkulls',
        'initialState.cornOnWheel',
        'initialState.buildings',
        'initialState.monuments',
        'initialState.wealthTiles',
        'initialState.constructions',
      ],
      notifications: index.groups
        .filter((g) => g.moveId <= distribution.moveId)
        .flatMap((g) =>
          g.entries.map((n) => ({
            packetIndex: n.packetIndex,
            notificationIndex: n.notificationIndex,
            uid: n.uid,
            type: n.type,
          })),
        ),
      comparedFields: Object.keys(expected),
      interpretations: {
        round: 'initialState.day + 1, with day=0 and no earlier rotation',
        age: 'day=0 and every original visible building is building_age_1',
        foodDays: 'day=0, original nextFoodDay=7, and no earlier feeding or rotation',
        turnOrder: 'initialState.playerorder rotated to original firstPlayer, mapped by roster IDs',
        turnIndex: 'position of the last original state4 active_player in that order',
        gears:
          'owned source workers are unplaced; null-owned neutral workers retain original gear positions, with no placement before state4',
        skullSpaces: 'original skull placement list is empty, with no play before state4',
      },
      scope:
        'First Playing after two wealth tiles per actor; calendar interpretation uses the declared basic profile, not authenticated options. Initial board/counters are from the source ledger, never native output.',
    },
  };
}

/** Distinguish the observed paid target from a free theology route without fitting final resources. */
export function observedChichenRoute(rows, position, corn) {
  const direct = rows.filter(
    (r) =>
      r.action.type === 'useAction' &&
      r.action.gear === 'chichenItza' &&
      r.action.position === position &&
      r.action.cornCost === corn,
  );
  const ahead =
    corn === 0 && position < 10
      ? rows.filter(
          (r) =>
            r.action.type === 'useAction' &&
            r.action.gear === 'chichenItza' &&
            r.action.position === 10 &&
            r.action.source === 'theologyAhead' &&
            r.action.cornCost === 0,
        )
      : [];
  if (direct.length + ahead.length !== 1)
    fail('observed Chichen action has ambiguous or unavailable legal routes');
  return direct.length === 1 ? 'direct' : 'theologyAhead';
}

/** Match source automatic building goods to an exact queued native effect, not a resource choice. */
export function queuedBuildingResources(entries, pending, sourceId) {
  const construct = entries.filter(
    (n) => n.type === 'construct' && String(n.args.player_id) === sourceId,
  );
  const delayed = pending?.after?.[0];
  if (
    construct.length !== 1 ||
    delayed?.type !== 'effects' ||
    delayed.effects.length !== 1 ||
    delayed.effects[0].type !== 'resources'
  )
    return null;
  const b = construct[0].args.building;
  const rewards = entries.filter(
    (n) =>
      n.type === 'takeGoods' &&
      String(n.args.player_id) === sourceId &&
      n.args.origin === `${b.type}_b${b.id}`,
  );
  const resources = {};
  for (const n of rewards) {
    if (
      !R.includes(n.args.type) ||
      !Number.isSafeInteger(Number(n.args.quantity)) ||
      Number(n.args.quantity) < 1 ||
      Object.hasOwn(resources, n.args.type)
    )
      return null;
    resources[n.args.type] = (resources[n.args.type] ?? 0) + Number(n.args.quantity);
  }
  return rewards.length && key(resources) === key(delayed.effects[0].resources)
    ? { resources, rewards }
    : null;
}

/** Corn-payment discount names may be indistinguishable; prove every complete outcome equal. */
export function proveEquivalentCornBuilds(rows, task, apply) {
  const comparable = (row) => {
    const rest = { ...row.action };
    delete rest.discountResource;
    return rest;
  };
  if (
    task?.type !== 'build' ||
    task.cornPayment !== true ||
    rows.length < 2 ||
    rows.length > 3 ||
    rows.some(
      (r) =>
        r.action.type !== 'build' ||
        r.action.architecture !== true ||
        !['wood', 'stone', 'gold'].includes(r.action.discountResource) ||
        r.action.cost.slice(1).some((n) => n !== 0) ||
        !isDeepStrictEqual(comparable(r), comparable(rows[0])),
    ) ||
    new Set(rows.map((r) => r.action.discountResource)).size !== rows.length
  )
    fail('construction alternatives lack the same observed corn-payment contract');
  const ordered = rows
    .slice()
    .sort((a, b) => R.indexOf(a.action.discountResource) - R.indexOf(b.action.discountResource));
  const responses = ordered.map(apply);
  if (
    responses.some((r) => r.error || !r.snapshot?.state || !r.observation) ||
    responses.slice(1).some((r) => !isDeepStrictEqual(r, responses[0]))
  )
    fail('construction alternatives have different complete native outcomes');
  return { row: ordered[0], response: responses[0], equivalentChoices: rows.length };
}

/** Prove an early source designation against every currently legal native rotation. */
export function proveCalendarDesignation(state, observation, entries, sourceToNative, dispatch) {
  const claimed = state.firstPlayerClaimed;
  const designation = entries.filter((n) => n.type === 'firstPlayer');
  const payout = entries.filter((n) => n.type === 'firstPlayerCorn');
  const selection = entries.filter((n) => n.type === 'gameStateChange' && Number(n.args.id) === 95);
  if (
    state.pending?.task.type !== 'rotation' ||
    claimed === null ||
    observation?.actor !== claimed ||
    designation.length !== 1 ||
    payout.length !== 1 ||
    selection.length !== 1 ||
    sourceToNative.get(String(payout[0].args.player_id)) !== claimed ||
    sourceToNative.get(String(selection[0].args.active_player)) !== claimed
  )
    fail('calendar claimant, corn recipient and rotation selector are not bound');
  const rows = observation.legalActions.filter((r) => r.action.type === 'rotate');
  if (!rows.length) fail('calendar has no legal native rotation');
  const projections = rows.map((row) => {
    const response = dispatch({
      operation: 'publicApply',
      state,
      actor: claimed,
      move: row.move,
      refills: { currentAge: [], age2: [] },
    });
    if (response.error) fail(response.error);
    const after = response.snapshot.state;
    if (
      after.firstPlayerClaimed !== null ||
      after.firstPlayer !== sourceToNative.get(String(designation[0].args.player_id))
    )
      fail('source next starter differs from legal native rotation result');
    return {
      days: row.action.days,
      nextFirstPlayer: after.firstPlayer,
      round: after.round,
      doubleAdvanceAvailable: after.players[claimed].doubleAdvanceAvailable,
    };
  });
  return {
    claimant: claimed,
    cornRecipient: claimed,
    rotationSelector: claimed,
    priorFirstPlayer: state.firstPlayer,
    nextFirstPlayer: projections[0].nextFirstPlayer,
    priorRound: state.round,
    projections,
  };
}

/** Strict bounded prefix decoding. Unknown decisions or first mismatch stop without state repairs. */
export function replayNetworkPrefix(index, { initial, initialMoveId, clientContract }, dispatch) {
  assertNetworkClockContract(index, clientContract);
  const clusters = inspectCancellationClusters(index, clientContract);
  if (
    initial.phase !== 'playing' ||
    initial.version !== 1 ||
    initial.additionalBuildings !== false ||
    initial.expansion != null
  )
    fail('unsupported initial profile');
  const distribution = index.groups.find((g) => g.entries.some((n) => n.type === 'wealthTiles'));
  if (
    !Number.isSafeInteger(initialMoveId) ||
    distribution?.moveId !== initialMoveId ||
    initial.pending !== null ||
    initial.turn.mode !== 'none' ||
    initial.turn.count !== 0 ||
    initial.turn.begged
  )
    fail('initial boundary must be the original completed wealth distribution');
  const mapper = networkCardMapper(index.initial, initial);
  const sourceToNative = new Map(
    Object.values(index.initial.players).map((p) => {
      const match = initial.players.filter((n) => n.name === p.name);
      if (match.length !== 1) fail('initial player mapping');
      return [p.id, match[0].id];
    }),
  );
  const ledger = initialLedger(index);
  const feedingEvents = sourceFeedingWitnesses(index);
  if (feedingEvents.some((e) => e.moveId <= initialMoveId))
    fail('source feeding precedes completed setup');
  ledger.feedingWitnesses = feedingEvents;
  ledger.buildings = index.initial.buildings.map((c) => mapper.bind(c));
  ledger.monuments = index.initial.monuments.map((c) => mapper.bind(c));
  const firstGroups = index.groups.filter((g) => g.moveId <= initialMoveId);
  if (!firstGroups.some((g) => g.entries.some((n) => n.type === 'wealthTiles')))
    fail('missing initial distribution');
  for (const group of firstGroups) for (const n of group.entries) updateLedger(ledger, n, mapper);
  const initialDiff = differences(ledger, initial, sourceToNative);
  if (initialDiff.length) fail(`initial boundary mismatch ${JSON.stringify(initialDiff)}`);
  const record = {
    schema: 'tzolkin-public-replay-v1',
    rulesVersion: 1,
    catalogHash: catalogFingerprint(catalogBytes),
    market: 'unlimited',
    source: { reference: index.sourceUrl, actionIds: [initialMoveId] },
    initial: copy(initial),
    initialCheckpoint: null,
    steps: [],
    terminalCheckpoint: null,
  };
  record.initialCheckpoint = {
    source: copy(record.source),
    expected: sourceCheckpoint(ledger, sourceToNative),
  };
  let initialCoverageEvidence = null;
  // Full board coverage needs canonical source geometry; reduced structural projections remain partial.
  if (
    sourceToNative.size === 4 ||
    ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'].every((g) =>
      Object.hasOwn(index.initial.gears, g),
    )
  ) {
    const connected = initialSourceBoardCheckpoint(index, ledger, sourceToNative);
    record.initialCheckpoint.expected = connected.expected;
    initialCoverageEvidence = connected.evidence;
  }
  let frame = dispatch({ operation: 'publicInspect', state: initial });
  if (frame.error) fail(frame.error);
  let calls = 1;
  const audit = [];
  const timingFences = [];
  const calendarTransitions = [];
  const terminalWitness = terminalPlayingCalendarWitness(index);
  let terminalCalendar = null,
    terminalCalendarFence = null;
  const terminalCalendarChecks = [];
  let openRotationFence = null;
  let openCornFence = null;
  let openRewardFences = [];
  let current = null;
  const native = () => frame.snapshot.state;
  const actions = () => frame.observation?.legalActions ?? [];
  function applyWhere(
    predicate,
    reason,
    refills = { currentAge: [], age2: [] },
    cornEquivalent = false,
  ) {
    const rows = actions().filter((row) => predicate(row.action));
    const actor = frame.observation.actor,
      preState = native();
    const apply = (row) => {
      if (++calls > 4000) fail('native call bound');
      return dispatch({
        operation: 'publicApply',
        state: preState,
        actor,
        move: row.move,
        refills,
      });
    };
    let row,
      response,
      equivalentChoices = 1;
    if (rows.length !== 1) {
      if (!cornEquivalent) fail(`${reason}: expected one legal action, found ${rows.length}`);
      ({ row, response, equivalentChoices } = proveEquivalentCornBuilds(
        rows,
        native().pending?.task,
        apply,
      ));
    } else {
      row = rows[0];
      response = apply(row);
    }
    if (response.error) fail(response.error);
    trackRewardQueues(
      openRewardFences,
      preState.pending,
      response.snapshot.state.pending,
      record.steps.length + 1,
    );
    record.steps.push({
      actor,
      move: copy(row.move),
      refills: copy(refills),
      sourceActionIds: [current.moveId],
    });
    audit.push({
      moveId: current.moveId,
      actionId: current.moveId,
      step: record.steps.length,
      actor,
      reason,
      action: copy(row.action),
      equivalentChoices,
      ...(equivalentChoices > 1
        ? {
            equivalentCandidateMoves: rows.map((r) => copy(r.move)),
            equivalenceReason:
              'Every original legal candidate has the same complete native response from one pre-state/actor/refill input.',
            sourceLabelBindingUnavailable: true,
            sourceLabelBindingReason:
              'Corn discount names are not observed; all complete native outcomes are identical.',
          }
        : {}),
      derivedFromVisibleBoundary:
        equivalentChoices > 1 ? null : row.action.type === 'endTurn' || row.action.type === 'skip',
      source: current.entries.map((n) => ({
        packetIndex: n.packetIndex,
        notificationIndex: n.notificationIndex,
        uid: n.uid,
      })),
      teacherCertified: false,
    });
    frame = response;
    return row.action;
  }
  const getActor = (n) =>
    sourceToNative.get(
      String(
        n.args.player_id ??
          Object.values(index.initial.players).find((p) => p.name === n.args.player_name)?.id,
      ),
    );
  function assertActor(n) {
    if (getActor(n) !== frame.observation?.actor) fail(`${n.type}: source/native actor mismatch`);
  }
  const stateId = () =>
    Number(current.entries.filter((n) => n.type === 'gameStateChange').at(-1)?.args.id);
  const payment = () =>
    amounts(
      Object.fromEntries(
        R.map((r) => [
          r,
          current.entries
            .filter((n) => n.type === 'payGoods' && n.args.type === r)
            .reduce((a, n) => a + Number(n.args.quantity), 0),
        ]),
      ),
    );
  function applyPayment() {
    const paid = payment();
    applyWhere(
      (a) => a.type === 'payment' && isDeepStrictEqual(a.resources, paid),
      'observed payment',
    );
  }
  for (const group of index.groups.filter((g) => g.moveId > initialMoveId)) {
    current = group;
    const before = copy(native());
    const stepsBefore = record.steps.length;
    let constructionTransition = null;
    try {
      for (const caption of group.entries.filter((n) => n.type === 'workerLimit')) {
        assertActor(caption);
        const actor = getActor(caption),
          sourceId = [...sourceToNative].find(([, id]) => id === actor)?.[0];
        const workerBuilding = group.entries.some(
          (n) =>
            n.type === 'construct' &&
            getActor(n) === actor &&
            catalog.BUILDINGS.find((b) => b.id === mapper.bind(n.args.building))?.effects.some(
              (e) => e.type === 'worker',
            ),
        );
        if (
          native().players[actor].workers !== 6 ||
          ledger.players[sourceId].workers !== 6 ||
          !workerBuilding ||
          group.entries.filter(
            (n) => n.type === 'updateScore' && getActor(n) === actor && Number(n.args.points) === 6,
          ).length !== 1
        )
          fail('worker cap caption lacks its owned-six-worker building and score receipt');
      }
      for (const caption of group.entries.filter((n) => n.type === 'theologyMax')) {
        assertActor(caption);
        const actor = getActor(caption),
          sameActor = (n) => getActor(n) === actor;
        if (
          group.entries.filter(
            (n) =>
              n.type === 'increaseTechnology' &&
              n.args.technology_id === 'theology' &&
              Number(n.args.level) === 4 &&
              sameActor(n),
          ).length !== 1 ||
          group.entries.filter(
            (n) =>
              n.type === 'decreaseTechnology' &&
              n.args.technology_id === 'theology' &&
              Number(n.args.level) === 3 &&
              sameActor(n),
          ).length !== 1 ||
          group.entries.filter(
            (n) =>
              n.type === 'takeGoods' &&
              n.args.type === 'skull' &&
              Number(n.args.quantity) === 1 &&
              n.args.origin === 'technology_theology_level_4' &&
              sameActor(n),
          ).length !== 1
        )
          fail('theology caption lacks its capped advance and skull receipt');
      }
      const rotations = group.entries.filter((n) => n.type === 'rotateWheel');
      const flips = group.entries.filter((n) => n.type === 'flipBoard');
      const payouts = group.entries.filter((n) => n.type === 'firstPlayerCorn');
      if (
        payouts.length > 1 ||
        payouts.some(
          (n) =>
            Number(n.args.corn) !== ledger.accumulatedCorn ||
            getActor(n) !== before.firstPlayerClaimed,
        )
      )
        fail('calendar corn receipt differs from source accumulation or native claimant');
      if (rotations.length > 1) fail('multiple calendar rotations in one source boundary');
      for (const n of flips) {
        if (n.args.side === 1) {
          if (
            flips.filter((f) => f.args.side === 1).length !== 1 ||
            rotations.length !== 1 ||
            rotations[0].args.twice !== 1 ||
            String(rotations[0].args.player_id) !== String(n.args.player_id)
          )
            fail('speed-up consumption lacks its same-source double rotation');
        } else if (
          !group.entries.some(
            (e) =>
              e.type === 'alterTemple' &&
              String(e.args.player_id) === String(n.args.player_id) &&
              Number(e.args.step) - 1 === catalog.TEMPLE_TRACKS[temple(e.args.temple_id)]?.max,
          )
        )
          fail('speed-up renewal lacks a source top-temple transition');
      }
      if (openRotationFence && !group.entries.some((n) => n.type === 'rotateWheel'))
        fail('deferred first-player boundary did not immediately resolve by rotation');
      if (
        openCornFence &&
        group.entries.some(
          (n) =>
            (['placeWorker', 'trade', 'construct', 'selectAction'].includes(n.type) ||
              (n.type === 'pickUpWorker' && n.args.city !== undefined)) &&
            getActor(n) === openCornFence.actor,
        )
      )
        fail('corn timing mismatch reached claimant decision');
      if (group.entries.some((n) => n.type === 'undoRestoreDISAB')) {
        const cancel = group.entries[0];
        assertActor(cancel);
        if (
          native().pending ||
          native().turn.mode !== 'none' ||
          native().turn.count !== 0 ||
          native().turn.begged
        )
          fail('caption cancellation is not an untouched native start turn');
      } else {
        const refillCards = group.entries
          .filter((n) => n.type === 'pickBuilding')
          .map((n) => n.args.building);
        const reset =
          group.entries.find((n) => n.type === 'resetAllBuildings')?.args.buildings ?? [];
        const refills = {
          currentAge: refillCards.map((c) => mapper.bind(c)),
          age2: reset.map((c) => mapper.bind(c)),
        };
        const nextRemoval = group.entries.find(
          (n) => n.type === 'pickUpWorker' && n.args.city !== undefined,
        );
        if (nextRemoval && native().pending) {
          assertActor(nextRemoval);
          for (let i = 0; native().pending && i < 4; i++)
            applyWhere(
              (x) => x.type === 'skip',
              'observed next worker closes optional previous action',
              refills,
            );
        }
        for (const n of group.entries) {
          const a = n.args;
          if (
            n.type === 'alterTemple' &&
            Number.isInteger(a.corn) &&
            a.corn > 0 &&
            a.corn <= 3 &&
            native().pending === null &&
            group.entries.some(
              (e) =>
                e.type === 'takeGoods' &&
                e.args.type === 'corn' &&
                Number(e.args.quantity) === a.corn &&
                String(e.args.player_id) === String(a.player_id),
            )
          ) {
            assertActor(n);
            applyWhere((x) => x.type === 'beg', 'observed begging before temple descent');
          }
          if (n.type === 'placeWorker') {
            assertActor(n);
            if (a.place === 'startingPlayer')
              applyWhere(
                (x) => x.type === 'firstPlayer' && x.cornCost === Number(a.corn),
                'observed first-player placement',
              );
            else
              applyWhere(
                (x) => x.type === 'place' && x.gear === a.place && x.cornCost === Number(a.corn),
                'observed worker placement',
              );
          } else if (n.type === 'pickUpWorker') {
            if (
              a.city === undefined &&
              a.position === undefined &&
              group.entries.some((e) => e.type === 'rotateWheel')
            ) {
              const worker = ledger.workers.get(String(a.worker_id));
              if (!worker || worker.place === null || String(worker.player) !== String(a.player_id))
                fail('calendar returned worker identity');
              continue;
            }
            assertActor(n);
            const gear = {
              Palenque: 'palenque',
              Yaxchilan: 'yaxchilan',
              Tikal: 'tikal',
              Uxmal: 'uxmal',
              'Chichen Itza': 'chichenItza',
            }[a.city];
            if (!gear) fail('unknown source city');
            applyWhere(
              (x) => x.type === 'remove' && x.gear === gear && x.position === Number(a.position),
              'observed worker removal',
            );
            let observedCorn = null;
            if (gear === 'chichenItza') {
              const paid = group.entries
                .filter(
                  (e) =>
                    e.type === 'payGoods' &&
                    e.args.type === 'corn' &&
                    String(e.args.player_id) === String(a.player_id),
                )
                .reduce((sum, e) => sum + Number(e.args.quantity), 0);
              observedCorn = Number(a.corn ?? paid);
              if (
                !Number.isSafeInteger(observedCorn) ||
                observedCorn < 0 ||
                (paid > 0 && paid !== observedCorn)
              )
                fail('Chichen action corn receipt mismatch');
              if (
                observedChichenRoute(actions(), Number(a.action_number), observedCorn) ===
                'theologyAhead'
              ) {
                applyWhere(
                  (x) =>
                    x.type === 'useAction' &&
                    x.gear === gear &&
                    x.position === 10 &&
                    x.source === 'theologyAhead' &&
                    x.cornCost === 0,
                  'observed zero-cost Chichen target requires the legal theology-ahead continuation',
                  refills,
                );
                const structural = audit.at(-1);
                structural.derivedFromVisibleBoundary = null;
                structural.sourceLabelBindingUnavailable = true;
                structural.sourceLabelBindingReason =
                  'Structural theology-ahead route; only the final Chichen target is observed.';
              }
            }
            applyWhere(
              (x) =>
                x.type === 'useAction' &&
                x.gear === gear &&
                x.position === Number(a.action_number) &&
                (observedCorn === null || x.cornCost === observedCorn),
              'observed gear action',
              refills,
            );
          } else if (n.type === 'selectAction') {
            assertActor(n);
            const gear = {
              Palenque: 'palenque',
              Yaxchilan: 'yaxchilan',
              Tikal: 'tikal',
              Uxmal: 'uxmal',
              'Chichen Itza': 'chichenItza',
            }[a.city];
            applyWhere(
              (x) =>
                x.type === 'useAction' && x.gear === gear && x.position === Number(a.action_number),
              'observed any action',
              refills,
            );
          } else if (
            n.type === 'increaseTechnology' &&
            native().pending?.task.type === 'technology'
          ) {
            assertActor(n);
            applyWhere(
              (x) => x.type === 'technology' && x.technology === a.technology_id,
              'observed technology',
            );
            if (native().pending?.task.type === 'payTechnology') applyPayment();
          } else if (n.type === 'payGoods' && native().pending?.task.type === 'payResource') {
            applyPayment();
          } else if (
            n.type === 'payGoods' &&
            native().pending?.task.type === 'theology' &&
            a.target === 'technology_theology_level_2'
          ) {
            assertActor(n);
            if (Number(a.quantity) !== 1 || !['wood', 'stone', 'gold'].includes(a.type))
              fail('unsupported theology offering');
            applyWhere(
              (x) => x.type === 'offering' && x.resource === a.type,
              'observed theology offering',
            );
          } else if (n.type === 'alterTemple' && native().pending?.task.type === 'temple') {
            assertActor(n);
            applyWhere(
              (x) =>
                x.type === 'temple' &&
                x.temple === temple(a.temple_id) &&
                x.direction ===
                  Math.sign(
                    Number(a.step) -
                      1 -
                      native().players[frame.observation.actor].temples[temple(a.temple_id)],
                  ),
              'observed temple choice',
            );
          } else if (
            n.type === 'takeGoods' &&
            native().pending?.task.type === 'resource' &&
            queuedBuildingResources(
              group.entries,
              native().pending,
              String(a.player_id),
            )?.rewards.includes(n)
          ) {
            assertActor(n);
            // Source already displays these goods; native queues them after the resource choice.
            continue;
          } else if (n.type === 'takeGoods' && native().pending?.task.type === 'resource') {
            assertActor(n);
            for (let i = 0; i < Number(a.quantity); i++)
              applyWhere(
                (x) => x.type === 'resource' && x.resource === a.type,
                'observed resource choice',
              );
          } else if (
            n.type === 'takeGoods' &&
            native().pending?.task.type === 'palenque' &&
            a.type === 'corn' &&
            /^action_palenque_\d+$/.test(a.origin ?? '') &&
            !group.entries.some((e) => e.type === 'takeTile')
          ) {
            assertActor(n);
            applyWhere(
              (x) => x.type === 'harvest' && x.kind === 'emptyCorn',
              'observed agricultural harvest without tile',
            );
          } else if (n.type === 'takeTile' && native().pending?.task.type === 'palenque') {
            assertActor(n);
            const kind = a.type === 'wood_tile' ? 'wood' : 'corn';
            applyWhere((x) => x.type === 'harvest' && x.kind === kind, 'observed harvest');
          } else if (n.type === 'trade') {
            assertActor(n);
            const resource1 = /resource_(corn|wood|stone|gold|skull)\b/.exec(a.resource1)?.[1];
            const resource2 = /resource_(corn|wood|stone|gold|skull)\b/.exec(a.resource2)?.[1];
            const buy = resource1 === 'corn',
              resource = buy ? resource2 : resource1;
            if (!['wood', 'stone', 'gold'].includes(resource) || (!buy && resource2 !== 'corn'))
              fail('unsupported trade');
            const quantity = Number(buy ? a.quantity2 : a.quantity1);
            if (!Number.isSafeInteger(quantity) || quantity < 1 || quantity > 100)
              fail('trade quantity bound');
            for (let i = 0; i < quantity; i++)
              applyWhere(
                (x) => x.type === 'trade' && x.resource === resource && x.buy === buy,
                'observed market exchange',
              );
          } else if (n.type === 'construct') {
            assertActor(n);
            if (
              native().pending?.task.type === 'trade' &&
              native().pending.after?.some((t) => t.type === 'build' && t.remaining > 0)
            ) {
              applyWhere(
                (x) => x.type === 'skip',
                'observed next construction closes prior optional market',
                refills,
              );
              if (native().pending?.task.type !== 'build')
                fail('market exit did not reach the witnessed construction');
              openRewardFences = reconcileRewardFences(
                openRewardFences,
                differences(ledger, native(), sourceToNative, terminalCalendar),
                native().pending,
                group.moveId,
              ).active;
            }
            const id = mapper.bind(a.building);
            const candidates = actions().filter(
              (x) => ['build', 'monument'].includes(x.action.type) && x.action.id === id,
            );
            if (
              candidates.length > 1 &&
              native().players[frame.observation.actor].technologies.architecture === 0
            )
              fail('architecture variants at zero level need an independent continuation contract');
            const architectureCorn = group.entries.some(
              (e) =>
                e.type === 'takeGoods' &&
                e.args.origin === 'technology_architecture_level_1' &&
                String(e.args.player_id) === String(a.player_id),
            );
            const paid = payment();
            const actor = frame.observation.actor;
            if (constructionTransition) fail('multiple construction transitions in one boundary');
            constructionTransition = {
              nativeId: id,
              beforePending: copy(native().pending),
              beforeScore: native().players[actor].score,
              architectureLevel: native().players[actor].technologies.architecture,
            };
            const selected = applyWhere(
              (x) =>
                ['build', 'monument'].includes(x.type) &&
                x.id === id &&
                isDeepStrictEqual(x.cost, paid) &&
                (candidates.length === 1 ||
                  x.type === 'monument' ||
                  x.architecture === architectureCorn),
              'observed construction cost and architecture bonus',
              refills,
              true,
            );
            Object.assign(constructionTransition, {
              action: selected,
              afterPending: copy(native().pending),
              afterScore: native().players[actor].score,
            });
          } else if (n.type === 'gameStateChange' && Number(a.id) === 90) {
            for (let i = 0; native().pending && i < 4; i++)
              applyWhere((x) => x.type === 'skip', 'observed exit from optional task', refills);
            applyWhere(
              (x) =>
                x.type === 'endTurn' && (x.doubleAdvance === null || x.doubleAdvance === false),
              'observed turn end',
              refills,
            );
          } else if (n.type === 'rotateWheel' && native().pending?.task.type === 'rotation') {
            const actor = frame.observation.actor;
            if (
              actor !== native().firstPlayerClaimed ||
              (a.twice === 1 &&
                (getActor(n) !== actor ||
                  flips.filter((f) => f.args.side === 1 && getActor(f) === actor).length !== 1))
            )
              fail('calendar rotation actor or consumed right mismatch');
            const start = copy(native());
            applyWhere(
              (x) => x.type === 'rotate' && x.days === 1 + a.twice,
              a.twice === 1 ? 'observed two-day rotation' : 'observed one-day rotation',
              refills,
            );
            if (terminalWitness?.start.moveId === group.moveId) {
              terminalCalendar = bindTerminalCalendarTransition(
                terminalWitness,
                ledger.day,
                start,
                native(),
                sourceToNative,
              );
              terminalCalendarFence = {
                startMoveId: group.moveId,
                resolvedMoveId: null,
                reason:
                  'Source physical final-day and native bounded Playing-round feature equivalence is not established.',
                differences: [
                  {
                    field: 'calendarCoordinates',
                    observed: terminalCalendar.sourcePlayingActualRound,
                    native: terminalCalendar.nativePlayingRound,
                  },
                ],
                terminalCalendarWitness: terminalCalendar,
              };
              timingFences.push(terminalCalendarFence);
            }
            if (
              openRotationFence &&
              (actor !== openRotationFence.proof.rotationSelector ||
                native().firstPlayer !== openRotationFence.proof.nextFirstPlayer)
            )
              fail('calendar designation did not resolve through its legal rotation');
            calendarTransitions.push({
              moveId: group.moveId,
              actor,
              days: 1 + a.twice,
              priorFirstPlayer: start.firstPlayer,
              nextFirstPlayer: native().firstPlayer,
              priorRound: start.round,
              round: native().round,
              rightsBefore: start.players.map((p) => p.doubleAdvanceAvailable),
              rightsAfter: native().players.map((p) => p.doubleAdvanceAvailable),
              foodDaysBefore: copy(start.foodDays),
              foodDaysAfter: copy(native().foodDays),
              source: group.entries
                .filter((e) => ['flipBoard', 'rotateWheel', 'pickUpWorker'].includes(e.type))
                .map((e) => ({
                  packetIndex: e.packetIndex,
                  notificationIndex: e.notificationIndex,
                  uid: e.uid,
                })),
            });
          }
        }
        // A source transition to worker removal ends the preceding optional subtask.
        if ([4, 7].includes(stateId()) && native().pending)
          for (let i = 0; native().pending && i < 4; i++)
            applyWhere((x) => x.type === 'skip', 'observed return to worker selection', refills);
      }
      for (const n of group.entries) updateLedger(ledger, n, mapper);
      if (native().phase === 'finished') {
        const result = group.entries.find(
          (n) => n.type === 'gameStateChange' && Number(n.args.id) === 99,
        )?.args.args?.result;
        if (Array.isArray(result))
          record.terminalDisplayCheckpoint = {
            source: { reference: index.sourceUrl, actionIds: [group.moveId] },
            mode: 'floorTotal',
            scores: result.map((p) => ({
              playerId: sourceToNative.get(String(p.id)),
              total: Number(p.score),
              rank: Number(p.rank),
            })),
          };
      }
      if (
        terminalCalendar &&
        native().phase === 'playing' &&
        ledger.day === terminalCalendar.sourcePlayingDay
      )
        terminalCalendarChecks.push({
          moveId: group.moveId,
          sourceDay: ledger.day,
          sourceActualRound: ledger.day + 1,
          nativeRound: native().round,
          witnessStartMoveId: terminalCalendar.start.moveId,
        });
      if (terminalCalendarFence && native().phase === 'finished') {
        if (
          group.moveId !== terminalCalendar.terminal.moveId ||
          ledger.day !== terminalCalendar.sourceFinalDay ||
          !native().foodDays.includes(27)
        )
          fail('terminal calendar did not reach its original final feeding and end boundary');
        terminalCalendarFence.resolvedMoveId = group.moveId;
      }
      let diff = differences(ledger, native(), sourceToNative, terminalCalendar);
      const originalDiff = copy(diff);
      const sourceActor = [...sourceToNative].find(
        ([, id]) => id === frame.observation?.actor,
      )?.[0];
      const queuedResourceReceipt = queuedBuildingResources(
        group.entries,
        native().pending,
        sourceActor,
      );
      if (
        constructionTransition &&
        ((native().pending?.task.type === 'technology' && native().pending.task.free === true) ||
          ['trade', 'anyAction'].includes(native().pending?.task.type) ||
          queuedResourceReceipt !== null)
      ) {
        const delayed = native().pending.after?.[0];
        const card = catalog.BUILDINGS.find((b) => b.id === constructionTransition.nativeId);
        const construct = group.entries.find((n) => n.type === 'construct');
        let candidate = null;
        if (
          delayed?.type === 'effects' &&
          delayed.effects.length === 1 &&
          delayed.effects[0].type === 'resources'
        ) {
          const fixed = card?.effects.filter((e) => e.type === 'resources');
          if (
            !queuedResourceReceipt ||
            fixed?.length !== 1 ||
            !isDeepStrictEqual(fixed[0], delayed.effects[0])
          )
            fail('building resources lack an exact source, catalog and queue binding');
          candidate = {
            startMoveId: group.moveId,
            sourceId: sourceActor,
            resources: queuedResourceReceipt.resources,
            points: 0,
            resolvedMoveId: null,
            excludedMoveIds: [],
            queueBinding: bindNewFixedReward(
              constructionTransition.beforePending,
              constructionTransition.afterPending,
              fixed[0],
            ),
            resourceReceiptProof: {
              sourceBuilding: copy(construct.args.building),
              nativeBuilding: card.id,
              catalogEffect: copy(fixed[0]),
              receipts: queuedResourceReceipt.rewards.map((n) => ({
                uid: n.uid,
                packetIndex: n.packetIndex,
                notificationIndex: n.notificationIndex,
                origin: n.args.origin,
                resource: n.args.type,
                amount: Number(n.args.quantity),
              })),
            },
            reason:
              native().pending.task.type === 'resource'
                ? 'source building resources precede its resource choice; native queued effects follow it'
                : 'source building resources precede free technology; native effects task follows it',
          };
        }
        if (
          delayed?.type === 'effects' &&
          delayed.effects.length === 1 &&
          delayed.effects[0].type === 'points' &&
          card?.effects.some((e) => e.type === 'points')
        ) {
          const proof = proveFixedBuildingPoints(
            group.entries,
            card,
            sourceActor,
            constructionTransition,
          );
          candidate = {
            startMoveId: group.moveId,
            sourceId: sourceActor,
            resources: {},
            points: proof.amount,
            resolvedMoveId: null,
            excludedMoveIds: [],
            queueBinding: proof.binding,
            pointReceiptProof: proof.proof,
            reason:
              'source building fixed points precede its pending choice; native effects follow',
          };
        }
        if (candidate) {
          appendRewardFence(openRewardFences, candidate, native().pending, openCornFence);
          timingFences.push(candidate);
        }
      }
      ({ differences: diff, active: openRewardFences } = reconcileRewardFences(
        openRewardFences,
        diff,
        native().pending,
        group.moveId,
      ));
      if (
        !openCornFence &&
        before.firstPlayerClaimed === before.currentPlayer &&
        ledger.accumulatedCorn > 0 &&
        native().accumulatedCorn === 0 &&
        group.entries.some((n) => n.type === 'gameStateChange' && Number(n.args.id) === 90)
      ) {
        const sourceId = [...sourceToNative].find(([, id]) => id === before.currentPlayer)?.[0];
        const amount = ledger.accumulatedCorn;
        if (
          diff.some(
            (d) =>
              d.field === `player.${sourceId}.resources.corn` && d.native === d.observed + amount,
          )
        ) {
          openCornFence = {
            startMoveId: group.moveId,
            actor: before.currentPlayer,
            sourceId,
            amount,
            resolvedMoveId: null,
            reason: 'native pays accumulated corn at claimant turn end; source pays at round end',
            excludedMoveIds: [],
          };
          if (
            openRewardFences.some((f) =>
              rewardFields(f).includes(`player.${sourceId}.resources.corn`),
            )
          )
            fail('calendar corn overlaps an outstanding building resource');
          timingFences.push(openCornFence);
        }
      }
      if (openCornFence) {
        const f = openCornFence;
        const payout = group.entries.find(
          (n) => n.type === 'firstPlayerCorn' && String(n.args.player_id) === f.sourceId,
        );
        if (payout) {
          if (Number(payout.args.corn) !== f.amount)
            fail('source accumulated corn payout differs from timing fence');
          f.resolvedMoveId = group.moveId;
          openCornFence = null;
        } else {
          const corn = diff.find((d) => d.field === `player.${f.sourceId}.resources.corn`);
          const accumulated = diff.find((d) => d.field === 'accumulatedCorn');
          if (
            !corn ||
            corn.native !== corn.observed + f.amount ||
            !accumulated ||
            accumulated.observed !== f.amount ||
            accumulated.native !== 0
          )
            fail('corn timing fence no longer describes exact observed difference');
          f.excludedMoveIds.push(group.moveId);
          diff = diff.filter((d) => d !== corn && d !== accumulated);
        }
      }
      if (diff.length) {
        const claimed = native().firstPlayerClaimed;
        const allowed =
          stateId() === 95 &&
          native().pending?.task.type === 'rotation' &&
          claimed !== null &&
          diff.length === 1 &&
          diff.some((d) => d.field === 'firstPlayer' && d.native === before.firstPlayer);
        if (!allowed) fail(`public state mismatch ${JSON.stringify(diff)}`);
        const proof = proveCalendarDesignation(
          native(),
          frame.observation,
          group.entries,
          sourceToNative,
          (request) => {
            if (++calls > 4000) fail('native call bound');
            return dispatch(request);
          },
        );
        openRotationFence = {
          startMoveId: group.moveId,
          reason: 'source changes first-player designation before native rotation choice',
          differences: diff,
          resolvedMoveId: null,
          proof,
        };
        timingFences.push(openRotationFence);
        audit.push({
          moveId: group.moveId,
          publicStateMatched: false,
          deferredFirstPlayer: true,
          teacherCertified: false,
        });
        continue;
      }
      if (openRotationFence) {
        openRotationFence.resolvedMoveId = group.moveId;
        openRotationFence = null;
      }
      // Finished BGA scores are integer display totals, not exact native fractional scores.
      // The existing floorTotal terminal contract above retains that measurement without promotion.
      if (
        originalDiff.length === 0 &&
        record.steps.length > stepsBefore &&
        native().phase !== 'finished'
      ) {
        record.steps.at(-1).checkpoint = {
          source: { reference: index.sourceUrl, actionIds: [group.moveId] },
          expected: sourceCheckpoint(ledger, sourceToNative),
        };
        const operation = audit.findLast((row) => row.step === record.steps.length)?.action.type;
        const full = feedingSourceBoardCheckpoint(ledger, sourceToNative, group.moveId, {
          operationType: operation,
          unresolvedTimingFence: timingFences.some(
            (f) => f.startMoveId <= group.moveId && f.resolvedMoveId == null,
          ),
        });
        if (full) {
          record.steps.at(-1).checkpoint.expected = full.expected;
          ledger.pendingFeeding.boardCheckpoint = {
            ...full.evidence,
            replayStepIndex: record.steps.length - 1,
            checkpointSource: copy(record.steps.at(-1).checkpoint.source),
          };
          ledger.pendingFeeding = null;
        }
      }
      audit.push({
        moveId: group.moveId,
        publicStateMatched: originalDiff.length === 0,
        deferredCorn: !!openCornFence,
        deferredBuildingResources: openRewardFences.length > 0,
        teacherCertified: false,
        differences: originalDiff,
        nativeSteps: record.steps.length - stepsBefore,
        cancellation: group.entries[0].type === 'undoRestoreDISAB',
      });
    } catch (error) {
      if (native().phase === 'finished') {
        const result = group.entries.find(
          (n) => n.type === 'gameStateChange' && Number(n.args.id) === 99,
        )?.args.args?.result;
        if (Array.isArray(result))
          record.terminalDisplayCheckpoint = {
            source: { reference: index.sourceUrl, actionIds: [group.moveId] },
            mode: 'floorTotal',
            scores: result.map((p) => ({
              playerId: sourceToNative.get(String(p.id)),
              total: Number(p.score),
              rank: Number(p.rank),
            })),
          };
      }
      return {
        schema: 'tzolkin-bga-network-prefix-v1',
        tableId: index.tableId,
        rawSha256: index.rawSha256,
        finished: false,
        terminalReached: native().phase === 'finished',
        semanticReplayVerified: false,
        teacherCompleteness: false,
        teacherCertified: false,
        record,
        audit,
        calls,
        clientContract,
        initialCoverageEvidence,
        cancellationClusters: clusters,
        timingFences,
        calendarTransitions,
        feedingSourceEvidence: feedingEvents,
        ...(index.clockCaptionEntries?.length
          ? { clockCaptionEvidence: index.clockCaptionEntries }
          : {}),
        ...(terminalCalendar
          ? { terminalCalendarProjection: terminalCalendar, terminalCalendarChecks }
          : {}),
        blocked: {
          moveId: group.moveId,
          source: group.entries,
          reason: error.message,
          beforeState: before,
          afterState: native(),
          legalActions: actions(),
        },
        cardMappings: mapper.entries(),
      };
    }
  }
  return {
    schema: 'tzolkin-bga-network-prefix-v1',
    tableId: index.tableId,
    rawSha256: index.rawSha256,
    finished:
      native().phase === 'finished' &&
      !openRotationFence &&
      !openCornFence &&
      !openRewardFences.length,
    terminalReached: native().phase === 'finished',
    semanticReplayVerified: false,
    teacherCompleteness: false,
    teacherCertified: false,
    record,
    audit,
    calls,
    clientContract,
    initialCoverageEvidence,
    cancellationClusters: clusters,
    timingFences,
    calendarTransitions,
    feedingSourceEvidence: feedingEvents,
    ...(index.clockCaptionEntries?.length
      ? { clockCaptionEvidence: index.clockCaptionEntries }
      : {}),
    ...(terminalCalendar
      ? { terminalCalendarProjection: terminalCalendar, terminalCalendarChecks }
      : {}),
    blocked: null,
    cardMappings: mapper.entries(),
    terminal: native(),
  };
}

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { link, lstat, mkdir, mkdtemp, open, readFile, rm } from 'node:fs/promises';
import { dirname, join, parse, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import catalog from '../crates/tzolkin-core/data/catalog.json' with { type: 'json' };
import { parseMessage } from './bga-import.mjs';
import {
  auditReconstruction,
  coreDispatcher,
  MAX_INPUT_BYTES,
  MAX_TOTAL_EVIDENCE_BYTES,
} from './bga-replay.mjs';

export const QUALIFICATION_SCHEMA = 'tzolkin-human-bc-qualification-v1';
export const MAX_REPORT_BYTES = 32 * 1024 * 1024;
export const MAX_DEPENDENCY_IDS = 100_000;
const REPORT_SCHEMA = 'tzolkin-human-bc-audit-v1';
const FEATURE_SOURCE_SHA256 = 'f93d7f102822af80d93ccfde2eac5fd8ee7664e5af3ccf3252caebe7fdd3f1a8';
const FEATURE_SOURCE_CHAIN = [
  {
    path: 'crates/tzolkin-inference/src/features.rs',
    sha256: 'f93d7f102822af80d93ccfde2eac5fd8ee7664e5af3ccf3252caebe7fdd3f1a8',
  },
  {
    path: 'crates/tzolkin-inference/src/lib.rs',
    sha256: 'f8e3aa4ef2612533291f19ccacee3ed2b3d8e077c08d1862886f27e341f91489',
  },
  {
    path: 'crates/tzolkin-inference/Cargo.toml',
    sha256: '0cb53a4db4ae4533c0dde17c31f723a88bb9d7e325d5bcc01fdccf3cd492b119',
  },
  {
    path: 'crates/tzolkin-ai/src/features.rs',
    sha256: '43261972b339d3c7606a2b5b154931af7c14486b26ced0aa0005abd006b46d7a',
  },
  {
    path: 'crates/tzolkin-ai/Cargo.toml',
    sha256: 'd608d70ab4cd872e5c66501e7408694bbce0499b8572fe48f8b03d2834d63d76',
  },
];
const GEARS = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'];
const RESOURCES = ['corn', 'wood', 'stone', 'gold', 'skull'];
const TEMPLES = ['chaac', 'quetzalcoatl', 'kukulkan'];
const TECHNOLOGIES = ['agriculture', 'extraction', 'architecture', 'theology'];
const PUBLIC_FIELDS = [
  'schema',
  'moveSchema',
  'actor',
  'turnPlayer',
  'phase',
  'round',
  'age',
  'additionalBuildings',
  'players',
  'firstPlayer',
  'turnOrder',
  'turnIndex',
  'turn',
  'gears',
  'jungle',
  'skullSupply',
  'skullSpaces',
  'firstPlayerClaimed',
  'accumulatedCorn',
  'buildings',
  'buildingDeckCount',
  'age2DeckCount',
  'monuments',
  'pendingTask',
  'foodDays',
  'expansion',
  'legalActions',
];
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
export function tableFamilyId(tableId) {
  if (typeof tableId !== 'string' || !/^[1-9][0-9]{0,15}$/.test(tableId))
    fail('Invalid canonical table ID');
  return sha256(`tzolkin-bga-table-family-v1\0boardgamearena:${tableId}`);
}
const object = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
const canonical = (v) =>
  JSON.stringify(v, (_key, item) =>
    object(item)
      ? Object.fromEntries(
          Object.keys(item)
            .sort()
            .map((key) => [key, item[key]]),
        )
      : item,
  );
const same = (a, b) => canonical(a) === canonical(b);
const fail = (message) => {
  throw new Error(message);
};
function keys(value, allowed, context) {
  if (!object(value) || Object.keys(value).some((key) => !allowed.includes(key)))
    fail(`${context}: unexpected field/object`);
}
function array(v, max, context) {
  if (!Array.isArray(v) || v.length > max) fail(`${context}: invalid/big array`);
  return v;
}
function integer(v, min, max, context) {
  if (!Number.isSafeInteger(v) || v < min || v > max) fail(`${context}: invalid integer`);
  return v;
}
function checksum(v) {
  if (typeof v !== 'string' || !/^[a-f0-9]{64}$/.test(v)) fail('Invalid SHA-256');
}
function tableUrl(value, tableId) {
  const url = new URL(value);
  if (
    url.protocol !== 'https:' ||
    url.username ||
    url.password ||
    url.port ||
    url.hash ||
    !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
      url.hostname,
    ) ||
    !(
      url.pathname === '/table' ||
      url.pathname === '/gamereview' ||
      /^\/archive\/replay\/[a-zA-Z0-9_-]+\/$/.test(url.pathname)
    ) ||
    url.searchParams.getAll('table').length !== 1 ||
    url.searchParams.get('table') !== tableId
  )
    fail('Source URL/table mismatch');
}
function localPath(value) {
  if (
    typeof value !== 'string' ||
    !value ||
    value.includes('\0') ||
    /^\w+:\/\//.test(value) ||
    /^[\\/]{2}/.test(value)
  )
    fail('Expected an explicit local path');
  return value;
}
function css(style, property) {
  if (typeof style !== 'string') fail('Missing exact CSS token');
  const found = [...style.matchAll(new RegExp(`(?:^|;)\\s*${property}:\\s*([^;]+);`, 'g'))];
  if (found.length !== 1) fail(`Missing/ambiguous CSS ${property}`);
  return found[0][1].trim();
}
function one(values, predicate, context) {
  const matches = values.filter(predicate);
  if (matches.length !== 1) fail(`Missing/ambiguous ${context}`);
  return matches[0];
}
function turn(mode = 'none', count = 0) {
  return {
    mode,
    count,
    begged: false,
    placedWorkers: [],
    tribeAbilityUsed: false,
    placementDiscountUsed: false,
    skippedGear: null,
    skippedPosition: null,
  };
}
function projection(observation) {
  keys(observation, [...PUBLIC_FIELDS, 'observationKey', 'private'], 'observation');
  const result = Object.fromEntries(PUBLIC_FIELDS.map((field) => [field, observation[field]]));
  result.turn = { ...turn(), ...observation.turn };
  result.players = observation.players.map((p) => ({ ...p, wealth: [...p.wealth].sort() }));
  return result;
}

// A closed, deliberately small interpretation table, independently inspected from the
// original age-I/monument atlas. Unlisted artwork is refused; manual companion IDs
// are NOT consulted. Identical small farms use ascending physical serial order.
const BUILDING_SPRITES = {
  '-198px -82px': ['b12'],
  '0px 0px': ['b01', 'b02'],
  '-198px 0px': ['b14'],
  '-66px 0px': ['b04'],
  '-66px -164px': ['b07'],
};
const MONUMENT_SPRITES = {
  '-132px -82px': 'm08',
  '-66px 0px': 'm02',
  '-66px -164px': 'm12',
  '-198px -82px': 'm07',
  '-264px 0px': 'm04',
  '0px -164px': 'm10',
};
function market(board) {
  const cards = array(board.market, 12, 'initial market');
  const groups = new Map();
  const seen = new Set();
  for (const card of cards) {
    if (typeof card.id !== 'string' || seen.has(card.id)) fail('Duplicate/missing card ID');
    seen.add(card.id);
    if (card.id.startsWith('building_age_1_')) {
      if (!/^building_age_1_\d+$/.test(card.id)) fail('Unknown building ID');
      const token = css(card.style, 'background-position');
      if (!BUILDING_SPRITES[token]) fail('Uncalibrated initial building artwork');
      if (!groups.has(token)) groups.set(token, []);
      groups.get(token).push(card.id);
    } else if (
      !/^monument_\d+$/.test(card.id) ||
      !MONUMENT_SPRITES[css(card.style, 'background-position')]
    )
      fail('Uncalibrated initial monument artwork');
  }
  const mapping = new Map();
  for (const [token, ids] of groups) {
    if (ids.length > BUILDING_SPRITES[token].length) fail('Too many identical-effect copies');
    ids
      .sort((a, b) => Number(a.split('_').at(-1)) - Number(b.split('_').at(-1)))
      .forEach((id, i) => mapping.set(id, BUILDING_SPRITES[token][i]));
  }
  const buildings = cards.filter((c) => c.id.startsWith('building_')).map((c) => mapping.get(c.id));
  const monuments = cards
    .filter((c) => c.id.startsWith('monument_'))
    .map((c) => MONUMENT_SPRITES[css(c.style, 'background-position')]);
  if (buildings.length !== 6 || monuments.length !== 6 || new Set(monuments).size !== 6)
    fail('Initial six-plus-six public market required');
  return { buildings, monuments };
}
function wealthPair(player, markers) {
  const matches = [];
  const tiles = catalog.STARTING_WEALTH;
  for (let a = 0; a < tiles.length; a++)
    for (let b = a + 1; b < tiles.length; b++) {
      const p = {
        resources: [0, 0, 0, 0, 0],
        workers: 3,
        temples: [0, 0, 0],
        technologies: [0, 0, 0, 0],
        feedWorkers: 0,
      };
      for (const tile of [tiles[a], tiles[b]]) {
        RESOURCES.forEach((r, i) => {
          p.resources[i] += tile.resources[r] ?? 0;
        });
        for (const effect of tile.effects) {
          if (effect.type === 'worker') p.workers++;
          else if (effect.type === 'temple')
            p.temples[TEMPLES.indexOf(effect.temple)] += effect.steps;
          else if (effect.type === 'technology')
            p.technologies[TECHNOLOGIES.indexOf(effect.technology)] += effect.steps;
          else if (effect.type === 'feed') p.feedWorkers += effect.workers;
          else fail('Unsupported catalog wealth effect');
        }
      }
      if (
        same(
          p.resources,
          RESOURCES.map((r) => player.counters[r]),
        ) &&
        p.workers === player.counters.worker &&
        same(p.temples, markers.temples) &&
        same(p.technologies, markers.technologies)
      )
        matches.push({ ...p, wealth: [tiles[a].id, tiles[b].id] });
    }
  if (matches.length !== 1)
    fail('Initial chosen wealth is not uniquely determined by public effects');
  return matches[0];
}

function settings(document, tableId) {
  if (document.schema !== 'bga-visible-table-settings-evidence-v1' || document.tableId !== tableId)
    fail('Settings document identity/schema mismatch');
  tableUrl(document.url, tableId);
  if (new URL(document.url).pathname !== '/table')
    fail('Fixed settings must come from the table page');
  const rows = array(document.visibleConfiguration, 64, 'visibleConfiguration');
  const required = {
    ゲームモード: 'ノーマルモード',
    ウシュマルコーンの制限: '制限なし',
    追加建物: '無効',
    部族: '無効',
    予言: '無効',
    クイックアクション: '無効',
  };
  const reasons = [];
  for (const [label, value] of Object.entries(required)) {
    const matches = rows.filter((row) => row.label === label);
    if (matches.length !== 1 || matches[0].value !== value)
      reasons.push(`settings: explicit ${label}=${value} not observed`);
  }
  // Other document fields, including a claimed explicitlyAttested/qualified flag,
  // result columns and rankings, grant no proof. Only actual rendered rows count.
  return reasons;
}
function initialOrder(document, tableId, names) {
  keys(document, ['schema', 'tableId', 'url', 'label', 'text'], 'initial order document');
  if (
    document.schema !== 'bga-initial-order-visible-text-v1' ||
    document.tableId !== tableId ||
    document.label !== '初期手番順（ゲーム開始時）'
  )
    fail('No explicit fixed initial order field');
  tableUrl(document.url, tableId);
  if (new URL(document.url).pathname !== '/table')
    fail('Fixed initial order must come from the table page');
  const order = typeof document.text === 'string' ? document.text.split(' → ') : [];
  if (
    order.length !== names.length ||
    new Set(order).size !== names.length ||
    !same([...order].sort(), [...names].sort())
  )
    fail('Initial order text/roster mismatch');
  return order;
}

// Decode only retained first-day turn boundaries. The order becomes usable
// after the third distinct actor's prompt, never retroactively for earlier rows.
export function causalTurnOrder(companion, documents, record, names) {
  if (names.length !== 4 || new Set(names).size !== 4)
    fail('Causal turn order requires four distinct players');
  const read = (witnessId) => {
    const witness = one(companion.witnesses, (w) => w.id === witnessId, 'order witness');
    const definition = one(
      companion.evidenceFiles,
      (f) => f.id === witness.fileId,
      'order evidence',
    );
    const board = JSON.parse(documents.get(witness.fileId))[witness.index];
    if (board.sourceActionId !== witness.actionId || board.round !== 1)
      fail('Causal order needs a first-day source boundary');
    return { witness, definition, board };
  };
  const initial = read(companion.initialWitnessId);
  const identities = names.map((name) => {
    const player = one(initial.board.players, (p) => p.name === name, 'order source actor');
    if (typeof player.id !== 'string' || !/^[1-9][0-9]{0,15}$/.test(player.id))
      fail('Invalid order source actor ID');
    return { name, id: player.id };
  });
  if (new Set(identities.map((p) => p.id)).size !== 4) fail('Duplicate order source actor ID');
  const actor = ({ board, definition }) => {
    if (
      !same(
        board.players
          ?.map((p) => ({ name: p.name, id: p.id }))
          .sort((a, b) => a.id.localeCompare(b.id)),
        [...identities].sort((a, b) => a.id.localeCompare(b.id)),
      )
    )
      fail('Causal order source roster changed');
    const suffix = 'はワーカーを置くか取り除くかしてください';
    if (board.title === `あなた${suffix}`) {
      const viewers = new URL(definition.reference).searchParams.getAll('player');
      if (viewers.length !== 1) fail('Ambiguous causal order viewer');
      return one(identities, (p) => p.id === viewers[0], 'causal order viewer').name;
    }
    return one(identities, (p) => board.title === `${p.name}${suffix}`, 'causal order prompt').name;
  };
  const first = actor(initial);
  const firstIdentity = identities.find((p) => p.name === first);
  if (initial.board.firstPlayer !== `firstplayer_${firstIdentity.id}`)
    fail('Causal initial prompt/first-player token mismatch');
  const order = [first];
  const boundaries = [];
  const receipt = ({ witness, definition }) => ({
    witnessId: witness.id,
    fileId: witness.fileId,
    index: witness.index,
    actionId: witness.actionId,
    sha256: definition.sha256,
  });
  let previousId = initial.witness.actionId;
  for (const step of record.steps) {
    if (names[step.actor] !== order.at(-1)) fail('Causal order prior actor mismatch');
    if (step.move.type === 'place') continue;
    if (
      step.move.type !== 'endTurn' ||
      step.move.doubleAdvance ||
      step.sourceActionIds.length !== 1
    )
      fail('Unsupported operation before causal order is known');
    const node = one(
      companion.timeline,
      (n) => n.kind === 'move' && same(n.step, step),
      'causal end-turn operation',
    );
    const boundary = read(node.witnessId);
    if (
      boundary.witness.actionId !== step.sourceActionIds[0] ||
      boundary.witness.actionId <= previousId
    )
      fail('Causal order boundary is not strictly ordered');
    const next = actor(boundary);
    if (order.includes(next)) fail('Repeated actor before first-day order is known');
    order.push(next);
    boundaries.push(receipt(boundary));
    previousId = boundary.witness.actionId;
    if (order.length === 3) {
      // The unchanged basic four-player cycle fixes the sole remaining actor.
      order.push(one(identities, (p) => !order.includes(p.name), 'remaining cycle actor').name);
      return {
        names: order,
        knownAfterActionId: previousId,
        initial: receipt(initial),
        boundaries,
      };
    }
  }
  fail('Causal turn order has not reached three distinct actor prompts');
}

// Only four-player, initial setup-complete board captures are supported. This
// decodes DOM counters/styles/empty spaces, not the reconstructed native state.
export function decodeInitialBoard(board, names) {
  if (
    names.length !== 4 ||
    new Set(names).size !== 4 ||
    board.round !== 1 ||
    board.workers?.length !== 0
  )
    fail('Requires four-player initial empty board, day 1');
  array(board.players, 4, 'board players');
  if (board.players.length !== 4 || new Set(board.players.map((p) => p.id)).size !== 4)
    fail('Initial player IDs must be unique');
  const markers = array(board.markers, 28, 'initial markers');
  if (markers.length !== 28 || new Set(markers.map((m) => m.id)).size !== 28)
    fail('Requires every initial marker exactly once');
  const players = names.map((name, id) => {
    const p = one(board.players, (p) => p.name === name, 'player');
    if (!/^\d+$/.test(p.id) || p.score !== 0) fail('Invalid initial identity/score');
    RESOURCES.forEach((r) => integer(p.counters?.[r], 0, 100, `counter ${r}`));
    integer(p.counters.worker, 3, 6, 'worker counter');
    const temples = TEMPLES.map((t) => {
      const suffix = t === 'kukulkan' ? 'kukulcan' : t;
      const token = css(
        one(markers, (m) => m.id === `marker_${p.id}_temple_${suffix}`, 'temple marker').style,
        'top',
      );
      const level = { '191px': 0, '167px': 1 }[token];
      if (level === undefined) fail('Uncalibrated initial temple token');
      return level;
    });
    const technologies = TECHNOLOGIES.map((t) => {
      const token = css(
        one(markers, (m) => m.id === `marker_${p.id}_technology_${t}`, 'technology marker').style,
        'left',
      );
      const level = { '1px': 0, '38px': 1 }[token];
      if (level === undefined) fail('Uncalibrated initial technology token');
      return level;
    });
    const chosen = one(
      array(board.wealth, 4, 'chosen wealth'),
      (w) => w.id === `player_wealthtiles_${p.id}`,
      'chosen wealth',
    );
    if (chosen.tiles?.length !== 2 || new Set(chosen.tiles.map((t) => t.id)).size !== 2)
      fail('Initial two visible wealth tiles required');
    const effects = wealthPair(p, { temples, technologies });
    return {
      id,
      resources: effects.resources,
      scoreQuarters: 0,
      workers: effects.workers,
      availableWorkers: effects.workers,
      temples,
      technologies,
      buildings: [],
      monuments: [],
      wealth: effects.wealth.sort(),
      tribe: null,
      feedWorkers: effects.feedWorkers,
      feedAll: false,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
      skullsPlaced: 0,
      buildingSkulls: 0,
      doubleAdvanceAvailable: true,
      templePoints: 0,
    };
  });
  const expectedPlaces = [
    'workerplace_startingPlayer_0',
    ...GEARS.flatMap((g) =>
      Array.from({ length: g === 'chichenItza' ? 13 : 10 }, (_, i) => `workerplace_${g}_${i}`),
    ),
  ];
  if (!same(board.workerPlaces?.map((p) => p.id).sort(), expectedPlaces.sort()))
    fail('Initial worker-place capture is incomplete');
  const skulls = array(board.skullSpaces, 9, 'skull spaces');
  if (
    skulls.length !== 9 ||
    !same(
      skulls.map((s) => s.id).sort(),
      Array.from({ length: 9 }, (_, i) => `skullplace_${i + 1}`).sort(),
    ) ||
    skulls.some((s) => s.html !== '')
  )
    fail('Initial skull spaces not visibly empty');
  integer(board.skullSupply, 0, 13, 'skull supply');
  if (board.skullSupply + players.reduce((sum, p) => sum + p.resources[4], 0) !== 13)
    fail('Initial skull conservation mismatch');
  const jungle = {};
  const boxes = array(board.jungle, 4, 'initial jungle');
  if (boxes.length !== 4) fail('Incomplete jungle');
  for (let i = 2; i <= 5; i++) {
    const box = one(boxes, (b) => b.id === `jungle_${i}`, 'jungle box');
    if (
      box.tiles?.length !== 4 ||
      new Set(box.tiles.map((t) => t.id)).size !== 4 ||
      box.tiles.some((t) => !t.class?.split(' ').includes(i === 2 ? 'corn_tile' : 'wood_tile'))
    )
      fail('Unrecognized initial jungle tiles');
    jungle[i] = { corn: 4, wood: i === 2 ? 0 : 4 }; // basic-rule hidden corn under each wood
  }
  const first = one(
    board.players,
    (p) => board.firstPlayer === `firstplayer_${p.id}`,
    'first-player token',
  );
  if (
    first.name !== names[0] ||
    board.title !== `${first.name}はワーカーを置くか取り除くかしてください`
  )
    fail('Initial current/first-player prompt mismatch');
  const cards = market(board);
  const age1 = catalog.BUILDINGS.filter((b) => b.age === 1).length;
  const age2 = catalog.BUILDINGS.filter((b) => b.age === 2).length;
  if (age1 !== 14 || age2 !== 18) fail('Unsupported base catalog counts');
  return {
    schema: 1,
    moveSchema: 1,
    actor: 0,
    turnPlayer: 0,
    phase: 'playing',
    round: 1,
    age: 1,
    additionalBuildings: false,
    players,
    firstPlayer: 0,
    turnOrder: [0, 1, 2, 3],
    turnIndex: 0,
    turn: turn(),
    gears: Object.fromEntries(
      GEARS.map((g) => [g, Array(g === 'chichenItza' ? 13 : 10).fill(null)]),
    ),
    jungle,
    skullSupply: board.skullSupply,
    skullSpaces: Array(10).fill(null),
    firstPlayerClaimed: null,
    accumulatedCorn: 0,
    ...cards,
    buildingDeckCount: age1 - 6,
    age2DeckCount: age2,
    pendingTask: null,
    foodDays: [],
    expansion: null,
  };
}
export function placementLegalActions(publicInput) {
  const p = publicInput.players[publicInput.actor];
  if (
    publicInput.phase !== 'playing' ||
    publicInput.round !== 1 ||
    publicInput.age !== 1 ||
    publicInput.pendingTask !== null ||
    publicInput.expansion !== null ||
    publicInput.additionalBuildings ||
    p.tribe !== null ||
    !['none', 'place'].includes(publicInput.turn.mode)
  )
    fail('Unsupported placement context');
  // First turns have no own old gear worker and no mercy/beg prompt. Refuse
  // rather than approximate an additional action family outside this decoder.
  if (
    publicInput.turn.mode === 'none' &&
    (p.resources[0] <= 2 ||
      GEARS.some((g) => publicInput.gears[g].some((w) => w?.playerId === p.id && !w.dummy)))
  )
    fail('Initial-turn beg/removal context is not supported');
  const result = [];
  if (p.availableWorkers > 0) {
    for (const g of GEARS) {
      const pos = publicInput.gears[g].findIndex(
        (w, i) => w === null && i <= (g === 'chichenItza' ? 10 : 6),
      );
      const cornCost = pos + publicInput.turn.count;
      if (pos >= 0 && cornCost <= p.resources[0])
        result.push({
          action: { type: 'place', gear: g, cornCost, discount: false },
          move: { type: 'place', gear: g },
        });
    }
    if (publicInput.firstPlayerClaimed === null && publicInput.turn.count <= p.resources[0])
      result.push({
        action: { type: 'firstPlayer', cornCost: publicInput.turn.count },
        move: { type: 'firstPlayer' },
      });
  }
  if (publicInput.turn.count > 0)
    result.push({
      action: { type: 'endTurn', doubleAdvance: null },
      move: { type: 'endTurn' },
    });
  return result;
}
function advance(input, step) {
  const next = structuredClone(input);
  if (step.actor !== input.actor) fail('Source actor/cycle mismatch');
  if (step.move.type === 'place') {
    const legal = placementLegalActions(input).find((a) => same(a.move, step.move));
    if (!legal) fail('Source placement not in independently calculated mask');
    const p = next.players[next.actor];
    p.resources[0] -= legal.action.cornCost;
    p.availableWorkers--;
    const slots = next.gears[step.move.gear];
    slots[slots.findIndex((w) => w === null)] = { playerId: next.actor, dummy: false };
    next.turn.mode = 'place';
    next.turn.count++;
  } else if (
    step.move.type === 'endTurn' &&
    !step.move.doubleAdvance &&
    input.turn.count > 0 &&
    input.turnIndex < 3
  ) {
    next.turnIndex++;
    next.actor = next.turnPlayer = next.turnOrder[next.turnIndex];
    next.turn = turn();
  } else fail('Unsupported prior operation or first day has ended');
  return next;
}
function causalSource(raw, actionId, initialId, names) {
  for (const entry of raw.dom_entries.filter(
    (e) => e.action_id > initialId && e.action_id <= actionId,
  )) {
    for (const message of entry.messages) {
      const event = parseMessage(message, names);
      if (!['place', 'cancellation'].includes(event.kind))
        fail(`Unsupported dynamic source before decision: ${entry.action_id}/${event.kind}`);
    }
  }
}
function endTurnSource(step, input, companion, documents, names) {
  if (input.turnIndex === 3) fail('First day rotation is outside the input decoder');
  if (step.sourceActionIds.length !== 1) fail('End-turn boundary needs one source ID');
  const sourceId = step.sourceActionIds[0];
  const node = one(
    companion.timeline,
    (n) => n.kind === 'move' && same(n.step, step),
    'end-turn operation',
  );
  const w = one(companion.witnesses, (w) => w.id === node.witnessId, 'end-turn witness');
  const board = JSON.parse(documents.get(w.fileId))[w.index];
  if (board.sourceActionId !== sourceId) fail('End-turn source boundary mismatch');
  const nextName = names[input.turnOrder[input.turnIndex + 1]];
  const nextPlayer = one(board.players, (p) => p.name === nextName, 'next-player board');
  const viewer = new URL(
    companion.evidenceFiles.find((f) => f.id === w.fileId).reference,
  ).searchParams.get('player');
  const prompt = `${nextName}はワーカーを置くか取り除くかしてください`;
  if (
    board.title !== prompt &&
    !(viewer === nextPlayer.id && board.title === 'あなたはワーカーを置くか取り除くかしてください')
  )
    fail('No causal next-player prompt at the explicit end-turn boundary');
}

function initialSourceActorIds(companion, documents, names) {
  const witness = one(
    companion.witnesses,
    (w) => w.id === companion.initialWitnessId,
    'initial identity witness',
  );
  const board = JSON.parse(documents.get(witness.fileId))[witness.index];
  if (board.sourceActionId !== witness.actionId) fail('Initial identity source ID mismatch');
  const players = array(board.players, 4, 'initial source identities');
  if (players.length !== names.length) fail('Initial identity roster mismatch');
  const ids = names.map((name) => {
    const player = one(players, (p) => p.name === name, 'initial source actor');
    if (typeof player.id !== 'string' || !/^[1-9][0-9]{0,15}$/.test(player.id))
      fail('Invalid stable source actor ID');
    return player.id;
  });
  if (new Set(ids).size !== ids.length) fail('Duplicate stable source actor ID');
  return ids;
}

// Calculate prefix lengths and the aggregate cost BEFORE allocating any
// per-candidate dependency arrays. One sorted source list serves every label.
export function dependencyPrefixes(timeline, actionIds) {
  const ids = new Set();
  for (const node of timeline) {
    const sourceIds = node.kind === 'move' ? node.step.sourceActionIds : [node.actionId];
    for (const id of sourceIds) {
      integer(id, 1, Number.MAX_SAFE_INTEGER, 'dependency source ID');
      ids.add(id);
    }
  }
  const ordered = [...ids].sort((a, b) => a - b);
  const lengths = [];
  let cursor = 0;
  let total = 0;
  let previous = 0;
  for (const actionId of actionIds) {
    integer(actionId, Math.max(1, previous), Number.MAX_SAFE_INTEGER, 'ordered label source ID');
    previous = actionId;
    while (cursor < ordered.length && ordered[cursor] < actionId) cursor++;
    total += cursor;
    if (total > MAX_DEPENDENCY_IDS) fail('Aggregate dependency IDs exceed 100000');
    lengths.push(cursor);
  }
  return { prefixes: lengths.map((length) => ordered.slice(0, length)), total };
}

export function auditQualification(
  rawBytes,
  companion,
  strictDocuments,
  qualification,
  sourceDocuments,
  dispatch,
  reconstructionBytes = Buffer.from(JSON.stringify(companion)),
) {
  if (
    !Buffer.isBuffer(rawBytes) ||
    !Buffer.isBuffer(reconstructionBytes) ||
    rawBytes.length > MAX_INPUT_BYTES ||
    reconstructionBytes.length > MAX_INPUT_BYTES ||
    Buffer.byteLength(JSON.stringify(qualification)) > MAX_INPUT_BYTES
  )
    fail('Qualification input exceeds 16 MiB');
  keys(
    qualification,
    ['schema', 'tableId', 'rawSha256', 'reconstructionSha256', 'sourceFiles', 'proofs'],
    'qualification',
  );
  if (qualification.schema !== QUALIFICATION_SCHEMA || qualification.tableId !== companion.tableId)
    fail('Qualification schema/table mismatch');
  checksum(qualification.rawSha256);
  checksum(qualification.reconstructionSha256);
  if (
    qualification.rawSha256 !== sha256(rawBytes) ||
    qualification.reconstructionSha256 !== sha256(reconstructionBytes)
  )
    fail('Qualification input SHA-256 mismatch');
  if (
    FEATURE_SOURCE_CHAIN.some(
      ({ path, sha256: expected }) =>
        sha256(readFileSync(new URL('../' + path, import.meta.url))) !== expected,
    )
  )
    fail('Feature dependency decoder needs review for this encoder source');
  const strict = auditReconstruction(rawBytes, companion, strictDocuments, dispatch);
  const record = strict.record;
  const raw = JSON.parse(rawBytes);
  const sources = new Map();
  let totalBytes = [...strictDocuments.values()].reduce((sum, bytes) => sum + bytes.length, 0);
  for (const def of array(qualification.sourceFiles, 16, 'sourceFiles')) {
    keys(def, ['id', 'path', 'sha256', 'reference'], 'source file');
    if (!/^[a-zA-Z0-9_-]{1,80}$/.test(def.id) || sources.has(def.id))
      fail('Duplicate/invalid source file ID');
    localPath(def.path);
    checksum(def.sha256);
    tableUrl(def.reference, companion.tableId);
    const bytes = sourceDocuments.get(def.id);
    if (!Buffer.isBuffer(bytes) || bytes.length > MAX_INPUT_BYTES || sha256(bytes) !== def.sha256)
      fail('Qualification source file SHA-256/size mismatch');
    totalBytes += bytes.length;
    if (totalBytes > MAX_TOTAL_EVIDENCE_BYTES) fail('Aggregate evidence exceeds 32 MiB');
    sources.set(def.id, { bytes, document: JSON.parse(bytes), definition: def });
  }
  const reasons = [];
  const proofResults = [];
  const seenKinds = new Set();
  let fixedNames;
  let causalOrder;
  let initialBoard;
  const names = record.initial.players.map((p) => p.name);
  // Even rejected labels use the captured player's stable BGA ID. This is an
  // identity decoder only, not a substitute for any initial-input proof.
  const actorSourceIds = initialSourceActorIds(companion, strictDocuments, names);
  for (const proof of array(qualification.proofs, 16, 'proofs')) {
    keys(proof, ['kind', 'fileId', 'witnessId'], 'proof');
    if (seenKinds.has(proof.kind)) fail('Duplicate proof kind');
    seenKinds.add(proof.kind);
    try {
      if (proof.kind === 'fixedTableSettingsV1') {
        keys(proof, ['kind', 'fileId'], 'settings proof');
        const source = sources.get(proof.fileId) ?? fail('Missing settings source file');
        const gaps = settings(source.document, companion.tableId);
        reasons.push(...gaps);
        proofResults.push({
          kind: proof.kind,
          fileId: proof.fileId,
          sha256: source.definition.sha256,
          interpretation: 'fixed table-creation option rows; later scores/actions excluded',
          decodedRows: source.document.visibleConfiguration
            .filter((r) =>
              [
                'ゲームモード',
                'ウシュマルコーンの制限',
                '追加建物',
                '部族',
                '予言',
                'クイックアクション',
              ].includes(r.label),
            )
            .map((r) => ({ label: r.label, value: r.value })),
          missing: gaps,
        });
      } else if (proof.kind === 'initialTurnOrderV1') {
        keys(proof, ['kind', 'fileId'], 'order proof');
        const source = sources.get(proof.fileId) ?? fail('Missing initial order source file');
        fixedNames = initialOrder(source.document, companion.tableId, names);
        proofResults.push({
          kind: proof.kind,
          fileId: proof.fileId,
          sha256: source.definition.sha256,
          interpretation: 'explicit initial order field is immutable table setup metadata',
        });
      } else if (proof.kind === 'causalTurnOrderV1') {
        keys(proof, ['kind'], 'causal order proof');
        causalOrder = causalTurnOrder(companion, strictDocuments, record, names);
        fixedNames = causalOrder.names;
        proofResults.push({
          kind: proof.kind,
          ...causalOrder,
          interpretation:
            'retained day-one public prompts plus the basic four-player cycle; earlier labels deferred',
        });
      } else if (proof.kind === 'initialBoardV1') {
        keys(proof, ['kind', 'witnessId'], 'initial board proof');
        if (proof.witnessId !== companion.initialWitnessId)
          fail('Initial proof must reference strict initial witness');
        const w = companion.witnesses.find((w) => w.id === proof.witnessId);
        initialBoard = JSON.parse(strictDocuments.get(w.fileId))[w.index];
        if (initialBoard.sourceActionId !== w.actionId) fail('Initial source ID mismatch');
        const event = raw.dom_entries.find((e) => e.action_id === w.actionId);
        if (!event?.messages?.some((m) => m.raw_text === '各プレイヤーは対応する資源を手に入れた'))
          fail('No setup-complete initial resource-distribution message');
        // Verify no gameplay before this anchor, not merely an initial native phase.
        if (companion.coverage.some((c) => c.actionId < w.actionId && c.kind !== 'nonGame'))
          fail('Initial board is not before all recorded gameplay');
        decodeInitialBoard(initialBoard, names);
        proofResults.push({
          kind: proof.kind,
          witnessId: w.id,
          fileId: w.fileId,
          index: w.index,
          actionId: w.actionId,
          sha256: companion.evidenceFiles.find((f) => f.id === w.fileId).sha256,
          interpretation:
            'DOM counters/exact marker tokens/empty spaces plus basic setup and catalog formulas',
        });
      } else fail(`Unsupported proof decoder: ${proof.kind}`);
    } catch (error) {
      reasons.push(`${proof.kind}: ${error.message}`);
    }
  }
  for (const kind of ['fixedTableSettingsV1', 'initialBoardV1'])
    if (!seenKinds.has(kind)) reasons.push(`Missing proof decoder: ${kind}`);
  if (!seenKinds.has('initialTurnOrderV1') && !seenKinds.has('causalTurnOrderV1'))
    reasons.push('Missing proof decoder: initialTurnOrderV1');
  if (seenKinds.has('initialTurnOrderV1') && seenKinds.has('causalTurnOrderV1'))
    reasons.push('Choose exactly one fixed or causal order proof');
  if (fixedNames && !same(fixedNames, names))
    reasons.push('Initial roster must use the independently decoded cycle order');
  let publicInput;
  if (!reasons.length) publicInput = decodeInitialBoard(initialBoard, fixedNames);
  const native = dispatch({ operation: 'publicReplay', replay: record });
  if (
    !Array.isArray(native.frames) ||
    native.frames.length !== record.steps.length + 1 ||
    native.verifiedSteps !== strict.manifest.verifiedSteps ||
    native.trainingReady !== false
  )
    fail('Inconsistent authoritative replay revalidation');
  const candidates = [];
  let sequenceError;
  const familyId = tableFamilyId(companion.tableId);
  const logicalIds = new Set();
  const operationsPerSource = new Map();
  for (const node of companion.timeline)
    if (node.kind === 'move')
      for (const sourceId of new Set(node.step.sourceActionIds))
        operationsPerSource.set(sourceId, (operationsPerSource.get(sourceId) ?? 0) + 1);
  const dependencies = dependencyPrefixes(
    companion.timeline,
    record.steps
      .filter((step) => step.move.type === 'place')
      .map((step) => Math.min(...step.sourceActionIds)),
  );
  record.steps.forEach((step, index) => {
    const candidateReasons = [...reasons];
    if (sequenceError) candidateReasons.push(sequenceError);
    const sourceIds = step.sourceActionIds;
    if (step.move.type === 'place') {
      const nativeInput = dispatch({
        operation: 'publicInspect',
        state: native.frames[index].snapshot.state,
      }).observation;
      let chosen = null;
      if (causalOrder && Math.min(...sourceIds) <= causalOrder.knownAfterActionId)
        candidateReasons.push('Turn order is not yet established at this decision');
      if (nativeInput?.phase !== 'playing' || nativeInput.pendingTask !== null)
        candidateReasons.push('Setup/pending task labels are not supported');
      if (sourceIds.length !== 1)
        candidateReasons.push('Placement requires one explicit raw source event');
      if (sourceIds.some((id) => operationsPerSource.get(id) !== 1))
        candidateReasons.push('Placement source must map to exactly one native operation');
      if (initialBoard && Math.min(...sourceIds) <= initialBoard.sourceActionId)
        candidateReasons.push('Dynamic initial evidence is not before this decision');
      if (!candidateReasons.length) {
        try {
          causalSource(raw, Math.min(...sourceIds), initialBoard.sourceActionId, names);
          const legal = placementLegalActions(publicInput);
          const expected = { ...publicInput, legalActions: legal };
          const actual = projection(nativeInput);
          for (const field of PUBLIC_FIELDS)
            if (!same(expected[field], actual[field]))
              fail(`Unqualified V2 dependency/mask mismatch: ${field}`);
          chosen = legal.findIndex((a) => same(a.move, step.move));
          if (chosen < 0) fail('Chosen placement not in full ordered legal mask');
        } catch (error) {
          candidateReasons.push(error.message);
        }
      }
      const actionId = Math.min(...sourceIds);
      const actorSourceId = actorSourceIds[step.actor];
      const labelId = sha256(`${familyId}\0${actorSourceId}\0${sourceIds.join(',')}\0${0}`);
      if (logicalIds.has(labelId)) fail('Duplicate logical placement label ID');
      logicalIds.add(labelId);
      candidates.push({
        labelId,
        familyId,
        tableId: companion.tableId,
        actor: step.actor,
        actorSourceId,
        sourceActionIds: sourceIds,
        sourceOperationOrdinal: 0,
        nativeRecordStepIndex: index,
        move: step.move,
        inputBeforeActionId: actionId,
        inputDynamicActionIds: dependencies.prefixes[candidates.length],
        accepted: candidateReasons.length === 0,
        reasons: candidateReasons,
        nativeLegalCandidates: nativeInput?.legalActions?.length ?? 0,
        qualifiedLegalCandidates:
          candidateReasons.length === 0 ? nativeInput.legalActions.length : null,
        chosenIndex: chosen,
        value: null,
      });
    }
    if (publicInput && !sequenceError) {
      try {
        if (step.move.type === 'endTurn')
          endTurnSource(step, publicInput, companion, strictDocuments, names);
        publicInput = advance(publicInput, step);
      } catch (error) {
        sequenceError = `Prior source operation ${index}: ${error.message}`;
      }
    }
  });
  const accepted = candidates.filter((c) => c.accepted).length;
  return {
    schema: REPORT_SCHEMA,
    qualificationSchema: QUALIFICATION_SCHEMA,
    tableId: companion.tableId,
    familyId,
    featureSchema: 2,
    featureCount: 512,
    featureEncoderSourceSha256: FEATURE_SOURCE_SHA256,
    featureEncoderSourceChain: FEATURE_SOURCE_CHAIN.map((entry) => ({ ...entry })),
    dependencyDecoder: causalOrder
      ? 'public-v2-basic-four-player-first-day-causal-order-v1'
      : 'public-v2-basic-four-player-first-day-v1',
    rawSha256: sha256(rawBytes),
    reconstructionSha256: sha256(reconstructionBytes),
    recordSha256: strict.manifest.recordSha256,
    strictManifest: strict.manifest,
    sourceFiles: [...sources.values()].map((s) => ({ ...s.definition, bytes: s.bytes.length })),
    proofs: proofResults,
    dependencyFields: PUBLIC_FIELDS,
    publicationBounds: {
      maxReportBytes: MAX_REPORT_BYTES,
      maxDependencyIds: MAX_DEPENDENCY_IDS,
      dependencyIds: dependencies.total,
    },
    candidates,
    counts: {
      candidates: candidates.length,
      accepted,
      rejected: candidates.length - accepted,
      emittedSamples: 0,
      valueLabels: 0,
    },
    status: native.status,
    verifiedComplete: native.verifiedComplete,
    terminalMatched: native.terminalMatched,
    sourceAuditPassed: false,
    trainingReady: false,
    limitations: [
      'An audit report only: no feature arrays, samples, model or value supervision are exported.',
      'Checksums bind local evidence bytes; they do not authenticate BGA or certify how a capture was collected.',
      causalOrder
        ? 'Supported placement inputs require four players, basic rules, initial empty board, causally established initial cycle order and day 1.'
        : 'Supported placement inputs require four players, basic rules, initial empty board, fixed initial order and day 1.',
      causalOrder
        ? 'Causal order uses retained public turn prompts; rows before its evidence boundary stay rejected.'
        : 'Settings/order are explicitly named immutable setup fields; later dynamic state never supplies earlier inputs.',
      'Basic setup defaults and catalog effects are conditional derivations, not direct DOM observations.',
      'The closed artwork decoder refuses other sprite layouts; identical small-farm copies are canonical equivalents.',
      'Private offer/selection/key fields are excluded in Playing v2; they are not replaced with feature zeros.',
      'Accepted input/label checks do not promote complete-game, source-audit or training readiness flags.',
    ],
  };
}

async function localHierarchy(path, allowMissing = false) {
  const absolute = resolve(localPath(path));
  localPath(absolute);
  // Inspect ancestors from the root before traversing them: a directory
  // symlink/junction may otherwise silently send a "local" path to a share.
  const root = parse(absolute).root;
  let current = root;
  const parts = relative(root, absolute).split(sep);
  let info;
  for (const part of parts) {
    current = join(current, part);
    try {
      info = await lstat(current);
    } catch (error) {
      if (allowMissing && error.code === 'ENOENT') return { absolute, info: undefined };
      throw error;
    }
    if (info.isSymbolicLink()) fail('Source symlinks/junctions are not supported');
  }
  return { absolute, info };
}
async function boundedRead(path) {
  const { absolute, info } = await localHierarchy(path);
  if (!info?.isFile()) fail('Input must be a local regular file');
  if (info.size > MAX_INPUT_BYTES) fail('Input exceeds 16 MiB');
  const bytes = await readFile(absolute);
  if (bytes.length > MAX_INPUT_BYTES) fail('Input exceeds 16 MiB');
  return bytes;
}

export function serializeAuditReport(report) {
  let size = 0;
  const charge = (bytes) => {
    size += bytes;
    if (size > MAX_REPORT_BYTES) fail('Audit report exceeds 32 MiB');
  };
  const string = (value) => {
    charge(2);
    for (let i = 0; i < value.length; i++) {
      const code = value.charCodeAt(i);
      if (code === 34 || code === 92) charge(2);
      else if (code < 32) charge([8, 9, 10, 12, 13].includes(code) ? 2 : 6);
      else if (code < 128) charge(1);
      else if (code < 2048) charge(2);
      else if (code >= 0xd800 && code <= 0xdbff) {
        const next = value.charCodeAt(i + 1);
        if (next >= 0xdc00 && next <= 0xdfff) {
          charge(4);
          i++;
        } else charge(6);
      } else if (code >= 0xdc00 && code <= 0xdfff) charge(6);
      else charge(3);
    }
  };
  const visit = (value, depth) => {
    if (depth > 64) fail('Audit report exceeds nesting limit');
    if (value === null) charge(4);
    else if (typeof value === 'string') string(value);
    else if (typeof value === 'boolean') charge(value ? 4 : 5);
    else if (typeof value === 'number') {
      if (!Number.isFinite(value)) fail('Non-finite audit report number');
      charge(String(value).length);
    } else if (Array.isArray(value)) {
      if (
        Object.getPrototypeOf(value) !== Array.prototype ||
        Object.keys(value).some((key) => !/^(0|[1-9][0-9]*)$/.test(key))
      )
        fail('Audit report must contain plain JSON arrays');
      charge(2);
      for (let index = 0; index < value.length; index++) {
        if (index > 0) charge(1);
        const property = Object.getOwnPropertyDescriptor(value, index);
        if (property && !Object.hasOwn(property, 'value')) fail('Audit report accessor refused');
        const item = property?.value;
        visit(item === undefined ? null : item, depth + 1);
      }
    } else if (object(value) && Object.getPrototypeOf(value) === Object.prototype) {
      charge(2);
      let count = 0;
      for (const key of Object.keys(value)) {
        const property = Object.getOwnPropertyDescriptor(value, key);
        if (!Object.hasOwn(property, 'value')) fail('Audit report accessor refused');
        if (property.value === undefined) continue;
        if (count++ > 0) charge(1);
        string(key);
        charge(1);
        visit(property.value, depth + 1);
      }
    } else fail('Audit report must contain plain JSON values');
  };
  visit(report, 0); // Preflight encoded bytes BEFORE the complete string is allocated.
  const serialized = JSON.stringify(report);
  if (Buffer.byteLength(serialized) !== size) fail('Audit report serialization size mismatch');
  return serialized;
}

export async function publishAuditReport(report, outputPath) {
  const serialized = serializeAuditReport(report);
  const { absolute: output } = await localHierarchy(outputPath, true);
  await mkdir(dirname(output), { recursive: true });
  const temporary = await mkdtemp(join(dirname(output), '.bga-qualify-'));
  try {
    const staging = join(temporary, 'report.json');
    const file = await open(staging, 'wx');
    try {
      await file.writeFile(serialized);
      await file.sync();
    } finally {
      await file.close();
    }
    await link(staging, output); // Synced file -> atomic create-new; never overwrite.
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

export async function qualifyFiles(
  rawPath,
  reconstructionPath,
  qualificationPath,
  cliPath,
  outputPath,
) {
  const raw = await boundedRead(rawPath);
  const reconstruction = await boundedRead(reconstructionPath);
  const qualificationBytes = await boundedRead(qualificationPath);
  const companion = JSON.parse(reconstruction);
  const qualification = JSON.parse(qualificationBytes);
  const strictDocuments = new Map();
  const sourceDocuments = new Map();
  let total = 0;
  for (const [defs, base, destination] of [
    [companion.evidenceFiles, reconstructionPath, strictDocuments],
    [qualification.sourceFiles, qualificationPath, sourceDocuments],
  ])
    for (const file of array(defs, 32, 'evidence files')) {
      const bytes = await boundedRead(resolve(dirname(base), localPath(file.path)));
      total += bytes.length;
      if (total > MAX_TOTAL_EVIDENCE_BYTES) fail('Aggregate evidence exceeds 32 MiB');
      destination.set(file.id, bytes);
    }
  const corePath = await localHierarchy(cliPath);
  if (!corePath.info?.isFile()) fail('Core CLI must be a local regular file');
  const report = auditQualification(
    raw,
    companion,
    strictDocuments,
    qualification,
    sourceDocuments,
    coreDispatcher(corePath.absolute),
    reconstruction,
  );
  report.qualificationSha256 = sha256(qualificationBytes);
  await publishAuditReport(report, outputPath);
  return report;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length !== 5) {
    console.error(
      'Usage: node scripts/bga-qualify.mjs RAW RECONSTRUCTION QUALIFICATION RUST_CLI NEW_REPORT.json',
    );
    process.exitCode = 1;
  } else
    qualifyFiles(...args)
      .then((report) => console.log(JSON.stringify(report.counts)))
      .catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
      });
}

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeGame } from './bga-import.mjs';
import { observedBasicUnlimitedOptions, uniqueJson } from './bga-observed-options.mjs';
import {
  assertBasicUnlimitedOptions,
  catalogFingerprint,
  coreDispatcher,
  publishReplay,
  sourceEvents,
} from './bga-replay.mjs';

export const WITNESS_SCHEMA = 'tzolkin-bga-public-witnesses-v1';
const MAX_BYTES = 16 * 1024 * 1024;
const MAX_STEPS = 4000;
const MAX_CALLS = 12000;
const RESOURCES = ['corn', 'wood', 'stone', 'gold', 'skull'];
const TECHNOLOGIES = ['agriculture', 'extraction', 'architecture', 'theology'];
const catalogBytes = readFileSync(
  new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url),
);
const catalog = JSON.parse(catalogBytes);
const buildings = new Map(catalog.BUILDINGS.map((card) => [card.id, card]));
const templeMaximums = Object.entries(catalog.TEMPLE_TRACKS).map(([temple, track]) => [
  temple,
  track.points.length - 2,
]);
const catalogHash = catalogFingerprint(catalogBytes);
const TEMPLES = {
  Chaac: 'chaac',
  Quetzalcoatl: 'quetzalcoatl',
  Kukulkan: 'kukulkan',
  Kukulcan: 'kukulkan',
};
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const copy = (value) => structuredClone(value);
const fail = (message) => {
  throw new Error(message);
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function closed(value, allowed, label) {
  if (!object(value) || Object.keys(value).some((key) => !allowed.includes(key)))
    fail(`${label}: unknown field`);
}
function positive(value, label) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${label}: invalid positive integer`);
}
function matches(actual, expected) {
  if (object(expected))
    return (
      object(actual) &&
      Object.entries(expected).every(([key, value]) => matches(actual[key], value))
    );
  if (Array.isArray(expected))
    return (
      Array.isArray(actual) &&
      actual.length === expected.length &&
      expected.every((value, index) => matches(actual[index], value))
    );
  return actual === expected;
}
function mechanical(state) {
  const value = copy(state);
  delete value.log;
  return value;
}
// The witness declares calibrated local identities, not a physical card choice.
export function validateBuildingEquivalenceClass(witness) {
  const ids = witness.buildingEquivalenceClass;
  if (ids === undefined) return null;
  if (
    witness.buildings !== undefined ||
    !Array.isArray(ids) ||
    ids.length < 2 ||
    ids.length > 3 ||
    new Set(ids).size !== ids.length ||
    ids.some((id) => !buildings.has(id))
  )
    fail('Invalid building equivalence class');
  const signature = (id) =>
    Object.fromEntries(
      Object.entries(buildings.get(id)).filter(([key]) => !['id', 'name'].includes(key)),
    );
  if (ids.some((id) => !same(signature(id), signature(ids[0]))))
    fail('Building equivalence class has different catalog fields');
  return [...ids].sort();
}
function equivalentBuildingState(state, ids) {
  const value = mechanical(state);
  const membership = new Set(ids);
  const cards = (list) => ({
    nonClassIds: list.filter((id) => !membership.has(id)),
    classCount: list.filter((id) => membership.has(id)).length,
  });
  value.buildings = cards(value.buildings);
  for (const player of value.players) player.buildings = cards(player.buildings);
  return value;
}
function resourceAmounts(event) {
  const icons = event.detail.iconResources ?? event.iconResources;
  if (
    !Array.isArray(icons) ||
    !icons.length ||
    icons.some((resource) => !RESOURCES.includes(resource))
  )
    fail('Unsupported resource icons');
  const amount = event.detail.explicitAmount;
  if (
    amount !== null &&
    amount !== undefined &&
    (icons.length !== 1 || !Number.isSafeInteger(amount) || amount < 1)
  )
    fail('Ambiguous resource amount');
  const values = RESOURCES.map(() => 0);
  for (const resource of icons) values[RESOURCES.indexOf(resource)] += amount ?? 1;
  return values;
}
function accumulatedCornPayout(event) {
  if (event.kind !== 'unparsed' || event.actor === null) return null;
  const match = event.rawText.slice(event.actor.length).match(/^は(\d+)\s*を歯車から獲得した$/);
  const amount = match ? Number(match[1]) : null;
  return Number.isSafeInteger(amount) ? amount : null;
}
function paymentOf(events) {
  const result = RESOURCES.map(() => 0);
  for (const event of events.filter((item) => item.kind === 'resourcePayment'))
    resourceAmounts(event).forEach((amount, index) => {
      result[index] += amount;
    });
  return result;
}
function emptyPurchasePayment(event) {
  return (
    event.kind === 'unparsed' &&
    event.actor !== null &&
    event.iconResources.length === 0 &&
    /^は\s*を支払った$/.test(event.rawText.slice(event.actor.length))
  );
}
function typedResourceCost(action) {
  const values =
    action.type === 'useAction'
      ? [action.cornCost, 0, 0, 0, 0]
      : ['build', 'monument'].includes(action.type)
        ? action.cost
        : RESOURCES.map(() => 0);
  if (
    !Array.isArray(values) ||
    values.length !== RESOURCES.length ||
    values.some((amount) => !Number.isSafeInteger(amount) || amount < 0)
  )
    fail('Core typed resource cost is not a bounded nonnegative vector');
  return values;
}
function sourceGroups(events, initialId, actionIds) {
  const groups = new Map(actionIds.filter((id) => id > initialId).map((id) => [id, []]));
  for (const event of events) {
    if (event.actionId <= initialId) continue;
    if (!groups.has(event.actionId)) fail('Event has no matching raw source action ID');
    groups.get(event.actionId).push(event);
  }
  return [...groups.entries()];
}

/** Text determines constraints; the core alone supplies legal operations and effects.
 * Unknown decisions/reveals stop at the previous whole source event. A returned
 * replay is mechanical evidence, not authenticated source or training admission. */
export function decodePublicGame(rawBytes, witnesses, dispatch, tableBytes = null) {
  return decodeGame(rawBytes, witnesses, dispatch, tableBytes, null);
}

/** Conditional mechanical reconstruction. This declaration does not turn missing
 * source options into observed options and cannot admit human training rows. */
export function decodeConditionalPublicGame(
  rawBytes,
  witnesses,
  dispatch,
  profile,
  tableBytes = null,
) {
  closed(profile, ['schema', 'rules', 'market', 'sourceOptions'], 'declared mechanical profile');
  if (
    profile.schema !== 'tzolkin-bga-declared-mechanical-profile-v1' ||
    profile.rules !== 'basic' ||
    profile.market !== 'unlimited' ||
    profile.sourceOptions !== 'unknown'
  )
    fail('Unsupported declared mechanical profile');
  return decodeGame(rawBytes, witnesses, dispatch, tableBytes, profile);
}

function decodeGame(rawBytes, witnesses, dispatch, tableBytes, declaredProfile) {
  if (!Buffer.isBuffer(rawBytes) || rawBytes.length > MAX_BYTES) fail('Raw input exceeds 16 MiB');
  closed(
    witnesses,
    [
      'schema',
      'tableId',
      'rawSha256',
      'catalogHash',
      'initialActionId',
      'initial',
      'initialExpected',
      'witnesses',
    ],
    'witnesses',
  );
  if (witnesses.schema !== WITNESS_SCHEMA || witnesses.rawSha256 !== sha(rawBytes))
    fail('Witness schema/raw SHA mismatch');
  positive(witnesses.initialActionId, 'initialActionId');
  if (!Array.isArray(witnesses.witnesses) || witnesses.witnesses.length > MAX_STEPS)
    fail('Invalid witness list');
  const raw =
    declaredProfile === null ? JSON.parse(rawBytes) : uniqueJson(rawBytes, MAX_BYTES, 'raw log');
  const table =
    declaredProfile === null
      ? null
      : tableBytes === null
        ? null
        : uniqueJson(tableBytes, MAX_BYTES, 'table evidence');
  if (declaredProfile !== null) observedBasicUnlimitedOptions(raw, table);
  const game = normalizeGame(raw, sha(rawBytes), tableBytes);
  const observedOptions =
    declaredProfile === null ? null : observedBasicUnlimitedOptions(raw, table, game.context);
  if (game.source.adapter !== 'bga-public-ui-dom-v1' && tableBytes !== null)
    fail('Legacy input requires embedded table settings, not separate table evidence');
  if (
    game.tableId !== witnesses.tableId ||
    !game.rawEntries.some((entry) => entry.action_id === witnesses.initialActionId)
  )
    fail('Witness table/initial source mismatch');
  if (declaredProfile === null)
    assertBasicUnlimitedOptions(
      game.source.adapter === 'bga-public-ui-dom-v1'
        ? tableBytes === null
          ? ''
          : JSON.parse(tableBytes).optionsText
        : raw.table_details_text.split('ゲーム構成\n').slice(1).join('ゲーム構成\n'),
    );
  const knownActionIds = game.rawEntries.map((entry) => entry.action_id);
  const knownIds = new Set(knownActionIds);
  const eventIds = new Set(game.events.map((event) => event.actionId));
  const normalizedEvents =
    game.source.adapter === 'bga-public-ui-dom-v1'
      ? game.events
      : sourceEvents(
          {
            entries: game.rawEntries.filter(
              (entry) => entry.raw_text !== '' || eventIds.has(entry.action_id),
            ),
          },
          game,
          knownActionIds.at(-1),
        );
  const source = {
    reference: game.source.reference,
    actionIds: [witnesses.initialActionId],
  };
  const record = {
    schema: 'tzolkin-public-replay-v1',
    rulesVersion: 1,
    catalogHash: witnesses.catalogHash,
    market: 'unlimited',
    source,
    initial: copy(witnesses.initial),
    initialCheckpoint: null,
    steps: [],
    terminalCheckpoint: null,
  };
  if (witnesses.initialExpected)
    record.initialCheckpoint = {
      source: { reference: source.reference, actionIds: [witnesses.initialActionId] },
      expected: copy(witnesses.initialExpected),
    };
  const observed = new Map();
  for (const witness of witnesses.witnesses) {
    closed(
      witness,
      [
        'actionId',
        'expected',
        'refills',
        'buildings',
        'buildingEquivalenceClass',
        'monuments',
        'pendingTask',
      ],
      'witness',
    );
    positive(witness.actionId, 'witness.actionId');
    validateBuildingEquivalenceClass(witness);
    if (!knownIds.has(witness.actionId) || observed.has(witness.actionId))
      fail('Duplicate/unknown witness action ID');
    if (
      witness.expected !== undefined &&
      (!object(witness.expected) || !Object.keys(witness.expected).length)
    )
      fail('Empty witness checkpoint');
    if (
      witness.pendingTask !== undefined &&
      witness.pendingTask !== null &&
      typeof witness.pendingTask !== 'string'
    )
      fail('Invalid pending task witness');
    for (const field of ['buildings', 'monuments'])
      if (
        witness[field] !== undefined &&
        (!Array.isArray(witness[field]) ||
          witness[field].length > 6 ||
          witness[field].some((id) => typeof id !== 'string'))
      )
        fail(`Invalid ${field} witness`);
    if (witness.refills !== undefined) {
      closed(witness.refills, ['currentAge', 'age2'], 'refills');
      for (const field of ['currentAge', 'age2'])
        if (
          witness.refills[field] !== undefined &&
          (!Array.isArray(witness.refills[field]) ||
            witness.refills[field].length > 6 ||
            witness.refills[field].some((id) => typeof id !== 'string'))
        )
          fail('Invalid reveal list');
    }
    observed.set(witness.actionId, witness);
  }
  let calls = 0;
  const core = (request) => {
    if (++calls > MAX_CALLS) fail('Core call limit reached');
    const response = dispatch(request);
    if (response?.error) fail(response.error);
    return response;
  };
  let frame = core({ operation: 'publicReplay', replay: record }).frames[0];
  if (!frame?.observation || frame.snapshot.state.phase !== 'playing')
    fail('Initial Playing frame is required');
  const names = frame.snapshot.state.players.map((player) => player.name);
  if (
    names.length !== game.context.playerCount ||
    new Set(names).size !== names.length ||
    game.context.players.some((name) => !names.includes(name))
  )
    fail('Initial/source players differ');
  let rotationCredit = 0;
  let declaredRotation = null;
  let extractionReward = null;
  let queuedPurchase = null;
  let deferredScore = null;
  let calendarEvidence = {
    corn: null,
    rotation: null,
    feeding: [],
    initialAnnouncement: {
      actor: frame.snapshot.state.firstPlayer,
      round: frame.snapshot.state.round,
      consumed: false,
    },
    announcement: null,
  };
  let history = [
    {
      frame: copy(frame),
      steps: 0,
      rotationCredit,
      declaredRotation,
      calendarEvidence,
      extractionReward,
    },
  ];
  const audit = [];
  let context;
  const legal = () => frame.observation?.legalActions ?? [];
  const state = () => frame.snapshot.state;
  const actorId = (name) => names.indexOf(name);
  const pending = () => frame.observation?.pendingTask?.type ?? null;

  function architectureScoreCaption(event) {
    if (event.kind !== 'unparsed' || event.iconResources.length) return null;
    const actors = names.filter((name) => event.rawText.startsWith(`建築技術ボーナス: ${name}は`));
    if (actors.length !== 1) return null;
    const actor = actors[0];
    const match = event.rawText
      .slice(`建築技術ボーナス: ${actor}`.length)
      .match(/^は(\d+)点を獲得した$/);
    const amount = match ? Number(match[1]) : null;
    return Number.isSafeInteger(amount) && amount > 0 ? { actor, amount } : null;
  }
  function purchaseActor(event) {
    return architectureScoreCaption(event)?.actor ?? theologySkullCaption(event) ?? event.actor;
  }
  function cappedWorkerCaption(event) {
    return (
      event.kind === 'unparsed' &&
      event.actor !== null &&
      event.iconResources.length === 0 &&
      event.rawText === `${event.actor}は既にワーカー6個を持っています`
    );
  }
  function cappedPurchaseWorker(outcome, previous = state()) {
    const action = outcome.row.action;
    const next = outcome.frame.snapshot.state;
    const actor = previous.currentPlayer;
    return (
      action.type === 'build' &&
      buildings.get(action.id)?.effects.filter((effect) => effect.type === 'worker').length === 1 &&
      previous.players[actor].workers === 6 &&
      next.players[actor].workers === 6 &&
      next.currentPlayer === actor &&
      settledBuild(outcome, previous)
    );
  }
  function settledBuild(outcome, previous = state()) {
    const after = outcome.frame.snapshot.state;
    const before = previous.pending;
    const task = after.pending?.task;
    return (
      before?.task?.type === 'build' &&
      Array.isArray(before.after) &&
      (task == null
        ? before.after.length === 0
        : task.type === 'build' &&
          task.remaining === before.task.remaining - 1 &&
          same(after.pending.after, before.after))
    );
  }
  function cappedPurchaseAdvance(outcome, previous = state()) {
    const action = outcome.row.action;
    const actor = previous.currentPlayer;
    const before = previous.players[actor];
    const after = outcome.frame.snapshot.state;
    const next = after.players[actor];
    const card = buildings.get(action.id);
    const effects = card?.effects;
    if (
      witnesses.catalogHash !== catalogHash ||
      action.type !== 'build' ||
      typeof action.architecture !== 'boolean' ||
      action.renovation != null ||
      before.technologies.architecture !== 3 ||
      next.technologies.architecture !== 3 ||
      previous.pending?.task?.type !== 'build' ||
      after.currentPlayer !== actor ||
      after.round !== previous.round ||
      after.phase !== previous.phase ||
      !Array.isArray(effects) ||
      effects.filter((effect) => effect.type === 'technology').length !== 1 ||
      !effects.some(
        (effect) =>
          effect.type === 'technology' &&
          effect.technology === 'architecture' &&
          effect.steps === 1,
      ) ||
      effects.some(
        (effect) =>
          !['technology', 'points'].includes(effect.type) ||
          (effect.type === 'points' && (!Number.isSafeInteger(effect.amount) || effect.amount < 0)),
      )
    )
      return false;
    const points = effects
      .filter((effect) => effect.type === 'points')
      .reduce((sum, effect) => sum + effect.amount, 0);
    return (
      settledBuild(outcome, previous) &&
      same(next.buildings, [...before.buildings, action.id]) &&
      TECHNOLOGIES.every((track) => before.technologies[track] === next.technologies[track]) &&
      next.score - before.score === points + 3 + (action.architecture ? 2 : 0)
    );
  }

  function syncHistory() {
    Object.assign(history.at(-1), {
      rotationCredit,
      declaredRotation,
      calendarEvidence,
      extractionReward,
    });
  }
  function extractionTransition(nextFrame, action) {
    const before = state();
    const after = nextFrame.snapshot.state;
    const task = frame.observation.pendingTask;
    const nextTask = nextFrame.observation?.pendingTask;
    const actor = before.currentPlayer;
    const opens =
      before.players[actor].technologies.extraction === 3 &&
      ((action.type === 'technology' &&
        action.technology === 'extraction' &&
        task?.type === 'technology' &&
        task.free === true) ||
        (action.type === 'payment' &&
          task?.type === 'payTechnology' &&
          task.technology === 'extraction')) &&
      nextTask?.type === 'resource' &&
      nextTask.remaining === 2;
    if (extractionReward?.remaining > 0) {
      // Only the last actual material choice may also deliver this purchase's
      // already witnessed fixed effect. Settlement below checks the full state.
      const queuedGain =
        queuedPurchase?.stage === 'extraction' && extractionReward.remaining === 1
          ? queuedPurchase
          : null;
      if (
        opens ||
        action.type !== 'resource' ||
        actor !== extractionReward.actor ||
        before.round !== extractionReward.round ||
        task?.type !== 'resource' ||
        task.remaining !== extractionReward.remaining ||
        !['wood', 'stone', 'gold'].includes(action.resource) ||
        after.currentPlayer !== actor ||
        after.round !== before.round ||
        RESOURCES.some(
          (resource) =>
            after.players[actor].resources[resource] - before.players[actor].resources[resource] !==
            Number(resource === action.resource) +
              (queuedGain?.resource === resource ? queuedGain.amount : 0),
        ) ||
        (extractionReward.remaining > 1 &&
          (nextTask?.type !== 'resource' || nextTask.remaining !== extractionReward.remaining - 1))
      )
        fail('Extraction reward choice has no exact matching core material gain');
      extractionReward = {
        ...extractionReward,
        remaining: extractionReward.remaining - 1,
        materials: [...extractionReward.materials, action.resource],
      };
    } else {
      extractionReward = null;
      if (opens) {
        if (
          after.currentPlayer !== actor ||
          after.round !== before.round ||
          after.players[actor].technologies.extraction !== 3
        )
          fail('Extraction reward origin has no matching actor/core task');
        extractionReward = {
          actor,
          round: before.round,
          originActionId: context.id,
          remaining: 2,
          materials: [],
          captionActionId: null,
        };
      }
    }
  }
  function extractionCaption(event) {
    const actors = names.filter(
      (name) => event.rawText === `収集技術ボーナス: ${name}は好きな資源2個を得た`,
    );
    if (
      event.actor !== null ||
      event.iconResources.length ||
      actors.length !== 1 ||
      !extractionReward ||
      extractionReward.actor !== actorId(actors[0]) ||
      state().currentPlayer !== extractionReward.actor ||
      state().round !== extractionReward.round ||
      extractionReward.captionActionId !== null ||
      (extractionReward.remaining !== 0 && extractionReward.remaining !== 2) ||
      (extractionReward.remaining === 2 &&
        (pending() !== 'resource' || frame.observation.pendingTask.remaining !== 2)) ||
      (extractionReward.remaining === 0 && pending() === 'resource')
    )
      fail('Extraction caption has no unique unconsumed core reward receipt');
    extractionReward = { ...extractionReward, captionActionId: context.id };
    syncHistory();
  }
  function available(publicState, playerId) {
    return (
      publicState.players[playerId].workers -
      Object.values(publicState.gears)
        .flat()
        .filter((worker) => worker && !worker.dummy && worker.playerId === playerId).length -
      Number(publicState.firstPlayerClaimed === playerId)
    );
  }
  function newLogs(before, after) {
    for (let dropped = 0; dropped <= before.length; dropped++) {
      const retained = before.slice(dropped);
      if (retained.length <= after.length && retained.every((line, index) => line === after[index]))
        return after.slice(retained.length);
    }
    fail('Core log continuity is unknown');
  }
  function calendarTransition(before, after, action) {
    if (action.type === 'endTurn' && before.firstPlayerClaimed === before.currentPlayer) {
      const actor = before.currentPlayer;
      const amount = before.accumulatedCorn;
      if (
        after.accumulatedCorn !== 0 ||
        after.players[actor].resources.corn - before.players[actor].resources.corn < amount
      )
        fail('Accumulated corn gain is not separable from this core transition');
      calendarEvidence = {
        ...calendarEvidence,
        corn: { actor, amount, round: before.round, consumed: false },
      };
    }
    if (after.round > before.round || (before.phase !== 'finished' && after.phase === 'finished')) {
      const claimed = before.firstPlayerClaimed;
      const returned =
        claimed === null ? null : available(after, claimed) - available(before, claimed);
      if (
        claimed !== null &&
        (after.firstPlayerClaimed !== null ||
          returned < 1 ||
          after.players[claimed].workers !== before.players[claimed].workers)
      )
        fail('First-player worker return is not established by this core rotation');
      const announcement =
        calendarEvidence.announcement?.confirmed === false ? calendarEvidence.announcement : null;
      if (
        announcement &&
        (announcement.actor !== after.firstPlayer || announcement.round !== before.round)
      )
        fail('First-player announcement disagrees with the completed core rotation');
      calendarEvidence = {
        ...calendarEvidence,
        rotation: {
          actor: claimed,
          firstPlayer: after.firstPlayer,
          round: after.round,
          returned,
          consumed: false,
          announcementConsumed: announcement !== null,
        },
        announcement: announcement
          ? { ...announcement, round: after.round, confirmed: true }
          : null,
      };
    }
    const feeding = [];
    for (const line of newLogs(before.log, after.log)) {
      const player = before.players.find((value) =>
        line.startsWith(`食糧の日：${value.name} はコーン `),
      );
      if (!player) continue;
      const result = line
        .slice(`食糧の日：${player.name} はコーン `.length)
        .match(/^(\d+) を支払い(?:、未給食で −(\d+) 点)?$/);
      if (!result) fail('Unknown core feeding receipt');
      const cost = Number(result[1]);
      const penalty = Number(result[2] ?? 0);
      const free = player.feedAll ? player.workers : Math.min(player.feedWorkers, player.workers);
      const perWorker = Math.max(0, 2 - player.feedDiscount);
      const fed = perWorker === 0 ? player.workers : free + cost / perWorker;
      if (
        !Number.isSafeInteger(cost) ||
        !Number.isSafeInteger(fed) ||
        fed > player.workers ||
        penalty !== (player.workers - fed) * 3 ||
        (perWorker === 0 && cost !== 0)
      )
        fail('Core feeding cost/count cannot be established from the public player');
      feeding.push({
        actor: player.id,
        cost,
        fed,
        unfed: player.workers - fed,
        penalty,
        consumed: false,
        shortageConsumed: false,
      });
    }
    if (feeding.length) {
      if (![8, 14, 21, 27].includes(after.foodDays.at(-1)))
        fail('Core feeding receipt has no matching public food day');
      calendarEvidence = { ...calendarEvidence, feeding };
    }
    return feeding;
  }
  function feedingObserved(event) {
    const prefix = `食料供給日: ${event.actor}はワーカー`;
    const result =
      event.rawText.startsWith(prefix) &&
      event.rawText.slice(prefix.length).match(/^(\d+)個で\s*(\d+)\s*を支払った$/);
    const receipt = calendarEvidence.feeding.find((value) => value.actor === actorId(event.actor));
    if (
      !result ||
      !receipt ||
      receipt.consumed ||
      Number(result[1]) !== receipt.fed ||
      Number(result[2]) !== receipt.cost ||
      !same(event.iconResources, ['corn'])
    )
      fail('Observed feeding actor/count/payment has no matching core receipt');
    calendarEvidence = {
      ...calendarEvidence,
      feeding: calendarEvidence.feeding.map((value) =>
        value === receipt ? { ...value, consumed: true } : value,
      ),
    };
    syncHistory();
  }
  function shortageObserved(text) {
    const player = names.find((name) => text.startsWith(`食料供給日: ${name}はワーカー`));
    if (!player) return false;
    const result = text
      .slice(`食料供給日: ${player}はワーカー`.length)
      .match(/^(\d+)個分の食料が足りないため(\d+)点失った$/);
    const receipt = calendarEvidence.feeding.find((value) => value.actor === actorId(player));
    if (
      !result ||
      !receipt ||
      receipt.shortageConsumed ||
      Number(result[1]) !== receipt.unfed ||
      Number(result[2]) !== receipt.penalty
    )
      fail('Observed food shortage has no matching core receipt');
    calendarEvidence = {
      ...calendarEvidence,
      feeding: calendarEvidence.feeding.map((value) =>
        value === receipt ? { ...value, shortageConsumed: true } : value,
      ),
    };
    syncHistory();
    return true;
  }

  function trial(row) {
    const draws = { currentAge: [], age2: [] };
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return {
          frame: core({
            operation: 'publicApply',
            state: state(),
            actor: state().currentPlayer,
            move: row.move,
            refills: draws,
          }),
          draws,
        };
      } catch (error) {
        const need = error.message.match(
          /refills\.(currentAge|age2): expected (\d+) observed (?:age-II )?cards, got \d+/,
        );
        if (!need || draws[need[1]].length) throw error;
        const count = Number(need[2]);
        const pool = context.pools[need[1]];
        if (pool.length < count)
          fail(
            `Source ${context.id} requires ${count} observed ${need[1]} reveals at this operation`,
          );
        draws[need[1]] = pool.slice(0, count);
      }
    }
    fail('Refill planner did not converge');
  }
  function choose(
    predicate,
    reason,
    derived = false,
    visibleCalendar = false,
    outcomeFilter = null,
    validateSingleOutcome = false,
    allowOptionalSkip = true,
    buildingClass = null,
  ) {
    if (record.steps.length >= MAX_STEPS) fail('Move limit reached');
    let candidates = legal().filter((row) => predicate(row.action));
    // The following visible decision can establish declining an optional
    // continuation even when the same player remains the turn actor.
    for (
      let count = 0;
      !candidates.length &&
      allowOptionalSkip &&
      deferredScore?.kind !== 'action' &&
      count < 8 &&
      legal().some((row) => row.action.type === 'skip');
      count++
    ) {
      choose((action) => action.type === 'skip', `optional continuation before ${reason}`, true);
      candidates = legal().filter((row) => predicate(row.action));
    }
    if (!candidates.length) fail(`No legal operation for ${reason}; pending=${pending()}`);
    if (calls + candidates.length * 3 >= MAX_CALLS)
      fail('Core call budget exhausted before the final replay verification');
    // Trial every legal candidate before changing history, effects or reveal pools.
    let outcomes = candidates.map((row) => ({ row, ...trial(row) }));
    if (
      outcomeFilter !== null &&
      (validateSingleOutcome ||
        outcomes.some(
          (outcome) =>
            !same(
              mechanical(outcome.frame.snapshot.state),
              mechanical(outcomes[0].frame.snapshot.state),
            ),
        ))
    )
      outcomes = outcomes.filter(outcomeFilter(outcomes));
    if (!outcomes.length) fail(`No legal outcome matches source evidence for ${reason}`);
    if (buildingClass !== null) {
      const action = (row) => ({ ...row.action, id: buildingClass[0] });
      if (
        outcomes.some(
          ({ row }) =>
            row.action.renovation !== null || !same(action(row), action(outcomes[0].row)),
        )
      )
        fail('Equivalent building choices have different typed actions');
      outcomes.sort((a, b) =>
        a.row.action.id < b.row.action.id ? -1 : a.row.action.id > b.row.action.id ? 1 : 0,
      );
    }
    const outcomeState = (outcome) =>
      buildingClass === null
        ? mechanical(outcome.frame.snapshot.state)
        : equivalentBuildingState(outcome.frame.snapshot.state, buildingClass);
    if (
      outcomes.length > 1 &&
      outcomes.some((outcome) => !same(outcomeState(outcome), outcomeState(outcomes[0])))
    )
      fail(`Ambiguous legal operation for ${reason}`);
    const chosen = outcomes[0];
    const previous = state();
    if (deferredScore) deferredScoreTransition(chosen.frame, chosen.row.action);
    if (
      derived &&
      !visibleCalendar &&
      (chosen.frame.snapshot.state.round !== previous.round ||
        chosen.frame.snapshot.state.phase !== previous.phase)
    )
      fail('Derived actor boundary requires an observed calendar transition');
    if (
      derived &&
      visibleCalendar &&
      chosen.row.action.type === 'endTurn' &&
      chosen.frame.snapshot.state.round === previous.round &&
      chosen.frame.snapshot.state.phase === previous.phase &&
      (pending() === 'rotation' || chosen.frame.observation?.pendingTask?.type !== 'rotation')
    )
      fail('Calendar boundary has no completed or pending core rotation');
    extractionTransition(chosen.frame, chosen.row.action);
    const feeding = calendarTransition(previous, chosen.frame.snapshot.state, chosen.row.action);
    if (feeding.length) {
      context.feeding.push(...feeding);
      context.feedingDay = chosen.frame.snapshot.state.foodDays.at(-1);
    }
    const resourceCost = typedResourceCost(chosen.row.action);
    for (const player of previous.players) {
      const next = chosen.frame.snapshot.state.players[player.id];
      const food = feeding.find((receipt) => receipt.actor === player.id);
      if (
        player.id === previous.currentPlayer &&
        chosen.frame.snapshot.state.currentPlayer === player.id &&
        chosen.frame.snapshot.state.round === previous.round &&
        player.doubleAdvanceAvailable === false &&
        next.doubleAdvanceAvailable === true &&
        templeMaximums.some(
          ([temple, maximum]) =>
            player.temples[temple] < maximum && next.temples[temple] === maximum,
        )
      ) {
        const key = `${player.id}:doubleAdvanceAvailable:1`;
        context.effects[key] = (context.effects[key] ?? 0) + 1;
      }
      for (const field of [
        'workers',
        'score',
        ...RESOURCES.map((resource) => `resource:${resource}`),
        ...['chaac', 'quetzalcoatl', 'kukulkan'].map((temple) => `temple:${temple}`),
        ...TECHNOLOGIES.map((technology) => `technology:${technology}`),
      ]) {
        const delta = field.startsWith('resource:')
          ? next.resources[field.slice(9)] - player.resources[field.slice(9)]
          : field.startsWith('temple:')
            ? next.temples[field.slice(7)] - player.temples[field.slice(7)]
            : field.startsWith('technology:')
              ? next.technologies[field.slice(11)] - player.technologies[field.slice(11)]
              : next[field] - player[field];
        const cost = field.startsWith('resource:')
          ? (player.id === previous.currentPlayer
              ? resourceCost[RESOURCES.indexOf(field.slice(9))]
              : 0) + (field === 'resource:corn' ? (food?.cost ?? 0) : 0)
          : field === 'score'
            ? (food?.penalty ?? 0)
            : 0;
        const gain = delta + cost;
        for (const [direction, amount] of [
          [1, Math.max(0, gain)],
          [-1, cost + Math.max(0, -gain)],
        ])
          if (amount) {
            const key = `${player.id}:${field}:${direction}`;
            context.effects[key] = (context.effects[key] ?? 0) + amount;
          }
      }
    }
    rotationCredit += Math.max(0, chosen.frame.snapshot.state.round - previous.round);
    if (previous.phase !== 'finished' && chosen.frame.snapshot.state.phase === 'finished')
      rotationCredit++;
    for (const field of ['currentAge', 'age2'])
      context.pools[field].splice(0, chosen.draws[field].length);
    const actor = state().currentPlayer;
    record.steps.push({
      actor,
      move: copy(chosen.row.move),
      sourceActionIds: [context.id],
      refills: chosen.draws,
      checkpoint: null,
    });
    frame = chosen.frame;
    history.push({
      frame: copy(frame),
      steps: record.steps.length,
      rotationCredit,
      declaredRotation,
      calendarEvidence,
      extractionReward,
      unsettledPurchase: deferredScore !== null,
    });
    audit.push({
      actionId: context.id,
      step: record.steps.length,
      actor,
      action: chosen.row.action,
      derivedFromVisibleBoundary: derived,
      equivalentChoices: outcomes.length,
      ...(buildingClass === null
        ? {}
        : {
            physicalSelectionUnknown: true,
            buildingEquivalenceClass: copy(buildingClass),
            equivalentLegalTargets: outcomes.map(({ row }) => copy(row)),
            canonicalLocalTarget: copy(chosen.row),
          }),
    });
    return chosen.row.action;
  }
  function finishOptional(reason) {
    if (legal().some((row) => row.action.type === 'skip'))
      return choose((action) => action.type === 'skip', reason, true);
    fail(`Missing observed choice before ${reason}; pending=${pending()}`);
  }
  function alignActor(name) {
    const target = actorId(name);
    if (target < 0) fail('Unknown source actor');
    for (let count = 0; state().currentPlayer !== target && count < 8; count++) {
      if (legal().some((row) => row.action.type === 'endTurn' && row.action.doubleAdvance === null))
        choose(
          (action) => action.type === 'endTurn' && action.doubleAdvance === null,
          `visible actor switch to ${name}`,
          true,
        );
      else finishOptional(`visible actor switch to ${name}`);
    }
    if (state().currentPlayer !== target) fail('Actor switch did not resolve');
  }
  function calendar(days = null) {
    // A visible calendar boundary also witnesses the end of optional trading.
    if (pending() === 'trade') finishOptional('visible calendar boundary');
    if (legal().some((row) => row.action.type === 'endTurn'))
      choose(
        (action) => action.type === 'endTurn' && action.doubleAdvance === null,
        'visible end-of-round',
        true,
        true,
      );
    if (days !== null && pending() === 'rotation')
      choose(
        (action) => action.type === 'rotate' && action.days === days,
        `visible rotation ${days}`,
      );
    if (days !== null) {
      if (rotationCredit < days) fail('Source rotation has no matching core calendar transition');
      rotationCredit -= days;
      syncHistory();
    }
  }
  function pay(values) {
    if (!values.some(Boolean)) return false;
    if (!['theology', 'payTechnology', 'payResource'].includes(pending())) return false;
    if (context.paymentActor !== state().currentPlayer)
      fail('Observed payment actor differs from the pending payment actor');
    const remaining = values.map((amount, index) =>
      Math.max(
        0,
        amount - (context.effects[`${state().currentPlayer}:resource:${RESOURCES[index]}:-1`] ?? 0),
      ),
    );
    if (!remaining.some(Boolean)) {
      if (context.paymentOperations && pending() !== 'theology')
        fail('Observed payment was already consumed by another payment operation');
      return false;
    }
    if (pending() === 'theology') {
      const material = remaining.findIndex((amount) => amount !== 0);
      if (
        material < 1 ||
        material > 3 ||
        remaining.some((amount, index) => amount !== (index === material ? 1 : 0))
      )
        fail('Observed theology offering must be one wood, stone, or gold');
      choose(
        (action) => action.type === 'offering' && action.resource === RESOURCES[material],
        'observed theology offering',
      );
      context.paymentOperations++;
      return true;
    }
    choose(
      (action) =>
        action.type === 'payment' &&
        action.resources.every((amount, index) => amount <= remaining[index]),
      'observed payment',
    );
    context.paymentOperations++;
    return true;
  }
  function automatic(event, field, amount, direction = 1) {
    const id = actorId(event.actor);
    if (id < 0) fail('Automatic effect has no known actor');
    const key = `${id}:${field}:${direction}`;
    const consumed = context.claims[key] ?? 0;
    if ((context.effects[key] ?? 0) < consumed + amount)
      fail(`Source effect ${field}=${direction * amount} is not established by this macro`);
    context.claims[key] = consumed + amount;
  }
  function unclaimed(event, field, direction = 1) {
    const id = actorId(event.actor);
    if (id < 0) fail('Automatic effect has no known actor');
    const key = `${id}:${field}:${direction}`;
    return (context.effects[key] ?? 0) - (context.claims[key] ?? 0);
  }
  function fixedScoreCaption(event) {
    const match =
      event.kind === 'unparsed' &&
      event.actor !== null &&
      event.iconResources.length === 0 &&
      event.rawText.slice(event.actor.length).match(/^は(\d+)点を獲得した$/);
    const amount = match ? Number(match[1]) : null;
    return Number.isSafeInteger(amount) && amount > 0 && amount <= 1000 ? amount : null;
  }
  // Only this selected catalog pair can caption points before the actual Trade
  // exit. Every source checkpoint remains at its original macro boundary.
  function deferredScoreShape(outcome, event, events, witness, previous = state()) {
    const actor = actorId(event.actor);
    const action = outcome.row.action;
    const effects = buildings.get(action.id)?.effects;
    const next = outcome.frame.snapshot.state;
    const before = previous.pending;
    const after = next.pending?.after;
    const scoreEvents = events.filter((item) => fixedScoreCaption(item) !== null);
    if (
      actor < 0 ||
      witnesses.catalogHash !== catalogHash ||
      event.kind !== 'build' ||
      action.type !== 'build' ||
      typeof action.architecture !== 'boolean' ||
      action.renovation !== null ||
      !same(witness.buildings, [action.id]) ||
      !same(typedResourceCost(action), paymentOf(events)) ||
      events.some((item) => item.actor !== event.actor) ||
      events.filter((item) => item.kind === 'build').length !== 1 ||
      scoreEvents.length !== 1 ||
      events.at(-1) !== scoreEvents[0] ||
      events.some(
        (item) => item !== event && item !== scoreEvents[0] && item.kind !== 'resourcePayment',
      ) ||
      events.slice(events.indexOf(event) + 1).some((item) => item !== scoreEvents[0]) ||
      !same(effects, [
        { type: 'trade' },
        { type: 'points', amount: fixedScoreCaption(scoreEvents[0]) },
      ]) ||
      before?.task?.type !== 'build' ||
      before.task.remaining !== 2 ||
      !same(before.after, []) ||
      !same(next.pending?.task, { type: 'trade' }) ||
      !same(outcome.frame.observation?.pendingTask, next.pending.task) ||
      !Array.isArray(after) ||
      after.length !== 2 ||
      !same(after[0], { type: 'effects', effects: [effects[1]] }) ||
      !same(after[1], {
        type: 'build',
        remaining: 1,
        allowMonument: false,
        cornPayment: before.task.cornPayment,
        architectureAvailable: before.task.architectureAvailable !== false && !action.architecture,
        ...(before.task.mandatory ? { mandatory: true } : {}),
      }) ||
      next.currentPlayer !== actor ||
      previous.currentPlayer !== actor ||
      next.round !== previous.round ||
      next.phase !== previous.phase ||
      !same(next.players[actor].buildings, [...previous.players[actor].buildings, action.id]) ||
      !same(
        next.buildings,
        previous.buildings.filter((id) => id !== action.id),
      ) ||
      !same(
        { ...mechanical(next), players: null, pending: null, buildings: null },
        { ...mechanical(previous), players: null, pending: null, buildings: null },
      ) ||
      next.players.some((player, id) =>
        id === actor
          ? player.score !== previous.players[id].score ||
            RESOURCES.some(
              (resource, index) =>
                player.resources[resource] !==
                previous.players[id].resources[resource] - action.cost[index],
            ) ||
            !same(
              { ...player, resources: null, buildings: null },
              { ...previous.players[id], resources: null, buildings: null },
            )
          : !same(player, previous.players[id]),
      )
    )
      return null;
    return {
      actor,
      amount: effects[1].amount,
      scoreEvent: scoreEvents[0],
      pending: copy(next.pending),
      round: next.round,
      phase: next.phase,
      trades: 0,
      captionObserved: false,
    };
  }
  function deferredScoreTransition(nextFrame, action) {
    const receipt = deferredScore;
    if (receipt.kind === 'action') return deferredActionScoreTransition(nextFrame, action);
    const before = state();
    const next = nextFrame.snapshot.state;
    const exit = action.type === 'skip' && context.deferredScoreExit === true;
    if (
      (!exit && action.type !== 'trade') ||
      !same(before.pending, receipt.pending) ||
      before.currentPlayer !== receipt.actor ||
      before.round !== receipt.round ||
      before.phase !== receipt.phase ||
      next.currentPlayer !== receipt.actor ||
      next.round !== receipt.round ||
      next.phase !== receipt.phase ||
      !same(
        { ...mechanical(next), players: null, pending: null },
        { ...mechanical(before), players: null, pending: null },
      ) ||
      (exit
        ? !same(next.pending?.task, receipt.pending.after[1]) ||
          !same(nextFrame.observation.pendingTask, receipt.pending.after[1]) ||
          !same(next.pending?.after, [])
        : !same(next.pending, receipt.pending) ||
          !same(nextFrame.observation.pendingTask, receipt.pending.task)) ||
      next.players.some((player, id) =>
        id === receipt.actor
          ? player.score - before.players[id].score !== (exit ? receipt.amount : 0) ||
            !same(
              { ...player, resources: null, score: null },
              { ...before.players[id], resources: null, score: null },
            ) ||
            (exit
              ? !same(player.resources, before.players[id].resources)
              : RESOURCES.some(
                  (resource) =>
                    resource !== 'corn' &&
                    player.resources[resource] - before.players[id].resources[resource] !==
                      (resource === action.resource ? (action.buy ? 1 : -1) : 0),
                ))
          : !same(player, before.players[id]),
      )
    )
      fail('Deferred purchase score requires the exact Trade queue and isolated legal exit');
    if (!exit) deferredScore = { ...receipt, trades: receipt.trades + 1 };
  }
  function finishDeferredScore() {
    const receipt = deferredScore;
    if (!receipt.captionObserved || receipt.trades < 1)
      fail('Deferred purchase requires its one score caption and an actual observed Trade');
    context.deferredScoreExit = true;
    try {
      choose((action) => action.type === 'skip', 'observed Trade exit before next Build', true);
    } finally {
      context.deferredScoreExit = false;
    }
    automatic(receipt.scoreEvent, 'score', receipt.amount);
    audit.push({
      actionId: context.id,
      deferredScoreSourceActionId: receipt.sourceId,
      actor: receipt.actor,
      amount: receipt.amount,
      actualDelivered: true,
      humanPlayingBcReady: false,
    });
    deferredScore = null;
  }
  function actionScorePair(id) {
    const effects = buildings.get(id)?.effects;
    return Array.isArray(effects) &&
      effects.length === 2 &&
      same(effects[0], { type: 'action' }) &&
      same(effects[1], { type: 'points', amount: effects[1].amount }) &&
      Number.isSafeInteger(effects[1].amount) &&
      effects[1].amount > 0 &&
      effects[1].amount <= 1000
      ? effects
      : null;
  }
  // This receipt retains the printed points through one actual Anywhere action
  // and monument. It never moves the original purchase checkpoint forward.
  function deferredActionScoreShape(outcome, event, events, witness, previous = state()) {
    const action = outcome.row.action;
    const effects = actionScorePair(action.id);
    const actor = actorId(event.actor);
    const before = previous.pending;
    const next = outcome.frame.snapshot.state;
    const scores = events.filter((item) => fixedScoreCaption(item) !== null);
    const gains = events.filter((item) => item.kind === 'resourceGain');
    const observed = witness.expected?.players?.[actor];
    if (
      !effects ||
      actor < 0 ||
      witnesses.catalogHash !== catalogHash ||
      event.kind !== 'build' ||
      action.type !== 'build' ||
      action.architecture !== true ||
      action.renovation !== null ||
      previous.players[actor].technologies.architecture < 2 ||
      !same(witness.buildings, [action.id]) ||
      !Number.isFinite(observed?.score) ||
      !RESOURCES.every((resource) => Number.isSafeInteger(observed?.resources?.[resource])) ||
      !matches(next, witness.expected) ||
      !same(typedResourceCost(action), paymentOf(events)) ||
      events.some((item) => item.actor !== event.actor) ||
      events.filter((item) => item.kind === 'build').length !== 1 ||
      scores.length !== 2 ||
      fixedScoreCaption(scores[0]) !== 2 ||
      fixedScoreCaption(scores[1]) !== effects[1].amount ||
      events.at(-2) !== scores[0] ||
      events.at(-1) !== scores[1] ||
      gains.length !== 1 ||
      !same(resourceAmounts(gains[0]), [1, 0, 0, 0, 0]) ||
      events.indexOf(gains[0]) !== events.indexOf(event) + 1 ||
      events.some(
        (item) =>
          item !== event &&
          !scores.includes(item) &&
          item !== gains[0] &&
          ((!emptyPurchasePayment(item) && item.kind !== 'resourcePayment') ||
            events.indexOf(item) > events.indexOf(event)),
      ) ||
      before?.task?.type !== 'build' ||
      before.task.remaining !== 1 ||
      !same(before.after, []) ||
      !same(next.pending?.task, { type: 'anyAction', excludeSkulls: true, cost: 1 }) ||
      !same(outcome.frame.observation?.pendingTask, next.pending.task) ||
      !same(next.pending.after, [
        { type: 'effects', effects: [effects[1]] },
        {
          type: 'build',
          remaining: 0,
          allowMonument: false,
          cornPayment: before.task.cornPayment,
          architectureAvailable: false,
          ...(before.task.mandatory ? { mandatory: true } : {}),
        },
      ]) ||
      next.currentPlayer !== actor ||
      previous.currentPlayer !== actor ||
      next.round !== previous.round ||
      next.phase !== previous.phase ||
      !same(next.players[actor].buildings, [...previous.players[actor].buildings, action.id]) ||
      !same(
        next.buildings,
        previous.buildings.filter((id) => id !== action.id),
      ) ||
      !same(
        { ...mechanical(next), players: null, pending: null, buildings: null },
        { ...mechanical(previous), players: null, pending: null, buildings: null },
      ) ||
      next.players.some((player, id) =>
        id === actor
          ? player.score - previous.players[id].score !== 2 ||
            RESOURCES.some(
              (resource, index) =>
                player.resources[resource] !==
                previous.players[id].resources[resource] - action.cost[index] + Number(index === 0),
            ) ||
            !same(
              { ...player, resources: null, score: null, buildings: null },
              { ...previous.players[id], resources: null, score: null, buildings: null },
            )
          : !same(player, previous.players[id]),
      )
    )
      return null;
    return {
      kind: 'action',
      stage: 'anyAction',
      actor,
      amount: effects[1].amount,
      scoreEvent: scores[1],
      captionObserved: false,
      pending: copy(next.pending),
      round: next.round,
      phase: next.phase,
    };
  }
  function deferredActionScoreTransition(nextFrame, action) {
    const receipt = deferredScore;
    const before = state();
    const next = nextFrame.snapshot.state;
    const anywhere = receipt.stage === 'anyAction';
    const validAction = anywhere
      ? action.type === 'useAction' &&
        action.source === 'anywhere' &&
        action.gear === 'tikal' &&
        action.position === 4 &&
        action.cornCost === 1
      : receipt.stage === 'monument' &&
        action.type === 'monument' &&
        action.renovation === null &&
        same(typedResourceCost(action), paymentOf(context.events));
    if (
      !receipt.captionObserved ||
      !validAction ||
      !same(before.pending, receipt.pending) ||
      before.currentPlayer !== receipt.actor ||
      before.round !== receipt.round ||
      before.phase !== receipt.phase ||
      next.currentPlayer !== receipt.actor ||
      next.round !== receipt.round ||
      next.phase !== receipt.phase ||
      (anywhere
        ? !same(next.pending?.task, {
            type: 'build',
            remaining: 2,
            allowMonument: true,
            cornPayment: false,
          }) ||
          !same(nextFrame.observation?.pendingTask, next.pending.task) ||
          !same(next.pending.after, before.pending.after)
        : next.pending !== null || nextFrame.observation?.pendingTask !== null) ||
      !same(
        { ...mechanical(next), players: null, pending: null, monuments: null },
        { ...mechanical(before), players: null, pending: null, monuments: null },
      ) ||
      !same(
        next.monuments,
        anywhere ? before.monuments : before.monuments.filter((id) => id !== action.id),
      ) ||
      next.players.some((player, id) =>
        id === receipt.actor
          ? player.score - before.players[id].score !== (anywhere ? 0 : receipt.amount) ||
            !same(
              player.monuments,
              anywhere
                ? before.players[id].monuments
                : [...before.players[id].monuments, action.id],
            ) ||
            RESOURCES.some(
              (resource, index) =>
                player.resources[resource] !==
                before.players[id].resources[resource] - typedResourceCost(action)[index],
            ) ||
            !same(
              { ...player, resources: null, score: null, monuments: null },
              { ...before.players[id], resources: null, score: null, monuments: null },
            )
          : !same(player, before.players[id]),
      )
    )
      fail('Deferred purchase requires its exact observed Anywhere action and monument completion');
    deferredScore = {
      ...receipt,
      stage: anywhere ? 'monument' : 'complete',
      pending: copy(next.pending),
    };
  }
  // A source can caption a fixed card resource before its required technology
  // choice. Keep that purchase tentative until the next actual choice delivers
  // the resource. This normalizes log order, not Human BC decision features.
  function cornTechnologyPair(id) {
    const effects = buildings.get(id)?.effects;
    const amount = effects?.[1]?.resources?.corn;
    return Number.isSafeInteger(amount) &&
      amount > 0 &&
      amount <= 1000 &&
      same(effects, [
        { type: 'technology', technology: 'any', steps: 1 },
        { type: 'resources', resources: { corn: amount } },
      ])
      ? amount
      : null;
  }
  function theologySkullCaption(event) {
    if (event.kind !== 'unparsed' || event.iconResources.length) return null;
    const actors = names.filter(
      (name) => event.rawText === `神学技術ボーナス: ${name}は追加で水晶ドクロ1個を得た`,
    );
    return actors.length === 1 ? actors[0] : null;
  }
  function cappedTheologyPurchase(outcome, event, events, witness, previous = state()) {
    const action = outcome.row.action;
    const actor = actorId(event.actor);
    const next = outcome.frame.snapshot.state;
    const before = previous.players[actor];
    const effects = buildings.get(action.id)?.effects;
    const temples = effects?.slice(1);
    if (
      witnesses.catalogHash !== catalogHash ||
      actor < 0 ||
      action.type !== 'build' ||
      typeof action.architecture !== 'boolean' ||
      action.renovation != null ||
      !same(witness.buildings, [action.id]) ||
      !same(effects?.[0], { type: 'technology', technology: 'theology', steps: 1 }) ||
      !Array.isArray(temples) ||
      temples.length > 3 ||
      temples.some(
        (effect) =>
          !Object.values(TEMPLES).includes(effect.temple) ||
          !same(effect, { type: 'temple', temple: effect.temple, steps: 1 }),
      ) ||
      new Set(temples.map((effect) => effect.temple)).size !== temples.length ||
      before.technologies.theology !== 3 ||
      previous.currentPlayer !== actor ||
      previous.pending?.task?.type !== 'build' ||
      previous.pending.task.remaining !== 1 ||
      previous.pending.task.cornPayment !== false ||
      !same(previous.pending.after, []) ||
      next.pending !== null ||
      outcome.frame.observation.pendingTask !== null ||
      previous.skullSupply < 1 ||
      next.skullSupply !== previous.skullSupply - 1 ||
      events.filter((item) => theologySkullCaption(item) === event.actor).length !== 1 ||
      events.filter(
        (item) =>
          item.kind === 'technology' &&
          item.actor === event.actor &&
          item.detail.technology === 'theology' &&
          item.detail.steps === 1,
      ).length !== 1
    )
      return false;
    const cost = RESOURCES.map((resource) => buildings.get(action.id).cost[resource] ?? 0);
    if (cost.some((amount) => !Number.isSafeInteger(amount) || amount < 0)) return false;
    if (action.architecture) {
      const index = RESOURCES.indexOf(action.discountResource);
      if (before.technologies.architecture !== 3 || index < 1 || index > 3 || cost[index] < 1)
        return false;
      cost[index]--;
    } else if (action.discountResource != null) return false;
    const own = witness.expected?.players?.[actor];
    return (
      same(typedResourceCost(action), cost) &&
      same(paymentOf(events), cost) &&
      same(
        next.buildings,
        previous.buildings.filter((id) => id !== action.id),
      ) &&
      same(
        { ...mechanical(next), players: null, pending: null, buildings: null, skullSupply: null },
        {
          ...mechanical(previous),
          players: null,
          pending: null,
          buildings: null,
          skullSupply: null,
        },
      ) &&
      next.players.every((player, id) =>
        id === actor
          ? same(player.buildings, [...before.buildings, action.id]) &&
            same(player.technologies, before.technologies) &&
            Object.values(TEMPLES).every(
              (temple) =>
                player.temples[temple] ===
                before.temples[temple] + Number(temples.some((effect) => effect.temple === temple)),
            ) &&
            player.score === before.score + (action.architecture ? 2 : 0) &&
            RESOURCES.every(
              (resource, index) =>
                player.resources[resource] ===
                  before.resources[resource] -
                    cost[index] +
                    (resource === 'skull'
                      ? 1
                      : resource === 'corn'
                        ? Number(action.architecture)
                        : 0) && own?.resources?.[resource] === player.resources[resource],
            ) &&
            same(
              {
                ...player,
                resources: null,
                technologies: null,
                temples: null,
                score: null,
                buildings: null,
              },
              {
                ...before,
                resources: null,
                technologies: null,
                temples: null,
                score: null,
                buildings: null,
              },
            )
          : same(player, previous.players[id]),
      )
    );
  }
  function queuedCornTechnology(previous, actor) {
    // Bind the choice to the next source group, never to a later board result.
    const [sourceId, events] = context.nextSource ?? [];
    const first = events?.[0];
    if (
      first?.kind !== 'technology' ||
      first.detail.steps !== 1 ||
      actorId(first.actor) !== actor ||
      !TECHNOLOGIES.includes(first.detail.technology)
    )
      return null;
    const technology = first.detail.technology;
    const level = previous.players[actor].technologies[technology];
    if (!Number.isSafeInteger(level) || level < 0 || level > 3) return null;
    if (
      level < 3
        ? events.length !== 1
        : technology !== 'extraction' ||
          events.length !== 2 ||
          events[1].kind !== 'unparsed' ||
          events[1].actor !== null ||
          events[1].iconResources.length !== 0 ||
          events[1].rawText !== `収集技術ボーナス: ${first.actor}は好きな資源2個を得た`
    )
      return null;
    return { kind: 'cornTechnology', technology, technologySourceId: sourceId };
  }
  function queuedCornPurchase(outcome, witness, events, previous, actor, amount, continuation) {
    const action = outcome.row.action;
    const next = outcome.frame.snapshot.state;
    const cost = RESOURCES.map((resource) => buildings.get(action.id)?.cost[resource] ?? 0);
    const own = witness.expected?.players?.[actor];
    return (
      witnesses.catalogHash === catalogHash &&
      cornTechnologyPair(action.id) === amount &&
      action.architecture === false &&
      previous.pending.task.cornPayment === false &&
      same(witness.buildings, [action.id]) &&
      same(typedResourceCost(action), cost) &&
      same(paymentOf(events), cost) &&
      (continuation !== null ||
        (object(own?.resources) && !Object.hasOwn(own.resources, 'corn'))) &&
      previous.currentPlayer === actor &&
      (continuation !== null || previous.players[actor].technologies.theology === 3) &&
      same(next.pending.after, [
        { type: 'effects', effects: [{ type: 'resources', resources: { corn: amount } }] },
        {
          type: 'build',
          remaining: 1,
          allowMonument: false,
          cornPayment: false,
          architectureAvailable: previous.pending.task.architectureAvailable !== false,
          ...(previous.pending.task.mandatory ? { mandatory: true } : {}),
        },
      ]) &&
      same(next.players[actor].buildings, [...previous.players[actor].buildings, action.id]) &&
      same(
        next.buildings,
        previous.buildings.filter((id) => id !== action.id),
      ) &&
      same(
        { ...mechanical(next), players: null, pending: null, buildings: null },
        { ...mechanical(previous), players: null, pending: null, buildings: null },
      ) &&
      next.players.every((player, id) =>
        id === actor
          ? RESOURCES.every(
              (resource, index) =>
                player.resources[resource] ===
                previous.players[id].resources[resource] - cost[index],
            ) &&
            same(
              { ...player, resources: null, buildings: null },
              { ...previous.players[id], resources: null, buildings: null },
            )
          : same(player, previous.players[id]),
      )
    );
  }
  function queuedPurchaseShape(outcome, event, events, witness, previous = state()) {
    const actor = actorId(event.actor);
    const before = previous.pending;
    const next = outcome.frame.snapshot.state;
    const task = next.pending?.task;
    const after = next.pending?.after;
    const observed = witness.expected?.players;
    const own = observed?.[actor];
    const gainEvents = events.filter((item) => item.kind === 'resourceGain');
    const action = outcome.row.action;
    const corn = after?.[0]?.effects?.[0]?.resources?.corn !== undefined;
    if (
      actor < 0 ||
      event.kind !== 'build' ||
      action.type !== 'build' ||
      typeof action.architecture !== 'boolean' ||
      action.renovation !== null ||
      action.discountResource !== null ||
      events.some((item) => item.actor !== event.actor) ||
      events.filter((item) => item.kind === 'build').length !== 1 ||
      events.some((item) => !['resourcePayment', 'build', 'resourceGain'].includes(item.kind)) ||
      gainEvents.length !== 1 ||
      events.at(-1) !== gainEvents[0] ||
      before?.task?.type !== 'build' ||
      before.task.remaining !== 2 ||
      !Array.isArray(before.after) ||
      before.after.length !== 0 ||
      task?.type !== 'technology' ||
      task.remaining !== 1 ||
      task.free !== true ||
      task.mandatory === true ||
      !same(task, outcome.frame.observation?.pendingTask) ||
      !Array.isArray(after) ||
      after.length !== 2 ||
      after[0]?.type !== 'effects' ||
      !Array.isArray(after[0].effects) ||
      after[0].effects.length !== 1 ||
      after[0].effects[0].type !== 'resources' ||
      !object(after[0].effects[0].resources) ||
      after[1]?.type !== 'build' ||
      after[1].remaining !== 1 ||
      after[1].allowMonument !== false ||
      after[1].cornPayment !== before.task.cornPayment ||
      after[1].architectureAvailable !==
        (before.task.architectureAvailable !== false && !action.architecture) ||
      Boolean(after[1].mandatory) !== Boolean(before.task.mandatory) ||
      next.currentPlayer !== actor ||
      next.round !== previous.round ||
      next.phase !== previous.phase ||
      (witness.pendingTask !== undefined && witness.pendingTask !== 'technology')
    )
      return null;
    const resources = Object.entries(after[0].effects[0].resources);
    if (resources.length !== 1) return null;
    const [resource, amount] = resources[0];
    if (
      !(corn ? resource === 'corn' : RESOURCES.slice(1).includes(resource)) ||
      !Number.isSafeInteger(amount) ||
      amount < 1 ||
      amount > 1000 ||
      !same(
        resourceAmounts(gainEvents[0]),
        RESOURCES.map((r) => (r === resource ? amount : 0)),
      )
    )
      return null;
    const continuation = corn ? queuedCornTechnology(previous, actor) : null;
    // A source-bound cornTechnology choice has an exact whole-state transition.
    // Omitted counters remain unknown; supplied checkpoints are still compared.
    // Keep the older material and capped Theology observation requirements.
    if (
      continuation === null &&
      (!Array.isArray(observed) ||
        observed.length !== previous.players.length ||
        (!corn && (!Number.isSafeInteger(own?.resources?.corn) || own.resources.corn < 0)) ||
        !Number.isFinite(own?.score))
    )
      return null;
    if (
      corn &&
      !queuedCornPurchase(outcome, witness, events, previous, actor, amount, continuation)
    )
      return null;
    return {
      actor,
      resource,
      amount,
      pending: copy(next.pending),
      gainEvent: gainEvents[0],
      ...(corn ? (continuation ?? { kind: 'cornTheology', bonusClaimed: false }) : {}),
      ...(continuation
        ? {
            purchaseCounterEvidence: {
              score: Object.hasOwn(own ?? {}, 'score') ? 'sourceCheckpoint' : 'unknown',
              resourceFields: object(own?.resources) ? Object.keys(own.resources) : [],
              coreState: 'derivedFromVerifiedPrefix',
            },
          }
        : {}),
    };
  }
  function finishQueuedPurchase(event) {
    const receipt = queuedPurchase;
    const next = state();
    const before = context.before;
    const actor = receipt.actor;
    if (receipt.stage === 'extraction') return finishQueuedExtraction(event);
    const technology = event.detail.technology;
    const expected = receipt.pending.after[1];
    const corn = receipt.kind === 'cornTheology';
    if (
      !corn &&
      technology === 'extraction' &&
      before.players[actor].technologies.extraction === 3
    ) {
      const after = [{ ...receipt.pending.task, remaining: 0 }, ...receipt.pending.after];
      const unchanged = mechanical(before);
      unchanged.pending = copy(next.pending);
      if (
        record.steps.length !== receipt.stepCount + 1 ||
        next.pending?.task?.type !== 'resource' ||
        next.pending.task.remaining !== 2 ||
        !same(next.pending.task, frame.observation.pendingTask) ||
        !same(next.pending.after, after) ||
        !same(mechanical(next), unchanged) ||
        extractionReward?.actor !== actor ||
        extractionReward.originActionId !== context.id ||
        extractionReward.captionActionId !== context.id ||
        extractionReward.remaining !== 2 ||
        extractionReward.materials.length !== 0
      )
        fail('Queued purchase has no exact capped Extraction reward and retained effect queue');
      queuedPurchase = {
        ...receipt,
        stage: 'extraction',
        technologySourceId: context.id,
        resourceSourceIds: [],
      };
      return false;
    }
    const uncapped = receipt.kind === 'cornTechnology' ? mechanical(before) : null;
    if (uncapped) {
      uncapped.players[actor].technologies[technology]++;
      uncapped.players[actor].resources.corn += receipt.amount;
      uncapped.pending = copy(next.pending);
    }
    if (
      record.steps.length !== receipt.stepCount + 1 ||
      next.currentPlayer !== actor ||
      next.round !== receipt.round ||
      next.phase !== before.phase ||
      !same(next.pending?.task, expected) ||
      !same(frame.observation.pendingTask, expected) ||
      !same(next.pending?.after, []) ||
      next.players[actor].technologies[technology] !==
        before.players[actor].technologies[technology] + Number(!corn) ||
      next.players.some((player, id) =>
        RESOURCES.some(
          (resource) =>
            player.resources[resource] !==
            before.players[id].resources[resource] +
              (id === actor
                ? resource === receipt.resource
                  ? receipt.amount
                  : corn && resource === 'skull'
                    ? 1
                    : 0
                : 0),
        ),
      ) ||
      next.players[actor].score !== before.players[actor].score ||
      (uncapped !== null && !same(mechanical(next), uncapped)) ||
      (corn &&
        (!receipt.bonusClaimed ||
          technology !== 'theology' ||
          before.players[actor].technologies.theology !== 3 ||
          before.skullSupply < 1 ||
          next.skullSupply !== before.skullSupply - 1 ||
          !same(
            { ...mechanical(next), players: null, pending: null, skullSupply: null },
            { ...mechanical(before), players: null, pending: null, skullSupply: null },
          ) ||
          next.players.some(
            (player, id) =>
              !same({ ...player, resources: null }, { ...before.players[id], resources: null }),
          ))) ||
      unclaimed(receipt.gainEvent, `resource:${receipt.resource}`) !== receipt.amount
    )
      fail('Queued purchase resource did not settle through the observed technology choice');
    automatic(receipt.gainEvent, `resource:${receipt.resource}`, receipt.amount);
    audit.push({
      actionId: context.id,
      queuedPurchaseSourceActionId: receipt.sourceId,
      actor,
      resource: receipt.resource,
      amount: receipt.amount,
      actualDelivered: true,
      ...(corn ? { cappedTheologySkull: 1, humanPlayingBcReady: false } : {}),
      ...(uncapped
        ? {
            technologySourceActionId: receipt.technologySourceId,
            continuation: 'uncappedTechnology',
            humanPlayingBcReady: false,
          }
        : {}),
      ...(receipt.purchaseCounterEvidence
        ? { purchaseCounterEvidence: copy(receipt.purchaseCounterEvidence) }
        : {}),
    });
    return true;
  }
  function finishQueuedExtraction(event) {
    const receipt = queuedPurchase;
    const before = context.before;
    const next = state();
    const remaining = receipt.pending.task.remaining - 1;
    const values = resourceAmounts(event);
    const resource = RESOURCES[values.findIndex((amount) => amount === 1)];
    const expected = mechanical(before);
    expected.players[receipt.actor].resources[resource]++;
    if (remaining === 0) {
      expected.players[receipt.actor].resources[receipt.resource] += receipt.amount;
      if (receipt.resource === 'skull') expected.skullSupply -= receipt.amount;
    }
    expected.pending = copy(next.pending);
    if (
      record.steps.length !== receipt.stepCount + 1 ||
      !same(mechanical(next), expected) ||
      !same(next.pending?.task, frame.observation.pendingTask) ||
      !same(
        next.pending?.task,
        remaining ? { type: 'resource', remaining } : receipt.pending.after.at(-1),
      ) ||
      !same(next.pending?.after, remaining ? receipt.pending.after : []) ||
      extractionReward?.originActionId !== receipt.technologySourceId ||
      extractionReward.captionActionId !== receipt.technologySourceId ||
      extractionReward.remaining !== remaining ||
      extractionReward.materials.length !== 2 - remaining
    )
      fail('Queued Extraction choice did not preserve or deliver the exact purchase continuation');
    if (remaining) {
      queuedPurchase = {
        ...receipt,
        resourceSourceIds: [...receipt.resourceSourceIds, context.id],
      };
      return false;
    }
    if (unclaimed(receipt.gainEvent, `resource:${receipt.resource}`) !== receipt.amount)
      fail('Queued purchase resource was not delivered exactly once');
    automatic(receipt.gainEvent, `resource:${receipt.resource}`, receipt.amount);
    audit.push({
      actionId: context.id,
      queuedPurchaseSourceActionId: receipt.sourceId,
      technologySourceActionId: receipt.technologySourceId,
      resourceChoiceSourceActionIds: [...receipt.resourceSourceIds, context.id],
      actor: receipt.actor,
      resource: receipt.resource,
      amount: receipt.amount,
      actualDelivered: true,
      continuation: 'cappedExtraction',
      humanPlayingBcReady: false,
      ...(receipt.purchaseCounterEvidence
        ? { purchaseCounterEvidence: copy(receipt.purchaseCounterEvidence) }
        : {}),
    });
    return true;
  }
  function queuedPurchaseEvents(events) {
    if (queuedPurchase.kind === 'technologyBuild')
      return same(events, queuedPurchase.sources[queuedPurchase.sourceIndex].events);
    const first = events[0];
    if (queuedPurchase.stage === 'extraction') {
      if (events.length !== 1 || first?.kind !== 'resourceGain') return false;
      const amounts = resourceAmounts(first);
      return (
        amounts.reduce((sum, amount) => sum + amount, 0) === 1 &&
        amounts.some(
          (amount, index) => amount === 1 && ['wood', 'stone', 'gold'].includes(RESOURCES[index]),
        )
      );
    }
    if (
      first?.kind !== 'technology' ||
      first.detail.steps !== 1 ||
      !TECHNOLOGIES.includes(first.detail.technology)
    )
      return false;
    if (queuedPurchase.kind === 'cornTheology')
      return (
        events.length === 2 &&
        theologySkullCaption(events[1]) === first.actor &&
        first.detail.technology === 'theology' &&
        state().players[queuedPurchase.actor].technologies.theology === 3
      );
    if (
      queuedPurchase.kind === 'cornTechnology' &&
      (context.id !== queuedPurchase.technologySourceId ||
        first.detail.technology !== queuedPurchase.technology)
    )
      return false;
    const level = state().players[queuedPurchase.actor].technologies[first.detail.technology];
    if (level < 3) return events.length === 1;
    return (
      first.detail.technology === 'extraction' &&
      events.length === 2 &&
      events[1].kind === 'unparsed' &&
      events[1].actor === null &&
      events[1].iconResources.length === 0 &&
      events[1].rawText === `収集技術ボーナス: ${first.actor}は好きな資源2個を得た`
    );
  }
  // A pure free-technology card can preserve an unused building bonus. Only
  // immediately observed free choices and the same Build's actual payment may
  // establish that branch; future checkpoints never select an earlier move.
  function queuedTechnologyBuild(outcome, event, events, witness, previous = state()) {
    const exact = (a, b) => matches(a, b) && matches(b, a);
    const action = outcome.row.action;
    const actor = actorId(event.actor);
    const effects = buildings.get(action.id)?.effects;
    const steps = effects?.[0]?.steps;
    const before = previous.pending;
    const next = outcome.frame.snapshot.state;
    const level = previous.players[actor]?.technologies.architecture;
    if (
      witnesses.catalogHash !== catalogHash ||
      actor < 0 ||
      previous.currentPlayer !== actor ||
      previous.version !== 1 ||
      previous.additionalBuildings !== false ||
      previous.expansion != null ||
      ![1, 2].includes(steps) ||
      !exact(effects, [{ type: 'technology', technology: 'any', steps }]) ||
      ![1, 2].includes(level) ||
      action.type !== 'build' ||
      typeof action.architecture !== 'boolean' ||
      action.discountResource !== null ||
      action.renovation !== null ||
      !same(witness.buildings, [action.id]) ||
      before?.task?.type !== 'build' ||
      before.task.remaining !== 2 ||
      before.task.cornPayment !== false ||
      !same(before.after, []) ||
      events.at(-1) !== event ||
      events.some(
        (item) => item.actor !== event.actor || !['resourcePayment', 'build'].includes(item.kind),
      ) ||
      events.filter((item) => item.kind === 'build').length !== 1 ||
      !same(typedResourceCost(action), paymentOf(events))
    )
      return null;
    const continuation = {
      type: 'build',
      remaining: 1,
      allowMonument: false,
      cornPayment: false,
      architectureAvailable: before.task.architectureAvailable !== false && !action.architecture,
      ...(before.task.mandatory ? { mandatory: true } : {}),
    };
    const freeTask = (task, remaining) =>
      task?.type === 'technology' &&
      task.remaining === remaining &&
      task.free === true &&
      task.mandatory !== true;
    if (
      !freeTask(next.pending?.task, steps) ||
      !exact(next.pending.task, outcome.frame.observation?.pendingTask) ||
      !exact(next.pending.after, [continuation])
    )
      return null;
    const expected = mechanical(previous);
    expected.pending = copy(next.pending);
    expected.buildings = previous.buildings.filter((id) => id !== action.id);
    const player = expected.players[actor];
    player.buildings.push(action.id);
    for (const [i, resource] of RESOURCES.entries()) player.resources[resource] -= action.cost[i];
    player.resources.corn += Number(action.architecture);
    player.score += action.architecture && level >= 2 ? 2 : 0;
    if (!exact(mechanical(next), expected)) return null;

    const sources = context.nextSources;
    if (sources.length < steps + 1) return null;
    let preview = outcome.frame;
    const evidence = [];
    for (let index = 0; index < steps; index++) {
      const [sourceId, sourceEvents] = sources[index];
      const choice = sourceEvents[0];
      const old = preview.snapshot.state;
      const technology = choice?.detail?.technology;
      if (
        sourceEvents.length !== 1 ||
        choice.kind !== 'technology' ||
        choice.actor !== event.actor ||
        choice.detail.steps !== 1 ||
        !TECHNOLOGIES.includes(technology) ||
        !freeTask(old.pending?.task, steps - index) ||
        !Number.isSafeInteger(old.players[actor].technologies[technology]) ||
        old.players[actor].technologies[technology] >= 3 ||
        (observed.get(sourceId)?.refills?.currentAge?.length ?? 0) !== 0 ||
        (observed.get(sourceId)?.refills?.age2?.length ?? 0) !== 0
      )
        return null;
      const rows = preview.observation.legalActions.filter(
        (row) => row.action.type === 'technology' && row.action.technology === technology,
      );
      if (rows.length !== 1) return null;
      if (calls + 1 >= MAX_CALLS)
        fail('Core call budget exhausted before queued technology preview');
      preview = core({
        operation: 'publicApply',
        state: old,
        actor,
        move: rows[0].move,
        refills: { currentAge: [], age2: [] },
      });
      const after = preview.snapshot.state;
      const expectedAfter = mechanical(old);
      expectedAfter.players[actor].technologies[technology]++;
      expectedAfter.pending = copy(after.pending);
      if (
        !exact(mechanical(after), expectedAfter) ||
        !exact(after.pending?.task, preview.observation?.pendingTask) ||
        (index + 1 < steps
          ? !freeTask(after.pending?.task, steps - index - 1) ||
            !exact(after.pending.after, [continuation])
          : !exact(after.pending?.task, continuation) || !same(after.pending?.after, []))
      )
        return null;
      evidence.push({
        sourceId,
        events: copy(sourceEvents),
        move: copy(rows[0].move),
        expectedState: mechanical(after),
      });
    }
    const [sourceId, sourceEvents] = sources[steps];
    const nextWitness = observed.get(sourceId) ?? {};
    const purchase = sourceEvents.find((item) => item.kind === 'build');
    const prefix = sourceEvents.slice(0, sourceEvents.indexOf(purchase));
    if (
      !purchase ||
      sourceEvents.filter((item) => ['build', 'monument'].includes(item.kind)).length !== 1 ||
      sourceEvents.some((item) => purchaseActor(item) !== event.actor) ||
      !prefix.length ||
      prefix.some((item) => item.kind !== 'resourcePayment' && !emptyPurchasePayment(item)) ||
      !prefix.some(emptyPurchasePayment) ||
      sourceEvents
        .slice(prefix.length)
        .some((item) => item.kind === 'resourcePayment' || emptyPurchasePayment(item)) ||
      nextWitness.buildings?.length !== 1 ||
      nextWitness.buildingEquivalenceClass !== undefined ||
      (nextWitness.monuments?.length ?? 0) !== 0 ||
      (nextWitness.refills?.currentAge?.length ?? 0) !== 0 ||
      (nextWitness.refills?.age2?.length ?? 0) !== 0
    )
      return null;
    const payment = paymentOf(prefix);
    const rows = preview.observation.legalActions.filter(
      (row) =>
        row.action.type === 'build' &&
        row.action.id === nextWitness.buildings[0] &&
        row.action.renovation === null &&
        same(typedResourceCost(row.action), payment),
    );
    if (rows.length === 0) return false;
    if (rows.length !== 1) return null;
    evidence.push({ sourceId, events: copy(sourceEvents), move: copy(rows[0].move) });
    return {
      kind: 'technologyBuild',
      actor,
      sources: evidence,
      sourceIndex: 0,
      pending: copy(next.pending),
    };
  }
  function finishQueuedTechnologyBuild() {
    const receipt = queuedPurchase;
    const source = receipt.sources[receipt.sourceIndex];
    if (
      record.steps.length !== receipt.stepCount + 1 ||
      !same(record.steps.at(-1).move, source.move) ||
      (source.expectedState &&
        (!matches(mechanical(state()), source.expectedState) ||
          !matches(source.expectedState, mechanical(state()))))
    )
      fail('Queued technology purchase differs from its actual observed continuation');
    if (receipt.sourceIndex + 1 < receipt.sources.length) {
      queuedPurchase = { ...receipt, sourceIndex: receipt.sourceIndex + 1 };
      return false;
    }
    if (pending() !== null) fail('Queued technology purchase did not settle its final Build');
    audit.push({
      actionId: context.id,
      queuedPurchaseSourceActionId: receipt.sourceId,
      technologySourceActionIds: receipt.sources.slice(0, -1).map((item) => item.sourceId),
      paymentSourceActionId: source.sourceId,
      actor: receipt.actor,
      actualDelivered: true,
      continuation: 'pureTechnologyBuild',
      humanPlayingBcReady: false,
    });
    return true;
  }
  function purchaseOutcome(event, events, witness, outcomes = []) {
    const actor = actorId(event.actor);
    const remaining = events.slice(events.indexOf(event) + 1);
    const workerCaps = remaining.filter(cappedWorkerCaption);
    const gains = {};
    if (
      actor < 0 ||
      events.filter((item) => ['build', 'monument'].includes(item.kind)).length !== 1 ||
      remaining.some((item) => purchaseActor(item) !== event.actor) ||
      workerCaps.length > 1
    )
      fail('Purchase source requires one purchased card and one actor');
    // Earlier operations and claims have already been validated. Only the
    // unconsumed suffix can constrain this purchase's trial outcome.
    for (const item of remaining) {
      if (['resourcePayment', 'build', 'monument'].includes(item.kind)) continue;
      if (cappedWorkerCaption(item)) continue; // No worker delta or score conversion is claimed.
      if (item.kind === 'resourceGain') {
        for (const [index, amount] of resourceAmounts(item).entries())
          if (amount) {
            const field = `resource:${RESOURCES[index]}`;
            gains[field] = (gains[field] ?? 0) + amount;
          }
      } else if (item.kind === 'workerGain') gains.workers = (gains.workers ?? 0) + 1;
      else if (item.kind === 'technology') {
        if (item.detail.steps !== 1 || !TECHNOLOGIES.includes(item.detail.technology))
          fail('Purchase source has an unsupported technology advance');
        const field = `technology:${item.detail.technology}`;
        gains[field] = (gains[field] ?? 0) + 1;
      } else {
        if (theologySkullCaption(item)) {
          gains['resource:skull'] = (gains['resource:skull'] ?? 0) + 1;
          continue;
        }
        const bonus = architectureScoreCaption(item);
        if (bonus) {
          gains.score = (gains.score ?? 0) + bonus.amount;
          continue;
        }
        const temple =
          item.kind === 'unparsed'
            ? item.rawText
                .slice(item.actor.length)
                .match(/^は(Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を1段上げた$/)
            : null;
        if (temple) {
          const field = `temple:${TEMPLES[temple[1]]}`;
          gains[field] = (gains[field] ?? 0) + 1;
          continue;
        }
        const score =
          item.kind === 'unparsed'
            ? item.rawText.slice(item.actor.length).match(/^は(\d+)点を獲得した$/)
            : null;
        if (!score) fail('Purchase source has an unsupported or independent operation');
        const amount = Number(score[1]);
        if (!Number.isSafeInteger(amount)) fail('Observed purchase score is out of bounds');
        gains.score = (gains.score ?? 0) + amount;
      }
    }
    if (Object.values(gains).some((amount) => !Number.isSafeInteger(amount) || amount < 0))
      fail('Observed purchase gain is out of bounds');
    // A branch is selected only when every candidate has the same bounded
    // causal shape and exactly one can pay the next observed purchase. A
    // partial proof leaves the original candidates ambiguous.
    const technologyBuildProofs = outcomes.map((outcome) =>
      queuedTechnologyBuild(outcome, event, events, witness),
    );
    const provedTechnologyBuild =
      technologyBuildProofs.length > 1 &&
      technologyBuildProofs.every((proof) => proof !== null) &&
      technologyBuildProofs.filter(Boolean).length === 1
        ? outcomes[technologyBuildProofs.findIndex(Boolean)]
        : null;
    return (outcome) => {
      const task = outcome.frame.observation?.pendingTask?.type ?? null;
      if (workerCaps.length && !cappedPurchaseWorker(outcome)) return false;
      if (remaining.some((item) => theologySkullCaption(item)))
        return (
          cappedTheologyPurchase(outcome, event, events, witness) &&
          matches(outcome.frame.snapshot.state, witness.expected)
        );
      if (task === 'anyAction' && actionScorePair(outcome.row.action.id))
        return deferredActionScoreShape(outcome, event, events, witness) !== null;
      const envelope = events.some(
        (item) =>
          emptyPurchasePayment(item) || architectureScoreCaption(item) || cappedWorkerCaption(item),
      );
      if (envelope && (event.kind !== 'build' || !settledBuild(outcome))) return false;
      if (task === 'technology' && cornTechnologyPair(outcome.row.action.id) !== null) {
        const receipt = queuedPurchaseShape(outcome, event, events, witness);
        return (
          ['cornTheology', 'cornTechnology'].includes(receipt?.kind) &&
          ((receipt.kind === 'cornTechnology' && witness.expected === undefined) ||
            matches(outcome.frame.snapshot.state, witness.expected))
        );
      }
      if (task === 'technology' && queuedPurchaseShape(outcome, event, events, witness))
        return matches(outcome.frame.snapshot.state, witness.expected);
      if (
        task === 'technology' &&
        buildings.get(outcome.row.action.id)?.effects.length === 1 &&
        buildings.get(outcome.row.action.id).effects[0].technology === 'any'
      )
        return provedTechnologyBuild === null || outcome === provedTechnologyBuild;
      if (
        task === 'trade' &&
        (gains.score ?? 0) > 0 &&
        same(buildings.get(outcome.row.action.id)?.effects, [
          { type: 'trade' },
          { type: 'points', amount: gains.score },
        ])
      )
        return (
          deferredScoreShape(outcome, event, events, witness) !== null &&
          (!witness.expected || matches(outcome.frame.snapshot.state, witness.expected))
        );
      // Later resource/temple/technology choices can change the observed fields.
      // Leave these candidates ambiguous instead of filtering with a future end board.
      if (task !== null && task !== 'build') return true;
      const previous = state().players[actor];
      const next = outcome.frame.snapshot.state.players[actor];
      const cost = typedResourceCost(outcome.row.action);
      const capped = cappedPurchaseAdvance(outcome);
      if (
        envelope &&
        ((gains.score ?? 0) !== next.score - previous.score || !same(cost, paymentOf(events)))
      )
        return false;
      for (const [field, amount] of Object.entries(gains)) {
        const resource = field.startsWith('resource:') ? field.slice(9) : null;
        const temple = field.startsWith('temple:') ? field.slice(7) : null;
        const technology = field.startsWith('technology:') ? field.slice(11) : null;
        const delta = resource
          ? next.resources[resource] - previous.resources[resource]
          : temple
            ? next.temples[temple] - previous.temples[temple]
            : technology
              ? next.technologies[technology] - previous.technologies[technology]
              : next[field] - previous[field];
        const knownCost = resource ? cost[RESOURCES.indexOf(resource)] : 0;
        const key = `${actor}:${field}:1`;
        const gross =
          (context.effects[key] ?? 0) +
          Math.max(0, delta + knownCost) +
          (field === 'technology:architecture' && capped ? 1 : 0);
        if (gross - (context.claims[key] ?? 0) !== amount) return false;
      }
      // A witnessed remaining Build can be the original decremented continuation.
      // The unchanged queue excludes nested card choices whose effects are not settled.
      // Only the actor's directly observed counters constrain this pending purchase.
      const beforePending = state().pending;
      const nextPending = outcome.frame.snapshot.state.pending;
      const observedPlayers = witness.expected?.players;
      const observedPlayer = observedPlayers?.[actor];
      if (
        event.kind === 'build' &&
        task === 'build' &&
        witness.pendingTask === 'build' &&
        beforePending?.task?.type === 'build' &&
        Number.isSafeInteger(beforePending.task.remaining) &&
        beforePending.task.remaining > 1 &&
        nextPending?.task?.type === 'build' &&
        nextPending.task.remaining === beforePending.task.remaining - 1 &&
        Array.isArray(beforePending.after) &&
        Array.isArray(nextPending.after) &&
        same(beforePending.after, nextPending.after) &&
        Array.isArray(observedPlayers) &&
        observedPlayers.length === state().players.length &&
        object(observedPlayer?.resources) &&
        Number.isSafeInteger(observedPlayer.resources.corn) &&
        observedPlayer.resources.corn >= 0 &&
        Number.isFinite(observedPlayer.score)
      )
        return (
          matches(next.resources, observedPlayer.resources) && next.score === observedPlayer.score
        );
      // Other unresolved tasks cannot use an end board to infer an earlier choice.
      return (
        task !== null ||
        !witness.expected ||
        matches(outcome.frame.snapshot.state, witness.expected)
      );
    };
  }

  function rollback(saved) {
    frame = saved.frame;
    record.steps = saved.steps;
    record.steps.length = saved.stepCount;
    history = saved.history;
    history.length = saved.historyCount;
    history.at(-1).rotationCredit = saved.historyLastCredit;
    history.at(-1).declaredRotation = saved.historyLastDeclaration;
    history.at(-1).calendarEvidence = saved.historyLastEvidence;
    history.at(-1).extractionReward = saved.historyLastReward;
    audit.length = saved.audit;
    rotationCredit = saved.rotationCredit;
    declaredRotation = saved.declaredRotation;
    calendarEvidence = saved.calendarEvidence;
    extractionReward = saved.extractionReward;
    queuedPurchase = saved.queuedPurchase;
    deferredScore = saved.deferredScore;
  }
  let blocked = null;
  const groups = sourceGroups(normalizedEvents, witnesses.initialActionId, knownActionIds);
  for (const [groupIndex, [id, events]] of groups.entries()) {
    const witness = observed.get(id) ?? {};
    const saved = {
      frame: copy(frame),
      steps: record.steps,
      stepCount: record.steps.length,
      history,
      historyCount: history.length,
      historyLastCredit: history.at(-1).rotationCredit,
      historyLastDeclaration: history.at(-1).declaredRotation,
      historyLastEvidence: history.at(-1).calendarEvidence,
      historyLastReward: history.at(-1).extractionReward,
      audit: audit.length,
      rotationCredit,
      declaredRotation,
      calendarEvidence,
      extractionReward,
      queuedPurchase,
      deferredScore,
    };
    context = {
      id,
      events,
      nextSource: groups[groupIndex + 1],
      nextSources: groups.slice(groupIndex + 1, groupIndex + 4),
      before: copy(state()),
      pools: {
        currentAge: [...(witness.refills?.currentAge ?? [])],
        age2: [...(witness.refills?.age2 ?? [])],
      },
      claims: {},
      effects: {},
      feeding: [],
      feedingDay: null,
      foodScoreClaims: new Set(),
      ageAnnouncement: false,
      paymentActor: null,
      paymentOperations: 0,
      purchase: null,
      architectureAdvance: null,
    };
    try {
      if (!events.length) fail('Raw source action has no observed messages');
      if (deferredScore) {
        if (
          events.some((event) => actorId(event.actor) !== deferredScore.actor) ||
          !same(state().pending, deferredScore.pending) ||
          (witness.refills?.currentAge?.length ?? 0) !== 0 ||
          (witness.refills?.age2?.length ?? 0) !== 0
        )
          fail('Deferred purchase requires the same actor and unchanged continuation');
        if (deferredScore.kind === 'action') {
          const operation = events.find((event) => event.kind !== 'resourcePayment');
          if (
            deferredScore.stage === 'anyAction'
              ? events.length !== 1 ||
                operation?.kind !== 'gearAction' ||
                operation.detail.source !== 'selectedAction' ||
                operation.detail.gear !== 'tikal' ||
                operation.detail.actionPosition !== 4 ||
                operation.detail.cornCost !== 1
              : deferredScore.stage !== 'monument' ||
                operation?.kind !== 'monument' ||
                events.at(-1) !== operation ||
                events.some((event) => event !== operation && event.kind !== 'resourcePayment') ||
                witness.monuments?.length !== 1
          )
            fail('Deferred purchase requires the next observed Anywhere action, then monument');
        } else if (!events.every((event) => event.kind === 'trade')) {
          if (
            events.filter((event) => event.kind === 'build').length !== 1 ||
            events.some((event) => ['trade', 'cancellation', 'monument'].includes(event.kind)) ||
            witness.buildings?.length !== 1
          )
            fail('Deferred purchase requires observed Trades followed by the next actual Build');
          finishDeferredScore();
        }
      }
      if (
        queuedPurchase &&
        (id !== queuedPurchase.nextSourceId ||
          !queuedPurchaseEvents(events) ||
          actorId(events[0].actor) !== queuedPurchase.actor ||
          state().round !== queuedPurchase.round ||
          state().currentPlayer !== queuedPurchase.actor ||
          !same(state().pending, queuedPurchase.pending) ||
          (witness.refills?.currentAge?.length ?? 0) !== 0 ||
          (witness.refills?.age2?.length ?? 0) !== 0 ||
          ((witness.buildings?.length ?? 0) !== 0 &&
            !(
              queuedPurchase.kind === 'technologyBuild' &&
              queuedPurchase.sourceIndex === queuedPurchase.sources.length - 1
            )) ||
          (witness.monuments?.length ?? 0) !== 0)
      )
        fail('Queued purchase requires the next same-actor technology or exact material choice');
      for (const [field, kind] of [
        ['buildings', 'build'],
        ['buildingEquivalenceClass', 'build'],
        ['monuments', 'monument'],
      ])
        if (witness[field]?.length && !events.some((event) => event.kind === kind))
          fail(`Unused ${field} witness without a matching source event`);
      if (events.some((event) => event.kind === 'cancellation')) {
        alignActor(events[0].actor);
        if (
          events.length !== 1 ||
          !witness.expected?.gears ||
          !Number.isSafeInteger(witness.expected?.round) ||
          !witness.expected?.players?.every(
            (player) => player.resources && player.workers !== undefined,
          )
        )
          fail('Cancellation requires an observed settled day/worker/resource board');
        const alternatives = history.filter(
          (entry) =>
            !entry.unsettledPurchase &&
            entry.frame.snapshot.state.currentPlayer === actorId(events[0].actor) &&
            matches(entry.frame.snapshot.state, witness.expected) &&
            (witness.pendingTask === undefined ||
              (entry.frame.observation?.pendingTask?.type ?? null) === witness.pendingTask),
        );
        if (
          !alternatives.length ||
          alternatives.some(
            (entry) =>
              !same(
                mechanical(entry.frame.snapshot.state),
                mechanical(alternatives[0].frame.snapshot.state),
              ),
          )
        )
          fail('Cancellation rollback is unknown/ambiguous; observe the turn and pending prompt');
        const selected = alternatives.at(-1);
        frame = copy(selected.frame);
        const removed = record.steps.length - selected.steps;
        record.steps = record.steps.slice(0, selected.steps);
        history = history.filter((entry) => entry.steps <= selected.steps);
        rotationCredit = selected.rotationCredit;
        declaredRotation = selected.declaredRotation;
        calendarEvidence = selected.calendarEvidence;
        extractionReward = selected.extractionReward;
        audit.push({
          actionId: id,
          cancellation: true,
          removedSteps: removed,
          observedBoardMatched: true,
        });
      } else {
        const payments = paymentOf(events);
        const emptyPayments = events.filter(emptyPurchasePayment);
        if (emptyPayments.length) {
          const purchases = events.filter((event) => event.kind === 'build');
          const event = purchases[0];
          if (
            emptyPayments.length !== 1 ||
            purchases.length !== 1 ||
            pending() !== 'build' ||
            events.indexOf(emptyPayments[0]) >= events.indexOf(event) ||
            events
              .slice(0, events.indexOf(event))
              .some((item) => item.kind !== 'resourcePayment' && !emptyPurchasePayment(item)) ||
            emptyPayments[0].actor !== event.actor
          )
            fail('Empty payment requires one same-actor pending Build purchase');
        }
        const paymentActors = events
          .filter((event) => event.kind === 'resourcePayment')
          .map((event) => actorId(event.actor));
        if (paymentActors.some((actor) => actor < 0 || actor !== paymentActors[0]))
          fail('Observed payments do not have one unambiguous actor');
        context.paymentActor = paymentActors[0] ?? null;
        const firstOperation = events.findIndex(
          (event) =>
            ['place', 'gearAction', 'technology', 'trade', 'build', 'monument', 'beg'].includes(
              event.kind,
            ) ||
            (event.kind === 'resourceGain' && pending() === 'resource') ||
            (event.kind === 'resourcePayment' &&
              ['payTechnology', 'payResource', 'theology'].includes(pending())) ||
            (event.kind === 'unparsed' &&
              event.actor !== null &&
              /は(?:スタートプレイヤースペースに|(?:Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を|(?:corn|コーン|wood)タイルを獲得した)/.test(
                event.rawText.slice(event.actor.length),
              )),
        );
        if (
          firstOperation >= 0 &&
          events
            .slice(firstOperation + 1)
            .some((event) => ['gearAdvanced', 'feeding', 'doubleAdvance'].includes(event.kind))
        )
          fail('Actor operation precedes a calendar message in the same source');
        const startsCalendar = events.some((event) =>
          ['gearAdvanced', 'feeding'].includes(event.kind),
        );
        const doubleEvents = events.filter((event) => event.kind === 'doubleAdvance');
        const double = doubleEvents.length > 0;
        if (
          double &&
          (doubleEvents.length !== 1 ||
            state().firstPlayerClaimed === null ||
            actorId(doubleEvents[0].actor) !== state().firstPlayerClaimed)
        )
          fail('Observed double advancement actor differs from the core claimer');
        const payouts = events.filter((event) => accumulatedCornPayout(event) !== null);
        const announcements = events.filter((event) => event.kind === 'firstPlayer');
        let payoutBoundary = false;
        if (
          !startsCalendar &&
          !double &&
          pending() === null &&
          payouts.length &&
          announcements.length
        ) {
          const receipt = calendarEvidence.corn;
          const claimed = state().firstPlayerClaimed;
          const future = claimed === state().firstPlayer ? (claimed + 1) % names.length : claimed;
          if (
            events.length !== 2 ||
            payouts.length !== 1 ||
            announcements.length !== 1 ||
            !receipt ||
            receipt.consumed ||
            receipt.round !== state().round ||
            claimed === null ||
            receipt.actor !== claimed ||
            future === state().firstPlayer ||
            actorId(payouts[0].actor) !== receipt.actor ||
            // The current first player keeps their payout; claiming again passes
            // the next round's first-player marker to their successor.
            actorId(announcements[0].actor) !== future ||
            accumulatedCornPayout(payouts[0]) !== receipt.amount ||
            !same(payouts[0].iconResources, ['corn']) ||
            announcements[0].iconResources.length ||
            announcements[0].rawText !==
              `${announcements[0].actor}がこのラウンドのスタートプレイヤーです`
          )
            fail('Corn/first-player boundary has no matching unconsumed core payout receipt');
          payoutBoundary = true;
        }
        if (startsCalendar || double || payoutBoundary) calendar();
        if (double) {
          choose(
            (action) => action.type === 'rotate' && action.days === 2,
            'observed double advancement',
          );
          declaredRotation = 2;
          syncHistory();
        }
        if (events.some((event) => event.kind === 'gearAdvanced')) {
          calendar(declaredRotation ?? 1);
          declaredRotation = null;
          syncHistory();
        }
        for (const event of events) {
          const detail = event.detail;
          if (['gearAdvanced', 'doubleAdvance', 'resourcePayment'].includes(event.kind)) continue;
          if (emptyPurchasePayment(event)) continue; // Validated against the actual purchase below.
          if (event.kind === 'feeding') feedingObserved(event);
          else if (event.kind === 'firstPlayer') {
            const actor = actorId(event.actor);
            const claimed = state().firstPlayerClaimed;
            const rotation = calendarEvidence.rotation;
            const initial = calendarEvidence.initialAnnouncement;
            if (actor < 0) fail('First-player announcement disagrees with the public calendar');
            if (pending() === 'rotation') {
              const predicted =
                claimed === state().firstPlayer ? (claimed + 1) % names.length : claimed;
              if (
                claimed === null ||
                actor !== predicted ||
                calendarEvidence.announcement?.confirmed === false
              )
                fail('First-player announcement disagrees with the pending core rotation');
            } else if (rotation && rotation.round === state().round) {
              if (actor !== rotation.firstPlayer || rotation.announcementConsumed)
                fail('First-player announcement disagrees with the current core rotation receipt');
              calendarEvidence = {
                ...calendarEvidence,
                rotation: { ...rotation, announcementConsumed: true },
              };
            } else {
              if (actor !== initial.actor || state().round !== initial.round || initial.consumed)
                fail('First-player announcement disagrees with the known initial round');
              calendarEvidence = {
                ...calendarEvidence,
                initialAnnouncement: { ...initial, consumed: true },
              };
            }
            calendarEvidence = {
              ...calendarEvidence,
              announcement: {
                actor,
                round: state().round,
                actionId: id,
                confirmed: pending() !== 'rotation',
              },
            };
            syncHistory();
          } else if (event.kind === 'gameEnd') {
            if (state().phase !== 'finished')
              fail('Observed game end precedes the core terminal state');
          } else if (event.kind === 'place') {
            alignActor(event.actor);
            choose(
              (action) =>
                action.type === 'place' &&
                action.gear === detail.gear &&
                action.cornCost === detail.cornCost &&
                !action.discount,
              'placement',
            );
          } else if (event.kind === 'gearAction') {
            alignActor(event.actor);
            const inlineCosts = [...event.rawText.matchAll(/([+-]?\d+)\s*コーンを支払って/g)];
            if (inlineCosts.length > 1) fail('Ambiguous inline corn payment');
            const cornCost = inlineCosts.length ? Number(inlineCosts[0][1]) : detail.cornCost;
            if (
              inlineCosts.length &&
              (!Number.isSafeInteger(cornCost) ||
                cornCost < 0 ||
                (detail.cornCost !== undefined && detail.cornCost !== cornCost))
            )
              fail('Invalid/conflicting inline corn payment');
            if (detail.source === 'removal')
              choose(
                (action) =>
                  action.type === 'remove' &&
                  action.gear === detail.gear &&
                  action.position === detail.workerPosition,
                'worker removal',
              );
            choose(
              (action) =>
                action.type === 'useAction' &&
                action.gear === detail.gear &&
                action.position === detail.actionPosition &&
                (cornCost === undefined || action.cornCost === cornCost),
              'gear action',
            );
          } else if (event.kind === 'technology') {
            alignActor(event.actor);
            if (detail.steps !== 1)
              fail('Multiple technology advances need separate observed choices');
            const field = `technology:${detail.technology}`;
            const capped =
              (detail.technology === 'architecture'
                ? context.purchase?.capped
                : detail.technology === 'theology' && context.purchase?.theologyCapped) &&
              context.purchase?.actor === actorId(event.actor) &&
              !context.purchase.advanceClaimed;
            if (capped) context.purchase.advanceClaimed = true;
            else if (unclaimed(event, field) < 1) {
              const before = copy(state());
              const selectedAtCap =
                detail.technology === 'architecture' &&
                before.players[actorId(event.actor)].technologies.architecture === 3 &&
                pending() === 'technology';
              choose(
                (action) => action.type === 'technology' && action.technology === detail.technology,
                'technology advance',
              );
              pay(payments);
              if (selectedAtCap) {
                const actor = actorId(event.actor);
                const after = state();
                if (
                  (context.architectureAdvance && !context.architectureAdvance.claimed) ||
                  after.currentPlayer !== actor ||
                  after.round !== before.round ||
                  after.players[actor].technologies.architecture !== 3 ||
                  after.players[actor].score - before.players[actor].score !== 3
                )
                  fail('Selected capped Architecture requires its actual same-source three points');
                context.architectureAdvance = { actor, claimed: false };
              }
            }
            // A fixed effect may already have advanced this track. Claim its
            // same-macro delta once; a genuine choice at level three has no level delta.
            if (!capped && unclaimed(event, field) > 0) automatic(event, field, 1);
          } else if (event.kind === 'trade') {
            alignActor(event.actor);
            const buy = detail.from?.resource === 'corn';
            const resource = buy ? detail.to?.resource : detail.from?.resource;
            const amount = buy ? detail.to?.amount : detail.from?.amount;
            if (
              !['wood', 'stone', 'gold'].includes(resource) ||
              (buy ? detail.from?.resource : detail.to?.resource) !== 'corn' ||
              !Number.isSafeInteger(amount) ||
              amount < 1 ||
              amount > 1000
            )
              fail('Unsupported market exchange');
            const corn = state().players[actorId(event.actor)].resources.corn;
            for (let count = 0; count < amount; count++)
              choose(
                (action) =>
                  action.type === 'trade' && action.resource === resource && action.buy === buy,
                'market exchange',
              );
            const cornDelta = state().players[actorId(event.actor)].resources.corn - corn;
            if (cornDelta !== (buy ? -detail.from.amount : detail.to.amount))
              fail('Observed market price mismatch');
          } else if (event.kind === 'build' || event.kind === 'monument') {
            alignActor(event.actor);
            const buildingClass = validateBuildingEquivalenceClass(witness);
            const ids =
              buildingClass ?? witness[event.kind === 'build' ? 'buildings' : 'monuments'];
            if (
              buildingClass !== null &&
              (event.kind !== 'build' ||
                state().version !== 1 ||
                state().additionalBuildings !== false ||
                state().expansion != null)
            )
              fail('Equivalent building witness requires base building rules');
            if (buildingClass === null && ids?.length !== 1)
              fail('Observed purchased card identity is required');
            const beforePurchase = copy(state());
            const action = choose(
              (action) =>
                action.type === event.kind &&
                (buildingClass === null ? action.id === ids[0] : ids.includes(action.id)) &&
                (!payments.some(Boolean) || same(action.cost, payments)),
              `observed ${event.kind}`,
              false,
              false,
              (outcomes) => purchaseOutcome(event, events, witness, outcomes),
              buildingClass !== null ||
                events.some(
                  (item) =>
                    emptyPurchasePayment(item) ||
                    architectureScoreCaption(item) ||
                    cappedWorkerCaption(item),
                ) ||
                actionScorePair(ids[0]) !== null ||
                (buildings.get(ids[0])?.effects[0]?.type === 'trade' &&
                  events.some((item) => fixedScoreCaption(item) !== null)),
              actionScorePair(ids[0]) === null,
              buildingClass,
            );
            context.purchase = {
              actor: actorId(event.actor),
              architecture:
                action.architecture === true &&
                beforePurchase.players[actorId(event.actor)].technologies.architecture >= 2,
              capped: cappedPurchaseAdvance({ frame, row: { action } }, beforePurchase),
              theologyCapped: cappedTheologyPurchase(
                { frame, row: { action } },
                event,
                events,
                witness,
                beforePurchase,
              ),
              advanceClaimed: false,
              bonusClaimed: false,
              workerCapped: cappedPurchaseWorker({ frame, row: { action } }, beforePurchase),
              workerCapClaimed: false,
            };
            const receipt =
              queuedPurchaseShape(
                { frame, row: { action } },
                event,
                events,
                witness,
                beforePurchase,
              ) ??
              queuedTechnologyBuild(
                { frame, row: { action } },
                event,
                events,
                witness,
                beforePurchase,
              );
            if (
              cornTechnologyPair(action.id) !== null &&
              !['cornTheology', 'cornTechnology'].includes(receipt?.kind)
            )
              fail(
                'Corn plus technology purchase requires its exact observed continuation receipt',
              );
            if (receipt) {
              if (saved.deferredScore || saved.queuedPurchase?.kind === 'technologyBuild')
                fail('A deferred purchase cannot nest another unresolved receipt');
              const nextSourceId = groups[groupIndex + 1]?.[0];
              if (nextSourceId === undefined) fail('Queued purchase has no next observed source');
              queuedPurchase = {
                ...receipt,
                sourceId: id,
                kinds: events.map((item) => item.kind),
                nextSourceId,
                round: state().round,
                stepCount: record.steps.length,
                saved,
              };
              history.at(-1).unsettledPurchase = true;
              if (receipt.kind === 'technologyBuild') {
                audit.at(-1).causalPreviewSourceActionIds = receipt.sources.map(
                  (item) => item.sourceId,
                );
                audit.at(-1).humanPlayingBcReady = false;
              }
            }
            const scoreReceipt = deferredScoreShape(
              { frame, row: { action } },
              event,
              events,
              witness,
              beforePurchase,
            );
            if (scoreReceipt) {
              if (saved.deferredScore || queuedPurchase)
                fail('A deferred purchase cannot nest another unresolved receipt');
              deferredScore = {
                ...scoreReceipt,
                sourceId: id,
                kinds: events.map((item) => item.kind),
                saved,
              };
              history.at(-1).unsettledPurchase = true;
            }
            const actionScoreReceipt = deferredActionScoreShape(
              { frame, row: { action } },
              event,
              events,
              witness,
              beforePurchase,
            );
            if (actionScorePair(action.id) && !actionScoreReceipt)
              fail('Action plus Points purchase requires its exact deferred receipt');
            if (actionScoreReceipt) {
              if (deferredScore || queuedPurchase || saved.deferredScore?.kind === 'action')
                fail('A deferred purchase cannot nest another unresolved receipt');
              // Closing Trade at this source does not commit its tentative chain.
              // Keep its earlier anchor until this observed continuation settles.
              deferredScore = {
                ...actionScoreReceipt,
                sourceId: id,
                kinds: events.map((item) => item.kind),
                saved: saved.deferredScore?.saved ?? saved,
                rollbackSourceId: saved.deferredScore?.sourceId ?? id,
                rollbackKinds: saved.deferredScore?.kinds ?? events.map((item) => item.kind),
              };
              history.at(-1).unsettledPurchase = true;
            }
          } else if (event.kind === 'workerGain') automatic(event, 'workers', 1);
          else if (event.kind === 'resourceGain') {
            if (queuedPurchase?.sourceId === id && queuedPurchase.gainEvent === event) continue;
            const values = resourceAmounts(event);
            if (pending() === 'resource') {
              alignActor(event.actor);
              for (const [index, amount] of values.entries())
                for (let count = 0; count < amount; count++)
                  choose(
                    (action) => action.type === 'resource' && action.resource === RESOURCES[index],
                    'resource selection',
                  );
            }
            for (const [index, amount] of values.entries())
              if (amount) automatic(event, `resource:${RESOURCES[index]}`, amount);
          } else if (event.kind === 'beg') {
            const body = event.rawText.slice(event.actor?.length ?? 0);
            const temple = body.match(
              /^は(Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を(?:1段)?下げ(?:て|ることで|、)コーン([1-3])個を受け取った$/,
            );
            if (!temple) fail('Begging requires one unambiguous temple in the actor-free phrase');
            if (event.iconResources.length)
              fail('Begging caption requires icon-free corn evidence');
            alignActor(event.actor);
            const templeId = TEMPLES[temple[1]];
            const before = state().players[actorId(event.actor)];
            const beforeCorn = before.resources.corn;
            const beforeTemple = before.temples[templeId];
            const gained = Number(temple[2]);
            if (
              !legal().some((row) => row.action.type === 'beg') ||
              !Number.isSafeInteger(beforeCorn) ||
              beforeCorn < 0 ||
              beforeCorn > 2 ||
              gained !== 3 - beforeCorn
            )
              fail('Begging caption corn gain differs from the legal base Beg');
            choose((action) => action.type === 'beg', 'observed begging');
            const lowerTemple = (action) =>
              action.type === 'temple' && action.temple === templeId && action.direction === -1;
            if (pending() !== 'temple' || !legal().some((row) => lowerTemple(row.action)))
              fail('Begging requires the actual named temple-down choice');
            choose(lowerTemple, 'observed begging temple');
            const after = state().players[actorId(event.actor)];
            if (
              after.resources.corn - beforeCorn !== gained ||
              after.temples[templeId] - beforeTemple !== -1
            )
              fail('Begging caption differs from the actual corn and temple transition');
            automatic(event, `temple:${templeId}`, 1, -1);
            automatic(event, 'resource:corn', gained);
          } else if (event.kind === 'unparsed') {
            const architectureBonus = architectureScoreCaption(event);
            const theologyBonus = theologySkullCaption(event);
            const body =
              event.actor === null ? event.rawText : event.rawText.slice(event.actor.length);
            if (cappedWorkerCaption(event)) {
              if (
                context.purchase?.actor !== actorId(event.actor) ||
                !context.purchase.workerCapped ||
                context.purchase.workerCapClaimed
              )
                fail('Worker cap caption has no unique matching settled purchase');
              context.purchase.workerCapClaimed = true;
              continue;
            }
            const first = body.match(
              /^はスタートプレイヤースペースに(\d+)\s*を支払ってワーカーを置いた$/,
            );
            const temple = body.match(
              /^は(Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を1段(上げた|下げた)$/,
            );
            const harvest = body.match(/^は(corn|コーン|wood)タイルを獲得した$/);
            const paidRemoval = body.match(
              /^は(Palenque|Yaxchilan|Tikal|Uxmal|Chichen Itza) (\d+)のワーカーを取り除き、\s*(\d+)コーンを支払ってn°(\d+)\s*アクションを行った$/,
            );
            const leaderScore = body.match(
              /^は(chaac|quetzalcoatl|kukulkan|kukulcan)の信仰が一番高い(?:うちの一人なので、半分の点を獲得した: (\d+)点|ので、(\d+)点を獲得した)$/,
            );
            const trackScore = body.match(/^は信仰により(-?\d+)点を獲得した$/);
            if (body === 'は再び歯車を高速化する権利を得た') {
              if (event.iconResources.length)
                fail('Double-advance rights caption requires icon-free evidence');
              automatic(event, 'doubleAdvanceAvailable', 1);
            } else if (theologyBonus && queuedPurchase?.kind === 'cornTheology') {
              if (
                queuedPurchase.nextSourceId !== id ||
                queuedPurchase.actor !== actorId(theologyBonus) ||
                queuedPurchase.bonusClaimed
              )
                fail('Theology bonus has no matching unconsumed queued purchase');
              automatic({ ...event, actor: theologyBonus }, 'resource:skull', 1);
              queuedPurchase.bonusClaimed = true;
            } else if (theologyBonus) {
              const receipt = context.purchase;
              if (
                !receipt?.theologyCapped ||
                receipt.actor !== actorId(theologyBonus) ||
                !receipt.advanceClaimed ||
                receipt.bonusClaimed
              )
                fail('Theology bonus has no matching unconsumed fixed purchase receipt');
              automatic({ ...event, actor: theologyBonus }, 'resource:skull', 1);
              receipt.bonusClaimed = true;
            } else if (architectureBonus) {
              const selected = context.architectureAdvance;
              if (
                selected &&
                !selected.claimed &&
                selected.actor === actorId(architectureBonus.actor) &&
                architectureBonus.amount === 3
              ) {
                automatic({ ...event, actor: architectureBonus.actor }, 'score', 3);
                selected.claimed = true;
                continue;
              }
              const receipt = context.purchase;
              if (
                !receipt ||
                receipt.actor !== actorId(architectureBonus.actor) ||
                receipt.bonusClaimed ||
                !(
                  (architectureBonus.amount === 3 && receipt.capped && receipt.advanceClaimed) ||
                  (architectureBonus.amount === 2 && receipt.architecture)
                )
              )
                fail('Architecture bonus has no matching unconsumed purchase receipt');
              automatic(
                { ...event, actor: architectureBonus.actor },
                'score',
                architectureBonus.amount,
              );
              receipt.bonusClaimed = true;
            } else if (first) {
              alignActor(event.actor);
              choose(
                (action) => action.type === 'firstPlayer' && action.cornCost === Number(first[1]),
                'first-player placement',
              );
            } else if (paidRemoval) {
              alignActor(event.actor);
              const gear = {
                Palenque: 'palenque',
                Yaxchilan: 'yaxchilan',
                Tikal: 'tikal',
                Uxmal: 'uxmal',
                'Chichen Itza': 'chichenItza',
              }[paidRemoval[1]];
              choose(
                (action) =>
                  action.type === 'remove' &&
                  action.gear === gear &&
                  action.position === Number(paidRemoval[2]),
                'worker removal',
              );
              choose(
                (action) =>
                  action.type === 'useAction' &&
                  action.gear === gear &&
                  action.position === Number(paidRemoval[4]) &&
                  action.cornCost === Number(paidRemoval[3]),
                'paid gear action',
              );
            } else if (temple) {
              alignActor(event.actor);
              const direction = temple[2] === '上げた' ? 1 : -1;
              pay(payments);
              const field = `temple:${TEMPLES[temple[1]]}`;
              if (unclaimed(event, field, direction) < 1 && pending() === 'temple') {
                choose(
                  (action) =>
                    action.type === 'temple' &&
                    action.temple === TEMPLES[temple[1]] &&
                    action.direction === direction,
                  'temple advance',
                );
              }
              automatic(event, field, 1, direction);
            } else if (harvest) {
              alignActor(event.actor);
              const kind = harvest[1] === 'コーン' ? 'corn' : harvest[1];
              choose((action) => action.type === 'harvest' && action.kind === kind, 'harvest');
            } else if (event.rawText.startsWith('収集技術ボーナス: ')) {
              extractionCaption(event);
            } else if (event.rawText.startsWith('世代中間報酬: ')) {
              if (
                event.actor !== null ||
                event.iconResources.length ||
                !context.feeding.length ||
                ![8, 21].includes(context.feedingDay) ||
                state().foodDays.at(-1) !== context.feedingDay
              )
                fail('Resource-reward caption has no matching core food-day receipt');
            } else if (shortageObserved(event.rawText)) {
              // The exact actor/count/penalty is bound to a core feeding receipt.
            } else if (leaderScore || trackScore) {
              const actor = actorId(event.actor);
              const field = leaderScore
                ? `leader:${leaderScore[1] === 'kukulcan' ? 'kukulkan' : leaderScore[1]}`
                : 'track';
              const key = `${actor}:${field}`;
              const amount = Number(
                leaderScore ? (leaderScore[2] ?? leaderScore[3]) : trackScore[1],
              );
              if (
                actor < 0 ||
                event.iconResources.length ||
                !context.feeding.length ||
                ![14, 27].includes(context.feedingDay) ||
                state().foodDays.at(-1) !== context.feedingDay ||
                !Number.isSafeInteger(amount) ||
                context.foodScoreClaims.has(key)
              )
                fail('Food-day score caption has no unique matching core scoring receipt');
              if (amount < 0) {
                const receipt = context.feeding.find((value) => value.actor === actor);
                if (!receipt || unclaimed(event, 'score', -1) - receipt.penalty < -amount)
                  fail('Food-day temple loss is not established separately from feeding penalties');
              }
              automatic(event, 'score', Math.abs(amount), amount < 0 ? -1 : 1);
              context.foodScoreClaims.add(key);
            } else if (event.rawText === '建物が第2世代になりました') {
              if (
                event.actor !== null ||
                event.iconResources.length ||
                !context.feeding.length ||
                context.feedingDay !== 14 ||
                state().foodDays.at(-1) !== 14 ||
                context.before.age !== 1 ||
                state().age !== 2 ||
                context.ageAnnouncement
              )
                fail('Age-two announcement has no unique matching core food-day transition');
              context.ageAnnouncement = true;
            } else if (
              event.actor === null &&
              state().phase === 'finished' &&
              /^(最終スコア|記念碑: |残り資源: )/.test(event.rawText)
            ) {
              // Source display annotations do not change the official core score.
            } else if (/は\d+点を獲得した$/.test(event.rawText)) {
              if (deferredScore?.sourceId === id && deferredScore.scoreEvent === event) {
                if (deferredScore.captionObserved)
                  fail('Deferred purchase score was already observed');
                deferredScore = { ...deferredScore, captionObserved: true };
              } else
                automatic(event, 'score', Number(event.rawText.match(/(\d+)点を獲得した$/)[1]));
            } else if (
              /スタートプレイヤースペースに置かれているワーカーは.+の手元に戻った$/.test(
                event.rawText,
              )
            ) {
              const name = event.rawText.match(
                /^スタートプレイヤースペースに置かれているワーカーは(.+)の手元に戻った$/,
              )?.[1];
              const receipt = calendarEvidence.rotation;
              if (
                !receipt ||
                receipt.consumed ||
                receipt.actor !== actorId(name) ||
                receipt.returned < 1
              )
                fail('Worker return has no matching actor/claimed-worker core receipt');
              calendarEvidence = { ...calendarEvidence, rotation: { ...receipt, consumed: true } };
              syncHistory();
            } else if (accumulatedCornPayout(event) !== null) {
              const amount = accumulatedCornPayout(event);
              const receipt = calendarEvidence.corn;
              if (
                !receipt ||
                receipt.consumed ||
                receipt.actor !== actorId(event.actor) ||
                receipt.amount !== amount ||
                !same(event.iconResources, ['corn'])
              )
                fail('Accumulated corn has no matching actor/amount core EndTurn receipt');
              calendarEvidence = { ...calendarEvidence, corn: { ...receipt, consumed: true } };
              syncHistory();
            } else fail(`Unsupported source message: ${event.rawText.slice(0, 120)}`);
          } else fail(`Unsupported source effect ${event.kind}`);
        }
        pay(payments);
        if (
          context.purchase?.theologyCapped &&
          (!context.purchase.advanceClaimed || !context.purchase.bonusClaimed)
        )
          fail('Fixed Theology purchase did not consume its advance and skull captions');
        for (const event of events.filter((item) => item.kind === 'resourcePayment'))
          for (const [index, amount] of resourceAmounts(event).entries())
            if (amount) automatic(event, `resource:${RESOURCES[index]}`, amount, -1);
      }
      if (context.pools.currentAge.length || context.pools.age2.length)
        fail('Observed reveal pool was not consumed at this source event');
      if (Object.hasOwn(witness, 'pendingTask') && witness.pendingTask !== pending())
        fail('Observed macro-end pending task mismatch');
      if (witness.expected && !matches(state(), witness.expected))
        fail('Observed macro-end public checkpoint mismatch');
      if (record.steps.length > saved.stepCount && witness.expected)
        record.steps.at(-1).checkpoint = {
          source: { reference: source.reference, actionIds: [id] },
          expected: copy(witness.expected),
        };
      if (saved.queuedPurchase) {
        if (
          saved.queuedPurchase.kind === 'technologyBuild'
            ? finishQueuedTechnologyBuild()
            : finishQueuedPurchase(events[0])
        )
          queuedPurchase = null;
        else {
          const nextSourceId = groups[groupIndex + 1]?.[0];
          if (nextSourceId === undefined)
            fail('Queued purchase has no next observed material choice');
          queuedPurchase = {
            ...queuedPurchase,
            nextSourceId,
            stepCount: record.steps.length,
            pending: copy(state().pending),
          };
          history.at(-1).unsettledPurchase = true;
        }
      }
      if (deferredScore?.kind === 'action' && deferredScore.stage === 'complete') {
        automatic(deferredScore.scoreEvent, 'score', deferredScore.amount);
        audit.push({
          actionId: id,
          deferredScoreSourceActionId: deferredScore.sourceId,
          actor: deferredScore.actor,
          amount: deferredScore.amount,
          actualDelivered: true,
          continuation: 'anyActionMonument',
          humanPlayingBcReady: false,
        });
        deferredScore = null;
        history.at(-1).unsettledPurchase = false;
      }
    } catch (error) {
      const receipt =
        saved.deferredScore ?? deferredScore ?? queuedPurchase ?? saved.queuedPurchase;
      rollback(receipt?.saved ?? saved);
      blocked = {
        actionId: receipt?.rollbackSourceId ?? receipt?.sourceId ?? id,
        kinds: receipt?.rollbackKinds ?? receipt?.kinds ?? events.map((event) => event.kind),
        reason: receipt ? `Unsettled purchase at source ${id}: ${error.message}` : error.message,
      };
      break;
    }
  }
  if (!blocked && deferredScore) {
    const receipt = deferredScore;
    rollback(receipt.saved);
    blocked = {
      actionId: receipt.rollbackSourceId ?? receipt.sourceId,
      kinds: receipt.rollbackKinds ?? receipt.kinds,
      reason:
        receipt.kind === 'action'
          ? 'Deferred purchase has no observed action and monument completion; entire tentative chain was rolled back'
          : 'Deferred purchase has no observed Trade exit; entire tentative chain was rolled back',
    };
  }
  if (!blocked && state().phase === 'finished' && game.resultsDisplay.length === names.length)
    record.terminalDisplayCheckpoint = {
      source: { reference: source.reference, actionIds: [knownActionIds.at(-1)] },
      mode: 'floorTotal',
      scores: game.resultsDisplay.map((result) => ({
        playerId: actorId(result.player),
        total: result.scoreDisplay,
        rank: result.rank,
      })),
    };
  const report = core({ operation: 'publicReplay', replay: record });
  if (
    report.verifiedSteps !== record.steps.length ||
    report.frames?.length !== record.steps.length + 1 ||
    !same(report.frames.at(-1).snapshot, frame.snapshot) ||
    !same(report.frames.at(-1).observation, frame.observation)
  )
    fail('Final core replay differs from the incremental verified prefix');
  return {
    record,
    manifest: {
      schema: 'tzolkin-bga-decoded-replay-v1',
      tableId: game.tableId,
      sourceSha256: sha(rawBytes),
      witnessesSha256: sha(JSON.stringify(witnesses)),
      normalizedSchema: game.schema,
      verifiedSteps: report.verifiedSteps,
      checkpointsVerified: report.checkpointsVerified,
      frameCount: report.frames.length,
      mechanicallyFinished: report.frames.at(-1).snapshot.state.phase === 'finished',
      terminalDisplayComparison: report.terminalDisplayComparison ?? null,
      status: 'partial',
      sourceCoverage: report.sourceCoverage,
      sourceAuthenticated: false,
      strictExporterAccepted: false,
      trainingReady: false,
      ...(declaredProfile === null
        ? {}
        : {
            conditionalMechanicalProfile: copy(declaredProfile),
            observedOptions: copy(observedOptions),
            sourceOptionsObserved: observedOptions.evidence.length > 0,
            sourceOptionsFullyObserved: observedOptions.complete,
            policyTrainingReady: false,
            valueTrainingReady: false,
          }),
      blocked,
      coreCalls: calls,
      audit,
      pendingCalendarAnnouncement:
        calendarEvidence.announcement?.confirmed === false ? calendarEvidence.announcement : null,
      extractionRewardReceipt: extractionReward,
    },
  };
}

async function boundedRead(path) {
  const entry = await stat(path);
  if (!entry.isFile() || entry.size > MAX_BYTES)
    fail('Input must be a regular file at most 16 MiB');
  const bytes = await readFile(path);
  if (bytes.length !== entry.size || bytes.length > MAX_BYTES) fail('Input changed during read');
  return bytes;
}
export async function exportDecodedGame(rawPath, witnessesPath, tablePath, cliPath, outputPath) {
  const raw = await boundedRead(rawPath);
  const evidence = await boundedRead(witnessesPath);
  const table = tablePath === '-' ? null : await boundedRead(tablePath);
  const result = decodePublicGame(raw, JSON.parse(evidence), coreDispatcher(cliPath), table);
  result.manifest.witnessFileSha256 = sha(evidence);
  result.manifest.tableEvidenceSha256 = table === null ? null : sha(table);
  return publishReplay(result, outputPath);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length !== 5 || args.some((value) => !value)) {
    console.error(
      'Usage: node scripts/bga-decode.mjs RAW_JSON PUBLIC_WITNESSES_JSON TABLE_EVIDENCE_JSON RUST_CLI NEW_OUTPUT_DIR',
    );
    process.exitCode = 1;
  } else
    exportDecodedGame(
      ...args.map((value, index) => (index === 2 && value === '-' ? value : resolve(value))),
    )
      .then((manifest) =>
        console.log(
          JSON.stringify({
            tableId: manifest.tableId,
            verifiedSteps: manifest.verifiedSteps,
            mechanicallyFinished: manifest.mechanicallyFinished,
            blocked: manifest.blocked,
            trainingReady: false,
          }),
        ),
      )
      .catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
      });
}

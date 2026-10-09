import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeGame } from './bga-import.mjs';
import {
  assertBasicUnlimitedOptions,
  coreDispatcher,
  publishReplay,
  sourceEvents,
} from './bga-replay.mjs';

export const WITNESS_SCHEMA = 'tzolkin-bga-public-witnesses-v1';
const MAX_BYTES = 16 * 1024 * 1024;
const MAX_STEPS = 4000;
const MAX_CALLS = 12000;
const RESOURCES = ['corn', 'wood', 'stone', 'gold', 'skull'];
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
  const raw = JSON.parse(rawBytes);
  const game = normalizeGame(raw, sha(rawBytes), tableBytes);
  if (game.source.adapter !== 'bga-public-ui-dom-v1' && tableBytes !== null)
    fail('Legacy input requires embedded table settings, not separate table evidence');
  if (
    game.tableId !== witnesses.tableId ||
    !game.rawEntries.some((entry) => entry.action_id === witnesses.initialActionId)
  )
    fail('Witness table/initial source mismatch');
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
      ['actionId', 'expected', 'refills', 'buildings', 'monuments', 'pendingTask'],
      'witness',
    );
    positive(witness.actionId, 'witness.actionId');
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
    const nextTask = nextFrame.observation.pendingTask;
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
            Number(resource === action.resource),
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
  function choose(predicate, reason, derived = false, visibleCalendar = false) {
    if (record.steps.length >= MAX_STEPS) fail('Move limit reached');
    let candidates = legal().filter((row) => predicate(row.action));
    // The following visible decision can establish declining an optional
    // continuation even when the same player remains the turn actor.
    for (
      let count = 0;
      !candidates.length && count < 8 && legal().some((row) => row.action.type === 'skip');
      count++
    ) {
      choose((action) => action.type === 'skip', `optional continuation before ${reason}`, true);
      candidates = legal().filter((row) => predicate(row.action));
    }
    if (!candidates.length) fail(`No legal operation for ${reason}; pending=${pending()}`);
    if (calls + candidates.length * 3 >= MAX_CALLS)
      fail('Core call budget exhausted before the final replay verification');
    const outcomes = candidates.map((row) => ({ row, ...trial(row) }));
    if (
      outcomes.length > 1 &&
      outcomes.some(
        (outcome) =>
          !same(
            mechanical(outcome.frame.snapshot.state),
            mechanical(outcomes[0].frame.snapshot.state),
          ),
      )
    )
      fail(`Ambiguous legal operation for ${reason}`);
    const chosen = outcomes[0];
    const previous = state();
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
    const cornCost = chosen.row.action.type === 'useAction' ? chosen.row.action.cornCost : 0;
    if (!Number.isSafeInteger(cornCost) || cornCost < 0)
      fail('Core useAction corn cost is not a bounded nonnegative integer');
    for (const player of previous.players) {
      const next = chosen.frame.snapshot.state.players[player.id];
      const food = feeding.find((receipt) => receipt.actor === player.id);
      for (const field of [
        'workers',
        'score',
        ...RESOURCES.map((resource) => `resource:${resource}`),
        ...['chaac', 'quetzalcoatl', 'kukulkan'].map((temple) => `temple:${temple}`),
      ]) {
        const delta = field.startsWith('resource:')
          ? next.resources[field.slice(9)] - player.resources[field.slice(9)]
          : field.startsWith('temple:')
            ? next.temples[field.slice(7)] - player.temples[field.slice(7)]
            : next[field] - player[field];
        const cost =
          field === 'resource:corn'
            ? (player.id === previous.currentPlayer ? cornCost : 0) + (food?.cost ?? 0)
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
    });
    audit.push({
      actionId: context.id,
      step: record.steps.length,
      actor,
      action: chosen.row.action,
      derivedFromVisibleBoundary: derived,
      equivalentChoices: outcomes.length,
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

  let blocked = null;
  for (const [id, events] of sourceGroups(
    normalizedEvents,
    witnesses.initialActionId,
    knownActionIds,
  )) {
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
    };
    context = {
      id,
      before: copy(state()),
      pools: {
        currentAge: [...(witness.refills?.currentAge ?? [])],
        age2: [...(witness.refills?.age2 ?? [])],
      },
      claims: {},
      effects: {},
      feeding: [],
      feedingDay: null,
      paymentActor: null,
      paymentOperations: 0,
    };
    try {
      if (!events.length) fail('Raw source action has no observed messages');
      for (const [field, kind] of [
        ['buildings', 'build'],
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
            receipt.actor !== future ||
            future === state().firstPlayer ||
            actorId(payouts[0].actor) !== receipt.actor ||
            actorId(announcements[0].actor) !== receipt.actor ||
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
            choose(
              (action) => action.type === 'technology' && action.technology === detail.technology,
              'technology advance',
            );
            pay(payments);
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
            const ids = witness[event.kind === 'build' ? 'buildings' : 'monuments'];
            if (ids?.length !== 1) fail('Observed purchased card identity is required');
            choose(
              (action) =>
                action.type === event.kind &&
                action.id === ids[0] &&
                (!payments.some(Boolean) || same(action.cost, payments)),
              `observed ${event.kind}`,
            );
          } else if (event.kind === 'workerGain') automatic(event, 'workers', 1);
          else if (event.kind === 'resourceGain') {
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
              /^は(Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を(?:1段)?下げ(?:て|ることで|、)コーン3個を受け取った$/,
            );
            if (!temple) fail('Begging requires one unambiguous temple in the actor-free phrase');
            alignActor(event.actor);
            choose((action) => action.type === 'beg', 'observed begging');
            choose(
              (action) =>
                action.type === 'temple' &&
                action.temple === TEMPLES[temple[1]] &&
                action.direction === -1,
              'observed begging temple',
            );
            automatic(event, `temple:${TEMPLES[temple[1]]}`, 1, -1);
          } else if (event.kind === 'unparsed') {
            const body =
              event.actor === null ? event.rawText : event.rawText.slice(event.actor.length);
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
            if (first) {
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
            } else if (/は\d+点を獲得した$/.test(event.rawText))
              automatic(event, 'score', Number(event.rawText.match(/(\d+)点を獲得した$/)[1]));
            else if (
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
            } else if (
              state().phase === 'finished' &&
              /^(最終スコア|記念碑: |残り資源: )/.test(event.rawText)
            ) {
              // Source display annotations do not change the official core score.
            } else fail(`Unsupported source message: ${event.rawText.slice(0, 120)}`);
          } else fail(`Unsupported source effect ${event.kind}`);
        }
        pay(payments);
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
    } catch (error) {
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
      blocked = { actionId: id, kinds: events.map((event) => event.kind), reason: error.message };
      break;
    }
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

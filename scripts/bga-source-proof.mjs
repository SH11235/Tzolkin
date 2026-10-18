import { isDeepStrictEqual as same } from 'node:util';

const RESOURCES = ['corn', 'wood', 'stone', 'gold', 'skull'];
const FOOD_DAYS = [8, 14, 21, 27];
const fail = (reason) => {
  throw new Error(`Source derivation: ${reason}`);
};

/** Interpret calendar notifications, never engine frames. The caller supplies
 * a separately witnessed day-one boundary; this does not authenticate that input.
 * Required Food Days and age changes can also follow from the basic calendar;
 * this does not invent missing resource payments or rewards. */
export function deriveBasicCalendar(events, initialActionId) {
  if (!Number.isSafeInteger(initialActionId) || initialActionId < 1 || events.length > 20000)
    fail('invalid calendar boundary');
  const groups = new Map();
  let previous = initialActionId;
  for (const event of events) {
    if (event.actionId <= initialActionId) continue;
    if (!Number.isSafeInteger(event.actionId) || typeof event.rawText !== 'string')
      fail('invalid calendar event');
    if (event.actionId < previous) fail('calendar events are not in source order');
    previous = event.actionId;
    if (!groups.has(event.actionId)) groups.set(event.actionId, []);
    groups.get(event.actionId).push(event);
  }
  let round = 1;
  let age = 1;
  let finished = false;
  let nextFirstActor = null;
  const foodDays = [];
  const receipts = [];
  for (const [actionId, messages] of groups) {
    if (finished) fail('source continues after game end');
    const announcements = messages.filter((e) => e.kind === 'firstPlayer');
    if (announcements.length > 1 || announcements.some((e) => typeof e.actor !== 'string'))
      fail(`ambiguous first-player announcement at ${actionId}`);
    if (announcements.length) {
      if (nextFirstActor !== null)
        fail(`duplicate pending first-player announcement at ${actionId}`);
      nextFirstActor = announcements[0].actor;
    }
    const hasFood = messages.some(
      (e) =>
        /^(食料供給日: |世代中間報酬: )/.test(e.rawText) ||
        /は信仰により-?\d+点を獲得した$/.test(e.rawText),
    );
    const advances = messages.filter((e) => ['gearAdvanced', 'doubleAdvance'].includes(e.kind));
    const ages = messages.filter((e) => e.rawText === '建物が第2世代になりました');
    const ends = messages.filter((e) => e.kind === 'gameEnd');
    if (advances.length > 1 || ages.length > 1 || ends.length > 1)
      fail(`duplicate calendar transition at ${actionId}`);
    const before = { round, age, foodDays: [...foodDays] };
    const due = FOOD_DAYS.find((day) => day <= round && !foodDays.includes(day));
    const performsFood = hasFood || ages.length > 0 || (advances.length > 0 && due !== undefined);
    if (performsFood) {
      if (due === undefined) fail(`unexpected food notification at ${actionId}`);
      foodDays.push(due);
    }
    if (ages.length) {
      if (!performsFood || due !== 14 || age !== 1)
        fail(`age announcement has no age-one food day at ${actionId}`);
    }
    if (performsFood && due === 14) age = 2;
    if (advances.length) {
      const days = advances[0].kind === 'doubleAdvance' ? 2 : 1;
      if (round + days > 27 && !ends.length)
        fail(`calendar passes final food day without game end at ${actionId}`);
      round = Math.min(27, round + days);
    }
    if (ends.length) {
      if (round !== 27 || !same(foodDays, FOOD_DAYS) || !advances.length)
        fail(`game end lacks complete calendar at ${actionId}`);
      finished = true;
    }
    if (hasFood || advances.length || ages.length || ends.length)
      receipts.push({
        actionId,
        before,
        expected: { round, age, foodDays: [...foodDays] },
        foodDay: performsFood ? due : null,
        foodDayBasis: performsFood
          ? hasFood || ages.length
            ? 'source notification'
            : 'basic calendar'
          : null,
        ageAnnouncementObserved: ages.length === 1,
        advancedDays: advances.length ? (advances[0].kind === 'doubleAdvance' ? 2 : 1) : 0,
        terminal: finished,
        announcedFirstActor: advances.length ? nextFirstActor : null,
      });
    if (advances.length) nextFirstActor = null;
  }
  return { receipts, finished };
}

function resourceVector(event) {
  const icons = event.detail?.iconResources ?? event.iconResources;
  const amount = event.detail?.explicitAmount;
  if (
    !Array.isArray(icons) ||
    !icons.length ||
    icons.some((r) => !RESOURCES.includes(r)) ||
    (amount != null && (icons.length !== 1 || !Number.isSafeInteger(amount) || amount < 1))
  )
    fail('ambiguous food-day resource caption');
  return Object.fromEntries(
    RESOURCES.map((r) => [r, icons.filter((x) => x === r).length * (amount ?? 1)]),
  );
}

/** Independent notification-side net flow, restricted to Food Day macros.
 * Before/after resources are engine evidence only on the comparison side. */
export function verifyFoodResourceFlow(messages, before, after) {
  return foodResourceFlow(messages, before, after);
}

function foodResourceFlow(messages, before, after, corn = null) {
  const names = before.players.map((p) => p.name);
  const deltas = names.map(() => Object.fromEntries(RESOURCES.map((r) => [r, 0])));
  let claims = 0;
  const add = (name, values, sign) => {
    const actor = names.indexOf(name);
    if (actor < 0) fail('unknown food-day resource actor');
    for (const r of RESOURCES) {
      deltas[actor][r] += values[r] * sign;
      if (!Number.isSafeInteger(deltas[actor][r])) fail('food-day resource overflow');
    }
    claims++;
  };
  for (const event of messages) {
    if (['resourceGain', 'resourcePayment'].includes(event.kind))
      add(event.actor, resourceVector(event), event.kind === 'resourceGain' ? 1 : -1);
    else if (event.kind === 'feeding') {
      const payment = event.rawText.match(
        /^食料供給日: (.+)はワーカー\d+個で\s*(\d+)\s*を支払った$/,
      );
      if (!payment || !same(event.iconResources, ['corn'])) fail('unreadable feeding payment');
      add(payment[1], { corn: Number(payment[2]), wood: 0, stone: 0, gold: 0, skull: 0 }, -1);
    } else if (event.actor !== null && event.kind === 'unparsed') {
      const payout = event.rawText
        .slice(event.actor.length)
        .match(/^は(\d+)\s*を歯車から獲得した$/);
      if (payout) {
        if (!same(event.iconResources, ['corn'])) fail('unreadable calendar payout');
        const receipt = corn?.byCaption.get(event);
        if (corn && (!receipt || receipt.index >= corn.last))
          fail('calendar payout has no preceding verified receipt');
        add(
          event.actor,
          {
            corn: receipt && receipt.index < corn.first ? 0 : Number(payout[1]),
            wood: 0,
            stone: 0,
            gold: 0,
            skull: 0,
          },
          1,
        );
      }
    }
  }
  for (const [i, player] of before.players.entries())
    for (const resource of RESOURCES) {
      const expected = player.resources[resource] + deltas[i][resource];
      if (expected !== after.players[i]?.resources[resource])
        fail(`food-day ${player.name} ${resource} conservation mismatch`);
    }
  return {
    sourceClaims: claims,
    deltas,
    ...(corn
      ? {
          calendarCorn: messages
            .filter((event) => corn.byCaption.has(event))
            .map((event) => {
              const receipt = corn.byCaption.get(event);
              return {
                ...receipt,
                disposition:
                  receipt.index < corn.first ? 'alreadyIncludedBeforeMacro' : 'creditedInsideMacro',
              };
            }),
        }
      : {}),
  };
}

/** Reconstruct receipts from the retained operations and native transitions,
 * independently of decoder calendarEvidence and rollback bookkeeping. */
function calendarCornReceipts(events, record, report, calendar) {
  const names =
    record.initial?.players?.map((p) => p.name) ??
    report.frames[0].snapshot.state.players.map((p) => p.name);
  const initialId = record.source.actionIds[0];
  const rounds = new Map();
  const boundaries = new Map(calendar.map((entry) => [entry.actionId, entry]));
  const groups = new Map();
  let round = 1;
  for (const event of events) {
    if (event.actionId <= initialId) continue;
    if (!groups.has(event.actionId)) {
      groups.set(event.actionId, []);
      rounds.set(event.actionId, round);
      const boundary = boundaries.get(event.actionId);
      if (boundary) round = boundary.expected.round;
    }
    groups.get(event.actionId).push(event);
  }
  const usedClaims = new Set();
  const claims = new Map();
  const receipts = [];
  const resources = (state) => state.players.map((p) => p.resources);
  const legal = (index, type) => {
    const step = record.steps[index];
    const observation = report.frames[index].observation;
    const rows = observation?.legalActions?.filter((row) => same(row.move, step.move));
    if (observation?.actor !== step.actor || rows?.length !== 1 || rows[0].action.type !== type)
      fail('calendar corn operation has no exact actor/legal-action binding');
    return rows[0].action;
  };
  for (const [index, step] of record.steps.entries()) {
    const before = report.frames[index]?.snapshot.state;
    const after = report.frames[index + 1]?.snapshot.state;
    if (step.move?.type === 'firstPlayer') {
      const action = legal(index, 'firstPlayer');
      const source = step.sourceActionIds;
      const actor = step.actor;
      const originals = groups
        .get(source[0])
        ?.filter(
          (event) =>
            event.actor === names[actor] &&
            event.rawText.startsWith(event.actor) &&
            /^はスタートプレイヤースペースに\d+\s*を支払ってワーカーを置いた$/.test(
              event.rawText.slice(event.actor.length),
            ),
        );
      const expected = structuredClone(resources(before));
      expected[actor].corn -= action.cornCost;
      if (
        source.length !== 1 ||
        originals?.length !== 1 ||
        usedClaims.has(originals[0]) ||
        originals[0].kind !== 'unparsed' ||
        !same(originals[0].iconResources, ['corn']) ||
        !Number.isSafeInteger(originals[0].messageIndex) ||
        originals[0].messageIndex < 0 ||
        !Number.isSafeInteger(action.cornCost) ||
        action.cornCost < 0 ||
        Number(originals[0].rawText.slice(names[actor].length).match(/\d+/)[0]) !==
          action.cornCost ||
        before.currentPlayer !== actor ||
        before.firstPlayerClaimed !== null ||
        after.firstPlayerClaimed !== actor ||
        before.round !== after.round ||
        rounds.get(source[0]) !== before.round ||
        claims.has(before.round) ||
        before.accumulatedCorn !== after.accumulatedCorn ||
        !same(resources(after), expected)
      )
        fail('calendar corn claim has no unique original operation/round transition');
      usedClaims.add(originals[0]);
      claims.set(before.round, { index, actor, source: source[0], event: originals[0] });
    }
    if (step.move?.type !== 'endTurn' || before.firstPlayerClaimed !== step.actor) continue;
    legal(index, 'endTurn');
    const claim = claims.get(before.round);
    const amount = before.accumulatedCorn;
    if (
      !claim ||
      claim.actor !== step.actor ||
      claim.index >= index ||
      before.currentPlayer !== step.actor ||
      step.sourceActionIds.length !== 1 ||
      rounds.get(step.sourceActionIds[0]) !== before.round ||
      !Number.isSafeInteger(amount) ||
      amount < 0 ||
      after.accumulatedCorn !== 0 ||
      (after.round !== before.round &&
        after.round !== boundaries.get(step.sourceActionIds[0])?.expected.round) ||
      receipts.some((receipt) => receipt.round === before.round)
    )
      fail('calendar corn EndTurn has no unique claim/round receipt');
    const food = !same(before.foodDays, after.foodDays);
    if (!food) {
      const expected = structuredClone(resources(before));
      expected[step.actor].corn += amount;
      if (!same(resources(after), expected))
        fail('calendar corn EndTurn resource transition differs');
    } else if (
      !same(after.foodDays.slice(0, -1), before.foodDays) ||
      after.foodDays.length !== before.foodDays.length + 1 ||
      !FOOD_DAYS.includes(after.foodDays.at(-1)) ||
      (after.pending !== null && after.pending?.task?.type !== 'rotation')
    )
      fail('calendar corn concurrent Food Day is not an atomic supported transition');
    receipts.push({
      index,
      actor: step.actor,
      amount,
      round: before.round,
      sourceActionId: step.sourceActionIds[0],
      claimIndex: claim.index,
      claimSourceActionId: claim.source,
      claimMessageIndex: claim.event.messageIndex,
      food,
    });
  }
  const byCaption = new Map();
  const used = new Set();
  const captionIds = new Set();
  for (const event of events) {
    if (event.actionId <= initialId) continue;
    const literal = event.rawText.match(/^(.+)は(\d+)\s*を歯車から獲得した$/);
    if (!literal) continue;
    const actor = names.indexOf(event.actor);
    const amount = Number(literal[2]);
    const identity = `${event.actionId}:${event.messageIndex}`;
    const matches = receipts.filter(
      (receipt) =>
        receipt.actor === actor &&
        receipt.amount === amount &&
        receipt.round === rounds.get(event.actionId) &&
        receipt.sourceActionId <= event.actionId,
    );
    if (
      event.kind !== 'unparsed' ||
      actor < 0 ||
      literal[1] !== event.actor ||
      !same(event.iconResources, ['corn']) ||
      !Number.isSafeInteger(amount) ||
      !Number.isSafeInteger(event.messageIndex) ||
      event.messageIndex < 0 ||
      captionIds.has(identity) ||
      matches.length !== 1 ||
      used.has(matches[0])
    )
      fail('calendar corn caption has no unique unconsumed actor/amount/round receipt');
    const receipt = matches[0];
    receipt.captionSourceActionId = event.actionId;
    receipt.captionMessageIndex = event.messageIndex;
    byCaption.set(event, receipt);
    used.add(receipt);
    captionIds.add(identity);
  }
  for (const receipt of receipts) {
    if (receipt.amount > 0 && !used.has(receipt))
      fail('calendar corn positive receipt has no caption');
    if (receipt.food) {
      const before = report.frames[receipt.index].snapshot.state;
      const after = report.frames[receipt.index + 1].snapshot.state;
      const boundary = boundaries.get(receipt.sourceActionId);
      if (!boundary || boundary.foodDay !== after.foodDays.at(-1))
        fail('calendar corn concurrent feeding has no original Food Day boundary');
      foodResourceFlow(groups.get(receipt.sourceActionId), before, after, {
        byCaption,
        first: receipt.index,
        last: receipt.index + 1,
      });
    }
  }
  return { receipts, byCaption };
}

/** Counterfactual diagnostic only: floor scores BEFORE applying the existing
 * workers-on-gears tie rule. A match here never establishes BGA's tie rule. */
export function compareFloorRankings(scores) {
  const rank = (score, floor) =>
    1 +
    scores.filter((other) => {
      const a = floor ? Math.floor(other.total) : other.total;
      const b = floor ? Math.floor(score.total) : score.total;
      return a > b || (a === b && other.workersOnGears > score.workersOnGears);
    }).length;
  const players = scores.map((score) => ({
    playerId: score.playerId,
    nativeRank: rank(score, false),
    floorBeforeRanking: rank(score, true),
  }));
  return {
    sameForThisReplay: players.every((p) => p.nativeRank === p.floorBeforeRanking),
    players,
  };
}

/** Recompute the basic final conversion from a supplied counter checkpoint,
 * independently of native FinalScore fields. The checkpoint's authenticity and
 * original resource provenance remain the caller's responsibility. */
export function verifyTerminalResourceFractions(observedPlayers, scores) {
  if (!Array.isArray(observedPlayers) || observedPlayers.length !== scores.length)
    fail('missing terminal resource counter checkpoint');
  return observedPlayers.map((player, i) => {
    const resources = player.resources;
    if (
      !resources ||
      RESOURCES.some((r) => !Number.isSafeInteger(resources[r]) || resources[r] < 0)
    )
      fail('incomplete terminal resource counters');
    const cornEquivalent =
      resources.corn + 2 * resources.wood + 3 * resources.stone + 4 * resources.gold;
    if (!Number.isSafeInteger(cornEquivalent)) fail('terminal resource conversion overflow');
    const resourcePoints = cornEquivalent / 4;
    const skullPoints = 3 * resources.skull;
    if (!Number.isSafeInteger(skullPoints)) fail('terminal skull conversion overflow');
    const score = scores.find((s) => s.playerId === i);
    if (!score || score.resourcePoints !== resourcePoints || score.skullPoints !== skullPoints)
      fail('terminal resource fraction differs from native score');
    return {
      playerId: i,
      cornEquivalent,
      resourcePoints,
      skullPoints,
      displayRemainder: Math.floor(resourcePoints + skullPoints),
    };
  });
}

/** A physical-piece ledger only. It consumes literal source worker operations
 * and the decoder's source-bound card choices/reveals; it does not reimplement
 * resource gains, technology benefits, feeding or scoring. */
export function verifyWorkerCardLedger(events, record, report, audit, calendar) {
  const initial = record.initial;
  if (
    !record.initialCheckpoint?.expected?.gears ||
    !record.initialCheckpoint.expected.players?.every((p) => Number.isSafeInteger(p.workers))
  )
    return {
      status: 'unavailable',
      reason: 'initial worker board/count observation is incomplete',
    };
  if (record.steps.some((s) => s.sourceActionIds.length !== 1))
    return {
      status: 'unavailable',
      reason: 'piece ledger requires single-source operation bindings',
    };
  const cancellations = events.filter((e) => e.kind === 'cancellation');
  for (const event of cancellations) {
    const receipts = audit.filter((a) => a.cancellation && a.actionId === event.actionId);
    if (!receipts.length)
      return { status: 'unavailable', reason: 'cancellation source-to-step receipt is missing' };
    if (receipts.length !== 1) fail('cancellation has no unique rollback receipt');
    if (receipts[0].removedSteps === 0 && receipts[0].discardedSourceActionIds?.length)
      fail('no-op cancellation contains discarded source actions');
    if (receipts[0].removedSteps !== 0)
      return {
        status: 'unavailable',
        reason: `nonempty rollback at ${event.actionId} needs source-observed retained history`,
      };
  }
  const gears = structuredClone(initial.gears);
  const workers = record.initialCheckpoint.expected.players.map((p) => p.workers);
  const buildings = [...initial.buildings];
  const monuments = [...initial.monuments];
  const skullSpaces = [...initial.skullSpaces];
  const jungle = structuredClone(initial.jungle);
  const owned = initial.players.map(() => ({ buildings: [], monuments: [] }));
  let claimed = null;
  let frameIndex = 0;
  let placements = 0;
  let removals = 0;
  let purchases = 0;
  let revealed = 0;
  const boundaries = [];
  let foodBoundary = null;
  const forestActions = new Map();
  const groups = new Map();
  for (const event of events) {
    if (event.actionId <= record.source.actionIds[0]) continue;
    if (!groups.has(event.actionId)) groups.set(event.actionId, []);
    groups.get(event.actionId).push(event);
  }
  const compare = (id) => {
    const state = report.frames[frameIndex].snapshot.state;
    for (const [key, value] of Object.entries({
      gears,
      skullSpaces,
      jungle,
      buildings,
      monuments,
      firstPlayerClaimed: claimed,
    }))
      if (!same(state[key], value)) fail(`piece ledger ${key} mismatch at ${id}`);
    for (const [actor, value] of owned.entries()) {
      if (
        workers[actor] !== state.players[actor].workers ||
        !same(value.buildings, state.players[actor].buildings) ||
        !same(value.monuments, state.players[actor].monuments)
      )
        fail(`piece ledger ownership mismatch at ${id}`);
    }
  };
  compare(record.source.actionIds[0]);
  const fragment = () =>
    structuredClone({
      gears,
      skullSpaces,
      jungle,
      buildings,
      monuments,
      firstPlayerClaimed: claimed,
      players: owned.map((p, i) => ({ ...p, workers: workers[i] })),
    });
  boundaries.push({ actionId: record.source.actionIds[0], expected: fragment() });
  for (const [id, messages] of groups) {
    for (const event of messages.filter((e) => e.kind === 'gearAction')) {
      if (event.detail.gear === 'palenque' && [2, 3, 4, 5].includes(event.detail.actionPosition))
        forestActions.set(event.actor, { box: event.detail.actionPosition, actionId: id });
      else forestActions.delete(event.actor);
    }
    const sourceActor = (step, event) => initial.players[step.actor]?.name === event.actor;
    const consumed = new Set();
    const claim = (predicate, label) => {
      const event = messages.find((e) => !consumed.has(e) && predicate(e));
      if (!event) fail(`${label} has no unused literal source at ${id}`);
      consumed.add(event);
    };
    while (record.steps[frameIndex]?.sourceActionIds[0] === id) {
      const step = record.steps[frameIndex];
      const move = step.move;
      if (move.type === 'place') {
        claim(
          (e) => e.kind === 'place' && sourceActor(step, e) && e.detail.gear === move.gear,
          'placement',
        );
        const occupied =
          Object.values(gears)
            .flat()
            .filter((w) => w && w.playerId === step.actor).length + Number(claimed === step.actor);
        if (occupied >= workers[step.actor])
          fail(`placement exceeds source-owned workers at ${id}`);
        const position = gears[move.gear]?.findIndex((w) => w === null);
        if (
          position === undefined ||
          position < 0 ||
          position > (move.gear === 'chichenItza' ? 10 : 7)
        )
          fail(`no basic worker placement cell at ${id}`);
        gears[move.gear][position] = { playerId: step.actor, dummy: false };
        placements++;
      } else if (move.type === 'remove') {
        claim(
          (e) =>
            e.kind === 'gearAction' &&
            sourceActor(step, e) &&
            e.detail.source === 'removal' &&
            e.detail.gear === move.gear &&
            e.detail.workerPosition === move.position,
          'removal physical position',
        );
        if (!same(gears[move.gear]?.[move.position], { playerId: step.actor, dummy: false }))
          fail(`removal does not own the physical source cell at ${id}`);
        gears[move.gear][move.position] = null;
        removals++;
      } else if (move.type === 'firstPlayer') {
        if (claimed !== null) fail(`starting worker space is occupied at ${id}`);
        if (
          Object.values(gears)
            .flat()
            .filter((w) => w && w.playerId === step.actor).length >= workers[step.actor]
        )
          fail(`starting placement exceeds source-owned workers at ${id}`);
        claim(
          (e) =>
            sourceActor(step, e) &&
            /^はスタートプレイヤースペースに\d+\s*を支払ってワーカーを置いた$/.test(
              e.rawText.slice(e.actor.length),
            ),
          'starting worker',
        );
        claimed = step.actor;
        placements++;
      }
      const card =
        move.type === 'choose' && move.choiceId.match(/^(build|monument):([^:]+)(?::|$)/);
      if (card) {
        const kind = card[1];
        const cardId = card[2];
        claim((e) => e.kind === kind && sourceActor(step, e), 'card acquisition');
        const market = kind === 'build' ? buildings : monuments;
        const position = market.indexOf(cardId);
        if (position < 0) fail(`acquired card absent from public market at ${id}`);
        market.splice(position, 1);
        owned[step.actor][kind === 'build' ? 'buildings' : 'monuments'].push(cardId);
        purchases++;
      }
      const refills = step.refills;
      if (refills?.currentAge?.length) {
        buildings.push(...refills.currentAge);
        revealed += refills.currentAge.length;
      }
      if (refills?.age2?.length) {
        buildings.splice(0, buildings.length, ...refills.age2);
        revealed += refills.age2.length;
      }
      const publicBuildings = [...buildings, ...owned.flatMap((p) => p.buildings)];
      if (buildings.length > 6 || new Set(publicBuildings).size !== publicBuildings.length)
        fail(`invalid public card refill at ${id}`);
      frameIndex++;
    }
    for (const event of messages)
      if (
        (['place', 'build', 'monument'].includes(event.kind) ||
          (event.kind === 'gearAction' && event.detail.source === 'removal')) &&
        !consumed.has(event)
      )
        fail(`source physical operation has no effective replay binding at ${id}`);
    for (const receipt of audit.filter((a) => a.cancellation && a.actionId === id))
      if (receipt.retainedSteps !== undefined && receipt.retainedSteps !== frameIndex)
        fail(`cancellation retained boundary differs from effective series at ${id}`);
    for (const event of messages.filter((e) => e.kind === 'workerGain')) {
      const actor = initial.players.findIndex((p) => p.name === event.actor);
      if (actor < 0 || ++workers[actor] > 6) fail(`invalid source worker increase at ${id}`);
    }
    for (const event of messages.filter(
      (e) => e.kind === 'gearAction' && e.detail.gear === 'chichenItza',
    )) {
      const position = event.detail.actionPosition;
      if (position < 1 || position > 9)
        return {
          status: 'unavailable',
          reason: `nonliteral skull action at ${id} needs a source choice`,
        };
      if (skullSpaces[position] !== null) fail(`source skull cell already occupied at ${id}`);
      const actor = initial.players.findIndex((p) => p.name === event.actor);
      if (actor < 0) fail(`source skull actor is unknown at ${id}`);
      skullSpaces[position] = actor;
    }
    for (const event of messages) {
      if (typeof event.actor !== 'string') continue;
      const tile = event.rawText
        .slice(event.actor.length)
        .match(/^は(コーン|corn|wood)タイルを獲得した$/);
      if (!tile) continue;
      const source = forestActions.get(event.actor);
      if (!source) fail(`forest tile has no source box at ${id}`);
      const box = source.box;
      forestActions.delete(event.actor);
      const kind = tile[1] === 'wood' ? 'wood' : 'corn';
      const burns = messages.filter(
        (notice) =>
          notice.kind === 'unparsed' &&
          notice.actor === event.actor &&
          /^は(Chaac|Quetzalcoatl|Kukulkan|Kukulcan)の信仰を1段下げることで焼き畑をしてコーンを手に入れた$/.test(
            notice.rawText.slice(event.actor.length),
          ),
      );
      if (burns.length) {
        if (
          burns.length !== 1 ||
          burns[0].iconResources.length ||
          kind !== 'corn' ||
          messages.indexOf(burns[0]) >= messages.indexOf(event) ||
          jungle[box].wood < 1 ||
          jungle[box].corn !== jungle[box].wood
        )
          fail(`burn has no unique covered-corn source at ${id}`);
        // The literal burn discards the covering wood tile, not a wood gain.
        jungle[box].wood--;
      }
      if (--jungle[box][kind] < 0) fail(`source harvest exceeds remaining forest tiles at ${id}`);
      if (jungle[box].corn < jungle[box].wood)
        return {
          status: 'unavailable',
          reason: `covered-corn harvest at ${id} needs explicit burn/technology evidence`,
        };
    }
    const days = calendar.find((r) => r.actionId === id)?.advancedDays ?? 0;
    if (days) {
      claimed = null;
      for (let n = 0; n < days; n++)
        for (const [gear, slots] of Object.entries(gears)) {
          const rotated = slots.map(() => null);
          for (const [position, worker] of slots.entries()) {
            if (!worker) continue;
            const next = position + 1;
            if (worker.dummy) rotated[next % slots.length] = worker;
            else if (next <= (gear === 'chichenItza' ? 10 : 7)) rotated[next] = worker;
          }
          gears[gear] = rotated;
        }
    }
    compare(id);
    const calendarReceipt = calendar.find((r) => r.actionId === id);
    if (calendarReceipt?.foodDay != null) foodBoundary = calendarReceipt.foodDay;
    if (foodBoundary !== null && days) {
      boundaries.push({ actionId: id, foodDay: foodBoundary, expected: fragment() });
      foodBoundary = null;
    }
  }
  if (frameIndex !== record.steps.length)
    fail('piece ledger did not consume the effective operation series');
  return {
    status: 'matched',
    sourceMacros: groups.size,
    placements,
    removals,
    purchases,
    revealed,
    noOpCancellationReceipts: cancellations.length,
    boundaries,
    limits:
      'Card identities/reveals and no-op cancellation observations still require caller-supplied source provenance; equivalent physical copies are not identified here.',
  };
}

/** Keep derived checks separate from the core's full-snapshot coverage contract.
 * Cancellation selection and card reveals still depend on the supplied witnesses.
 * This proof verifies a completed day-one replay only, not every unknown option. */
export function verifySourceDerivations(events, record, report, audit = []) {
  const last = report.frames.at(-1).snapshot.state;
  if (last.phase !== 'finished' || report.frames[0].snapshot.state.round !== 1)
    return {
      status: 'unavailable',
      reason: 'requires a finished replay and witnessed day-one anchor',
    };
  const initialId = record.source.actionIds[0];
  const derived = deriveBasicCalendar(events, initialId);
  if (!derived.finished && record.initialCheckpoint?.expected?.round !== 1)
    return {
      status: 'unavailable',
      reason: 'requires a complete source calendar and witnessed day-one anchor',
    };
  if (!derived.finished) fail('finished replay has no complete source calendar');
  const corn = calendarCornReceipts(events, record, report, derived.receipts);
  if (record.initialCheckpoint?.expected?.round !== 1)
    return {
      status: 'unavailable',
      reason: 'requires a witnessed day-one anchor for the complete source proof',
      calendarCornReceipts: corn.receipts,
      calendarCornBasis:
        'original captions and retained native day-one operations; conditional anchor',
      trainingReady: false,
    };
  const firstRotation = derived.receipts.find((r) => r.advancedDays)?.actionId;
  const firstActors = [];
  for (const event of events)
    if (
      event.actionId > initialId &&
      event.actionId < firstRotation &&
      ['place', 'gearAction'].includes(event.kind) &&
      !firstActors.some((e) => e.actor === event.actor)
    )
      firstActors.push({ actor: event.actor, actionId: event.actionId });
  const initial = report.frames[0].snapshot.state;
  const seatCycle = firstActors.map((e) => initial.players.findIndex((p) => p.name === e.actor));
  if (seatCycle.length !== initial.players.length || !same(seatCycle, initial.turnOrder))
    fail('initial source actor cycle differs from replay turn order');
  const initialExpected = {
    round: 1,
    age: 1,
    foodDays: [],
    turnIndex: 0,
    turnOrder: seatCycle,
    firstPlayer: seatCycle[0],
    currentPlayer: seatCycle[0],
  };
  for (const [key, value] of Object.entries(initialExpected))
    if (!same(initial[key], value)) fail(`initial ${key} differs from day-one source cycle`);
  const tops = record.initialCheckpoint.expected.jungle;
  let initialJungle = null;
  if (
    tops &&
    [2, 3, 4, 5].every((box) => tops[box]?.[box === 2 ? 'corn' : 'wood'] === initial.players.length)
  ) {
    initialJungle = Object.fromEntries(
      [2, 3, 4, 5].map((box) => [
        box,
        { corn: initial.players.length, wood: box === 2 ? 0 : initial.players.length },
      ]),
    );
    if (!same(initial.jungle, initialJungle))
      fail('initial covered corn differs from basic unharvested stacks');
  }
  const ranges = new Map();
  for (const [i, step] of record.steps.entries())
    for (const id of step.sourceActionIds) {
      const range = ranges.get(id) ?? { first: i, last: i + 1 };
      if (ranges.has(id) && range.last !== i)
        fail(`source ${id} operations do not form one contiguous macro`);
      range.last = i + 1;
      ranges.set(id, range);
    }
  const checked = [];
  let firstActor = firstActors[0].actor;
  for (const receipt of derived.receipts) {
    const range = ranges.get(receipt.actionId);
    if (!range) fail(`calendar source ${receipt.actionId} has no replay operation`);
    if (
      record.steps
        .slice(range.first, range.last)
        .some((step) => !same(step.sourceActionIds, [receipt.actionId]))
    )
      fail(`calendar source ${receipt.actionId} has an ambiguous operation boundary`);
    const state = report.frames[range.last].snapshot.state;
    for (const [key, value] of Object.entries(receipt.expected))
      if (!same(state[key], value)) fail(`${key} mismatch at source ${receipt.actionId}`);
    let turnCycle = null;
    if (receipt.advancedDays && !receipt.terminal) {
      if (receipt.announcedFirstActor !== null) firstActor = receipt.announcedFirstActor;
      const firstPlayer = initial.players.findIndex((p) => p.name === firstActor);
      const start = seatCycle.indexOf(firstPlayer);
      if (start < 0) fail('source first-player announcement names an unknown player');
      turnCycle = {
        firstPlayer,
        currentPlayer: firstPlayer,
        turnIndex: 0,
        turnOrder: [...seatCycle.slice(start), ...seatCycle.slice(0, start)],
      };
      for (const [key, value] of Object.entries(turnCycle))
        if (!same(state[key], value)) fail(`${key} cycle mismatch at source ${receipt.actionId}`);
    }
    let resources = null;
    if (receipt.foodDay !== null)
      resources = foodResourceFlow(
        events.filter((e) => e.actionId === receipt.actionId),
        report.frames[range.first].snapshot.state,
        state,
        { byCaption: corn.byCaption, first: range.first, last: range.last },
      );
    checked.push({ ...receipt, frame: range.last, resources, turnCycle });
  }
  return {
    status: 'matched',
    basis: 'basic calendar notifications and food-day resource conservation',
    initialActorCycle: firstActors,
    initialDerivations: {
      expected: initialExpected,
      jungle:
        initialJungle === null
          ? { status: 'unavailable', reason: 'visible initial top-stack checkpoint missing' }
          : {
              basis: 'basic day-one unharvested stacks plus supplied visible tops',
              expected: initialJungle,
            },
    },
    calendar: checked,
    calendarCornReceipts: corn.receipts,
    pieces: verifyWorkerCardLedger(events, record, report, audit, derived.receipts),
    terminalCounterConversion: record.steps
      .at(-1)
      ?.checkpoint?.expected?.players?.every(
        (p) => p.resources && RESOURCES.every((r) => Object.hasOwn(p.resources, r)),
      )
      ? {
          basis: 'supplied final counter checkpoint; source provenance not authenticated here',
          players: verifyTerminalResourceFractions(
            record.steps.at(-1).checkpoint.expected.players,
            last.finalScores,
          ),
        }
      : { status: 'unavailable', reason: 'final counter checkpoint is incomplete' },
    rankingDiagnostic: compareFloorRankings(last.finalScores),
    fullSnapshotCoverage: report.sourceCoverage,
    selectedActionsLegalUnderDeclaredProfile: true,
    fullLegalActionSetEstablished: false,
    sourceOptionsEstablished: false,
    missingRuleContract:
      'BGA release rule scope; unlisted core extensions are not assumed available on BGA',
    dependsOnSuppliedCancellationAndCardWitnesses: true,
    trainingReady: false,
  };
}

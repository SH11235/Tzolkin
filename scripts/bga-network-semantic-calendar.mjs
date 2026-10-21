import { isDeepStrictEqual } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';

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

export function createTerminalCalendarComparison({ index, sourceToNative }) {
  const sequence = originalNetworkSequence(index);
  const witness = terminalPlayingCalendarWitness(index);
  let obligation = null;
  let failure = null;
  const reject = (message) => {
    failure ??= message;
  };
  const original = (ref) => {
    if (!ref) return null;
    const saved = sequence?.groups.find(({ group }) => group.moveId === ref.moveId);
    const group = saved?.group;
    if (!group || !isDeepStrictEqual(group, saved.original)) return null;
    const entry = group?.entries.find(
      (n) =>
        n.uid === ref.uid &&
        n.packetIndex === ref.packetIndex &&
        n.notificationIndex === ref.notificationIndex,
    );
    const packet = entry && originalPacketEntry(entry);
    return packet?.rawSha256 === index.rawSha256 &&
      packet.packetType === 'resend' &&
      packet.tableId === index.tableId &&
      entry.channel === `/table/t${index.tableId}`
      ? entry
      : null;
  };
  function compare(context, { ledger, native }) {
    const fields = [],
      evidence = [];
    if (!witness) return { fields, evidence, failure };
    const { group, transitions } = context;
    if (group.moveId === witness.start.moveId) {
      const rotations = transitions.filter((t) => t.action.type === 'rotate');
      const transition = rotations[0];
      const refs = [
        witness.start,
        ...witness.priorFeeding,
        ...witness.turnStarts,
        ...witness.turnEnds,
        witness.finalFeeding,
        witness.finalRotation,
        witness.terminal,
      ];
      if (
        obligation ||
        rotations.length !== 1 ||
        !transition ||
        transition.action.days !== 2 ||
        !refs.every(original) ||
        !isDeepStrictEqual(transition.after, native)
      ) {
        reject('final-playing calendar entry lacks its original rotation or native transition');
      } else {
        let bound;
        try {
          bound = bindTerminalCalendarTransition(
            witness,
            witness.sourcePriorDay,
            transition.before,
            transition.after,
            sourceToNative,
          );
        } catch (error) {
          reject(error.message);
        }
        if (
          !bound ||
          transition.before.currentPlayer !== transition.actor ||
          !transition.legalActions.some((r) => isDeepStrictEqual(r.action, transition.action)) ||
          sourceToNative.get(String(original(witness.start)?.args.player_id)) !== transition.actor
        ) {
          reject('final-playing calendar is not bound to the actual selected rotation');
        } else {
          obligation = {
            startMoveId: group.moveId,
            terminalMoveId: witness.terminal.moveId,
            nativeActorOrder: copy(bound.nativeActorOrder),
            completedActors: [],
            closed: false,
            decisionTimeFeatureEquivalence: 'notEstablished',
          };
        }
      }
    }
    if (!obligation || failure) return { fields, evidence, failure };
    if (group.moveId > witness.terminal.moveId) {
      if (!obligation.closed || transitions.length)
        reject('native operations continue beyond the original final calendar');
      return { fields, evidence, failure };
    }
    if (group.moveId < witness.terminal.moveId && native.phase !== 'playing')
      reject('native final-playing cycle ends before the original terminal');
    const sourceEnds = witness.turnEnds.filter((r) => r.moveId === group.moveId);
    const nativeEnds = transitions.filter((t) => t.action.type === 'endTurn');
    if (sourceEnds.length !== nativeEnds.length)
      reject('final-playing actor cycle omits or adds a native turn');
    for (const [i, transition] of nativeEnds.entries()) {
      const position = obligation.completedActors.length,
        actor = obligation.nativeActorOrder[position];
      if (
        actor === undefined ||
        transition.actor !== actor ||
        transition.before.currentPlayer !== actor ||
        transition.before.turnIndex !== position ||
        transition.before.round !== 27 ||
        !isDeepStrictEqual(transition.before.turnOrder, obligation.nativeActorOrder) ||
        !isDeepStrictEqual(transition.before.foodDays, [8, 14, 21]) ||
        sourceToNative.get(String(original(sourceEnds[i])?.args.active_player)) !== actor
      ) {
        reject('final-playing native turn does not match the original actor order');
      } else obligation.completedActors.push(actor);
    }
    if (native.phase === 'playing') {
      if (
        ledger.day !== witness.sourcePlayingDay ||
        native.round !== witness.nativePlayingRound ||
        !isDeepStrictEqual(native.turnOrder, obligation.nativeActorOrder)
      )
        reject('final-playing coordinate mapping escaped its bounded cycle');
      else if (!failure) {
        fields.push('round');
        evidence.push({
          type: 'terminalPlayingCalendar',
          sourceRawSha256: index.rawSha256,
          sourceActualRound: ledger.day + 1,
          nativePlayingRound: native.round,
          start: copy(witness.start),
          currentMoveId: group.moveId,
          closurePending: true,
          decisionTimeFeatureEquivalence: 'notEstablished',
        });
      }
    }
    if (group.moveId === witness.terminal.moveId) {
      if (
        native.phase !== 'finished' ||
        ledger.day !== witness.sourceFinalDay ||
        !isDeepStrictEqual(native.foodDays, [8, 14, 21, 27]) ||
        !isDeepStrictEqual(obligation.completedActors, obligation.nativeActorOrder)
      ) {
        reject(
          'final-playing calendar did not close its actor cycle, food supply and terminal state',
        );
      } else if (!failure) {
        obligation.closed = true;
        evidence.push({
          type: 'terminalPlayingCalendarClosure',
          startMoveId: obligation.startMoveId,
          terminalMoveId: group.moveId,
          nativeActorOrder: copy(obligation.completedActors),
          terminalValidationRequired: true,
          decisionTimeFeatureEquivalence: 'notEstablished',
        });
      }
    }
    return { fields: failure ? [] : fields, evidence, failure };
  }
  return {
    compare,
    finish: () => ({
      obligation: copy(obligation),
      failure,
      allObligationsClosed: !failure && (!obligation || obligation.closed),
    }),
  };
}

import { isDeepStrictEqual } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';
import { provePostReceiptFood } from './bga-network-semantic-feeding.mjs';
import { createBuildingResourceComparison } from './bga-network-building-resources.mjs';
import { createTerminalCalendarComparison } from './bga-network-semantic-calendar.mjs';
import { terminalSettlementScoreUids } from './bga-network-terminal-settlement.mjs';
import { createDay14FoodProof, day14FoodCandidate } from './bga-network-day14-feeding.mjs';

const clone = structuredClone;
const sourceId = (entry) => String(entry.args.player_id);
const reference = (entry) => ({
  uid: entry.uid,
  packetIndex: entry.packetIndex,
  notificationIndex: entry.notificationIndex,
});

export function createSemanticComparison({
  index,
  initialMoveId,
  initial,
  sourceToNative,
  mapCard,
}) {
  const sequence = originalNetworkSequence(index);
  const remaining = sequence?.groups.filter(({ group }) => group.moveId > initialMoveId) ?? [];
  const mapping = new Map(sourceToNative);
  const day14Food = createDay14FoodProof({
    index,
    initial,
    initialMoveId,
    sourceToNative: mapping,
    mapCard,
  });
  const buildingResources = createBuildingResourceComparison({ index, sourceToNative: mapping });
  const calendarComparison = createTerminalCalendarComparison({ index, sourceToNative: mapping });
  const workers = new Map(index.initial.workers.map((w) => [String(w.id), clone(w)]));
  const owned = new Map(
    [...mapping.keys()].map((id) => [
      id,
      index.initial.workers.filter((w) => String(w.player) === id).length,
    ]),
  );
  const purchases = new Map([...mapping.keys()].map((id) => [id, []]));
  let cursor = 0;
  let order = clone(initial.turnOrder);
  let sourcePool = Number(index.initial.cornOnWheel);
  for (const group of index.groups.filter((g) => g.moveId <= initialMoveId))
    for (const n of group.entries) {
      if (n.type === 'cornOnWheel') sourcePool++;
      if (n.type === 'firstPlayer') sourcePool = 0;
      if (n.type === 'newWorker') {
        const id = sourceId(n);
        owned.set(id, owned.get(id) + 1);
        purchases.get(id)?.push(reference(n));
      }
    }
  let claim = null;
  let failure = sequence ? null : 'original notification sequence is not bound';
  const obligations = [];
  const decisions = [];
  const reject = (reason) => {
    failure ??= reason;
  };
  const publicEntry = (entry) => {
    const packet = originalPacketEntry(entry);
    return (
      packet?.packetType === 'resend' &&
      packet.rawSha256 === sequence?.rawSha256 &&
      packet.tableId === index.tableId &&
      entry.channel === `/table/t${index.tableId}`
    );
  };
  function begin(group, ledger, native, observation) {
    const bound = remaining[cursor++];
    if (!bound || bound.group !== group || !isDeepStrictEqual(bound.original, group))
      reject('original notification group changed or was omitted');
    return {
      group,
      original: clone(bound?.original),
      source: clone(ledger),
      native: clone(native),
      observation: clone(observation),
      transitions: [],
    };
  }
  function applied(context, transition) {
    if (
      ['rotate', 'endTurn', 'build', 'technology', 'resource', 'temple', 'skip'].includes(
        transition.action.type,
      )
    )
      context.transitions.push(clone(transition));
  }
  function compare(
    context,
    {
      ledger,
      native,
      observation,
      differences,
      applied,
      humanDecision,
      constructionTransition,
      skullShortageEvidence,
      terminalScoreSettlement,
    },
  ) {
    const { group, source: beforeSource, native: before, observation: priorObservation } = context;
    const entries = group.entries;
    if (!isDeepStrictEqual(context.original, group))
      reject('original notification group changed during its native transition');
    const allowed = new Set();
    const evidence = [];
    const placements = new Map();
    for (const entry of entries) {
      const id = sourceId(entry),
        workerId = String(entry.args.worker_id);
      if (entry.type === 'newWorker') {
        if (!publicEntry(entry) || !owned.has(id) || owned.get(id) >= 6)
          reject('new worker lacks a public original purchase or exceeds ownership');
        else {
          owned.set(id, owned.get(id) + 1);
          purchases.get(id).push(reference(entry));
        }
      } else if (entry.type === 'placeWorker') {
        const known = workers.get(workerId);
        const assigned = [...workers.values()].filter((w) => String(w.player) === id).length;
        if (
          !publicEntry(entry) ||
          !owned.has(id) ||
          !/^[1-9]\d*$/.test(workerId) ||
          (known ? String(known.player) !== id || known.place !== null : owned.get(id) <= assigned)
        )
          reject('worker identity lacks an unassigned owned slot or is already occupied');
        else {
          const binding = {
            source: reference(entry),
            workerId,
            owner: id,
            firstObservation: !known,
            owned: owned.get(id),
            previouslyAssigned: assigned,
            purchases: clone(purchases.get(id)),
          };
          placements.set(entry, binding);
          workers.set(workerId, {
            id: workerId,
            player: id,
            place: entry.args.place,
            position: Number(entry.args.position),
          });
          if (!known) evidence.push({ type: 'firstWorkerIdentity', ...binding });
        }
      } else if (entry.type === 'pickUpWorker' && workers.has(workerId)) {
        workers.get(workerId).place = null;
        workers.get(workerId).position = null;
      } else if (entry.type === 'rotateWheel') {
        for (const worker of entry.args.workers) workers.set(String(worker.id), clone(worker));
      }
    }
    for (const [id, count] of owned)
      if (ledger.players[id].workers !== count || native.players[mapping.get(id)].workers !== count)
        reject('original purchases do not match complete source and native worker counts');
    if (beforeSource.accumulatedCorn !== sourcePool)
      reject('central corn is not derived from the original notification sequence');
    const claims = entries.filter(
      (n) => n.type === 'placeWorker' && n.args.place === 'startingPlayer',
    );
    if (claims.length) {
      const entry = claims[0],
        owner = sourceId(entry),
        actor = mapping.get(owner);
      const workerBinding = placements.get(entry);
      if (
        claims.length !== 1 ||
        !publicEntry(entry) ||
        claim ||
        !workerBinding ||
        before.firstPlayerClaimed !== null ||
        priorObservation.actor !== actor ||
        native.firstPlayerClaimed !== actor ||
        beforeSource.accumulatedCorn !== before.accumulatedCorn ||
        !Number.isSafeInteger(beforeSource.accumulatedCorn) ||
        beforeSource.accumulatedCorn < 0 ||
        entries.some((n) => ['cornOnWheel', 'firstPlayerCorn', 'rotateWheel'].includes(n.type))
      ) {
        reject('starting-player claim is not bound to its original worker, actor and pool');
      } else {
        const oldStarter = order[0];
        claim = {
          sourceOwner: owner,
          owner: actor,
          worker: String(entry.args.worker_id),
          amount: beforeSource.accumulatedCorn,
          oldStarter,
          nextStarter: actor === oldStarter ? order[1] : actor,
          claim: reference(entry),
          claimMoveId: group.moveId,
          order: clone(order),
          cornOpen: false,
          cornClosed: false,
          markerOpen: false,
          closed: false,
          receipt: null,
          workerBinding,
        };
        obligations.push(claim);
      }
    }
    const rotation = entries.filter((n) => n.type === 'rotateWheel');
    if (!rotation.length && !isDeepStrictEqual(native.turnOrder, order))
      reject('current-round turn order changed before rotation');
    if (claim) {
      const c = claim;
      const receipts = entries.filter((n) => n.type === 'firstPlayerCorn');
      const credits = entries.filter(
        (n) => n.type === 'takeGoods' && sourceId(n) === c.sourceOwner && n.args.type === 'corn',
      );
      if (receipts.length) {
        const receipt = receipts[0],
          credit = credits[0];
        if (
          receipts.length !== 1 ||
          credits.length !== 1 ||
          !publicEntry(receipt) ||
          !publicEntry(credit) ||
          credit.args.origin !== null ||
          credit.packetIndex !== receipt.packetIndex ||
          credit.notificationIndex + 1 !== receipt.notificationIndex ||
          entries.indexOf(credit) + 1 !== entries.indexOf(receipt)
        )
          reject('corn receipt is not one adjacent public unsourced-credit pair');
      }
      if (receipts.length && c.amount === 0) {
        if (
          c.receipt ||
          receipts.length !== 1 ||
          sourceId(receipts[0]) !== c.sourceOwner ||
          Number(receipts[0].args.corn) !== 0 ||
          credits.length > 1 ||
          credits.some((n) => Number(n.args.quantity) !== 0)
        )
          reject('zero-corn settlement is not bound to one original claimant');
        else {
          c.receipt = { caption: reference(receipts[0]) };
          c.cornClosed = true;
        }
      }
      const claimantEnd = entries.some(
        (n) =>
          n.type === 'gameStateChange' &&
          Number(n.args.id) === 90 &&
          String(n.args.active_player) === c.sourceOwner,
      );
      if (
        !c.cornOpen &&
        !c.cornClosed &&
        c.amount > 0 &&
        claimantEnd &&
        native.accumulatedCorn === 0 &&
        (ledger.accumulatedCorn === c.amount || receipts.length)
      ) {
        c.cornOpen = true;
        c.cornStartMoveId = group.moveId;
        c.nativeResources = clone(before.players[c.owner].resources);
        if (before.accumulatedCorn === c.amount) c.nativeResources.corn += c.amount;
      }
      if (c.cornOpen && !c.cornClosed) {
        if (
          !receipts.length &&
          (observation?.actor === c.owner || mapping.get(ledger.activePlayer) === c.owner)
        )
          reject('claimant can act before the delayed corn receipt');
        let followingFood = receipts.length
          ? provePostReceiptFood({
              index,
              group,
              receipt: receipts[0],
              claim: c,
              beforeSource,
              before,
              ledger,
              native,
              transitions: context.transitions,
              sourceToNative: mapping,
              skullShortageEvidence,
            })
          : null;
        if (!followingFood && receipts.length && day14FoodCandidate(context, native)) {
          const input = { context, ledger, native, observation, claim: c };
          const proof = day14Food.prove(input);
          if (proof && day14Food.verify(proof, input)) followingFood = proof;
        }
        if (
          !isDeepStrictEqual(native.players[c.owner].resources, c.nativeResources) &&
          !followingFood
        )
          reject('claimant resources changed during delayed corn receipt');
        const ownership = ledger.workers.get(c.worker);
        if (
          (!ownership ||
            ownership.place !== 'startingPlayer' ||
            String(ownership.player) !== c.sourceOwner) &&
          !receipts.length
        )
          reject('starting-player ownership changed before corn settlement');
        if (
          entries.some(
            (n) =>
              sourceId(n) === c.sourceOwner &&
              n.type === 'payGoods' &&
              !followingFood?.authorizedEffectUids.includes(n.uid),
          )
        )
          reject('claimant paid resources before corn settlement');
        const terminalScoreUids = terminalSettlementScoreUids(terminalScoreSettlement, {
          context,
          ledger,
          native,
          claim: c,
        });
        const unexpected = entries.some(
          (n) =>
            sourceId(n) === c.sourceOwner &&
            [
              'takeGoods',
              'newWorker',
              'placeWorker',
              'increaseTechnology',
              'decreaseTechnology',
              'alterTemple',
              'construct',
              'updateScore',
            ].includes(n.type) &&
            !(receipts.length === 1 && credits.includes(n)) &&
            !(n.type === 'updateScore' && terminalScoreUids.includes(n.uid)) &&
            !followingFood?.authorizedEffectUids.includes(n.uid),
        );
        if (unexpected) reject('claimant received an unrelated effect before corn settlement');
        if (followingFood) evidence.push(followingFood);
        if (terminalScoreUids.length) evidence.push(clone(terminalScoreSettlement));
        if (entries.some((n) => n.type === 'undoRestoreDISAB')) {
          const onlyCaption =
            entries.length === 1 &&
            beforeSource.stateId === 4 &&
            beforeSource.activePlayer !== c.sourceOwner &&
            !before.pending &&
            before.turn.mode === 'none' &&
            before.turn.count === 0 &&
            !before.turn.begged &&
            isDeepStrictEqual(before, native) &&
            isDeepStrictEqual(beforeSource, ledger);
          if (!onlyCaption) reject('cancellation effect on the corn obligation is not established');
          else evidence.push({ type: 'unchangedCancellation', source: reference(entries[0]) });
        }
        if (receipts.length) {
          const receipt = receipts[0],
            credit = credits[0];
          if (
            receipts.length !== 1 ||
            (followingFood ? credits.length < 1 : credits.length !== 1) ||
            sourceId(receipt) !== c.sourceOwner ||
            Number(receipt.args.corn) !== c.amount ||
            Number(credit.args.quantity) !== c.amount ||
            credit.packetIndex !== receipt.packetIndex ||
            credit.notificationIndex >= receipt.notificationIndex ||
            ledger.accumulatedCorn !== 0 ||
            native.accumulatedCorn !== 0 ||
            ledger.players[c.sourceOwner].resources.corn !== native.players[c.owner].resources.corn
          ) {
            reject('delayed corn receipt does not close the original amount and owner');
          } else {
            c.cornClosed = true;
            c.cornClosedMoveId = group.moveId;
            c.receipt = { credit: reference(credit), caption: reference(receipt) };
          }
        } else {
          const field = `player.${c.sourceOwner}.resources.corn`;
          const corn = differences.find((d) => d.field === field);
          const pool = differences.find((d) => d.field === 'accumulatedCorn');
          if (
            !corn ||
            corn.native !== corn.observed + c.amount ||
            !pool ||
            pool.observed !== c.amount ||
            pool.native !== 0
          )
            reject('delayed corn difference is not the exact original entitlement');
          else {
            allowed.add(field);
            allowed.add('accumulatedCorn');
            evidence.push({
              type: 'delayedCorn',
              claim: c.claim,
              amount: c.amount,
              owner: c.sourceOwner,
              closurePending: true,
            });
          }
          const places = ledger.stateArgs?.places;
          if (places && Object.hasOwn(places, 'startingPlayer'))
            reject('occupied starting-player space is offered to another actor');
        }
      } else if (receipts.length && c.cornClosed && c.amount > 0) {
        reject('duplicate accumulated-corn receipt');
      }
      const marker = differences.find((d) => d.field === 'firstPlayer');
      if (marker) {
        if (
          native.pending?.task.type !== 'rotation' ||
          observation.actor !== c.owner ||
          mapping.get(ledger.activePlayer) !== c.owner ||
          ledger.stateId !== 95 ||
          marker.native !== c.oldStarter ||
          marker.observed !== c.nextStarter ||
          native.firstPlayerClaimed !== c.owner ||
          !observation.legalActions.some((row) => row.action.type === 'rotate')
        ) {
          reject('starter-marker timing does not preserve rotation authority');
        } else {
          allowed.add('firstPlayer');
          c.markerOpen = true;
          evidence.push({
            type: 'starterMarker',
            claim: c.claim,
            oldStarter: c.oldStarter,
            nextStarter: c.nextStarter,
            rotationActor: observation.actor,
            closurePending: true,
          });
        }
      }
      if (rotation.length) {
        const selected = applied.filter((row) => row.action?.type === 'rotate');
        const transitions = context.transitions.filter((row) => row.action.type === 'rotate');
        const transition = transitions[0];
        const offset = order.indexOf(c.nextStarter);
        const nextOrder = [...order.slice(offset), ...order.slice(0, offset)];
        if (
          rotation.length !== 1 ||
          selected.length !== 1 ||
          transitions.length !== 1 ||
          selected[0].actor !== c.owner ||
          selected[0].action.days !== 1 + rotation[0].args.twice ||
          transition.before.pending?.task.type !== 'rotation' ||
          transition.actor !== c.owner ||
          !isDeepStrictEqual(transition.action, selected[0].action) ||
          transition.step !== selected[0].step ||
          transition.before.firstPlayerClaimed !== c.owner ||
          !isDeepStrictEqual(transition.before.turnOrder, order) ||
          !isDeepStrictEqual(transition.after, native) ||
          !transition.legalActions.some(
            (row) => row.action.type === 'rotate' && row.action.days === selected[0].action.days,
          ) ||
          native.firstPlayerClaimed !== null ||
          native.firstPlayer !== c.nextStarter ||
          (native.phase !== 'finished' &&
            (!isDeepStrictEqual(native.turnOrder, nextOrder) ||
              native.currentPlayer !== c.nextStarter ||
              native.turnIndex !== 0)) ||
          !c.cornClosed
        ) {
          reject('rotation does not close the original claimant and next-round order');
        } else {
          c.closed = true;
          c.rotationTransition = clone(transition);
          c.closedMoveId = group.moveId;
          evidence.push({
            type: 'roundClosure',
            claim: c.claim,
            rotation: reference(rotation[0]),
            days: selected[0].action.days,
            nextStarter: c.nextStarter,
            nextOrder,
          });
          order = clone(native.turnOrder);
          claim = null;
        }
      }
    } else if (rotation.length) {
      if (
        before.firstPlayerClaimed !== null ||
        native.firstPlayer !== order[0] ||
        (native.phase !== 'finished' && !isDeepStrictEqual(native.turnOrder, order))
      )
        reject('unclaimed rotation changed the starter or round order');
    }
    const building = buildingResources.compare(context, {
      ledger,
      native,
      observation,
      differences,
      constructionTransition,
    });
    if (building.failure) reject(building.failure);
    for (const field of building.fields) allowed.add(field);
    evidence.push(...building.evidence);
    const calendar = calendarComparison.compare(context, { ledger, native });
    if (calendar.failure) reject(calendar.failure);
    for (const field of calendar.fields) allowed.add(field);
    evidence.push(...calendar.evidence);
    const remainingDifferences = differences.filter((d) => !allowed.has(d.field));
    for (const n of entries) {
      if (n.type === 'cornOnWheel' || n.type === 'firstPlayer') {
        if (!publicEntry(n)) reject('central corn update is not a public original notification');
        sourcePool = n.type === 'cornOnWheel' ? sourcePool + 1 : 0;
      }
    }
    if (ledger.accumulatedCorn !== sourcePool)
      reject('central corn does not match the original round receipts');
    const result = {
      sourceRawSha256: sequence?.rawSha256 ?? null,
      rawEqual: differences.length === 0,
      semanticMatch: !failure && remainingDifferences.length === 0,
      remainingDifferences: clone(remainingDifferences),
      evidence,
      unresolvedObligations: obligations
        .filter((c) => !c.closed)
        .map((c) => ({
          claimMoveId: c.claimMoveId,
          corn: c.cornOpen && !c.cornClosed,
          marker: c.markerOpen,
        }))
        .concat(building.unresolved ? [building.unresolved] : []),
      failure,
    };
    if (humanDecision) decisions.push({ moveId: group.moveId, matched: result.semanticMatch });
    return result;
  }
  function finish() {
    return {
      schema: 'tzolkin-bga-semantic-comparison-v1',
      sourceRawSha256: sequence?.rawSha256 ?? null,
      sourceGroupsConsumed: cursor,
      sourceGroupsExpected: remaining.length,
      sourceSequenceComplete: !!sequence && cursor === remaining.length,
      obligations: clone(obligations),
      buildingResources: buildingResources.finish(),
      terminalCalendar: calendarComparison.finish(),
      decisionBoundaries: clone(decisions),
      failure,
      allObligationsClosed:
        obligations.every((c) => c.closed) &&
        buildingResources.finish().allObligationsClosed &&
        calendarComparison.finish().allObligationsClosed,
      allDecisionBoundariesMatched: decisions.every((d) => d.matched),
      teacherCertified: false,
    };
  }
  return {
    begin,
    applied,
    compare,
    finish,
    terminalCalendarEvidence: calendarComparison.terminalClosure,
  };
}

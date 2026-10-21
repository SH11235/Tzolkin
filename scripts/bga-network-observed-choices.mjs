import { isDeepStrictEqual as same } from 'node:util';
import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';
import {
  technologyChoiceProof,
  begChoiceProof,
  forcedRotationProof,
  closedBurnChoiceProof,
  fixedBuildingResourceChoiceProof,
} from './bga-network-compound-choices.mjs';

const clone = structuredClone;
const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];
const materials = ['wood', 'stone', 'gold'];
const technologies = ['agriculture', 'extraction', 'architecture', 'theology'];
const cities = {
  Palenque: 'palenque',
  Yaxchilan: 'yaxchilan',
  Tikal: 'tikal',
  Uxmal: 'uxmal',
  'Chichen Itza': 'chichenItza',
};
const god = (x) => (x === 'kukulcan' ? 'kukulkan' : x);
const integer = (x) =>
  ['string', 'number'].includes(typeof x) &&
  /^(0|[1-9]\d*)$/.test(String(x)) &&
  Number.isSafeInteger(Number(x));
const empty = (x) =>
  x === null || (typeof x === 'object' && x !== null && Object.keys(x).length === 0);
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
export const choicePublicState = (state) =>
  Object.fromEntries(
    Object.entries(state)
      .filter(([k]) => !['pending', 'log'].includes(k))
      .map(([k, v]) => [k, clone(v)]),
  );
const constructionSourceFields = [
  'players',
  'workers',
  'buildings',
  'monuments',
  'jungle',
  'skullSupply',
  'skullSpaces',
  'day',
  'age',
  'firstPlayer',
  'accumulatedCorn',
  'foodDays',
  'activePlayer',
];
const constructionSourceState = (ledger) => {
  if (
    !ledger ||
    !(ledger.workers instanceof Map) ||
    !constructionSourceFields.every((k) => Object.hasOwn(ledger, k) && ledger[k] !== undefined)
  )
    return null;
  return Object.fromEntries(
    constructionSourceFields.map((k) => [
      k,
      k === 'workers' ? [...ledger.workers] : clone(ledger[k]),
    ]),
  );
};
const payments = (group, owner) =>
  resources.map((r) =>
    group.entries
      .filter(
        (n) => n.type === 'payGoods' && String(n.args.player_id) === owner && n.args.type === r,
      )
      .reduce((s, n) => (integer(n.args.quantity) ? s + Number(n.args.quantity) : NaN), 0),
  );
const sourceOwner = (n, index, fallback) =>
  n?.args?.player_id !== undefined
    ? String(n.args.player_id)
    : n?.args?.player_name !== undefined
      ? String(Object.values(index.initial.players).find((p) => p.name === n.args.player_name)?.id)
      : fallback;

const stages = {
  playerTurn: [null, ['place', 'firstPlayer', 'remove', 'beg']],
  placeWorker: [null, ['place', 'firstPlayer', 'endTurn']],
  pickUpWorker: [null, ['remove', 'endTurn']],
  harvest: ['palenque', ['harvest', 'skip']],
  technologyUpgrade: ['technology', ['technology', 'skip']],
  construct: ['build', ['build', 'skip']],
  constructChooseSparedResources: ['build', ['build']],
  tikal4: ['build', ['build', 'monument', 'skip']],
  tikal5: ['payResource', ['payment', 'skip']],
  trade: ['trade', ['trade', 'skip']],
  uxmal4: ['build', ['build', 'skip']],
  anyAction: ['anyAction', ['useAction', 'skip']],
  resourceReward: ['resource', ['resource']],
  templeReward: ['temple', ['temple']],
  freeTechnology: ['technology', ['technology']],
  constructWithoutArchitecture: ['build', ['build', 'skip']],
  theology2: ['theology', ['offering', 'skip']],
  turnWheel: ['rotation', ['rotate']],
};
const constructNames = new Set([
  'construct',
  'constructChooseSparedResources',
  'tikal4',
  'uxmal4',
  'constructWithoutArchitecture',
]);

export function observedStageArguments(boundary, operation, operations = [operation]) {
  const { name, args, actor } = boundary,
    before = operation.before,
    action = operation.row.action;
  const p = before.players[actor],
    task = before.pending?.task;
  if (!p || !stages[name]) return 'unknownStage';
  if ((task?.type ?? null) !== stages[name][0]) return 'sourceAndNativeStageDiffer';
  if (!stages[name][1].includes(action.type)) return 'chosenOperationNotInSourceStage';
  if (
    ['construct', 'uxmal4', 'constructWithoutArchitecture', 'resourceReward', 'turnWheel'].includes(
      name,
    ) &&
    !empty(args)
  )
    return 'unexpectedStageArguments';
  if (name === 'freeTechnology' || name === 'technologyUpgrade') {
    const tracks = name === 'freeTechnology' ? args : args?.technologies;
    if (
      !tracks ||
      typeof tracks !== 'object' ||
      Array.isArray(tracks) ||
      Object.entries(tracks).some(
        ([t, value]) =>
          !technologies.includes(t) || !integer(value) || Number(value) !== p.technologies[t],
      ) ||
      (action.type === 'technology' && !Object.hasOwn(tracks, action.technology))
    )
      return 'technologyArgumentsDiffer';
    if (task.free !== (name === 'freeTechnology')) return 'technologyPaymentModeDiffer';
    if (
      name === 'technologyUpgrade' &&
      materials.some(
        (r) => !integer(args?.resources?.[r]) || Number(args.resources[r]) !== p.resources[r],
      )
    )
      return 'technologyBudgetArgumentsDiffer';
  }
  if (name === 'templeReward' || name === 'tikal5') {
    const temples = name === 'templeReward' ? args : args?.temples;
    const selected = operations
      .filter((o) => o.row.action.type === 'temple')
      .map((o) => (o.row.action.temple === 'kukulkan' ? 'kukulcan' : o.row.action.temple));
    if (
      !temples ||
      !Object.keys(temples).length ||
      Object.entries(temples).some(
        ([t, value]) =>
          !['chaac', 'quetzalcoatl', 'kukulcan'].includes(t) ||
          !integer(value) ||
          Number(value) - 1 !== p.temples[god(t)],
      ) ||
      selected.some((t) => !Object.hasOwn(temples, t))
    )
      return 'templeArgumentsDiffer';
    if (
      name === 'tikal5' &&
      materials.some(
        (r) => !integer(args?.resources?.[r]) || Number(args.resources[r]) !== p.resources[r],
      )
    )
      return 'templePaymentBudgetDiffer';
  }
  if (
    name === 'theology2' &&
    materials.some((r) => !integer(args?.[r]) || Number(args[r]) !== p.resources[r])
  )
    return 'offeringBudgetArgumentsDiffer';
  if (
    name === 'trade' &&
    (!integer(args?.maxvalue) || Number(args.maxvalue) < 1 || Number(args.maxvalue) > 100)
  )
    return 'tradeLimitMissing';
  if (
    name === 'trade' &&
    action.type === 'trade' &&
    Number(action.buy ? operation.entry.args.quantity2 : operation.entry.args.quantity1) >
      Number(args.maxvalue)
  )
    return 'tradeQuantityExceedsSourceLimit';
  if (
    name === 'tikal4' &&
    (typeof args?.architecture !== 'boolean' ||
      args.architecture !==
        (task.architectureAvailable !== false && p.technologies.architecture > 0))
  )
    return 'architectureAvailabilityDiffer';
  if (name === 'constructWithoutArchitecture' && task.architectureAvailable !== false)
    return 'architectureDisabledStageDiffer';
  if (name === 'uxmal4' && task.cornPayment !== true) return 'cornConstructionStageDiffer';
  if (name === 'constructChooseSparedResources') {
    const candidates = operation.equivalentRows?.length
      ? operation.equivalentRows
      : [operation.row];
    if (
      !materials.every((r) => typeof args?.[r] === 'boolean') ||
      !candidates.some((r) => r.action.type === 'build' && args[r.action.discountResource] === true)
    )
      return 'discountResourceNotOffered';
  }
  if (
    ['playerTurn', 'placeWorker'].includes(name) &&
    ['place', 'firstPlayer'].includes(action.type)
  ) {
    const place = action.type === 'firstPlayer' ? 'startingPlayer' : action.gear;
    if (
      !integer(args?.places?.[place]) ||
      !integer(args?.corn) ||
      Number(args.places[place]) + Number(args.corn) !== action.cornCost
    )
      return 'placementQuoteArgumentsDiffer';
  }
  if (
    name === 'anyAction' &&
    action.type === 'useAction' &&
    (!integer(args?.[action.gear]?.[action.position]) ||
      Number(args[action.gear][action.position]) !== action.cornCost)
  )
    return 'anyActionQuoteDiffer';
  if (name === 'harvest' && action.type === 'harvest') {
    if (['corn', 'emptyCorn'].includes(action.kind)) {
      const forest = before.jungle?.[task.position],
        agriculture = p.technologies.agriculture >= 2;
      const canCorn =
        forest && Number.isSafeInteger(forest.corn) && Number.isSafeInteger(forest.wood)
          ? forest.corn > forest.wood
          : null;
      if (
        typeof args?.corn_tile_available !== 'boolean' ||
        typeof args?.agriculture_level_2 !== 'boolean' ||
        canCorn === null ||
        args.corn_tile_available !== canCorn ||
        args.agriculture_level_2 !== agriculture ||
        !(args.corn_tile_available || args.agriculture_level_2)
      )
        return 'sourceCornHarvestPermissionNotBound';
      if (action.kind === 'corn' ? !canCorn : canCorn || !agriculture)
        return 'nativeCornHarvestKindDiffer';
    }
    if (action.kind === 'burn' && args?.can_burn !== true) return 'burnNotOffered';
    const material = action.kind === 'wood' ? 'wood' : 'corn';
    if (
      !integer(args?.[material]) ||
      operation.after.players[actor].resources[material] - p.resources[material] !==
        Number(args[material])
    )
      return 'harvestYieldArgumentsDiffer';
  }
  return null;
}

export function observedOperationBinding({
  operation: op,
  group,
  index,
  sourceToNative,
  mapCard,
  audit,
  operations = [op],
  semanticEvidence = [],
}) {
  const n = op.entry,
    a = n?.args,
    x = op.row?.action,
    p = op.before.players[op.actor],
    q = op.after.players[op.actor];
  const owner = sourceOwner(n, index, op.sourceActor);
  const context = {
    operation: op,
    operations,
    group,
    index,
    sourceToNative,
    audit,
    semanticEvidence,
  };
  const forced = forcedRotationProof(context);
  const burn = closedBurnChoiceProof(context);
  if (
    !n ||
    !a ||
    !x ||
    !p ||
    !q ||
    (sourceToNative.get(owner) !== op.actor && !forced && !burn) ||
    op.actor !== op.observationActor ||
    !op.legalRows.some((r) => same(r, op.row))
  )
    return 'chosenActorOrLegalRowNotBound';
  if (op.equivalentRows?.some((r) => !op.legalRows.some((legal) => same(legal, r))))
    return 'equivalentRowNotLegal';
  if (op.equivalentRows?.length) {
    const materialIndependent = (a) =>
      Object.fromEntries(Object.entries(a).filter(([key]) => key !== 'discountResource'));
    if (
      x.type !== 'build' ||
      op.before.pending?.task.cornPayment !== true ||
      op.equivalentRows.some((r) => !same(materialIndependent(r.action), materialIndependent(x)))
    )
      return 'equivalentConstructionMeaningDiffer';
  }
  switch (x.type) {
    case 'place':
    case 'firstPlayer':
      return n.type === 'placeWorker' &&
        a.place === (x.type === 'firstPlayer' ? 'startingPlayer' : x.gear) &&
        integer(a.corn) &&
        Number(a.corn) === x.cornCost
        ? null
        : 'placementTargetOrQuoteDiffer';
    case 'remove':
      return n.type === 'pickUpWorker' &&
        cities[a.city] === x.gear &&
        integer(a.position) &&
        Number(a.position) === x.position
        ? null
        : 'removedWorkerTargetDiffer';
    case 'useAction': {
      const city = cities[a.city];
      if (!['pickUpWorker', 'selectAction'].includes(n.type) || city !== x.gear)
        return 'selectedGearDiffer';
      if (Number(a.action_number) === x.position) return null;
      return n.type === 'pickUpWorker' &&
        x.gear === 'chichenItza' &&
        x.position === 10 &&
        x.source === 'theologyAhead' &&
        x.cornCost === 0 &&
        same(choicePublicState(op.before), choicePublicState(op.after))
        ? null
        : 'selectedActionPositionDiffer';
    }
    case 'technology':
      return technologyChoiceProof({ ...context, owner }).error;
    case 'payment':
      return ['increaseTechnology', 'payGoods'].includes(n.type) &&
        ['payResource', 'payTechnology'].includes(op.before.pending?.task.type) &&
        group.entries
          .filter((e) => e.type === 'payGoods')
          .every((e) => String(e.args.player_id) === owner) &&
        same(x.resources, payments(group, owner))
        ? null
        : 'chosenPaymentAmountsDiffer';
    case 'offering':
      return n.type === 'payGoods' &&
        a.target === 'technology_theology_level_2' &&
        a.type === x.resource &&
        Number(a.quantity) === 1 &&
        p.resources[x.resource] - q.resources[x.resource] === 1
        ? null
        : 'offeringTargetOrAmountDiffer';
    case 'temple': {
      if (burn) return null;
      const direct =
        n.type === 'alterTemple'
          ? n
          : group.entries.find(
              (e) =>
                e.type === 'alterTemple' &&
                String(e.args.player_id) === owner &&
                god(e.args.temple_id) === x.temple,
            );
      return direct &&
        god(direct.args.temple_id) === x.temple &&
        Number(direct.args.step) - 1 === q.temples[x.temple] &&
        q.temples[x.temple] - p.temples[x.temple] === x.direction
        ? null
        : 'templeTargetOrDirectionDiffer';
    }
    case 'resource':
      return n.type === 'takeGoods' &&
        a.type === x.resource &&
        integer(a.quantity) &&
        Number(a.quantity) > 0
        ? null
        : 'resourceTargetOrQuantityMissing';
    case 'trade': {
      if (n.type !== 'trade') return 'tradeNotificationMissing';
      const from = /resource_(corn|wood|stone|gold)\b/.exec(a.resource1 ?? '')?.[1],
        to = /resource_(corn|wood|stone|gold)\b/.exec(a.resource2 ?? '')?.[1],
        buy = from === 'corn';
      const count = Number(buy ? a.quantity2 : a.quantity1),
        corn = Number(buy ? a.quantity1 : a.quantity2);
      return integer(count) &&
        count > 0 &&
        integer(corn) &&
        corn > 0 &&
        buy === x.buy &&
        (buy ? to : from) === x.resource &&
        (buy ? from : to) === 'corn' &&
        q.resources[x.resource] - p.resources[x.resource] === (buy ? 1 : -1) &&
        q.resources.corn - p.resources.corn === ((buy ? -1 : 1) * corn) / count
        ? null
        : 'tradeTargetDirectionOrPriceDiffer';
    }
    case 'build':
    case 'monument': {
      if (
        n.type !== 'construct' ||
        mapCard(a.building) !== x.id ||
        !same(x.cost, payments(group, owner))
      )
        return 'constructionTargetOrPaymentDiffer';
      if (x.type === 'build') {
        const corn = group.entries.some(
          (e) =>
            e.type === 'takeGoods' &&
            String(e.args.player_id) === owner &&
            e.args.type === 'corn' &&
            e.args.origin === 'technology_architecture_level_1',
        );
        if (x.architecture !== corn) return 'architectureBonusChoiceDiffer';
      }
      return null;
    }
    case 'beg':
      return begChoiceProof({ ...context, owner }) ? null : 'beggingAmountDiffer';
    case 'endTurn':
      return n.type === 'gameStateChange' && Number(a.id) === 90
        ? null
        : 'turnEndNotificationMissing';
    case 'rotate':
      return n.type === 'rotateWheel' && [0, 1].includes(a.twice) && x.days === 1 + a.twice
        ? null
        : 'rotationDistanceDiffer';
    case 'harvest': {
      if (x.kind === 'burn') return burn ? null : 'closedBurnWitnessMissing';
      if (
        n.type === 'takeTile' &&
        Number(a.action) === x.position &&
        (a.type === 'wood_tile' ? 'wood' : 'corn') === x.kind
      )
        return null;
      return n.type === 'takeGoods' &&
        a.type === 'corn' &&
        /^action_palenque_\d+$/.test(a.origin ?? '') &&
        x.kind === 'emptyCorn'
        ? null
        : 'harvestTargetDiffer';
    }
    case 'skip': {
      if (n.type === 'construct' && op.before.pending?.task.type === 'trade') return null;
      if (audit.some((row) => row.step === op.step && row.declinedRemovalEvidence)) return null;
      return n.type === 'gameStateChange' && [7, 70, 90].includes(Number(a.id))
        ? null
        : 'sourcePassEvidenceMissing';
    }
    default:
      return 'unsupportedObservedOperation';
  }
}

export function createObservedChoices({
  index,
  initialMoveId,
  sourceToNative,
  mapCard,
  initialBoundary,
}) {
  const sequence = originalNetworkSequence(index),
    originals = new Map(sequence?.groups.map((x) => [x.group, x.original]) ?? []),
    proofs = new Map(),
    operationEvidence = [];
  const capturedOperations = [],
    deferredTechnologies = [];
  const expected = [initialMoveId];
  let last = clone(initialBoundary),
    interval = null;
  const fail = (b, reason, extra = {}) => {
    if (proofs.get(b.moveId)?.verified === false) return;
    proofs.set(b.moveId, {
      moveId: b.moveId,
      stateId: b.stateId,
      name: b.name,
      verified: false,
      reason,
      ...extra,
    });
  };
  const pass = (b, kind, group, ops, extra = {}) => {
    if (proofs.get(b.moveId)?.verified === false) return;
    proofs.set(b.moveId, {
      moveId: b.moveId,
      stateId: b.stateId,
      name: b.name,
      verified: true,
      kind,
      responseMoveId: group.moveId,
      source: group.entries.map(ref),
      nativeSteps: ops.map((o) => o.step),
      ...extra,
    });
  };
  function trusted(group, entry) {
    const p = originalPacketEntry(entry);
    return (
      originals.has(group) &&
      same(originals.get(group), group) &&
      group.entries.includes(entry) &&
      p?.rawSha256 === sequence?.rawSha256 &&
      String(p.tableId) === String(index.tableId) &&
      p.packetType === 'resend' &&
      entry.channel === `/table/t${index.tableId}`
    );
  }
  function sourceExit(group) {
    const e = group.entries,
      s = index.initial.gamestates;
    return (
      last.name === 'trade' &&
      last.pending?.task.type === 'trade' &&
      same(last.pending.task, { type: 'trade' }) &&
      same(last.pending.after, [
        {
          type: 'build',
          remaining: 1,
          allowMonument: false,
          cornPayment: false,
          architectureAvailable: true,
        },
      ]) &&
      same(
        e.map((n) => n.type),
        ['gameStateChange', 'gameStateChange'],
      ) &&
      Number(e[0].args.id) === 9 &&
      Number(e[1].args.id) === 32 &&
      e.every((n) => trusted(group, n) && String(n.args.active_player) === last.sourceActor) &&
      s[9]?.type === 'game' &&
      s[32]?.name === 'construct' &&
      s[42]?.transitions?.nextAction === 9 &&
      s[42]?.possibleactions?.includes('pass')
    );
  }
  function emptyPaidTechnologyExit(group, before, after, operations, sourceBefore, sourceAfter) {
    const args = last.args,
      op = operations[0],
      s = index.initial.gamestates;
    if (
      last.stateId !== 31 ||
      last.name !== 'technologyUpgrade' ||
      !args ||
      !Object.hasOwn(args, 'technologies') ||
      !Array.isArray(args.technologies) ||
      args.technologies.length ||
      !same(Object.keys(args).sort(), ['resources', 'technologies']) ||
      !args.resources ||
      !same(Object.keys(args.resources).sort(), ['gold', 'stone', 'wood']) ||
      operations.length !== 1 ||
      !same(op?.row, { action: { type: 'skip' }, move: { type: 'choose', choiceId: 'skip' } }) ||
      op.actor !== last.actor ||
      op.sourceActor !== last.sourceActor ||
      sourceToNative.get(last.sourceActor) !== last.actor ||
      before.currentPlayer !== last.actor ||
      before.additionalBuildings !== false ||
      before.turn?.mode !== 'remove' ||
      !same(before.pending?.task, { type: 'technology', remaining: 1, free: false }) ||
      !same(before.pending.after, []) ||
      !same(last.pending, before.pending) ||
      !same(op.before, before) ||
      !same(op.after, after) ||
      !same(op.legalRows, [op.row]) ||
      !same(last.legalRows, op.legalRows) ||
      materials.some(
        (r) =>
          !integer(args.resources[r]) ||
          Number(args.resources[r]) !== 0 ||
          before.players[last.actor].resources[r] !== 0,
      )
    )
      return null;
    const expected = clone(before);
    expected.pending = null;
    if (!same(after, expected)) return null;
    const menuGroup = index.groups.find((g) => g.moveId === last.moveId),
      menu = menuGroup?.entries.findLast((n) => n.type === 'gameStateChange');
    if (
      !menu ||
      !trusted(menuGroup, menu) ||
      menu.args.id !== 31 ||
      menu.args.type !== 'activeplayer' ||
      menu.args.active_player !== last.sourceActor ||
      !Object.hasOwn(menu.args, 'args') ||
      !same(menu.args.args, args) ||
      group.moveId !== last.moveId + 1 ||
      sourceBefore?.initial !== index.initial ||
      sourceBefore.stateId !== 31 ||
      sourceBefore.activePlayer !== last.sourceActor ||
      !same(sourceBefore.stateArgs, args) ||
      sourceAfter?.initial !== index.initial ||
      sourceAfter.stateId !== 7 ||
      sourceAfter.activePlayer !== last.sourceActor ||
      s[31]?.name !== 'technologyUpgrade' ||
      s[31].type !== 'activeplayer' ||
      s[31].args !== 'argTechnologyUpgrade' ||
      !same(s[31].possibleactions, ['chooseTechnology', 'pass']) ||
      s[31].transitions?.nextAction !== 9 ||
      s[9]?.name !== 'solveStackedAction' ||
      s[9].type !== 'game' ||
      s[9].action !== 'stSolveStackedAction' ||
      s[9].transitions?.stackEmpty !== 7 ||
      s[7]?.name !== 'pickUpWorker' ||
      s[7].type !== 'activeplayer'
    )
      return null;
    const entries = group.entries;
    if (
      !same(
        entries.map((n) => n.type),
        ['gameStateChange', 'gameStateChange'],
      ) ||
      !same(
        entries.map((n) => n.args?.id),
        [9, 7],
      ) ||
      op.entry !== entries[1] ||
      entries.some((n, i) => {
        const p = originalPacketEntry(n);
        return (
          !trusted(group, n) ||
          p.packetNotificationCount !== 2 ||
          n.packetIndex !== entries[0].packetIndex ||
          n.notificationIndex !== i ||
          !n.visible ||
          n.administrative ||
          n.beforeCheckpoint ||
          n.args.active_player !== last.sourceActor ||
          n.args.type !== (i === 0 ? 'game' : 'activeplayer') ||
          !Object.hasOwn(n.args, 'args') ||
          Object.keys(n.args).some(
            (k) =>
              ![
                'id',
                'active_player',
                'args',
                'type',
                'reflexion',
                'updateGameProgression',
              ].includes(k),
          )
        );
      }) ||
      !(entries[0].args.args === null || same(entries[0].args.args, [])) ||
      !entries[1].args.args ||
      !same(Object.keys(entries[1].args.args), ['workers']) ||
      !same(sourceAfter.stateArgs, entries[1].args.args) ||
      !entries[1].args.args.workers ||
      typeof entries[1].args.args.workers !== 'object' ||
      Array.isArray(entries[1].args.args.workers) ||
      Object.values(entries[1].args.args.workers).some(
        (w) =>
          !integer(w.position) ||
          after.gears[w.place]?.[Number(w.position)]?.playerId !== last.actor,
      )
    )
      return null;
    return {
      sourceMenu: ref(menu),
      sourceExit: entries.map(ref),
      nativeStep: op.step,
      actor: op.actor,
      sourceActor: last.sourceActor,
      budget: clone(args.resources),
      structuralPassExit: true,
      humanClickObserved: false,
      publicStateUnchanged: true,
      nextPending: null,
    };
  }
  function observe({
    group,
    before,
    after,
    operations,
    boundary,
    audit,
    publicResultMatched,
    semanticEvidence = [],
    buildingResourceClosures = [],
    sourceBefore = null,
    sourceAfter = null,
  }) {
    const nextHuman = boundary.potentialHumanDecision;
    const next = {
      moveId: group.moveId,
      stateId: boundary.sourceStateId,
      name: index.initial.gamestates[boundary.sourceStateId]?.name,
      args: clone(boundary.sourceMenu),
      sourceActor: String(boundary.sourceActor),
      actor: boundary.nativeActor,
      pending: clone(boundary.nativePending),
      legalRows: clone(boundary.nativeLegalActions),
    };
    const sourceBound = originals.has(group) && same(originals.get(group), group);
    const unchanged = same(before, after);
    const actorSame =
      operations.every(
        (o) =>
          o.actor === last.actor ||
          forcedRotationProof({
            operation: o,
            operations,
            group,
            index,
            sourceToNative,
            semanticEvidence,
          }),
      ) &&
      (!nextHuman ||
        next.actor === last.actor ||
        operations.some((o) => ['endTurn', 'rotate'].includes(o.row.action.type)));
    let error = !sourceBound
      ? 'originalChoiceGroupNotBound'
      : !publicResultMatched
        ? 'publicResponseNotMatched'
        : !actorSame
          ? 'responseActorDiffer'
          : null;
    for (const op of operations) {
      const binding = trusted(group, op.entry)
        ? observedOperationBinding({
            operation: op,
            group,
            index,
            sourceToNative,
            mapCard,
            audit,
            operations,
            semanticEvidence,
          })
        : 'chosenNotificationNotOriginal';
      error ??= binding;
      capturedOperations.push({
        operation: op,
        group,
        boundary: clone(last),
        bindingError: binding,
        buildingResourceClosures: clone(buildingResourceClosures),
      });
      operationEvidence.push({
        step: op.step,
        moveId: group.moveId,
        actor: op.actor,
        source: op.entry ? ref(op.entry) : null,
        bindingError: binding,
        action: clone(op.row.action),
        legalRow: clone(op.row),
        legalBeforeRows: clone(op.legalRows),
        before: clone(op.before),
        after: clone(op.after),
      });
    }
    for (const op of operations.filter((o) => ['resource', 'trade'].includes(o.row.action.type))) {
      const sameEntry = operations.filter(
        (o) => o.entry === op.entry && o.row.action.type === op.row.action.type,
      );
      const a = op.entry.args,
        count =
          op.row.action.type === 'resource'
            ? Number(a.quantity)
            : Number(op.row.action.buy ? a.quantity2 : a.quantity1);
      if (sameEntry.length !== count) error ??= 'chosenQuantityDoesNotMatchOriginal';
    }
    if (error) {
      for (const b of [...(interval?.boundaries ?? []), last]) fail(b, error);
      interval = null;
    } else if (!operations.length) {
      const onlyStates = group.entries.every(
        (n) => n.type === 'gameStateChange' && trusted(group, n),
      );
      if (
        group.entries.length === 1 &&
        group.entries[0].type === 'undoRestoreDISAB' &&
        trusted(group, group.entries[0]) &&
        sourceOwner(group.entries[0], index, last.sourceActor) === last.sourceActor &&
        last.name === 'playerTurn' &&
        unchanged &&
        before.pending === null &&
        before.turn.mode === 'none' &&
        before.turn.count === 0 &&
        !before.turn.begged &&
        audit.some((a) => a.moveId === group.moveId && a.cancellation === true)
      ) {
        pass(last, 'untouchedCaptionCancellation', group, []);
      } else if (
        onlyStates &&
        unchanged &&
        nextHuman &&
        next.actor === last.actor &&
        (sourceExit(group) || (constructNames.has(last.name) && constructNames.has(next.name)))
      ) {
        if (last.name === 'constructChooseSparedResources' && next.name === 'construct') {
          const start = interval?.boundaries?.[0],
            definition = index.initial.gamestates;
          const opening = interval?.groups?.[0];
          const prior = interval?.sourceOpening;
          const currentBefore = constructionSourceState(sourceBefore),
            currentAfter = constructionSourceState(sourceAfter);
          const path = [opening, group];
          const valid =
            interval &&
            interval.boundaries.length === 1 &&
            interval.exit === null &&
            start?.name === 'construct' &&
            start.stateId === 32 &&
            last.stateId === 33 &&
            next.stateId === 32 &&
            definition[32]?.type === 'activeplayer' &&
            definition[32]?.name === 'construct' &&
            definition[32]?.transitions?.sparedResource === 33 &&
            definition[32]?.possibleactions?.includes('construct') &&
            definition[32]?.action == null &&
            definition[33]?.type === 'activeplayer' &&
            definition[33]?.name === 'constructChooseSparedResources' &&
            definition[33]?.transitions?.cancel === 32 &&
            definition[33]?.possibleactions?.includes('sparedResource') &&
            definition[33]?.action == null &&
            path.every(
              (g, i) =>
                g &&
                g.entries.length === 1 &&
                trusted(g, g.entries[0]) &&
                originalPacketEntry(g.entries[0])?.packetNotificationCount === 1 &&
                g.entries[0].type === 'gameStateChange' &&
                Number(g.entries[0].args.id) === (i === 0 ? 33 : 32) &&
                g.entries[0].args.type === 'activeplayer' &&
                String(g.entries[0].args.active_player) === start.sourceActor,
            ) &&
            sourceToNative.get(start.sourceActor) === start.actor &&
            last.actor === start.actor &&
            next.actor === start.actor &&
            prior &&
            currentBefore &&
            currentAfter &&
            same(prior, currentBefore) &&
            same(prior, currentAfter) &&
            prior.activePlayer === start.sourceActor &&
            same(start.args, next.args) &&
            empty(start.args) &&
            same(Object.keys(last.args ?? {}).sort(), materials.toSorted()) &&
            materials.every((r) => typeof last.args[r] === 'boolean') &&
            materials.some((r) => last.args[r]) &&
            same(interval.native, before) &&
            same(before, after) &&
            before.pending?.task.type === 'build' &&
            before.pending.task.architectureAvailable !== false &&
            before.players[start.actor].technologies.architecture >= 3 &&
            same(start.pending, before.pending) &&
            same(next.pending, before.pending) &&
            same(start.legalRows, last.legalRows) &&
            same(start.legalRows, next.legalRows) &&
            start.legalRows.length > 0 &&
            start.legalRows.every((r) => ['build', 'monument', 'skip'].includes(r.action.type));
          for (const b of [...(interval?.boundaries ?? []), last]) {
            if (valid)
              pass(b, 'uncommittedConstructionDraftCancelled', group, [], {
                intervalSource: [...interval.source, ...group.entries.map(ref)],
                sourcePublicBefore: clone(prior),
                sourcePublicAfter: clone(currentAfter),
                unknownDraftBuildingId: null,
                futureBuildingWasNotAssigned: true,
                nativeExecutionUnchanged: true,
              });
            else fail(b, 'constructionDraftReturnNotEstablished');
          }
          interval = null;
        } else {
          interval ??= {
            boundaries: [],
            native: clone(before),
            exit: null,
            source: [],
            groups: [],
            sourceOpening: constructionSourceState(sourceBefore),
          };
          interval.boundaries.push(last);
          interval.groups.push(group);
          interval.source.push(...group.entries.map(ref));
          if (sourceExit(group))
            interval.exit = {
              moveId: group.moveId,
              source: group.entries.map(ref),
              actor: last.actor,
            };
        }
      } else {
        for (const b of [...(interval?.boundaries ?? []), last])
          fail(b, 'sourceChoiceWithoutBoundNativeOperation');
        interval = null;
      }
    } else {
      const build = operations.find((o) => ['build', 'monument'].includes(o.row.action.type));
      if (interval) {
        const boundaries = [...interval.boundaries, last],
          skip = operations.find(
            (o) => o.row.action.type === 'skip' && o.before.pending?.task.type === 'trade',
          );
        let issue = !build ? 'constructionIntervalDidNotCommit' : null;
        if (skip) {
          if (
            !interval.exit ||
            skip.actor !== interval.exit.actor ||
            !same(choicePublicState(skip.before), choicePublicState(skip.after)) ||
            !same(skip.before.pending.after?.[0], skip.after.pending?.task) ||
            !same(skip.after.pending?.after, skip.before.pending.after.slice(1)) ||
            skip.after.pending.task.type !== 'build' ||
            !same(skip.before, interval.native)
          )
            issue ??= 'compoundTradeExitNotBound';
        } else if (interval.exit) issue ??= 'observedTradeExitHasNoActualSkip';
        const reachable = skip?.after ?? interval.native;
        if (build && !same(reachable, build.before))
          issue ??= 'constructionIntervalChangedBeforeCommit';
        for (const b of boundaries) {
          if (b.name === 'trade') {
            if (skip) issue ??= observedStageArguments(b, skip);
            continue;
          }
          const virtual = { ...build, before: reachable };
          if (build && observedStageArguments(b, virtual))
            issue ??= observedStageArguments(b, virtual);
          const offered = skip ? skip.afterLegalRows : b.legalRows;
          if (build && !offered.some((r) => same(r, build.row)))
            issue ??= 'committedBuildWasNotAvailable';
        }
        for (const b of boundaries)
          issue
            ? fail(b, issue)
            : pass(b, 'constructionSelectionInterval', group, operations, {
                intervalSource: clone(interval.source),
                exit: clone(interval.exit),
                selectedBuildingWasNotAddedToEarlierObservation: true,
              });
        interval = null;
      } else {
        const emptyTechnologyExit = emptyPaidTechnologyExit(
          group,
          before,
          after,
          operations,
          sourceBefore,
          sourceAfter,
        );
        let issue = emptyTechnologyExit
          ? null
          : observedStageArguments(last, operations[0], operations);
        if (last.name === 'playerTurn' || last.name === 'pickUpWorker') {
          const removal = operations.find((o) => o.row.action.type === 'remove');
          if (removal) {
            const worker = last.args?.workers?.[String(removal.entry.args.worker_id)];
            if (
              !worker ||
              worker.place !== removal.row.action.gear ||
              Number(worker.position) !== removal.row.action.position
            )
              issue ??= 'sourceWorkerMenuTargetMissing';
            const choice = operations.find(
              (o) =>
                o.row.action.type === 'useAction' &&
                o.row.action.position === Number(removal.entry.args.action_number),
            );
            if (
              choice &&
              (!integer(worker?.actions?.[choice.row.action.gear]?.[choice.row.action.position]) ||
                Number(worker.actions[choice.row.action.gear][choice.row.action.position]) !==
                  choice.row.action.cornCost)
            )
              issue ??= 'sourceWorkerActionQuoteDiffer';
          }
        }
        if (
          operations[0].row.action.type === 'skip' &&
          last.name !== 'trade' &&
          !index.initial.gamestates[last.stateId]?.possibleactions?.some((a) =>
            ['pass', 'endTurn'].includes(a),
          ) &&
          !audit.some((a) => a.declinedRemovalEvidence)
        )
          issue ??= 'sourceStageDoesNotOfferPass';
        const tech = operations.find((o) => o.row.action.type === 'technology');
        const proof = tech
          ? technologyChoiceProof({
              operation: tech,
              operations,
              group,
              index,
              owner: sourceOwner(tech.entry, index, tech.sourceActor),
            })
          : null;
        if (!issue && proof?.reward) {
          deferredTechnologies.push({
            boundary: clone(last),
            group,
            operations: [...operations],
            proof,
          });
        } else
          issue
            ? fail(last, issue)
            : pass(
                last,
                emptyTechnologyExit ? 'emptyPaidTechnologyExit' : 'observedLegalSelection',
                group,
                operations,
                emptyTechnologyExit ? { emptyPaidTechnologyExit: emptyTechnologyExit } : {},
              );
      }
    }
    if (nextHuman) {
      last = next;
      expected.push(next.moveId);
    }
  }
  function finish() {
    const consumedBuildingRewards = new Set();
    for (const deferred of deferredTechnologies) {
      const { proof, boundary, group } = deferred;
      let previous = proof.commit,
        issue = null;
      const fixedContributions = [];
      const following = capturedOperations
        .filter((c) => c.operation.step > proof.commit.step)
        .slice(0, proof.reward.remaining);
      if (following.length !== proof.reward.remaining) issue = 'technologyCapRewardNotClosed';
      for (let i = 0; i < following.length; i++) {
        const {
          operation: op,
          group: response,
          boundary: choice,
          bindingError,
          buildingResourceClosures,
        } = following[i];
        if (
          bindingError ||
          proofs.get(choice.moveId)?.verified !== true ||
          op.actor !== proof.reward.actor ||
          !same(op.before, previous.after) ||
          observedStageArguments(choice, op) ||
          !same(op.before.pending?.task, {
            ...proof.reward.task,
            remaining: proof.reward.remaining - i,
          })
        )
          issue ??= 'technologyCapRewardChoiceNotBound';
        if (proof.track === 'extraction') {
          if (
            op.row.action.type !== 'resource' ||
            op.entry.type !== 'takeGoods' ||
            op.entry.args.origin !== null ||
            op.after.players[op.actor].resources[op.row.action.resource] -
              op.before.players[op.actor].resources[op.row.action.resource] !==
              1
          ) {
            const bound =
              !issue &&
              fixedBuildingResourceChoiceProof({
                operation: op,
                activation: deferred.operations.find((o) => o.row.action.type === 'technology'),
                rewardOperations: following.map((c) => c.operation),
                group: response,
                index,
                sourceToNative,
                mapCard,
                closures: buildingResourceClosures,
              });
            if (
              !bound ||
              consumedBuildingRewards.has(bound.key) ||
              fixedContributions.some((c) => c.key === bound.key)
            )
              issue ??= 'technologyCapResourceRewardNotIsolated';
            else fixedContributions.push(bound);
          }
        } else if (op.row.action.type !== 'temple' || op.row.action.direction !== 1)
          issue ??= 'technologyCapTempleRewardNotBound';
        previous = op;
        if (!originals.has(response) || !same(originals.get(response), response))
          issue ??= 'technologyCapRewardSourceNotBound';
      }
      if (issue) fail(boundary, issue);
      else {
        for (const contribution of fixedContributions)
          consumedBuildingRewards.add(contribution.key);
        pass(
          boundary,
          'technologyCapRewardInterval',
          following.at(-1).group,
          [...deferred.operations, ...following.map((c) => c.operation)],
          {
            activationMoveId: group.moveId,
            activationSource: clone(proof.source),
            rewardSource: following.map((c) => ref(c.operation.entry)),
            decisionObservationsUnmodified: true,
            fixedBuildingContributions: clone(fixedContributions),
          },
        );
      }
    }
    const all = expected.map(
      (moveId) =>
        proofs.get(moveId) ?? {
          moveId,
          verified: false,
          reason: 'observedChoiceResponseNotClosed',
        },
    );
    return {
      schema: 'tzolkin-bga-observed-choices-v1',
      sourceRawSha256: sequence?.rawSha256 ?? null,
      expectedBoundaries: clone(expected),
      boundaries: clone(all),
      verifiedBoundaries: all.filter((b) => b.verified).length,
      unresolved: clone(all.filter((b) => !b.verified)),
      allObservedChoicesMatched: all.length > 0 && all.every((b) => b.verified),
      operations: clone(operationEvidence),
      candidateSetComparison: {
        state: 'notCompared',
        scope: 'Only the observed selected trajectory is checked.',
      },
      teacherCertified: false,
    };
  }
  return { observe, finish };
}

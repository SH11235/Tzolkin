import { readFileSync } from 'node:fs';
import { isDeepStrictEqual } from 'node:util';
import { originalPacketEntry, originalNetworkSequence } from './bga-network.mjs';

const catalog = JSON.parse(
  readFileSync(new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url)),
);
const supported = new Set(['b11', 'b12', 'b23', 'b24', 'b25', 'b26']);
const resources = ['corn', 'wood', 'stone', 'gold', 'skull'];
const materials = ['wood', 'stone', 'gold'];
const technologies = ['agriculture', 'extraction', 'architecture', 'theology'];
const copy = structuredClone;
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
const fixedTask = (task, fixed) =>
  task?.type === 'effects' &&
  task.effects.length === 1 &&
  isDeepStrictEqual(task.effects[0], { type: 'resources', resources: fixed });
const basic = (s) =>
  s.version === 1 &&
  !s.expansion &&
  s.additionalBuildings === false &&
  s.players.every((p) => !p.tribe);

function constructionContinuation(opening) {
  const task = opening.before.pending?.task;
  const inherited = opening.before.pending?.after;
  const tail = opening.after.pending?.after;
  if (
    task?.type !== 'build' ||
    ![1, 2].includes(task.remaining) ||
    typeof task.allowMonument !== 'boolean' ||
    typeof task.cornPayment !== 'boolean' ||
    (task.architectureAvailable !== undefined && typeof task.architectureAvailable !== 'boolean') ||
    (task.mandatory !== undefined && task.mandatory !== false) ||
    !Array.isArray(tail) ||
    !Array.isArray(inherited) ||
    inherited.length > 1 ||
    (inherited.length === 1 &&
      !isDeepStrictEqual(inherited[0], {
        type: 'build',
        remaining: 0,
        allowMonument: false,
        cornPayment: false,
        architectureAvailable: false,
      }))
  )
    return null;
  const follow = {
    type: 'build',
    remaining: task.remaining - 1,
    allowMonument: false,
    cornPayment: task.cornPayment,
    architectureAvailable: task.architectureAvailable !== false && !opening.action.architecture,
  };
  return isDeepStrictEqual(tail.slice(1), [follow, ...inherited])
    ? { follow, inherited: copy(inherited) }
    : null;
}

function noncompetingFixedSkull(opening, technology, fixed, continuation) {
  const levels = opening.before.players[opening.actor]?.technologies;
  if (
    opening.before.skullSupply !== 1 ||
    !isDeepStrictEqual(fixed, { skull: 1 }) ||
    technology.type !== 'technology' ||
    technology.technology !== 'any' ||
    technology.steps !== 1 ||
    !isDeepStrictEqual(levels, { agriculture: 0, extraction: 3, architecture: 3, theology: 0 }) ||
    opening.before.pending?.after?.length !== 0 ||
    opening.after.pending?.task?.type !== 'technology' ||
    opening.after.pending.task.remaining !== 1 ||
    opening.after.pending.task.free !== true ||
    !continuation ||
    continuation.inherited.length !== 0 ||
    continuation.follow.remaining !== 1
  )
    return null;
  return {
    technologies: copy(levels),
    freeTechnologyChoices: 1,
    fixedSkull: 1,
    openingSkullSupply: 1,
  };
}

export function createBuildingResourceComparison({ index, sourceToNative }) {
  const history = [];
  const sequence = originalNetworkSequence(index);
  const originalGroups = new Map(
    sequence?.groups.map(({ group, original }) => [group, original]) ?? [],
  );
  let active = null,
    failure = null;
  const reject = (reason) => {
    failure ??= `Fixed building resources: ${reason}`;
  };
  const publicEntry = (n) => {
    const p = originalPacketEntry(n);
    return (
      p?.rawSha256 === index.rawSha256 &&
      p.tableId === index.tableId &&
      p.packetType === 'resend' &&
      n.channel === `/table/t${index.tableId}`
    );
  };
  function queued(item, pending) {
    const tasks = pending?.after ?? [];
    const matches = tasks.flatMap((t, i) => (fixedTask(t, item.fixed) ? [i] : []));
    if (matches.length > 1) {
      reject('identical reward provenance is ambiguous');
      return false;
    }
    if (!matches.length) return false;
    const at = matches[0];
    if (
      !isDeepStrictEqual(tasks.slice(at), item.tail) ||
      tasks
        .slice(0, at)
        .some((t) => t.type !== 'technology' || t.remaining !== 0 || t.free !== true)
    )
      reject('reward continuation changed or contains an unproved task');
    return true;
  }
  function pendingProfile(state, actor) {
    if (!basic(state) || state.currentPlayer !== actor) return false;
    const task = state.pending?.task;
    return (
      (task?.type === 'technology' &&
        task.free === true &&
        task.remaining === 1 &&
        !task.mandatory) ||
      (task?.type === 'resource' && [1, 2].includes(task.remaining)) ||
      (task?.type === 'temple' &&
        task.remaining === 1 &&
        (task.direction === undefined || task.direction === 1) &&
        !task.reason &&
        !task.distinct)
    );
  }
  function sourceBoundary(group, owner) {
    if (
      group.entries.some(
        (n) =>
          n.type === 'gameStateChange' &&
          (!['game', 'activeplayer'].includes(index.initial.gamestates[n.args.id]?.type) ||
            (n.args.type !== undefined &&
              n.args.type !== index.initial.gamestates[n.args.id].type)),
      )
    )
      return false;
    const human = group.entries.filter(
      (n) =>
        n.type === 'gameStateChange' &&
        ['activeplayer', 'multipleactiveplayer'].includes(
          n.args.type ?? index.initial.gamestates[n.args.id]?.type,
        ),
    );
    const states = group.entries.filter((n) => n.type === 'gameStateChange');
    return (
      human.length <= 1 &&
      (!human.length ||
        (human[0] === states.at(-1) && String(human[0].args.active_player) === owner))
    );
  }
  function precedingMarketExit(context, opening, owner) {
    const prior = context.transitions.filter((t) => t.step < opening.step);
    if (prior.length !== 1) return null;
    const skip = prior[0],
      position = index.groups.indexOf(context.group),
      [menu, exit, discount] = index.groups.slice(position - 3, position),
      definitions = index.initial.gamestates;
    if (position < 3 || !menu || !exit || !discount) return null;
    if (
      !Array.isArray(skip.legalActions) ||
      !Array.isArray(opening.legalActions) ||
      !skip.before ||
      !skip.after
    )
      return null;
    const constructs = context.group.entries.filter((n) => n.type === 'construct');
    if (constructs.length !== 1 || String(constructs[0].args.player_id) !== owner) return null;
    const groups = [menu, exit, discount, context.group];
    const closed = (g) =>
      isDeepStrictEqual(originalGroups.get(g), g) &&
      g.entries.every((n, i) => {
        const packet = originalPacketEntry(n);
        return (
          publicEntry(n) &&
          packet.packetNotificationCount === g.entries.length &&
          n.packetIndex === g.entries[0].packetIndex &&
          n.notificationIndex === i &&
          n.visible &&
          !n.administrative &&
          !n.beforeCheckpoint
        );
      });
    const state = (n, id, args) =>
      n?.type === 'gameStateChange' &&
      n.args.id === id &&
      n.args.active_player === owner &&
      n.args.type === (id === 9 ? 'game' : 'activeplayer') &&
      Object.hasOwn(n.args, 'args') &&
      isDeepStrictEqual(n.args.args, args);
    const buildRows = skip.legalActions.filter((r) => r.action.type === 'skip');
    const unchanged = (s) =>
      Object.fromEntries(Object.entries(s).filter(([k]) => !['pending', 'log'].includes(k)));
    const offer = discount.entries[0]?.args?.args;
    if (
      !groups.every(closed) ||
      groups.some((g, i) => g.moveId !== menu.moveId + i) ||
      new Set(groups.flatMap((g) => g.entries.map((n) => n.uid))).size !==
        groups.reduce((n, g) => n + g.entries.length, 0) ||
      !isDeepStrictEqual(
        menu.entries.map((n) => n.type),
        ['trade', 'takeGoods', 'payGoods', 'gameStateChange'],
      ) ||
      !state(menu.entries[3], 42, { maxvalue: '20' }) ||
      menu.entries.slice(0, 3).some((n) => String(n.args.player_id) !== owner) ||
      exit.entries.length !== 2 ||
      !state(exit.entries[0], 9, null) ||
      !state(exit.entries[1], 32, null) ||
      discount.entries.length !== 1 ||
      !state(discount.entries[0], 33, offer) ||
      !isDeepStrictEqual(Object.keys(offer ?? {}).sort(), materials.slice().sort()) ||
      materials.some((r) => typeof offer[r] !== 'boolean') ||
      offer[opening.action.discountResource] !== true ||
      opening.action.architecture !== true ||
      definitions[42]?.name !== 'trade' ||
      definitions[42].type !== 'activeplayer' ||
      definitions[42].args !== 'argTrade' ||
      !isDeepStrictEqual(definitions[42].possibleactions, ['trade', 'pass']) ||
      definitions[42].transitions?.nextAction !== 9 ||
      definitions[9]?.type !== 'game' ||
      definitions[9].action !== 'stSolveStackedAction' ||
      definitions[9].transitions?.construct !== 32 ||
      definitions[32]?.name !== 'construct' ||
      definitions[32].type !== 'activeplayer' ||
      definitions[32].args !== 'argConstruct' ||
      !isDeepStrictEqual(definitions[32].possibleactions, ['construct', 'pass']) ||
      definitions[32].transitions?.sparedResource !== 33 ||
      definitions[32].transitions?.nextAction !== 9 ||
      definitions[33]?.name !== 'constructChooseSparedResources' ||
      definitions[33].type !== 'activeplayer' ||
      definitions[33].args !== 'argSparedResource' ||
      !isDeepStrictEqual(definitions[33].possibleactions, ['sparedResource']) ||
      definitions[33].transitions?.nextAction !== 9 ||
      skip.action.type !== 'skip' ||
      skip.actor !== opening.actor ||
      skip.before.currentPlayer !== opening.actor ||
      skip.after.currentPlayer !== opening.actor ||
      !Number.isSafeInteger(skip.step) ||
      skip.step + 1 !== opening.step ||
      !isDeepStrictEqual(skip.before, context.native) ||
      !isDeepStrictEqual(skip.before.pending?.task, { type: 'trade' }) ||
      !isDeepStrictEqual(skip.before.pending.after, [skip.after.pending?.task]) ||
      !isDeepStrictEqual(skip.after.pending?.after, []) ||
      !isDeepStrictEqual(skip.after.pending?.task, {
        type: 'build',
        remaining: 1,
        allowMonument: false,
        cornPayment: false,
        architectureAvailable: true,
      }) ||
      !isDeepStrictEqual(unchanged(skip.before), unchanged(skip.after)) ||
      !isDeepStrictEqual(skip.after, opening.before) ||
      buildRows.length !== 1 ||
      !isDeepStrictEqual(buildRows[0].action, skip.action) ||
      !opening.legalActions.some((r) => isDeepStrictEqual(r.action, opening.action))
    )
      return null;
    return {
      transition: skip,
      evidence: {
        type: 'precedingBuildingMarketExit',
        actor: opening.actor,
        sourceOwner: owner,
        menu: ref(menu.entries[3]),
        exit: exit.entries.map(ref),
        discount: ref(discount.entries[0]),
        constructionMoveId: context.group.moveId,
        skipStep: skip.step,
        buildStep: opening.step,
        construction: ref(constructs[0]),
        payments: context.group.entries.filter((n) => n.type === 'payGoods').map(ref),
        gameplayUnchanged: true,
        humanPassClickObserved: false,
      },
    };
  }
  function closingDecision(item, ledger, native, observation) {
    const state = index.initial.gamestates[ledger.stateId];
    if (state?.type === 'game') return null;
    if (
      state?.type !== 'activeplayer' ||
      String(ledger.activePlayer) !== item.sourceOwner ||
      observation.actor !== item.actor ||
      native.currentPlayer !== item.actor
    )
      return false;
    const rows = observation.legalActions ?? [];
    const possible = state.possibleactions ?? [];
    if (!native.pending) {
      if (state.name === 'pickUpWorker' && native.turn.mode === 'remove' && native.turn.count > 0)
        return (
          possible.includes('pickUpWorker') &&
          possible.includes('endTurn') &&
          rows.some((r) => r.action.type === 'endTurn') &&
          rows.every((r) => ['remove', 'endTurn'].includes(r.action.type))
        );
      if (state.name === 'placeWorker' && native.turn.mode === 'place' && native.turn.count > 0)
        return (
          possible.includes('placeWorker') &&
          possible.includes('endTurn') &&
          rows.some((r) => r.action.type === 'endTurn') &&
          rows.every((r) => ['place', 'endTurn'].includes(r.action.type))
        );
      if (state.name === 'playerTurn' && native.turn.mode === 'none' && native.turn.count === 0)
        return (
          possible.includes('placeWorker') &&
          possible.includes('pickUpWorker') &&
          rows.length > 0 &&
          rows.every((r) => ['place', 'remove', 'beg'].includes(r.action.type))
        );
      return false;
    }
    const task = native.pending.task;
    if (
      task.type !== 'build' ||
      task.remaining !== 1 ||
      task.mandatory ||
      !possible.includes('construct') ||
      !possible.includes('pass') ||
      !rows.some((r) => r.action.type === 'skip') ||
      !rows.every((r) => ['build', 'monument', 'skip'].includes(r.action.type))
    )
      return false;
    return task.architectureAvailable === false
      ? state.name === 'constructWithoutArchitecture'
      : state.name === 'construct';
  }
  function compare(context, { ledger, native, observation, differences, constructionTransition }) {
    const { group, source, transitions } = context;
    const evidence = [],
      fields = new Set();
    let precedingTransition = null;
    if (!isDeepStrictEqual(originalGroups.get(group), group))
      reject('original notification group changed or is not source-bound');
    const builds = transitions.filter(
      (t) => t.action.type === 'build' && supported.has(t.action.id),
    );
    const constructs = group.entries.filter((n) => n.type === 'construct');
    const opening = builds[0];
    if (
      opening &&
      opening.after.pending &&
      (opening.after.pending.after ?? []).some(
        (t) => t.type === 'effects' && t.effects.some((e) => e.type === 'resources'),
      )
    ) {
      const construct = constructs[0],
        actor = opening.actor;
      const sourceOwner = String(construct?.args.player_id);
      const card = catalog.BUILDINGS.find((c) => c.id === opening.action.id);
      const fixed = card.effects.find((e) => e.type === 'resources')?.resources;
      const effect = opening.after.pending.after[0];
      const beforeTasks = opening.before.pending?.after ?? [];
      const tail = opening.after.pending.after;
      const continuation = constructionContinuation(opening);
      const preceding = transitions.filter((t) => t.step < opening.step);
      const prefix = preceding.length ? precedingMarketExit(context, opening, sourceOwner) : null;
      const technology = card.effects[0];
      const noncompetingSkull = noncompetingFixedSkull(opening, technology, fixed, continuation);
      const expectedTask = technology.technology === 'extraction' ? 'resource' : 'technology';
      const payments = group.entries.filter((n) => n.type === 'payGoods');
      const receipts = group.entries.filter(
        (n) =>
          n.type === 'takeGoods' &&
          n.args.origin === `${construct?.args.building.type}_b${construct?.args.building.id}`,
      );
      const paid = resources.map((r) =>
        payments
          .filter((n) => n.args.type === r)
          .reduce((sum, n) => sum + Number(n.args.quantity), 0),
      );
      if (
        active ||
        builds.length !== 1 ||
        constructs.length !== 1 ||
        !construct ||
        !publicEntry(construct) ||
        sourceToNative.get(sourceOwner) !== actor ||
        constructionTransition?.nativeId !== card.id ||
        constructionTransition.sourceConstructUid !== construct.uid ||
        !isDeepStrictEqual(constructionTransition.action, opening.action) ||
        !opening.legalActions.some((r) => isDeepStrictEqual(r.action, opening.action)) ||
        !basic(opening.before) ||
        !basic(opening.after) ||
        opening.before.pending?.task.type !== 'build' ||
        !isDeepStrictEqual(
          source.players[sourceOwner]?.resources,
          opening.before.players[actor].resources,
        ) ||
        opening.before.skullSupply !== source.skullSupply ||
        !isDeepStrictEqual(paid, opening.action.cost) ||
        payments.some(
          (n) =>
            !publicEntry(n) ||
            String(n.args.player_id) !== sourceOwner ||
            n.args.target !== `${construct.args.building.type}_${construct.args.building.id}` ||
            group.entries.indexOf(n) >= group.entries.indexOf(construct) ||
            !resources.includes(n.args.type),
        ) ||
        resources.some((r, i) => opening.before.players[actor].resources[r] < paid[i]) ||
        beforeTasks.some(
          (t) => t.type === 'effects' && t.effects.some((e) => e.type === 'resources'),
        ) ||
        !fixedTask(effect, fixed) ||
        !continuation ||
        (preceding.length > 0 && !prefix) ||
        opening.after.pending.task.type !== expectedTask ||
        (expectedTask === 'resource' &&
          (opening.before.players[actor].technologies.extraction !== 3 ||
            opening.after.pending.task.remaining !== 2)) ||
        !pendingProfile(opening.after, actor) ||
        (fixed.skull && opening.before.skullSupply < 2 && !noncompetingSkull) ||
        receipts.length !== Object.keys(fixed).length ||
        receipts.some(
          (n) =>
            !publicEntry(n) ||
            group.entries.indexOf(n) <= group.entries.indexOf(construct) ||
            String(n.args.player_id) !== sourceOwner ||
            Number(n.args.quantity) !== fixed[n.args.type] ||
            !Number.isSafeInteger(Number(n.args.quantity)),
        )
      ) {
        reject('construction, payment, receipt or bounded basic profile is not established');
      } else {
        active = {
          sourceOwner,
          actor,
          building: card.id,
          fixed: copy(fixed),
          tail: copy(tail),
          sourceConstruction: ref(construct),
          sourceReceipts: receipts.map(ref),
          startMoveId: group.moveId,
          initialSkullSupply: opening.before.skullSupply,
          technology: technology.technology,
          consumedAtStep: null,
          closed: false,
          stage: expectedTask,
          decisions: [],
          sourceRewardOrigin: `${construct.args.building.type}_b${construct.args.building.id}`,
          ...(noncompetingSkull ? { noncompetingSkullProfile: copy(noncompetingSkull) } : {}),
          ...(continuation.inherited.length
            ? { inheritedBuildContinuation: copy(continuation) }
            : {}),
          ...(prefix ? { precedingMarketExit: copy(prefix.evidence) } : {}),
        };
        history.push(active);
        precedingTransition = prefix?.transition ?? null;
        if (prefix) evidence.push(copy(prefix.evidence));
        evidence.push({
          type: 'fixedBuildingEntitlement',
          building: card.id,
          construction: ref(construct),
          receipts: receipts.map(ref),
          fixed: copy(fixed),
          nativeBuildStep: opening.step,
          preRewardSkullSupply: opening.before.skullSupply,
          ...(noncompetingSkull ? { noncompetingSkullProfile: copy(noncompetingSkull) } : {}),
        });
      }
    }
    if (!active) return { fields: [], evidence, failure };
    const item = active;
    if (!sourceBoundary(group, item.sourceOwner) || group.entries.some((n) => !publicEntry(n)))
      reject('intervening source decision or nonpublic event is not established');
    const openingGroup = item.startMoveId === group.moveId;
    const actions = transitions.filter((t) => t !== opening && t !== precedingTransition);
    const directReceipts = [];
    const technologyChoices = [],
      templeChoices = [];
    let points = 0;
    for (const t of actions) {
      if (item.consumedAtStep !== null) {
        reject('another operation followed reward consumption in one source boundary');
        break;
      }
      const task = t.before.pending?.task;
      if (
        !pendingProfile(t.before, item.actor) ||
        task.type !== item.stage ||
        t.actor !== item.actor ||
        !queued(item, t.before.pending) ||
        !t.legalActions.some((r) => isDeepStrictEqual(r.action, t.action))
      ) {
        reject('intervening native actor, pending or legal selection is not bound');
        break;
      }
      const direct = Object.fromEntries(resources.map((r) => [r, 0]));
      let nextStage = null;
      if (
        t.action.type === 'technology' &&
        task.type === 'technology' &&
        technologies.includes(t.action.technology)
      ) {
        const technology = t.action.technology;
        const level = t.before.players[item.actor].technologies[technology];
        technologyChoices.push({
          technology,
          level,
          afterLevel: t.after.players[item.actor].technologies[technology],
        });
        if (
          t.legalActions.length !== 4 ||
          !technologies.every((technology) =>
            t.legalActions.some(
              (r) => r.action.type === 'technology' && r.action.technology === technology,
            ),
          )
        )
          reject('free technology options depend on an unproved profile');
        if (level === 3) {
          if (technology === 'theology') direct.skull = Math.min(1, t.before.skullSupply);
          if (technology === 'architecture') points += 3;
          if (technology === 'agriculture') nextStage = 'temple';
          if (technology === 'extraction') nextStage = 'resource';
        }
      } else if (
        t.action.type === 'resource' &&
        task.type === 'resource' &&
        materials.includes(t.action.resource)
      ) {
        direct[t.action.resource] = 1;
        if (
          t.legalActions.length !== 3 ||
          !materials.every((resource) =>
            t.legalActions.some(
              (r) => r.action.type === 'resource' && r.action.resource === resource,
            ),
          )
        )
          reject('capped extraction options are not the three unlimited materials');
        if (task.remaining === 2) nextStage = 'resource';
      } else if (!(
        t.action.type === 'temple' &&
        task.type === 'temple' &&
        t.action.direction === 1
      )) {
        reject('intervening payment, cancellation or task is outside the supported free choices');
        break;
      } else {
        if (
          t.legalActions.length !== 3 ||
          !Object.keys(catalog.TEMPLE_TRACKS).every((temple) =>
            t.legalActions.some(
              (r) =>
                r.action.type === 'temple' &&
                r.action.temple === temple &&
                r.action.direction === 1,
            ),
          )
        )
          reject('capped agriculture options are not the basic free temple choices');
        templeChoices.push({
          temple: t.action.temple,
          beforeLevel: t.before.players[item.actor].temples[t.action.temple],
          afterLevel: t.after.players[item.actor].temples[t.action.temple],
        });
      }
      const remains = queued(item, t.after.pending);
      const consumed = !remains;
      if (remains !== (nextStage !== null) || (remains && t.after.pending?.task.type !== nextStage))
        reject('free technology continuation does not match its selected bonus');
      item.stage = nextStage;
      if (
        !basic(t.after) ||
        resources.some(
          (r) =>
            t.after.players[item.actor].resources[r] - t.before.players[item.actor].resources[r] !==
            direct[r] + (consumed ? (item.fixed[r] ?? 0) : 0),
        ) ||
        t.before.skullSupply - t.after.skullSupply !==
          direct.skull + (consumed ? (item.fixed.skull ?? 0) : 0)
      ) {
        reject('actual choice and fixed reward contributions do not match their separate amounts');
        break;
      }
      if (item.fixed.skull && consumed && t.before.skullSupply < 1 + direct.skull)
        reject('skull allocation to both distinct effects is not established');
      item.decisions.push({
        step: t.step,
        action: copy(t.action),
        directContribution: direct,
        fixedContribution: consumed ? copy(item.fixed) : {},
        preSkullSupply: t.before.skullSupply,
        postSkullSupply: t.after.skullSupply,
        legalOptionsIndependent: true,
      });
      directReceipts.push(
        ...Object.entries(direct)
          .filter(([, amount]) => amount > 0)
          .map(([resource, amount]) => ({ resource, amount })),
      );
      if (consumed) item.consumedAtStep = t.step;
    }
    const permitted = new Set(['gameStateChange', 'updateReflexionTime']);
    if (openingGroup) {
      for (const type of ['payGoods', 'construct', 'takeGoods', 'updateScore']) permitted.add(type);
      if (item.technology === 'extraction')
        for (const type of ['increaseTechnology', 'decreaseTechnology', 'extractionMax'])
          permitted.add(type);
      const unrelated = group.entries.filter(
        (n) => n.type === 'takeGoods' && !item.sourceReceipts.some((r) => r.uid === n.uid),
      );
      if (
        unrelated.some(
          (n) =>
            n.args.origin !== 'technology_architecture_level_1' ||
            opening.action.architecture !== true ||
            Number(n.args.quantity) !== 1 ||
            n.args.type !== 'corn',
        )
      )
        reject('another source resource receipt overlaps the fixed reward');
      points =
        opening.action.architecture &&
        opening.before.players[item.actor].technologies.architecture >= 2
          ? 2
          : 0;
      if (item.technology === 'extraction')
        technologyChoices.push({
          technology: 'extraction',
          level: opening.before.players[item.actor].technologies.extraction,
          afterLevel: opening.after.players[item.actor].technologies.extraction,
        });
    } else {
      for (const type of [
        'increaseTechnology',
        'decreaseTechnology',
        'extractionMax',
        'agricultureMax',
        'architectureMax',
        'theologyMax',
        'takeGoods',
        'alterTemple',
        'updateScore',
        'flipBoard',
      ])
        permitted.add(type);
      const receipts = group.entries.filter((n) => n.type === 'takeGoods');
      const observed = resources.map((r) =>
        receipts
          .filter((n) => n.args.type === r)
          .reduce((sum, n) => sum + Number(n.args.quantity), 0),
      );
      const expected = resources.map((r) =>
        directReceipts.filter((n) => n.resource === r).reduce((sum, n) => sum + n.amount, 0),
      );
      if (
        !isDeepStrictEqual(observed, expected) ||
        receipts.some(
          (n) =>
            n.args.origin !==
            (n.args.type === 'skull' &&
            technologyChoices.some((c) => c.technology === 'theology' && c.level === 3)
              ? 'technology_theology_level_4'
              : null),
        )
      )
        reject('source choice receipts differ from actual direct gains');
    }
    const increases = group.entries.filter((n) => n.type === 'increaseTechnology');
    const decreases = group.entries.filter((n) => n.type === 'decreaseTechnology');
    if (
      increases.length !== technologyChoices.length ||
      technologyChoices.some(
        (c) =>
          increases.filter(
            (n) => n.args.technology_id === c.technology && Number(n.args.level) === c.level + 1,
          ).length !== 1,
      ) ||
      decreases.length !== technologyChoices.filter((c) => c.level === 3).length ||
      decreases.some(
        (n) =>
          !technologyChoices.some(
            (c) =>
              c.level === 3 && n.args.technology_id === c.technology && Number(n.args.level) === 3,
          ),
      )
    )
      reject('source technology changes do not describe the actual free advance');
    const alterations = group.entries.filter((n) => n.type === 'alterTemple');
    if (
      alterations.length !== templeChoices.filter((c) => c.afterLevel !== c.beforeLevel).length ||
      alterations.some(
        (n) =>
          !templeChoices.some(
            (c) =>
              c.temple === (n.args.temple_id === 'kukulcan' ? 'kukulkan' : n.args.temple_id) &&
              Number(n.args.step) === c.afterLevel + 1,
          ),
      )
    )
      reject('source temple changes are not the actual capped agriculture choice');
    const scores = group.entries.filter((n) => n.type === 'updateScore');
    if (scores.length !== (points ? 1 : 0) || scores.some((n) => Number(n.args.points) !== points))
      reject('source score receipts are not the actual automatic technology or architecture bonus');
    if (
      group.entries.some(
        (n) =>
          /Max$/.test(n.type) &&
          !technologyChoices.some((c) => c.level === 3 && n.type === `${c.technology}Max`),
      )
    )
      reject('source capped-technology caption has no corresponding advance');
    const flips = group.entries.filter((n) => n.type === 'flipBoard');
    if (
      flips.length > 1 ||
      flips.some(
        (n) =>
          n.args.side !== 0 ||
          !templeChoices.some(
            (c) =>
              c.beforeLevel < catalog.TEMPLE_TRACKS[c.temple].max &&
              c.afterLevel === catalog.TEMPLE_TRACKS[c.temple].max,
          ),
      )
    )
      reject('source speed-up right did not follow the actual temple summit');
    if (
      group.entries.some(
        (n) =>
          !permitted.has(n.type) ||
          (n.args.player_id !== undefined && String(n.args.player_id) !== item.sourceOwner),
      )
    )
      reject('unrelated source effect or actor entered the fixed reward interval');
    if (item.consumedAtStep === null) {
      const expectedSourceName = {
        technology: 'freeTechnology',
        resource: 'resourceReward',
        temple: 'templeReward',
      }[native.pending?.task.type];
      if (
        index.initial.gamestates[ledger.stateId]?.name !== expectedSourceName ||
        String(ledger.activePlayer) !== item.sourceOwner
      )
        reject('source and native choices are not the same free-technology stage');
      if (
        !queued(item, native.pending) ||
        !pendingProfile(native, item.actor) ||
        observation.actor !== item.actor
      )
        reject('unconsumed reward no longer precedes a supported choice');
      for (const [r, amount] of Object.entries(item.fixed)) {
        const field = `player.${item.sourceOwner}.resources.${r}`;
        const difference = differences.find((d) => d.field === field);
        if (!difference || difference.observed !== difference.native + amount)
          reject('unconsumed reward does not equal its exact original public entitlement');
        else fields.add(field);
      }
      if (item.fixed.skull) {
        const difference = differences.find((d) => d.field === 'skullSupply');
        if (!difference || difference.native !== difference.observed + item.fixed.skull)
          reject('reserved source skull and shared supply do not form one pair');
        else fields.add('skullSupply');
      }
    } else {
      if (
        queued(item, native.pending) ||
        Object.keys(item.fixed).some((r) =>
          differences.some((d) => d.field === `player.${item.sourceOwner}.resources.${r}`),
        ) ||
        (item.fixed.skull && differences.some((d) => d.field === 'skullSupply'))
      )
        reject('consumed fixed reward did not rejoin its original counters');
      const decision = closingDecision(item, ledger, native, observation);
      if (decision === false)
        reject('source and native remaining decisions differ at reward closure');
      if (!failure && decision === true) {
        item.closed = true;
        item.closedMoveId = group.moveId;
        evidence.push({
          type: 'fixedBuildingClosure',
          building: item.building,
          construction: item.sourceConstruction,
          consumedAtStep: item.consumedAtStep,
          decisions: copy(item.decisions),
        });
        active = null;
      }
    }
    return {
      fields: [...fields],
      evidence,
      failure,
      unresolved: active
        ? {
            kind: 'fixedBuildingResources',
            building: active.building,
            sourceOwner: active.sourceOwner,
            startMoveId: active.startMoveId,
            construction: active.sourceConstruction,
            fixed: copy(active.fixed),
            nativeConsumedAtStep: active.consumedAtStep,
            awaitingDecisionClosure: active.consumedAtStep !== null,
          }
        : null,
    };
  }
  return {
    compare,
    finish: () => ({
      obligations: copy(history),
      failure,
      allObligationsClosed: !failure && history.every((h) => h.closed),
      teacherCertified: false,
    }),
  };
}

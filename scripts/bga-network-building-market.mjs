import { isDeepStrictEqual as same } from 'node:util';
import { originalPacketEntry } from './bga-network.mjs';

const integerEquals = (value, expected) =>
  ['number', 'string'].includes(typeof value) &&
  /^(0|[1-9]\d*)$/.test(String(value)) &&
  Number.isSafeInteger(Number(value)) &&
  Number(value) === expected;
const proofs = new WeakMap();
const exits = new WeakMap();
const fail = () => {
  throw new Error('Building market: original structural exit is not established');
};
const tail = [
  {
    type: 'build',
    remaining: 1,
    allowMonument: false,
    cornPayment: false,
    architectureAvailable: false,
  },
  {
    type: 'build',
    remaining: 0,
    allowMonument: false,
    cornPayment: false,
    architectureAvailable: false,
  },
];
const supportedTail = (value) => same(value, tail) || same(value, tail.slice(0, 1));
const publicState = (value) =>
  Object.fromEntries(Object.entries(value).filter(([k]) => k !== 'log'));
const prices = { wood: 2, stone: 3, gold: 4 };
function boundOperation(op, { index, entry, actor, sourceActor, before, after, action, move }) {
  return (
    op?.actor === actor &&
    op.row &&
    Array.isArray(op.legalRows) &&
    op.observationActor === actor &&
    op.sourceActor === sourceActor &&
    op.entry === entry &&
    originalPacketEntry(entry)?.rawSha256 === index.rawSha256 &&
    same(op.before, publicState(before)) &&
    same(op.after, publicState(after)) &&
    same(op.row.action, action) &&
    same(op.row.move, move) &&
    op.legalRows.filter((r) => same(r.action, action) && same(r.move, move)).length === 1 &&
    op.legalRows.some((r) => same(r, op.row))
  );
}
function closed(index, group, types) {
  return (
    index.groups.includes(group) &&
    same(
      group.entries.map((n) => n.type),
      types,
    ) &&
    new Set(group.entries.map((n) => n.uid)).size === types.length &&
    group.entries.every((n, i) => {
      const source = originalPacketEntry(n);
      return (
        source?.packetType === 'resend' &&
        source.packetNotificationCount === types.length &&
        source.rawSha256 === index.rawSha256 &&
        String(source.tableId) === String(index.tableId) &&
        n.packetIndex === group.entries[0].packetIndex &&
        n.notificationIndex === i &&
        n.moveId === group.moveId &&
        n.visible &&
        !n.administrative &&
        !n.beforeCheckpoint &&
        n.channel === `/table/t${index.tableId}`
      );
    })
  );
}
function state(n, id, actor, index) {
  const a = n?.args;
  if (
    !a ||
    a.id !== id ||
    a.active_player !== actor ||
    typeof actor !== 'string' ||
    a.type !== (id === 9 ? 'game' : 'activeplayer') ||
    Object.keys(a).some(
      (k) =>
        !['id', 'active_player', 'args', 'type', 'reflexion', 'updateGameProgression'].includes(k),
    )
  )
    return false;
  if (id === 42 ? !same(a.args, { maxvalue: '20' }) : !Array.isArray(a.args) || a.args.length !== 0)
    return false;
  if (
    a.updateGameProgression !== undefined &&
    (!Number.isSafeInteger(a.updateGameProgression) ||
      a.updateGameProgression < 0 ||
      a.updateGameProgression > 100)
  )
    return false;
  if (a.reflexion !== undefined) {
    if (
      !a.reflexion ||
      !same(Object.keys(a.reflexion), ['total']) ||
      !a.reflexion.total ||
      !same(Object.keys(a.reflexion.total).sort(), Object.keys(index.initial.players).sort()) ||
      Object.values(a.reflexion.total).some(
        (v) =>
          !['number', 'string'].includes(typeof v) ||
          !/^\d+$/.test(String(v)) ||
          !Number.isSafeInteger(Number(v)),
      )
    )
      return false;
  }
  return true;
}
export function trackBuildingMarket(
  previous,
  { index, group, before, after, transition, sourceActor, sourceBefore, differences, operations },
) {
  if (differences.length) return null;
  if (
    transition?.action.type === 'build' &&
    transition.action.id === 'b18' &&
    transition.action.renovation === null &&
    sourceBefore?.initial === index.initial &&
    sourceBefore.stateId === 37 &&
    sourceBefore.activePlayer === sourceActor &&
    same(sourceBefore.stateArgs, { wood: false, stone: true, gold: false }) &&
    transition.action.architecture === true &&
    transition.architectureLevel === 3 &&
    transition.action.discountResource === 'stone' &&
    same(transition.action.cost, [0, 0, 2, 0, 0]) &&
    supportedTail(after.pending?.after) &&
    same(after.pending?.task, { type: 'trade' }) &&
    after.players[after.currentPlayer].buildings.at(-1) === 'b18' &&
    closed(index, group, [
      'payGoods',
      'construct',
      'takeGoods',
      'updateScore',
      'updateScore',
      'gameStateChange',
      'gameStateChange',
    ])
  ) {
    const e = group.entries;
    if (
      e[0].args.player_id !== sourceActor ||
      e[0].args.type !== 'stone' ||
      !integerEquals(e[0].args.quantity, 2) ||
      e[1].args.player_id !== sourceActor ||
      !integerEquals(e[1].args.building?.id, Number(e[1].args.building?.id)) ||
      Number(e[1].args.building.id) < 1 ||
      e[1].args.building?.type !== 'building_age_2' ||
      !integerEquals(e[1].args.building.type_arg, 11) ||
      e[0].args.target !== `building_age_2_${e[1].args.building.id}` ||
      e[2].args.player_id !== sourceActor ||
      e[2].args.type !== 'corn' ||
      !integerEquals(e[2].args.quantity, 1) ||
      e[2].args.origin !== 'technology_architecture_level_1' ||
      !e
        .slice(3, 5)
        .every(
          (n, i) => n.args.player_id === sourceActor && integerEquals(n.args.points, [2, 6][i]),
        ) ||
      !state(e[5], 9, sourceActor, index) ||
      !state(e[6], 42, sourceActor, index) ||
      operations?.length !== 1 ||
      !boundOperation(operations[0], {
        index,
        entry: e[1],
        actor: after.currentPlayer,
        sourceActor,
        before,
        after,
        action: transition.action,
        move: { type: 'choose', choiceId: 'build:b18:stone' },
      })
    )
      return null;
    const token = {};
    proofs.set(token, {
      index,
      actor: after.currentPlayer,
      sourceActor,
      state: structuredClone(after),
      lastMove: group.moveId,
      constructionMove: group.moveId,
      constructionUids: e.map((n) => n.uid),
      transactions: [],
      tail: structuredClone(after.pending.after),
      uids: new Set(e.map((n) => n.uid)),
      title: transition.beforePending.title,
    });
    return token;
  }
  const proof = proofs.get(previous);
  if (
    !proof ||
    proof.index !== index ||
    !same(proof.state, before) ||
    group.moveId !== proof.lastMove + 1 ||
    !closed(index, group, ['trade', 'takeGoods', 'payGoods', 'gameStateChange']) ||
    group.entries.some((n) => proof.uids.has(n.uid))
  )
    return null;
  const e = group.entries,
    a = e[0].args;
  const icon = (resource) => `<div class='tz_icon resource_${resource}'></div>`;
  const buy = a.resource1 === icon('corn');
  const resource = Object.keys(prices).find((r) => (buy ? a.resource2 : a.resource1) === icon(r));
  if (!resource || (!buy && a.resource2 !== icon('corn'))) return null;
  const units = Number(buy ? a.quantity2 : a.quantity1);
  if (!Number.isSafeInteger(units) || units < 1 || units > 100) return null;
  const payment = buy ? 'corn' : resource,
    receipt = buy ? resource : 'corn',
    cost = buy ? prices[resource] : 1,
    gain = buy ? 1 : prices[resource];
  if (
    a.player_id !== proof.sourceActor ||
    !integerEquals(a.quantity1, cost * units) ||
    !integerEquals(a.quantity2, gain * units) ||
    e[1].args.player_id !== proof.sourceActor ||
    e[1].args.type !== receipt ||
    !integerEquals(e[1].args.quantity, gain * units) ||
    e[1].args.origin !== 'action_uxmal_2' ||
    e[2].args.player_id !== proof.sourceActor ||
    e[2].args.type !== payment ||
    !integerEquals(e[2].args.quantity, cost * units) ||
    e[2].args.target !== 'action_uxmal_2' ||
    !state(e[3], 42, proof.sourceActor, index) ||
    operations?.length !== units
  )
    return null;
  const expected = structuredClone(before);
  for (const op of operations) {
    const unitBefore = structuredClone(expected);
    if (expected.players[proof.actor].resources[payment] < cost) return null;
    expected.players[proof.actor].resources[payment] -= cost;
    expected.players[proof.actor].resources[receipt] += gain;
    if (
      !boundOperation(op, {
        index,
        entry: e[0],
        actor: proof.actor,
        sourceActor: proof.sourceActor,
        before: unitBefore,
        after: expected,
        action: { type: 'trade', resource, buy },
        move: { type: 'choose', choiceId: `${buy ? 'buy' : 'sell'}:${resource}` },
      })
    )
      return null;
  }
  if (!same(expected, after)) return null;
  const token = {};
  proofs.set(token, {
    ...proof,
    state: structuredClone(after),
    lastMove: group.moveId,
    transactions: [
      ...proof.transactions,
      { sourceMove: group.moveId, uids: e.map((n) => n.uid), resource, buy, units, cost, gain },
    ],
    uids: new Set([...proof.uids, ...e.map((n) => n.uid)]),
  });
  return token;
}
export function proveBuildingMarketExit(
  token,
  { index, group, ledger, before, legal, differences },
) {
  const p = proofs.get(token),
    s = index.initial.gamestates;
  if (
    !p ||
    p.index !== index ||
    p.lastMove + 1 !== group.moveId ||
    differences.length ||
    ledger.initial !== index.initial ||
    ledger.stateId !== 42 ||
    ledger.activePlayer !== p.sourceActor ||
    !same(ledger.stateArgs, { maxvalue: '20' }) ||
    !same(before, p.state) ||
    before.currentPlayer !== p.actor ||
    !same(before.pending?.after, p.tail) ||
    !supportedTail(p.tail) ||
    !same(before.pending?.task, { type: 'trade' }) ||
    !closed(index, group, ['gameStateChange', 'gameStateChange']) ||
    !state(group.entries[0], 9, p.sourceActor, index) ||
    !state(group.entries[1], 70, p.sourceActor, index) ||
    group.entries.some((n) => p.uids.has(n.uid)) ||
    group.entries[0].args.type !== 'game' ||
    group.entries[1].args.type !== 'activeplayer' ||
    s[42]?.name !== 'trade' ||
    s[42].type !== 'activeplayer' ||
    !same(s[42].possibleactions, ['trade', 'pass']) ||
    s[42].transitions?.nextAction !== 9 ||
    s[42].transitions?.nextTrade !== 42 ||
    s[9]?.name !== 'solveStackedAction' ||
    s[9].type !== 'game' ||
    s[9].transitions?.constructWithoutArchitecture !== 70 ||
    s[70]?.name !== 'constructWithoutArchitecture' ||
    s[70].type !== 'activeplayer' ||
    !same(s[70].possibleactions, ['construct', 'pass'])
  )
    fail();
  const skips = legal.filter((x) => x.action.type === 'skip');
  if (
    skips.length !== 1 ||
    !same(skips[0].move, { type: 'choose', choiceId: 'skip' }) ||
    legal.some(
      (x) =>
        x.action.type !== 'skip' &&
        (x.action.type !== 'trade' ||
          !['wood', 'stone', 'gold'].includes(x.action.resource) ||
          typeof x.action.buy !== 'boolean' ||
          x.move.type !== 'choose' ||
          x.move.choiceId !== `${x.action.buy ? 'buy' : 'sell'}:${x.action.resource}`),
    )
  )
    fail();
  const result = {
    move: structuredClone(skips[0].move),
    actor: p.actor,
    sourceActor: p.sourceActor,
    source: group.moveId,
    title: p.title,
    constructionMove: p.constructionMove,
    constructionUids: structuredClone(p.constructionUids),
    transactions: structuredClone(p.transactions),
    exitUids: group.entries.map((n) => n.uid),
  };
  exits.set(result, { proof: structuredClone(result), before: structuredClone(before) });
  return result;
}
export function bindBuildingMarketExit(proof, before, after) {
  const bound = exits.get(proof);
  if (!bound || !same(proof, bound.proof) || !same(before, bound.before)) fail();
  exits.delete(proof);
  const expected = structuredClone(before);
  expected.pending = {
    title: proof.title,
    task: expected.pending.after[0],
    after: expected.pending.after.slice(1),
  };
  if (!same(after, expected)) fail();
  return {
    sourceMove: proof.source,
    sourceActor: proof.sourceActor,
    actor: proof.actor,
    constructionMove: proof.constructionMove,
    constructionUids: proof.constructionUids,
    transactions: proof.transactions,
    exitUids: proof.exitUids,
    scoreDelta: 0,
    derivedReplayTransition: true,
    humanSkipObserved: false,
  };
}

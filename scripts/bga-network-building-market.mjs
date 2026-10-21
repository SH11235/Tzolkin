import { isDeepStrictEqual as same } from 'node:util';
import { originalPacketEntry } from './bga-network.mjs';

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
function closed(index, group, types) {
  return (
    index.groups.includes(group) &&
    same(
      group.entries.map((n) => n.type),
      types,
    ) &&
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
  { index, group, before, after, transition, sourceActor, sourceBefore, differences },
) {
  if (differences.length) return null;
  if (
    transition?.action.id === 'b18' &&
    sourceBefore?.initial === index.initial &&
    sourceBefore.stateId === 37 &&
    sourceBefore.activePlayer === sourceActor &&
    same(sourceBefore.stateArgs, { wood: false, stone: true, gold: false }) &&
    transition.action.architecture === true &&
    transition.architectureLevel === 3 &&
    transition.action.discountResource === 'stone' &&
    same(transition.action.cost, [0, 0, 2, 0, 0]) &&
    same(after.pending?.after, tail) &&
    after.pending?.task.type === 'trade' &&
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
      Number(e[0].args.quantity) !== 2 ||
      e[1].args.player_id !== sourceActor ||
      e[1].args.building?.type !== 'building_age_2' ||
      Number(e[1].args.building.type_arg) !== 11 ||
      e[0].args.target !== `building_age_2_${e[1].args.building.id}` ||
      e[2].args.player_id !== sourceActor ||
      e[2].args.type !== 'corn' ||
      Number(e[2].args.quantity) !== 1 ||
      e[2].args.origin !== 'technology_architecture_level_1' ||
      !same(
        e.slice(3, 5).map((n) => [n.args.player_id, Number(n.args.points)]),
        [
          [sourceActor, 2],
          [sourceActor, 6],
        ],
      ) ||
      !state(e[5], 9, sourceActor, index) ||
      !state(e[6], 42, sourceActor, index)
    )
      return null;
    const token = {};
    proofs.set(token, {
      index,
      actor: after.currentPlayer,
      sourceActor,
      state: structuredClone(after),
      lastMove: group.moveId,
      sales: 0,
      title: transition.beforePending.title,
    });
    return token;
  }
  const proof = proofs.get(previous);
  if (
    !proof ||
    proof.index !== index ||
    proof.sales >= 4 ||
    !same(proof.state, before) ||
    group.moveId !== proof.lastMove + 1 ||
    !closed(index, group, ['trade', 'takeGoods', 'payGoods', 'gameStateChange'])
  )
    return null;
  const e = group.entries,
    a = e[0].args;
  if (
    a.player_id !== proof.sourceActor ||
    Number(a.quantity1) !== 1 ||
    Number(a.quantity2) !== 4 ||
    a.resource1 !== "<div class='tz_icon resource_gold'></div>" ||
    a.resource2 !== "<div class='tz_icon resource_corn'></div>" ||
    e[1].args.player_id !== proof.sourceActor ||
    e[1].args.type !== 'corn' ||
    Number(e[1].args.quantity) !== 4 ||
    e[1].args.origin !== 'action_uxmal_2' ||
    e[2].args.player_id !== proof.sourceActor ||
    e[2].args.type !== 'gold' ||
    Number(e[2].args.quantity) !== 1 ||
    e[2].args.target !== 'action_uxmal_2' ||
    !state(e[3], 42, proof.sourceActor, index)
  )
    return null;
  const expected = structuredClone(before);
  expected.players[proof.actor].resources.gold -= 1;
  expected.players[proof.actor].resources.corn += 4;
  if (!same(expected, after)) return null;
  const token = {};
  proofs.set(token, {
    ...proof,
    state: structuredClone(after),
    lastMove: group.moveId,
    sales: proof.sales + 1,
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
    p.sales !== 4 ||
    p.lastMove + 1 !== group.moveId ||
    differences.length ||
    ledger.initial !== index.initial ||
    ledger.stateId !== 42 ||
    ledger.activePlayer !== p.sourceActor ||
    !same(ledger.stateArgs, { maxvalue: '20' }) ||
    !same(before, p.state) ||
    before.currentPlayer !== p.actor ||
    !same(before.pending?.after, tail) ||
    !closed(index, group, ['gameStateChange', 'gameStateChange']) ||
    !state(group.entries[0], 9, p.sourceActor, index) ||
    !state(group.entries[1], 70, p.sourceActor, index) ||
    group.entries[0].args.type !== 'game' ||
    group.entries[1].args.type !== 'activeplayer' ||
    s[42]?.name !== 'trade' ||
    !same(s[42].possibleactions, ['trade', 'pass']) ||
    s[42].transitions?.nextAction !== 9 ||
    s[42].transitions?.nextTrade !== 42 ||
    s[70]?.name !== 'constructWithoutArchitecture' ||
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
    scoreDelta: 0,
    derivedReplayTransition: true,
    humanSkipObserved: false,
  };
}

import { isDeepStrictEqual as equal } from 'node:util';
import { bindNewFixedReward } from './bga-network-rewards.mjs';

const fail = () => {
  throw new Error('Network architecture: automatic corn lacks an exact construction binding');
};
const reference = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});

export function bindAppliedArchitectureCorn({
  index,
  group,
  sourceId,
  actor,
  transition,
  sourceStateMatched,
  unresolvedTiming,
  card,
}) {
  if (!transition || !sourceStateMatched || unresolvedTiming) fail();
  const { beforeState: before, afterState: after, action, matchingOptions } = transition;
  const entries = group.entries;
  const previous = index.groups[index.groups.indexOf(group) - 1];
  const priorState = previous?.entries.filter((n) => n.type === 'gameStateChange').at(-1)?.args;
  if (
    !before ||
    !after ||
    !card ||
    matchingOptions !== 1 ||
    before.currentPlayer !== actor ||
    after.currentPlayer !== actor ||
    Number(priorState?.id) !== 44 ||
    String(priorState?.active_player) !== sourceId ||
    !equal(
      entries.map((n) => n.type),
      [
        'payGoods',
        'construct',
        'takeGoods',
        'updateScore',
        'takeGoods',
        'increaseTechnology',
        'extractionMax',
        'decreaseTechnology',
        'gameStateChange',
        'gameStateChange',
      ],
    ) ||
    new Set(entries.map((n) => n.uid)).size !== entries.length ||
    new Set(entries.map((n) => n.packetIndex)).size !== 1 ||
    entries.some((n, i) => n.notificationIndex !== i) ||
    entries.some((n) => n.args.player_id != null && String(n.args.player_id) !== sourceId) ||
    !equal(action, {
      type: 'build',
      id: card.id,
      cost: [6, 0, 0, 0, 0],
      architecture: true,
      discountResource: null,
      renovation: null,
    }) ||
    !equal(card.cost, { wood: 2, stone: 1 }) ||
    !equal(card.effects, [
      { type: 'technology', technology: 'extraction', steps: 1 },
      { type: 'resources', resources: { gold: 1 } },
    ]) ||
    before.players[actor]?.technologies.architecture !== 2 ||
    before.players[actor]?.technologies.extraction !== 3 ||
    before.buildings.filter((id) => id === card.id).length !== 1 ||
    before.players[actor].buildings.includes(card.id) ||
    !equal(before.pending?.task, {
      type: 'build',
      remaining: 1,
      allowMonument: false,
      cornPayment: true,
    }) ||
    !equal(before.pending.after, [])
  )
    fail();
  const [payment, construct, corn, score, gold, inc, cap, dec, interim, ending] = entries;
  const building = construct.args.building;
  const definition = index.initial.constructions?.[building?.type]?.[building?.type_arg];
  if (
    building?.type !== 'building_age_1' ||
    !equal(definition, { cost: { wood: 2, stone: 1 }, extraction: 1, gold: 1, kind: 'civic' }) ||
    payment.args.type !== 'corn' ||
    Number(payment.args.quantity) !== 6 ||
    payment.args.target !== `${building.type}_${building.id}` ||
    corn.args.type !== 'corn' ||
    Number(corn.args.quantity) !== 1 ||
    corn.args.origin !== 'technology_architecture_level_1' ||
    Number(score.args.points) !== 2 ||
    gold.args.type !== 'gold' ||
    Number(gold.args.quantity) !== 1 ||
    gold.args.origin !== `${building.type}_b${building.id}` ||
    inc.args.technology_id !== 'extraction' ||
    Number(inc.args.level) !== 4 ||
    dec.args.technology_id !== 'extraction' ||
    Number(dec.args.level) !== 3 ||
    cap.args.player_name !== before.players[actor].name ||
    Number(interim.args.id) !== 9 ||
    String(interim.args.active_player) !== sourceId ||
    Number(ending.args.id) !== 60 ||
    String(ending.args.active_player) !== sourceId
  )
    fail();
  const expected = structuredClone(before);
  expected.players[actor].resources.corn -= 6;
  expected.players[actor].resources.corn += 1;
  expected.players[actor].score += 2;
  expected.players[actor].buildings.push(card.id);
  expected.buildings = expected.buildings.filter((id) => id !== card.id);
  expected.pending = {
    title: '資源を選択（残り 2 個）',
    task: { type: 'resource', remaining: 2 },
    after: [
      { type: 'effects', effects: [{ type: 'resources', resources: { gold: 1 } }] },
      {
        type: 'build',
        remaining: 0,
        allowMonument: false,
        cornPayment: true,
        architectureAvailable: false,
      },
    ],
  };
  expected.log.push(`${before.players[actor].name} が「${card.name}」を建設`);
  if (!equal(after, expected)) fail();
  const queue = bindNewFixedReward(before.pending, after.pending, card.effects[1]);
  return {
    construct: reference(construct),
    receipt: reference(corn),
    sourceId,
    actor,
    nativeBuilding: card.id,
    sourceBuilding: structuredClone(building),
    selectedAction: structuredClone(action),
    automaticCorn: 1,
    automaticPoints: 2,
    fixedResources: { gold: 1 },
    queueBinding: queue,
    fullBuildStateVerified: true,
  };
}

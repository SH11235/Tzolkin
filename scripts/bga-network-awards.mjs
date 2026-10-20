import { isDeepStrictEqual } from 'node:util';

export const skullShortageCaption = 'Not enough Skulls available for everyone: no one get a Skull.';
export const singleSkullCaption = 'There is only 1 crystal skull left for ${player_name}';
const fail = (why) => {
  throw new Error(`Skull award: ${why}`);
};
const ref = (n) => ({
  uid: n.uid,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
});
const signature = (receipts) => receipts.map((r) => JSON.stringify(r)).sort();
const number = (x) =>
  ['string', 'number'].includes(typeof x) &&
  /^(0|[1-9]\d*)$/.test(String(x)) &&
  Number.isSafeInteger(Number(x));

const exactKeys = (value, keys) =>
  value &&
  !Array.isArray(value) &&
  typeof value === 'object' &&
  isDeepStrictEqual(Object.keys(value).sort(), [...keys].sort());

export function sourceSingleSkullWitness({
  ledger,
  group,
  before,
  observation,
  sourceToNative,
  unresolvedTiming,
  sourceStateMatched,
}) {
  const captions = group.entries.filter((n) => n.type === 'noSkullsLeft');
  if (!captions.length) return null;
  const entries = group.entries,
    [payment, selection, caption, receipt, transition, ending] = entries;
  if (
    captions.length !== 1 ||
    entries.length !== 6 ||
    captions[0] !== caption ||
    !isDeepStrictEqual(
      entries.map((n) => n.type),
      [
        'payGoods',
        'selectAction',
        'noSkullsLeft',
        'takeGoods',
        'gameStateChange',
        'gameStateChange',
      ],
    ) ||
    entries.some(
      (n, i) =>
        n.packetIndex !== caption.packetIndex ||
        n.moveId !== group.moveId ||
        n.notificationIndex !== i,
    ) ||
    !exactKeys(payment.args, [
      'type',
      'resources',
      'name',
      'player_id',
      'player_name',
      'quantity',
      'target',
    ]) ||
    !exactKeys(selection.args, ['player_name', 'city', 'action_number']) ||
    !exactKeys(caption.args, ['player_name']) ||
    !exactKeys(receipt.args, [
      'type',
      'resources',
      'player_id',
      'player_name',
      'quantity',
      'origin',
    ]) ||
    caption.singleSkullCaption !== singleSkullCaption ||
    payment.args.type !== 'corn' ||
    payment.args.name !== 'corn' ||
    payment.args.target !== null ||
    payment.args.player_name !== null ||
    !number(payment.args.quantity) ||
    Number(payment.args.quantity) !== 1 ||
    selection.args.city !== 'Yaxchilan' ||
    !number(selection.args.action_number) ||
    Number(selection.args.action_number) !== 4 ||
    receipt.args.type !== 'skull' ||
    !number(receipt.args.quantity) ||
    Number(receipt.args.quantity) !== 1 ||
    receipt.args.origin !== 'action_yaxchilan_4' ||
    Number(transition.args.id) !== 9 ||
    Number(ending.args.id) !== 7
  )
    fail('single-skull caption is outside its closed paid-action boundary');
  const sourceId = String(receipt.args.player_id),
    actor = sourceToNative.get(sourceId);
  const source = ledger.players[sourceId],
    player = before.players[actor];
  const names = Object.entries(ledger.initial.players).filter(
    ([, p]) => p.name === caption.args.player_name,
  );
  const task = before.pending?.task;
  if (
    names.length !== 1 ||
    names[0][0] !== sourceId ||
    !source ||
    !player ||
    unresolvedTiming !== false ||
    sourceStateMatched !== true ||
    ledger.stateId !== 45 ||
    ledger.activePlayer !== sourceId ||
    before.phase !== 'playing' ||
    before.additionalBuildings !== false ||
    before.expansion != null ||
    before.currentPlayer !== actor ||
    observation?.actor !== actor ||
    ledger.skullSupply !== 1 ||
    before.skullSupply !== 1 ||
    source.technologies.theology !== 3 ||
    player.technologies.theology !== 3 ||
    !isDeepStrictEqual(source.resources, player.resources) ||
    !isDeepStrictEqual(source.technologies, player.technologies) ||
    task?.type !== 'anyAction' ||
    task.cost !== 1 ||
    task.excludeSkulls !== true ||
    !isDeepStrictEqual(before.pending.after, []) ||
    String(payment.args.player_id) !== sourceId ||
    selection.args.player_name !== caption.args.player_name ||
    receipt.args.player_name !== caption.args.player_name ||
    String(transition.args.active_player) !== sourceId ||
    String(ending.args.active_player) !== sourceId
  )
    fail('single-skull source and native actor, pool or pending state differ');
  const rows = observation.legalActions.filter(
    (r) =>
      r.action.type === 'useAction' &&
      r.action.gear === 'yaxchilan' &&
      r.action.position === 4 &&
      r.action.source === 'anywhere' &&
      r.action.cornCost === 1,
  );
  if (
    rows.length !== 1 ||
    !isDeepStrictEqual(rows[0].move, { type: 'choose', choiceId: 'any:yaxchilan:4' })
  )
    fail('single-skull source choice has no unique existing legal move');
  return {
    sourceId,
    actor,
    moveId: group.moveId,
    move: structuredClone(rows[0].move),
    caption: ref(caption),
    resourceReceipt: ref(receipt),
    nominalSkulls: 2,
    actualSkulls: 1,
    cornCost: 1,
  };
}

export function bindSingleSkullOutcome(proof, before, after, steps) {
  if (!proof) return null;
  if (
    steps.length !== 1 ||
    steps[0].actor !== proof.actor ||
    !isDeepStrictEqual(steps[0].move, proof.move) ||
    !isDeepStrictEqual(steps[0].sourceActionIds, [proof.moveId]) ||
    !isDeepStrictEqual(steps[0].refills, { currentAge: [], age2: [] }) ||
    after.skullSupply !== 0 ||
    after.pending !== null ||
    !isDeepStrictEqual(Object.keys(before).sort(), Object.keys(after).sort())
  )
    fail('single-skull grant did not use its one original native action');
  for (const key of Object.keys(before))
    if (
      !['players', 'skullSupply', 'pending', 'log'].includes(key) &&
      !isDeepStrictEqual(before[key], after[key])
    )
      fail('single-skull action changed an unrelated native field');
  if (before.players.length !== after.players.length) fail('single-skull roster changed');
  for (let i = 0; i < before.players.length; i++) {
    const expected = structuredClone(before.players[i]);
    if (i === proof.actor) {
      expected.resources.corn -= 1;
      expected.resources.skull += 1;
    }
    if (!isDeepStrictEqual(expected, after.players[i]))
      fail('single-skull action differs from its exact observed resource grant');
  }
  return {
    ...proof,
    nativePoolBefore: before.skullSupply,
    nativePoolAfter: after.skullSupply,
    sourceCountersModified: false,
    qualificationRaised: false,
  };
}

export function sourceSkullShortageWitness({
  ledger,
  group,
  before,
  catalog,
  sourceToNative,
  unresolvedTiming,
}) {
  const shortages = group.entries.filter((n) => n.type === 'noskulltoday');
  if (!shortages.length) return null;
  const n = shortages[0],
    i = group.entries.indexOf(n),
    entries = group.entries;
  const state = (e, id) => e?.type === 'gameStateChange' && Number(e.args.id) === id;
  if (
    shortages.length !== 1 ||
    i < 3 ||
    !state(entries[i - 2], 93) ||
    entries[i - 1].type !== 'middleOfAgeRewards' ||
    !Array.isArray(entries[i - 1].args) ||
    entries[i - 1].args.length ||
    !state(entries.at(-1), 95) ||
    n.skullShortageCaption !== skullShortageCaption ||
    !Array.isArray(n.args) ||
    n.args.length ||
    entries
      .slice(i - 2)
      .some(
        (e, j) =>
          e.packetIndex !== n.packetIndex ||
          e.notificationIndex !== entries[i - 2].notificationIndex + j,
      )
  )
    fail('caption is outside its complete original temple award boundary');
  const prefix = entries.slice(0, i - 2);
  if (
    !isDeepStrictEqual(
      prefix.filter((e) => e.type === 'gameStateChange').map((e) => Number(e.args.id)),
      [90, 91, 92],
    ) ||
    prefix.some(
      (e) =>
        ![
          'gameStateChange',
          'takeGoods',
          'payGoods',
          'firstPlayerCorn',
          'firstPlayer',
          'foodDayCost',
          'updateScore',
        ].includes(e.type) ||
        (['takeGoods', 'payGoods'].includes(e.type) && e.args.type !== 'corn') ||
        (e.type === 'updateScore' && !(Number(e.args.points) < 0)),
    )
  )
    fail('earlier source effects do not preserve the pre-award skull and temple state');
  const day = ledger.day + 1;
  if (
    ![8, 21].includes(day) ||
    before.round !== day ||
    before.phase !== 'playing' ||
    before.pending !== null ||
    unresolvedTiming ||
    before.skullSupply !== ledger.skullSupply ||
    !Number.isSafeInteger(ledger.skullSupply) ||
    ledger.skullSupply < 0 ||
    before.foodDays.includes(day) ||
    ![3, 4].includes(before.players.length) ||
    sourceToNative.size !== before.players.length
  )
    fail('source and native pre-award boundaries are not synchronized');
  const expected = [],
    players = [],
    demands = [];
  for (const [id, player] of Object.entries(ledger.players)) {
    const actor = sourceToNative.get(id),
      nativePlayer = before.players[actor];
    if (
      !Number.isSafeInteger(actor) ||
      !nativePlayer ||
      !isDeepStrictEqual(player.temples, nativePlayer.temples)
    )
      fail('pre-award source temples differ from native');
    for (const buildingId of nativePlayer.buildings) {
      const building = catalog.BUILDINGS.find((b) => b.id === buildingId);
      if (
        !building ||
        building.effects.some((e) => ['foodReward', 'foodRewardSwitch'].includes(e.type))
      )
        fail('building rewards can change the earlier FoodDay boundary');
    }
    let skulls = 0;
    for (const [temple, level] of Object.entries(player.temples)) {
      const track = catalog.TEMPLE_TRACKS[temple];
      if (
        !track ||
        !Number.isSafeInteger(level) ||
        level < -1 ||
        level + 1 >= track.resourceRewards.length
      )
        fail('unknown source temple reward level');
      for (let step = 0; step <= level + 1; step++)
        for (const [resource, amount] of Object.entries(track.resourceRewards[step])) {
          if (resource === 'skull') skulls += amount;
          else if (amount > 0)
            expected.push({
              playerId: id,
              resource,
              quantity: amount,
              origin: `temple_${temple === 'kukulkan' ? 'kukulcan' : temple}_step_${step}`,
            });
        }
    }
    demands.push({ sourceActorId: id, skulls });
    players.push({
      sourceActorId: id,
      actor,
      temples: structuredClone(player.temples),
      skulls: nativePlayer.resources.skull,
    });
  }
  const demand = demands.reduce((sum, p) => sum + p.skulls, 0);
  if (!(demand > ledger.skullSupply))
    fail('original supply is not below the complete temple demand');
  const receipts = entries.slice(i + 1, -1);
  const actual = receipts.map((e) => {
    if (
      e.type !== 'takeGoods' ||
      !sourceToNative.has(String(e.args.player_id)) ||
      !number(e.args.quantity)
    )
      fail('unknown temple reward receipt');
    return {
      playerId: String(e.args.player_id),
      resource: e.args.type,
      quantity: Number(e.args.quantity),
      origin: e.args.origin,
    };
  });
  if (!isDeepStrictEqual(signature(expected), signature(actual)))
    fail('complete original temple reward receipts differ from the catalog');
  return {
    source: ref(n),
    boundary: [ref(entries[i - 2]), ref(entries.at(-1))],
    sourcePhysicalDay: ledger.day,
    scheduledDay: day,
    sourceSkullSupply: ledger.skullSupply,
    totalSkullDemand: demand,
    demands,
    players,
    receipts: receipts.map(ref),
    expectedRewards: expected,
    originalCaption: n.skullShortageCaption,
  };
}

export function bindSkullShortageOutcome(proof, before, after) {
  if (!proof) return null;
  if (
    !isDeepStrictEqual(after.foodDays, [...before.foodDays, proof.scheduledDay]) ||
    after.skullSupply !== proof.sourceSkullSupply ||
    proof.players.some((p) => after.players[p.actor].resources.skull !== p.skulls)
  )
    fail('native FoodDay does not retain the witnessed shortage outcome');
  return {
    ...proof,
    nativeFoodDaysBefore: structuredClone(before.foodDays),
    nativeFoodDaysAfter: structuredClone(after.foodDays),
    nativeSkullSupplyBefore: before.skullSupply,
    nativeSkullSupplyAfter: after.skullSupply,
    individualSkullsUnchanged: true,
  };
}

import { isDeepStrictEqual } from 'node:util';

export const skullShortageCaption = 'Not enough Skulls available for everyone: no one get a Skull.';
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

import { isDeepStrictEqual } from 'node:util';
import { originalHarvestEntry } from './bga-network.mjs';

const fail = (why) => {
  throw new Error(`Network burn harvest: ${why}`);
};
const equal = isDeepStrictEqual;
const keys = (x, names) => x && equal(Object.keys(x).sort(), [...names].sort());
const stateKeys = (n) =>
  Object.keys(n.args).every((k) =>
    ['id', 'active_player', 'args', 'type', 'reflexion', 'updateGameProgression'].includes(k),
  );
const emptyArgs = (x) => x === null || equal(x, []);
const integer = (x) =>
  ['string', 'number'].includes(typeof x) &&
  /^(0|[1-9]\d*)$/.test(String(x)) &&
  Number.isSafeInteger(Number(x));
const godName = (x) => (x === 'kukulcan' ? 'kukulkan' : x);

export function sourceBurnHarvestWitness({
  index,
  group,
  ledger,
  before,
  observation,
  sourceToNative,
  sourceStateMatched,
  unresolvedTiming,
}) {
  if (!group.entries.some((n) => n.type === 'takeTile' && n.args.player_id === null)) return null;
  if (!sourceStateMatched || unresolvedTiming) return null;
  const previous = index.groups[index.groups.indexOf(group) - 1];
  const entries = group.entries;
  const [discard, penalty, tile, goods, transition, ending] = entries;
  const captions = [
    '',
    '${player_name} burn the forest and move down 1 step in ${temple} temple',
    '${player_name} takes a ${name} tile',
    '${player_name} takes ${resources}',
    '',
    '',
  ];
  if (
    !previous ||
    previous.moveId + 1 !== group.moveId ||
    !equal(
      previous.entries.map((n) => n.type),
      ['pickUpWorker', 'gameStateChange', 'gameStateChange'],
    ) ||
    !equal(
      entries.map((n) => n.type),
      ['takeTile', 'alterTemple', 'takeTile', 'takeGoods', 'gameStateChange', 'gameStateChange'],
    ) ||
    entries.some((n, i) => {
      const raw = originalHarvestEntry(n);
      return (
        !raw ||
        raw.packetType !== 'resend' ||
        raw.caption !== captions[i] ||
        n.notificationIndex !== i ||
        n.packetIndex !== discard.packetIndex ||
        n.moveId !== group.moveId ||
        n.channel !== `/table/t${index.tableId}`
      );
    })
  )
    fail('incomplete original harvest boundary');
  const [pickup, priorTransition, prompt] = previous.entries;
  const prior = prompt.args;
  const operator = String(prior.active_player);
  const actor = sourceToNative.get(operator);
  const player = before.players[actor];
  const source = ledger.players[operator];
  const position = Number(discard.args.action);
  const god = godName(penalty.args.temple_id);
  const named = (n) =>
    String(n.args.player_id) === operator &&
    n.args.player_name === index.initial.players[operator]?.name;
  if (
    previous.entries.some((n, i) => {
      const raw = originalHarvestEntry(n);
      return (
        !raw ||
        raw.packetType !== 'resend' ||
        n.packetIndex !== pickup.packetIndex ||
        n.notificationIndex !== i ||
        n.channel !== `/table/t${index.tableId}` ||
        raw.caption !==
          (i === 0
            ? '${player_name} picks up a worker in ${city} (position ${position}) and plays action n°${action_number}'
            : '')
      );
    }) ||
    !keys(pickup.args, [
      'player_id',
      'player_name',
      'worker_id',
      'city',
      'position',
      'action_number',
    ]) ||
    !named(pickup) ||
    pickup.args.city !== 'Palenque' ||
    Number(pickup.args.position) !== position ||
    Number(pickup.args.action_number) !== position ||
    String(ledger.workers.get(String(pickup.args.worker_id))?.player) !== operator ||
    Number(priorTransition.args.id) !== 9 ||
    String(priorTransition.args.active_player) !== operator ||
    Number(prior.id) !== 10 ||
    prior.type !== 'activeplayer' ||
    ledger.stateId !== 10 ||
    ledger.activePlayer !== operator ||
    actor === undefined ||
    before.currentPlayer !== actor ||
    observation.actor !== actor ||
    !player ||
    !source ||
    ![priorTransition, prompt, transition, ending].every(stateKeys) ||
    !emptyArgs(priorTransition.args.args) ||
    !emptyArgs(transition.args.args) ||
    !keys(ending.args.args, ['workers']) ||
    !keys(prior.args, [
      'corn',
      'wood',
      'corn_tile_available',
      'agriculture_level_2',
      'can_burn',
      'temples',
    ]) ||
    prior.args.can_burn !== true ||
    prior.args.corn_tile_available !== false ||
    prior.args.agriculture_level_2 !== player.technologies.agriculture >= 2 ||
    !integer(prior.args.corn) ||
    !integer(prior.args.wood) ||
    Number(prior.args.wood) <= 0 ||
    !keys(prior.args.temples, ['chaac', 'quetzalcoatl', 'kukulcan']) ||
    Object.entries(prior.args.temples).some(
      ([t, n]) => !integer(n) || Number(n) - 1 !== player.temples[godName(t)],
    ) ||
    !equal(before.pending?.task, { type: 'palenque', position }) ||
    !equal(before.pending?.after, []) ||
    ![3, 4, 5].includes(position) ||
    !keys(discard.args, ['player_id', 'player_name', 'type', 'action', 'tile_number']) ||
    discard.args.player_id !== null ||
    discard.args.player_name !== '' ||
    discard.args.type !== 'wood_tile' ||
    !integer(discard.args.action) ||
    !integer(discard.args.tile_number) ||
    Number(discard.args.tile_number) !== before.jungle[position]?.wood ||
    before.jungle[position]?.wood <= 0 ||
    before.jungle[position].corn !== before.jungle[position].wood ||
    !equal(ledger.jungle[position], before.jungle[position]) ||
    !keys(penalty.args, ['temple_id', 'temple', 'player_id', 'player_name', 'step']) ||
    !['chaac', 'quetzalcoatl', 'kukulcan'].includes(penalty.args.temple_id) ||
    penalty.args.temple !==
      { chaac: 'Chaac', quetzalcoatl: 'Quetzalcoatl', kukulcan: 'Kukulcan' }[
        penalty.args.temple_id
      ] ||
    !named(penalty) ||
    !integer(penalty.args.step) ||
    Number(penalty.args.step) - 1 !== player.temples[god] - 1 ||
    !keys(tile.args, [
      'i18n',
      'player_id',
      'player_name',
      'name',
      'type',
      'action',
      'tile_number',
    ]) ||
    !equal(tile.args.i18n, ['name']) ||
    !named(tile) ||
    tile.args.type !== 'corn_tile' ||
    tile.args.name !== 'corn' ||
    tile.args.action !== discard.args.action ||
    tile.args.tile_number !== discard.args.tile_number ||
    !keys(goods.args, ['type', 'resources', 'player_id', 'player_name', 'quantity', 'origin']) ||
    !named(goods) ||
    goods.args.type !== 'corn' ||
    !integer(goods.args.quantity) ||
    Number(goods.args.quantity) <= 0 ||
    goods.args.origin !== `action_palenque_${position}` ||
    Number(goods.args.quantity) !== Number(prior.args.corn) ||
    Number(transition.args.id) !== 9 ||
    String(transition.args.active_player) !== operator ||
    transition.args.type !== 'game' ||
    Number(ending.args.id) !== 7 ||
    String(ending.args.active_player) !== operator ||
    ending.args.type !== 'activeplayer'
  )
    fail('source operator, discarded tile, reward or pending differs');
  const legal = observation.legalActions.filter(
    (r) =>
      r.action.type === 'harvest' && r.action.kind === 'burn' && r.action.position === position,
  );
  if (legal.length !== 1) fail('source burn lacks its unique legal move');
  return {
    actor,
    sourceId: operator,
    position,
    god,
    corn: Number(goods.args.quantity),
    moveId: group.moveId,
    move: structuredClone(legal[0].move),
    source: entries.map(({ uid, packetIndex, notificationIndex }) => ({
      uid,
      packetIndex,
      notificationIndex,
    })),
  };
}

export function bindBurnHarvestOutcome(proof, before, middle, after, steps) {
  const pending = middle.snapshot.state.pending;
  if (
    !equal(pending?.task, { type: 'temple', remaining: 1, direction: -1, reason: 'burn' }) ||
    !equal(pending.after, []) ||
    middle.observation.actor !== proof.actor ||
    steps.length !== 2 ||
    steps.some(
      (s) =>
        s.actor !== proof.actor ||
        !equal(s.sourceActionIds, [proof.moveId]) ||
        !equal(s.refills, { currentAge: [], age2: [] }),
    ) ||
    !equal(steps[0].move, proof.move)
  )
    fail('burn changed its native continuation');
  const rows = middle.observation.legalActions.filter(
    (r) => r.action.type === 'temple' && r.action.temple === proof.god && r.action.direction === -1,
  );
  if (rows.length !== 1 || !equal(steps[1].move, rows[0].move))
    fail('burn penalty lacks its unique named legal move');
  const expected = structuredClone(before);
  expected.players[proof.actor].resources.corn += proof.corn;
  expected.players[proof.actor].cornTiles++;
  expected.jungle[proof.position].corn--;
  expected.jungle[proof.position].wood--;
  expected.pending = structuredClone(pending);
  if (!equal(expected, middle.snapshot.state)) fail('burn changed an unrelated intermediate field');
  expected.players[proof.actor].temples[proof.god]--;
  expected.pending = null;
  if (!equal(expected, after)) fail('burn differs from its complete observed effect');
  return {
    ...proof,
    sourceCountersModified: false,
    qualificationRaised: false,
    intermediateObservationCertified: false,
  };
}

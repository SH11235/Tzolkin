const scheduledDays = [8, 14, 21, 27];
const ref = (n, moveId) => ({
  moveId,
  packetIndex: n.packetIndex,
  notificationIndex: n.notificationIndex,
  uid: n.uid,
  type: n.type,
});
const integer = (value, signed = false) =>
  ['number', 'string'].includes(typeof value) &&
  (signed ? /^(0|-?[1-9]\d*)$/ : /^(0|[1-9]\d*)$/).test(String(value)) &&
  Number.isSafeInteger(Number(value));
const fail = (reason) => {
  throw new Error(`Source feeding: ${reason}`);
};

/** Literal source receipt binding, not a claim about native per-player feeding events. */
export function sourceFeedingWitnesses(index) {
  const events = [],
    consumed = new Set();
  let day = Number(index.initial.day);
  for (const group of index.groups)
    for (const [i, n] of group.entries.entries()) {
      if (n.type === 'rotateWheel') day = Number(n.args.day);
      if (n.type !== 'gameStateChange' || Number(n.args.id) !== 92) continue;
      const scheduledDay = scheduledDays[events.length];
      if (scheduledDay === undefined) fail('too many original feeding boundaries');
      const endIndex = group.entries.findIndex((e, j) => j > i && e.type === 'gameStateChange');
      const end = endIndex < 0 ? null : group.entries[endIndex];
      if (end && Number(end.args.id) === 92) fail('duplicate original feeding boundary');
      const expectedEnd = [8, 21].includes(scheduledDay) ? 93 : 94;
      const missing = [];
      if (!end || Number(end.args.id) !== expectedEnd)
        missing.push({
          code: 'unsupportedBoundary',
          field: 'gameStateChange.id',
          expected: expectedEnd,
          observed: end?.args.id ?? null,
        });
      const entries = group.entries.slice(i + 1, endIndex < 0 ? group.entries.length : endIndex);
      const actors = new Map(
        Object.entries(index.initial.players).map(([id, p]) => [
          id,
          { sourceActorId: id, name: p.name, pay: [], cost: [], loss: [], penalty: [] },
        ]),
      );
      for (const [offset, e] of entries.entries()) {
        if (!['payGoods', 'updateScore', 'foodDayCost'].includes(e.type))
          fail('unsupported effect inside the original feeding stage');
        const a = e.args,
          actor = actors.get(String(a.player_id));
        if (
          !actor ||
          (a.player_name !== null && a.player_name !== undefined && a.player_name !== actor.name)
        )
          fail('receipt actor/name is not the original roster');
        const position = i + 1 + offset,
          source = ref(e, group.moveId);
        if (e.type === 'payGoods') {
          if (a.type !== 'corn' || a.target !== null || !integer(a.quantity))
            fail('invalid original corn payment');
          actor.pay.push({ amount: Number(a.quantity), position, source });
        } else if (e.type === 'updateScore') {
          if (!integer(a.points, true) || Number(a.points) >= 0)
            fail('invalid original feeding loss');
          actor.loss.push({ amount: -Number(a.points), position, source });
        } else {
          consumed.add(e.uid);
          if (Object.hasOwn(a, 'cost') === Object.hasOwn(a, 'points') || !integer(a.nbr))
            fail('ambiguous original feeding caption');
          const field = Object.hasOwn(a, 'cost') ? 'cost' : 'points';
          if (!integer(a[field])) fail('invalid original feeding caption amount');
          actor[field === 'cost' ? 'cost' : 'penalty'].push({
            amount: Number(a[field]),
            workers: Number(a.nbr),
            position,
            source,
          });
        }
      }
      const receipts = [];
      for (const actor of actors.values()) {
        for (const [effect, caption] of [
          ['pay', 'cost'],
          ['loss', 'penalty'],
        ]) {
          const effects = actor[effect],
            captions = actor[caption];
          if (effects.length > 1 || captions.length > 1)
            fail('duplicate per-actor feeding receipt');
          if (effects.length !== captions.length)
            missing.push({
              code: 'sourceMissing',
              sourceActorId: actor.sourceActorId,
              field: effects.length ? caption : effect,
            });
          else if (
            effects.length &&
            (effects[0].amount !== captions[0].amount ||
              effects[0].position >= captions[0].position)
          )
            fail('payment/loss and original caption disagree in amount or order');
        }
        receipts.push({
          sourceActorId: actor.sourceActorId,
          state:
            actor.pay.length || actor.loss.length || actor.cost.length || actor.penalty.length
              ? 'observedSourceEntries'
              : 'noIndividualReceipt',
          paidCorn: actor.pay[0]?.amount ?? null,
          penaltyPoints: actor.loss[0]?.amount ?? null,
          payments: actor.pay,
          costCaptions: actor.cost,
          losses: actor.loss,
          penaltyCaptions: actor.penalty,
          nativeIndividualReceiptMatch: 'notConnected',
        });
      }
      events.push({
        schema: 'tzolkin-bga-source-feeding-witness-v1',
        scheduledDay,
        sourcePhysicalDay: day,
        sourceActualRound: day + 1,
        moveId: group.moveId,
        start: ref(n, group.moveId),
        end: end ? ref(end, group.moveId) : null,
        eligible: missing.length === 0,
        classification: missing.length ? missing[0].code : 'sourceStageBound',
        missing,
        receipts,
        sourceRawSha256: index.rawSha256,
        scheduleBasis:
          'Ordinal explicit source state92 boundaries under the declared basic profile; physical day is retained separately.',
        scope:
          'Original stage payments and losses are paired with original captions before temple rewards. Missing individual entries stay null, never inferred as zero. Native internal receipts remain unconnected.',
      });
    }
  for (const group of index.groups)
    for (const e of group.entries)
      if (e.type === 'foodDayCost' && !consumed.has(e.uid))
        fail('feeding caption outside an explicit original feeding boundary');
  return events;
}

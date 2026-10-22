import { originalNetworkSequence, originalPacketEntry } from './bga-network.mjs';

export function terminalRoundSourceClaim(index, group) {
  const sequence = originalNetworkSequence(index);
  if (!sequence || index.groups.at(-1) !== group) return null;
  let claims = [],
    amount = Number(index.initial.cornOnWheel);
  for (const g of index.groups) {
    if (g === group) break;
    for (const n of g.entries) {
      if (n.type === 'rotateWheel') claims = [];
      if (n.type === 'cornOnWheel') amount++;
      if (n.type === 'firstPlayer') amount = 0;
      if (n.type === 'placeWorker' && n.args.place === 'startingPlayer')
        claims.push({ entry: n, amount });
    }
  }
  if (claims.length !== 1) return null;
  const claim = claims[0],
    packet = originalPacketEntry(claim.entry);
  return packet?.packetType === 'resend' &&
    packet.tableId === index.tableId &&
    packet.rawSha256 === index.rawSha256 &&
    claim.entry.channel === `/table/t${index.tableId}`
    ? { ...claim, currentAmount: amount }
    : null;
}

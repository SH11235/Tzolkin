import assert from 'node:assert/strict';
import test from 'node:test';
import { terminalComponentFixture } from './fixtures/bga-terminal-components.mjs';
import { inspectTerminalProjection } from './bga-network-terminal.mjs';
import { finalMonumentMetric, knownFinalMetricRefusal } from './bga-network-final-scoring.mjs';

const technologies = { agriculture: 3, extraction: 3, architecture: 0, theology: 0 };
const temples = { chaac: 2, quetzalcoatl: 2, kukulkan: 4 };
function fixture(changeRaw = () => {}, negativeTemples = false) {
  const f = terminalComponentFixture({
    changeRaw(raw) {
      const entries = raw.gameLogs.data.data.at(-1).data;
      let uid = 1000;
      const pair = (points, type, extra = {}, log = '', id = '1') => [
        {
          uid: String(++uid),
          type: 'updateScore',
          log: '',
          args: { player_id: id, player_name: null, points },
        },
        {
          uid: String(++uid),
          type,
          log,
          args: { player_id: id, player_name: 'P' + id, points, ...extra },
        },
      ];
      for (let i = entries.length - 1; i >= 0; i--)
        if (entries[i].type === 'templeLeader') entries.splice(i - 1, 2);
      const temple = entries.find((n) => n.type === 'templeScore' && n.args.player_id === '1');
      temple.args.points = negativeTemples ? -6 : 13;
      entries[entries.indexOf(temple) - 1].args.points = temple.args.points;
      entries.splice(
        entries.findIndex((n) => n.type === 'templeScore') - 1,
        0,
        ...(negativeTemples
          ? ['2', '3'].flatMap((id) => [
              ...pair(1, 'templeLeader', { temple: 'chaac' }, '', id),
              ...pair(3, 'templeLeader', { temple: 'quetzalcoatl' }, '', id),
              ...pair(2, 'templeLeader', { temple: 'kukulcan' }, '', id),
            ])
          : [
              ...pair(2, 'templeLeader', { temple: 'chaac' }),
              ...pair(6, 'templeLeader', { temple: 'quetzalcoatl' }),
              ...pair(4, 'templeLeader', { temple: 'kukulcan' }),
            ]),
      );
      const resource = entries.findIndex(
        (n) => n.type === 'finalScoring' && n.log.startsWith('Remaining resources:'),
      );
      entries.splice(
        resource - 1,
        0,
        ...pair(
          20,
          'finalScoring',
          { nbr: 2 },
          'Monument: ${player_name} scores ${points} for ${nbr} technologies at 3rd level',
        ),
        ...pair(
          negativeTemples ? -6 : 13,
          'finalScoring',
          {},
          'Monument: ${player_name} scores ${points} for points in the temples',
        ),
      );
      changeRaw(raw);
    },
  });
  for (const p of [
    f.context.source.players['1'],
    f.ledger.players['1'],
    f.context.native.players[0],
    f.native.players[0],
  ]) {
    p.monuments = ['m07', 'm11'];
    p.technologies = structuredClone(technologies);
    p.temples = negativeTemples
      ? { chaac: -1, quetzalcoatl: -1, kukulkan: -1 }
      : structuredClone(temples);
  }
  f.context.native.players[0].templePoints = 7;
  f.native.players[0].templePoints = negativeTemples ? 1 : 20;
  f.native.players.forEach((p, i) => {
    p.score = (negativeTemples ? [102.25, 99.25, 98.25] : [152.25, 93.25, 92.25])[i];
  });
  f.native.finalScores.forEach((s, i) => {
    s.pointsBeforeFinal = (negativeTemples ? [85, 96, 95] : [116, 90, 89])[i];
    s.monumentPoints = i === 0 ? (negativeTemples ? 14 : 33) : 0;
    s.total = f.native.players[i].score;
  });
  f.terminalDisplayCheckpoint.scores.forEach((s) => {
    s.total = Math.floor(f.native.finalScores[s.playerId].total);
  });
  return f;
}

test('technology and current-temple monuments bind original source receipts through the formal gate', () => {
  const f = fixture(),
    result = inspectTerminalProjection(f);
  assert.equal(result.completed, true, JSON.stringify(result));
  assert.deepEqual(
    result.components.finalScoring.players[0].monuments.map((m) => [m.monument, m.points]),
    [
      ['m07', 20],
      ['m11', 13],
    ],
  );
  assert.equal(Object.hasOwn(result.components.finalScoring.players[0].monuments[1], 'nbr'), false);
  assert.equal(f.native.players[0].templePoints, 20);
});

test('max-technology monument uses all four capped original tracks', () => {
  const f = fixture(),
    p = f.context.native.players[0],
    card = f.catalog.MONUMENTS.find((m) => m.id === 'm07');
  for (const [count, points] of [0, 9, 20, 33, 33].entries()) {
    p.technologies = Object.fromEntries(
      Object.keys(technologies).map((t, i) => [t, i < count ? 3 : 2]),
    );
    assert.deepEqual(finalMonumentMetric(f.context.native, p, card, f.catalog), {
      nbr: count,
      points,
    });
  }
  p.technologies.theology = 4;
  assert.throws(() => finalMonumentMetric(f.context.native, p, card, f.catalog), /technology/);
});

test('signed temple monument receipt preserves negative current-position points through the formal gate', () => {
  const result = inspectTerminalProjection(fixture(() => {}, true));
  assert.equal(result.completed, true, JSON.stringify(result));
  assert.equal(
    result.components.finalScoring.players[0].monuments.find((m) => m.monument === 'm11').points,
    -6,
  );
});

test('temple monument uses current signed position points rather than cumulative points or leader awards', () => {
  const f = fixture(),
    p = f.context.native.players[0],
    card = f.catalog.MONUMENTS.find((m) => m.id === 'm11');
  p.templePoints = 99;
  assert.deepEqual(finalMonumentMetric(f.context.native, p, card, f.catalog), { points: 13 });
  p.temples = { chaac: -1, quetzalcoatl: -1, kukulkan: -1 };
  assert.deepEqual(finalMonumentMetric(f.context.native, p, card, f.catalog), { points: -6 });
  p.temples = { chaac: 5, quetzalcoatl: 7, kukulkan: 6 };
  assert.deepEqual(finalMonumentMetric(f.context.native, p, card, f.catalog), { points: 31 });
  p.temples.chaac = 6;
  assert.throws(
    () => finalMonumentMetric(f.context.native, p, card, f.catalog),
    /temple positions/,
  );
});

test('equal-total monument cancellation is rejected before completion', () => {
  const f = fixture((raw) => {
    const entries = raw.gameLogs.data.data.at(-1).data;
    const rows = entries.filter((n) => n.type === 'finalScoring' && n.log.startsWith('Monument:'));
    for (const [i, n] of rows.entries()) {
      n.args.points += i === 0 ? -1 : 1;
      entries[entries.indexOf(n) - 1].args.points = n.args.points;
    }
  });
  const result = inspectTerminalProjection(f);
  assert.equal(
    result.scores.every((r) => r.matched),
    true,
  );
  assert.equal(result.completed, false);
});

test('monument receipt identity rejects unknown captions artificial quantities actors missing and duplicate rows', () => {
  for (const kind of ['caption', 'quantity', 'actor', 'missing', 'duplicate']) {
    const f = fixture((raw) => {
      const entries = raw.gameLogs.data.data.at(-1).data;
      const at = entries.findIndex(
        (n) => n.type === 'finalScoring' && n.log.endsWith('points in the temples'),
      );
      if (kind === 'caption')
        entries[at].log = 'Monument: ${player_name} scores ${points} for something else';
      if (kind === 'quantity') entries[at].args.nbr = 13;
      if (kind === 'actor') {
        entries[at].args.player_id = '2';
        entries[at].args.player_name = 'P2';
        entries[at - 1].args.player_id = '2';
      }
      if (kind === 'missing') entries.splice(at - 1, 2);
      if (kind === 'duplicate') {
        const pair = structuredClone(entries.slice(at - 1, at + 1));
        pair[0].uid = '9000';
        pair[1].uid = '9001';
        entries.splice(at + 1, 0, ...pair);
      }
    });
    assert.equal(inspectTerminalProjection(f).completed, false, kind);
  }
});

test('new monument metrics reject changed catalog identities and ambiguous ownership', () => {
  for (const id of ['m07', 'm11'])
    for (const kind of ['card', 'ownership', 'duplicate']) {
      const f = fixture(),
        p = f.context.native.players[0],
        card = structuredClone(f.catalog.MONUMENTS.find((m) => m.id === id));
      if (kind === 'card') card.cost = {};
      if (kind === 'ownership') p.monuments = p.monuments.filter((m) => m !== id);
      if (kind === 'duplicate') p.monuments.push(id);
      assert.throws(
        () => finalMonumentMetric(f.context.native, p, card, f.catalog),
        (e) => {
          assert.equal(knownFinalMetricRefusal(e), null);
          return true;
        },
      );
    }
});

test('all thirteen monument metrics retain their independently counted public quantities', () => {
  const f = fixture(),
    catalog = f.catalog;
  const expected = {
    m01: 8,
    m02: 8,
    m03: 8,
    m04: 12,
    m05: 10,
    m06: 8,
    m07: 20,
    m08: 21,
    m09: 12,
    m10: 8,
    m11: 13,
    m12: 12,
    m13: 9,
  };
  for (const card of catalog.MONUMENTS) {
    const player = {
      monuments: [card.id],
      buildings: ['graveyard', 'municipal', 'shrine'].map(
        (c) => catalog.BUILDINGS.find((b) => b.category === c).id,
      ),
      workers: 5,
      technologies: { agriculture: 3, extraction: 3, architecture: 1, theology: 0 },
      temples: structuredClone(temples),
      cornTiles: 2,
      woodTiles: 3,
    };
    const state = {
      players: [player, { monuments: [card.id === 'm09' ? 'm06' : 'm09'] }, { monuments: [] }],
      monuments: [],
      skullSpaces: [null, 0, 1, 2, ...Array(6).fill(null)],
    };
    assert.equal(
      finalMonumentMetric(state, player, card, catalog).points,
      expected[card.id],
      card.id,
    );
  }
});

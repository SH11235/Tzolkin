import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { decodeConditionalPublicGame, decodePublicGame } from './bga-decode.mjs';
import {
  ATLAS_SCHEMA,
  CONTROL_SCHEMA,
  LATER_WITNESS_SCHEMA,
  decodeProjectedInitial,
  projectCapturedInitial,
  projectPlayingFiles,
} from './bga-playing-project.mjs';

const bytes = (value) => Buffer.from(JSON.stringify(value));
const sha = (value) => createHash('sha256').update(value).digest('hex');
const pin = (value) => ({ bytes: value.length, sha256: sha(value) });
const catalogBytes = readFileSync(
  new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url),
);
const GEARS = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'];
const resourceNames = ['corn', 'wood', 'stone', 'gold', 'skull'];
const markerNames = ['agriculture', 'extraction', 'architecture', 'theology'];
const templeNames = ['chaac', 'quetzalcoatl', 'kukulcan'];
const element = (tag, id, content = '', attributes = '') =>
  `<${tag} id="${id}"${attributes}>${content}</${tag}>`;
const dom = (id, html, rect = { x: 0, y: 0, width: 20, height: 20 }) => ({
  id,
  html,
  visible: true,
  rect,
});

// Independent synthetic counters for w01+w02, w03+w04, w05+w06, w07+w08.
// Sprite coordinates and geometry are invented, not copied from a public game.
function fixture(n) {
  const ids = ['100', '200', '300', '400'].slice(0, n);
  const names = ['A', 'B', 'C', 'D'].slice(0, n);
  const resources = [
    [8, 0, 0, 1, 0],
    [9, 0, 2, 1, 0],
    [2, 3, 0, 0, 0],
    [7, 3, 1, 0, 0],
  ];
  const technology = [
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
  ];
  const temples = [
    [0, 2, 0],
    [0, 0, 0],
    [0, 0, 2],
    [0, 0, 0],
  ];
  const counters = [],
    selected = [],
    tech = [],
    temple = [];
  const playersHtml = ids
    .map((id, index) => {
      const values = [
        ['worker', 3],
        ...resourceNames.map((name, j) => [name, resources[index][j]]),
      ];
      const counterHtml = values
        .map(([name, value]) => {
          const key = `${name}_number_player_${id}`;
          const html = element('span', key, String(value));
          counters.push({ ...dom(key, html), valueText: String(value) });
          return html;
        })
        .join('');
      const first = element(
        'div',
        `firstplayer_${id}`,
        index === 0 ? element('div', 'firstplayer') : '',
      );
      return element(
        'div',
        `overall_player_board_${id}`,
        element('div', `player_name_${id}`, `<a style="color: #008000">${names[index]}</a>`) +
          element('span', `player_score_${id}`, '0') +
          counterHtml +
          first,
      );
    })
    .join('');
  for (let index = 0; index < n; index++) {
    const id = ids[index];
    const tiles = [index * 2 + 1, index * 2 + 2]
      .map((serial) =>
        element(
          'div',
          `wealth_tile_${serial}`,
          '',
          ` class="wealth_tile" style="background-position: ${serial * -10}px 0px;"`,
        ),
      )
      .join('');
    const key = `player_wealthtiles_${id}`;
    selected.push(dom(key, element('div', key, tiles, ' class="player_wealthtiles"')));
    for (let j = 0; j < markerNames.length; j++) {
      const key = `marker_${id}_technology_${markerNames[j]}`;
      tech.push(
        dom(
          key,
          element('div', key, '', ` class="marker" style="left: ${technology[index][j] * 10}px;"`),
        ),
      );
    }
    for (let j = 0; j < templeNames.length; j++) {
      const key = `marker_${id}_temple_${templeNames[j]}`;
      temple.push(
        dom(
          key,
          element(
            'div',
            key,
            '',
            ` class="marker" style="top: ${100 - temples[index][j] * 10}px;"`,
          ),
        ),
      );
    }
  }
  const gearNodes = ['central', ...GEARS].map((gear) => {
    const id = `gear_${gear}`;
    return dom(
      id,
      element('img', id, '', ' style="transform: rotate(0deg);"').replace('</img>', ''),
    );
  });
  const places = [
    'workerplace_startingPlayer_0',
    ...GEARS.flatMap((gear) =>
      Array.from(
        { length: gear === 'chichenItza' ? 13 : 10 },
        (_v, i) => `workerplace_${gear}_${i}`,
      ),
    ),
  ].map((id, index) =>
    dom(id, element('div', id, '', ' class="workerplace"'), {
      x: index * 25,
      y: 50,
      width: 20,
      height: 20,
    }),
  );
  const workers =
    n === 3
      ? // These slots can be produced by the unselected w07, w09 and w13.
        [11, 16, 22, 27, 32, 37].map((placeIndex, index) => {
          const id = `worker_${index + 1}`;
          return dom(id, element('div', id, '', ' class="worker worker_dummy"'), {
            x: places[placeIndex].rect.x + 2,
            y: 45,
            width: 10,
            height: 15,
          });
        })
      : [];
  const skullSpaces = Array.from({ length: 9 }, (_v, i) => {
    const id = `skullplace_${i + 1}`;
    return dom(id, element('div', id, '', ' class="skullplace"'));
  });
  const skullHtml = element('div', 'skull_left_wrap', element('span', 'skull_left', '13'));
  const jungle = [2, 3, 4, 5]
    .map((box) =>
      element(
        'div',
        `jungle_${box}`,
        Array.from({ length: n }, (_v, i) =>
          element(
            'div',
            `jungle_${box}_tile_${i + 1}`,
            '',
            ` class="board_tile ${box === 2 ? 'corn_tile' : 'wood_tile'}"`,
          ),
        ).join(''),
      ),
    )
    .join('');
  const buildingsHtml = element(
    'div',
    'buildings',
    ['b01', 'b02', 'b03', 'b04', 'b05', 'b06']
      .map((_id, index) =>
        element(
          'div',
          `building_age_1_${index + 1}`,
          '',
          ` class="construction" style="background-position: ${index < 2 ? 0 : index * -10}px -10px;"`,
        ),
      )
      .join(''),
  );
  const monumentsHtml = element(
    'div',
    'monuments',
    Array.from({ length: n + 2 }, (_v, index) =>
      element(
        'div',
        `monument_${index + 1}`,
        '',
        ` class="construction" style="background-position: ${index * -10}px -20px;"`,
      ),
    ).join(''),
  );
  const boardHtml = element(
    'div',
    'board',
    [...gearNodes, ...places, ...workers, ...tech, ...temple, ...skullSpaces]
      .map((node) => node.html)
      .join('') +
      skullHtml +
      jungle +
      element('span', 'current_round', '1'),
  );
  const capture = {
    schema:
      n === 3
        ? 'tzolkin-bga-public-setup-complete-playing-board-v1'
        : 'tzolkin-bga-public-playing-initial-board-witness-v1',
    tableId: '123',
    viewPlayerId: '100',
    move: 'Move #7',
    url: 'https://boardgamearena.com/archive/replay/fixture/?table=123&player=100&comments=',
    header: 'Aはワーカーを置くか取り除くかしてください',
    roundsText: '経過日 1 / 27',
    offerNodesRemaining: 0,
    gameAreaHtml: element(
      'div',
      'game_play_area',
      boardHtml +
        selected.map((node) => node.html).join('') +
        buildingsHtml +
        monumentsHtml +
        '<script type="text/javascript">const template = `<div id="worker_${id}" class="worker worker_dummy"></div><div id="wealth_tile_${id}" class="wealth_tile"></div>`;</script>',
    ),
    playerBoardsHtml: element('div', 'player_boards', playersHtml),
    gears: gearNodes,
    gearWorkerNodes: workers,
    workerPlaces: places,
    boardGeometry: dom('board', boardHtml, { x: 0, y: 0, width: 1500, height: 300 }),
    publicCounters: counters,
    selectedPlayerTiles: selected,
    technologyMarkers: tech,
    templeMarkers: temple,
    skullRemainingHtml: skullHtml,
    skullSpaces,
    buildingsHtml,
    monumentsHtml,
    marketDomNodes: [],
  };
  const atlas = {
    schema: ATLAS_SCHEMA,
    archiveRelease: 'fixture',
    catalogSha256: sha(catalogBytes),
    faces: [
      ...Array.from({ length: n * 2 }, (_v, i) => ({
        kind: 'wealth',
        position: [(i + 1) * -10, 0],
        catalogIds: [`w${String(i + 1).padStart(2, '0')}`],
      })),
      { kind: 'building', position: [0, -10], catalogIds: ['b01', 'b02'] },
      ...[2, 3, 4, 5].map((i) => ({
        kind: 'building',
        position: [i * -10, -10],
        catalogIds: [`b${String(i + 1).padStart(2, '0')}`],
      })),
      ...Array.from({ length: n + 2 }, (_v, i) => ({
        kind: 'monument',
        position: [i * -10, -20],
        catalogIds: [`m${String(i + 1).padStart(2, '0')}`],
      })),
    ],
    markerPositions: {
      templeTop: [0, 1, 2].map((level) => ({ level, pixels: 100 - level * 10 })),
      technologyLeft: [0, 1].map((level) => ({ level, pixels: level * 10 })),
    },
    geometry:
      n === 3
        ? {
            captureSha256: '',
            coordinateSpace: 'retained-client-rect',
            boardSize: { width: 1500, height: 300 },
            workerSize: { width: 10, height: 15 },
            offset: { x: 2, y: -5 },
            tolerance: 0.01,
          }
        : null,
  };
  const raw = {
    schema_version: '1.0',
    table_id: '123',
    source_type: 'public_ui_visible_log',
    export_scope: 'all_observed_log_entries',
    source_url: 'https://boardgamearena.com/gamereview?table=123',
    captured_on: 'synthetic',
    entry_count: 1,
    entries: [{ action_id: 7, raw_text: '', timestamp_display: '12:00:00' }],
    dom_entries: [{ action_id: 7, timestamp_display: '12:00:00', messages: [] }],
    metadata: {
      table_id: '123',
      player_count: n,
      players: names.map((text) => ({ text })),
      history_text: '',
    },
    table_details_text: '',
  };
  const control = {
    schema: CONTROL_SCHEMA,
    capture: {},
    calibration: {},
    rawLog: {},
    seatOrder: ids,
    mechanicalProfile: {
      schema: 'tzolkin-bga-declared-mechanical-profile-v1',
      rules: 'basic',
      market: 'unlimited',
      sourceOptions: 'unknown',
    },
    turnOrderEvidence: { kind: 'declared', knownAfterActionId: null },
  };
  return { capture, atlas, raw, control };
}
function inputs(value) {
  const capture = bytes(value.capture);
  if (value.atlas.geometry !== null) value.atlas.geometry.captureSha256 = sha(capture);
  const atlas = bytes(value.atlas),
    raw = bytes(value.raw);
  value.control.capture = pin(capture);
  value.control.calibration = pin(atlas);
  value.control.rawLog = pin(raw);
  return [capture, atlas, raw, bytes(value.control)];
}
function legacyFixture() {
  const value = fixture(4);
  const modern = value.capture;
  const playerPanels = value.control.seatOrder.flatMap((id, index) => {
    const name = element(
      'div',
      `player_name_${id}`,
      `<a style="color: #008000">${'ABCD'[index]}</a>`,
    );
    const board = element(
      'div',
      `player_board_${id}`,
      element('span', `player_score_${id}`, '0') +
        modern.publicCounters
          .filter((counter) => counter.id.endsWith(`_${id}`))
          .map((counter) => counter.html)
          .join('') +
        element('div', `firstplayer_${id}`, index === 0 ? element('div', 'firstplayer') : ''),
    );
    return [
      {
        id: `player_board_inner_${id}`,
        html: element('div', `player_board_inner_${id}`, name + board),
      },
      { id: `player_board_${id}`, html: board },
    ];
  });
  value.capture = {
    schema: 'tzolkin-bga-public-initial-selection-observation-v1',
    url: modern.url,
    move: modern.move,
    header: modern.header,
    roundsText: modern.roundsText,
    gameAreaHtml: modern.gameAreaHtml,
    selectedPlayerTiles: modern.selectedPlayerTiles.map(({ id, html }) => ({ id, html })),
    playerPanels,
    wealthTilesHtml: null,
    trainingReady: false,
  };
  return value;
}

function fragmentFixture() {
  const value = fixture(4);
  const modern = value.capture;
  value.capture = {
    schema: 'bga-retained-visible-initial-dom-v1',
    sourceKind: 'public_ui_dom_only',
    tableId: modern.tableId,
    sourceUrl: modern.url,
    moveText: '7',
    roundText: '1',
    prompt: modern.header,
    boardHtml:
      modern.boardGeometry.html
        .slice(0, -6)
        .replace(
          element('span', 'current_round', '1'),
          element('div', 'rounds_wrap', `経過日 ${element('span', 'current_round', '1')} / 27`),
        ) +
      modern.buildingsHtml +
      modern.monumentsHtml +
      element('div', 'centralgearcorns') +
      '</div>',
    playerBoards: value.control.seatOrder.map((id, index) => {
      const wealthTilesHtml = modern.selectedPlayerTiles[index].html;
      const buildingsHtml = element('div', `player_buildings_${id}`);
      return {
        playerId: id,
        boardHtml: element(
          'div',
          `player_board_${id}`,
          element('span', `player_score_${id}`, '0') +
            modern.publicCounters
              .filter((node) => node.id.endsWith(`_${id}`))
              .map((node) => node.html)
              .join('') +
            element('div', `firstplayer_${id}`, index === 0 ? element('div', 'firstplayer') : ''),
        ),
        rowHtml: `<div class="whiteblock"><h3 style="color:#008000">${'ABCD'[index]}:</h3>${buildingsHtml}${wealthTilesHtml}</div>`,
        buildingsHtml,
        wealthTilesHtml,
      };
    }),
    renderedReplayControls: [],
    completeNativeReplayQualified: false,
    humanBcQualified: false,
  };
  return value;
}

test('legacy nested initial panels project only consistent retained state with unknown options', () => {
  const value = legacyFixture();
  const projected = projectCapturedInitial(...inputs(value));
  assert.deepEqual(projected.record, projectCapturedInitial(...inputs(fixture(4))).record);
  assert.equal(
    projected.provenance.captureAdapter.schema,
    'tzolkin-bga-legacy-initial-capture-adapter-v1',
  );
  assert.equal(projected.provenance.captureAdapter.visibilityVerified, false);
  assert.equal(projected.provenance.captureAdapter.geometryVerified, false);
  assert.deepEqual(projected.provenance.sourceOptions, {
    tribes: null,
    prophecies: null,
    additionalBuildings: null,
    quickActions: null,
    marketLimit: null,
  });
  for (const field of [
    'sourceAuthenticated',
    'coreInitialVerified',
    'fullGameReplayed',
    'strictHumanQualification',
    'policyTrainingReady',
    'valueTrainingReady',
  ])
    assert.equal(projected.provenance[field], false);
  const fragments = projectCapturedInitial(...inputs(fragmentFixture()));
  assert.deepEqual(fragments.record.initial, projected.record.initial);
  const observed = fragments.witness.initialExpected;
  assert.deepEqual(fragments.record.initialCheckpoint.expected, observed);
  assert.deepEqual(fragments.record.initialCheckpoint.source.actionIds, [7]);
  assert.deepEqual(observed.players[0].resources, projected.record.initial.players[0].resources);
  assert.deepEqual(observed.players[0].wealth, ['w01', 'w02']);
  assert.equal(observed.players[0].temples.quetzalcoatl, 2);
  assert.deepEqual(observed.jungle, {
    2: { corn: 4 },
    3: { wood: 4 },
    4: { wood: 4 },
    5: { wood: 4 },
  });
  assert.deepEqual(observed.monuments, fragments.record.initial.monuments);
  for (const field of ['age', 'foodDays', 'turnOrder', 'turnIndex', 'buildings'])
    assert.equal(Object.hasOwn(observed, field), false);
  assert.equal(fragments.provenance.captureAdapter.visibilityVerified, false);
  assert.equal(fragments.provenance.captureAdapter.geometryVerified, false);
  assert.equal(fragments.provenance.initialObservation.completeSourceBoard, false);
  assert.equal(fragments.provenance.policyTrainingReady, false);
  assert.equal(fragments.provenance.sourceOptions.additionalBuildings, null);
});

test('legacy panels refuse conflicting duplicates, missing state and foreign identities', () => {
  for (const [label, mutate, pattern] of [
    [
      'duplicate conflict',
      (v) => {
        v.capture.playerPanels[1].html = v.capture.playerPanels[1].html.replace('>8<', '>9<');
      },
      /Conflicting duplicate/,
    ],
    [
      'missing name',
      (v) => {
        v.capture.playerPanels = v.capture.playerPanels.filter(
          (p) => p.id.startsWith('player_board_') && !p.id.startsWith('player_board_inner_'),
        );
      },
      /Missing legacy player name/,
    ],
    [
      'missing counter',
      (v) => {
        for (const p of v.capture.playerPanels)
          p.html = p.html.replace(element('span', 'corn_number_player_100', '8'), '');
      },
      /Incomplete player setup fields/,
    ],
    [
      'foreign counter',
      (v) => {
        v.capture.playerPanels[0].html = v.capture.playerPanels[0].html.replace(
          'corn_number_player_100',
          'corn_number_player_999',
        );
      },
      /crosses player identity/,
    ],
    [
      'discarded first-player token',
      (v) => {
        v.capture.playerPanels[2].html = v.capture.playerPanels[2].html.replace(
          '<div id="player_board_200">',
          `${element('div', 'firstplayer')}<div id="player_board_200">`,
        );
      },
      /setup node outside retained board\/name/,
    ],
    [
      'missing selection',
      (v) => {
        v.capture.selectedPlayerTiles.pop();
      },
      /Incomplete legacy selected/,
    ],
    [
      'detached selection',
      (v) => {
        v.capture.selectedPlayerTiles[0].html += ' ';
      },
      /detached legacy selected/,
    ],
    [
      'foreign view',
      (v) => {
        v.capture.url = v.capture.url.replace('player=100', 'player=999');
      },
      /roster\/view mismatch/,
    ],
    [
      'foreign table',
      (v) => {
        v.capture.url = v.capture.url.replace('table=123', 'table=124');
      },
      /boundary mismatch/,
    ],
    [
      'foreign release',
      (v) => {
        v.capture.url = v.capture.url.replace('/fixture/', '/other/');
      },
      /release\/catalog mismatch/,
    ],
    [
      'duplicate view',
      (v) => {
        v.capture.url += '&player=100';
      },
      /archive identity/,
    ],
    [
      'unobserved geometry',
      (v) => {
        v.capture.gameAreaHtml += element('div', 'worker_999', '', ' class="worker worker_dummy"');
      },
      /lacks retained worker geometry/,
    ],
    [
      'readiness claim',
      (v) => {
        v.capture.trainingReady = true;
      },
      /readiness mismatch/,
    ],
  ]) {
    const value = legacyFixture();
    mutate(value);
    assert.throws(() => projectCapturedInitial(...inputs(value)), pattern, label);
  }
  for (const [label, mutate, pattern] of [
    [
      'detached selected fragment',
      (v) => {
        v.capture.playerBoards[0].wealthTilesHtml += ' ';
      },
      /Detached\/nonempty/,
    ],
    [
      'nested foreign owner',
      (v) => {
        const p = v.capture.playerBoards[0];
        const corn = element('span', 'corn_number_player_100', '8');
        p.boardHtml = p.boardHtml.replace(corn, element('div', 'player_board_999', corn));
      },
      /crosses owner identity/,
    ],
    [
      'duplicate across retained siblings',
      (v) => {
        const p = v.capture.playerBoards[0];
        p.rowHtml = p.rowHtml.replace(
          '<h3',
          `${element('span', 'corn_number_player_100', '8')}<h3`,
        );
      },
      /Conflicting retained sibling/,
    ],
    [
      'hidden row ancestor',
      (v) => {
        const p = v.capture.playerBoards[0];
        p.rowHtml = p.rowHtml.replace(
          'class="whiteblock"',
          'class="whiteblock" style="DISPLAY: none ! IMPORTANT"',
        );
      },
      /Hidden retained/,
    ],
    [
      'placed worker',
      (v) => {
        v.capture.boardHtml =
          v.capture.boardHtml.slice(0, -6) +
          element('div', 'worker_99', '', ' class="worker"') +
          '</div>';
      },
      /lack placed-worker geometry/,
    ],
    [
      'wrong source table',
      (v) => {
        v.capture.sourceUrl = v.capture.sourceUrl.replace('table=123', 'table=124');
      },
      /identity\/boundary/,
    ],
    [
      'false qualification',
      (v) => {
        v.capture.completeNativeReplayQualified = true;
      },
      /identity\/boundary/,
    ],
    [
      'missing first-player corn observation',
      (v) => {
        v.capture.boardHtml = v.capture.boardHtml.replace(element('div', 'centralgearcorns'), '');
      },
      /Missing retained DOM node/,
    ],
    [
      'malformed score node',
      (v) => {
        const p = v.capture.playerBoards[0];
        p.boardHtml = p.boardHtml.replace(
          element('span', 'player_score_100', '0'),
          element('div', 'player_score_100', '0'),
        );
      },
      /Malformed retained score/,
    ],
    [
      'unknown option inserted',
      (v) => {
        v.capture.additionalBuildings = false;
      },
      /missing\/unknown field/,
    ],
  ]) {
    const value = fragmentFixture();
    mutate(value);
    assert.throws(() => projectCapturedInitial(...inputs(value)), pattern, label);
  }
  for (const market of ['buildings', 'monuments']) {
    for (const attributes of [
      ' hidden',
      ' inert',
      ' popover',
      ' popover="manual"',
      ' style="opacity:0%"',
      ' style="opacity:-1"',
      ' style="opacity:-5%"',
    ]) {
      const value = fragmentFixture();
      const html = fixture(4).capture[`${market}Html`];
      value.capture.boardHtml = value.capture.boardHtml.replace(
        html,
        `<div${attributes}>${html}</div>`,
      );
      assert.throws(
        () => projectCapturedInitial(...inputs(value)),
        /Hidden retained/,
        `${market} with hidden outer wrapper ${attributes}`,
      );
    }
  }
  for (const [tag, attributes] of [
    ['datalist', ''],
    ['dialog', ''],
    ['details', ''],
    ['details', ' open'],
    ['div', ' popover'],
    ['meter', ''],
    ['progress', ''],
    ['rp', ''],
    ['svg', ''],
    ['math', ''],
    ['head', ''],
  ]) {
    const market = fragmentFixture();
    const marketHtml = fixture(4).capture.monumentsHtml;
    market.capture.boardHtml = market.capture.boardHtml.replace(
      marketHtml,
      `<${tag}${attributes}>${marketHtml}</${tag}>`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(market)),
      /Hidden retained|unsupported retained initial fragment container/,
      `${tag} market wrapper ${attributes}`,
    );
    const counter = fragmentFixture();
    const counterHtml = element('span', 'corn_number_player_100', '8');
    counter.capture.playerBoards[0].boardHtml = counter.capture.playerBoards[0].boardHtml.replace(
      counterHtml,
      `<${tag}${attributes}>${counterHtml}</${tag}>`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(counter)),
      /Hidden retained|unsupported retained initial fragment container/,
      `${tag} counter wrapper ${attributes}`,
    );
  }
  for (const style of [
    'display:none/**/',
    String.raw`d\isplay:none`,
    'display:none{}',
    'display:none@layer',
    '--d:none;display:var(--d)',
    '--c:hidden;content-visibility:var(--c)',
    'all:initial',
  ]) {
    const value = fragmentFixture();
    const html = fixture(4).capture.monumentsHtml;
    value.capture.boardHtml = value.capture.boardHtml.replace(
      html,
      `<div style="${style}">${html}</div>`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(value)),
      /Unsupported retained initial fragment style/,
      'ambiguous fragment CSS declarations',
    );
  }
  for (const style of [
    'display:var(--d,none)',
    'display:inherit',
    'visibility:var(--v,hidden)',
    'visibility:inherit',
    'content-visibility:var(--c,hidden)',
    'content-visibility:inherit',
    'content-visibility:auto',
  ]) {
    const value = fragmentFixture();
    const html = fixture(4).capture.monumentsHtml;
    value.capture.boardHtml = value.capture.boardHtml.replace(
      html,
      `<div style="${style}">${html}</div>`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(value)),
      /Unsupported retained initial fragment visibility/,
      `unsupported fragment visibility ${style}`,
    );
  }
  for (const field of ['boardHtml', 'rowHtml']) {
    const value = fragmentFixture();
    const owner = field === 'boardHtml' ? value.capture : value.capture.playerBoards[0];
    owner[field] = owner[field].replace('</div>', '<style>#monuments{display:none}</style></div>');
    assert.throws(
      () => projectCapturedInitial(...inputs(value)),
      /unsupported retained initial fragment container/,
      `retained stylesheet in ${field}`,
    );
  }
  for (const opacity of ['calc(0)', 'var(--hidden)', 'inherit', 'initial', '1e999', '0x0']) {
    const value = fragmentFixture();
    const html = fixture(4).capture.monumentsHtml;
    value.capture.boardHtml = value.capture.boardHtml.replace(
      html,
      `<div style="opacity:${opacity}">${html}</div>`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(value)),
      /Unsupported retained initial fragment opacity/,
      `unsupported fragment opacity ${opacity}`,
    );
  }
  for (const style of [
    'opacity:.5',
    'opacity:50%',
    'opacity:1e-1',
    'display:block;visibility:visible;content-visibility:visible',
  ]) {
    const value = fragmentFixture();
    const html = fixture(4).capture.monumentsHtml;
    value.capture.boardHtml = value.capture.boardHtml.replace(
      html,
      `<div style="${style}">${html}</div>`,
    );
    assert.deepEqual(
      projectCapturedInitial(...inputs(value)).record.initial,
      projectCapturedInitial(...inputs(fragmentFixture())).record.initial,
    );
  }
  const modern = fixture(4).capture;
  for (const html of [
    modern.templeMarkers[0].html,
    modern.technologyMarkers[0].html,
    modern.gears[0].html,
    element('div', 'centralgearcorns'),
  ]) {
    const value = fragmentFixture();
    value.capture.boardHtml = value.capture.boardHtml.replace(html, '');
    value.capture.playerBoards[0].rowHtml = value.capture.playerBoards[0].rowHtml.replace(
      '<h3',
      `${html}<h3`,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(value)),
      /outside original board fragment|Missing retained DOM node/,
      'board field supplied only by a player row',
    );
  }
  const misplacedRound = fragmentFixture();
  misplacedRound.capture.boardHtml = misplacedRound.capture.boardHtml.replace(
    element('div', 'rounds_wrap', `経過日 ${element('span', 'current_round', '1')} / 27`),
    element('div', 'rounds_wrap', '経過日 / 27') + element('span', 'current_round', '1'),
  );
  assert.throws(
    () => projectCapturedInitial(...inputs(misplacedRound)),
    /outside retained rounds wrapper/,
  );
  for (const [id, scalar, board] of [
    ['corn_number_player_100', '8', false],
    ['worker_number_player_100', '3', false],
    ['player_score_100', '0', false],
    ['current_round', '1', true],
    ['skull_left', '13', true],
  ]) {
    for (const attributes of [' hidden', ' inert']) {
      const value = fragmentFixture();
      const target = board ? value.capture : value.capture.playerBoards[0];
      target.boardHtml = target.boardHtml.replace(
        element('span', id, scalar),
        element('span', id, `<span${attributes}>${scalar}</span>`),
      );
      assert.throws(
        () => projectCapturedInitial(...inputs(value)),
        /nonscalar DOM counter|Hidden retained/,
        `${id} with hidden numeric descendant ${attributes}`,
      );
    }
  }
});
function currentUi(value) {
  value.raw = {
    ...value.raw,
    schema_version: 1,
    source_type: 'public_ui_dom',
    export_scope: 'all_rendered_gamereview_log_entries',
    entries: [{ action_id: 7, raw_text: '行動 7 :\n12:00:00\n', timestamp_display: '12:00:00' }],
    dom_entry_count: value.raw.dom_entries.length,
    metadata: {
      tableUrl: 'https://boardgamearena.com/table?table=123',
      playerCount: value.raw.metadata.player_count,
      players: value.raw.metadata.players.map(({ text }) => ({ name: text })),
    },
  };
  return value;
}

function withLaterWitnesses(value, witnesses, table = null) {
  if (table !== null) value.control.table = pin(table);
  const prepared = inputs(value);
  const initial = projectCapturedInitial(...prepared, table);
  const collection = {
    schema: LATER_WITNESS_SCHEMA,
    tableId: initial.witness.tableId,
    rawLogReference: initial.record.source.reference,
    initialCaptureReference: value.capture.url,
    rawLogSha256: sha(prepared[2]),
    initialCaptureSha256: sha(prepared[0]),
    calibrationSha256: sha(prepared[1]),
    tableEvidenceSha256: table === null ? null : sha(table),
    archiveRelease: initial.provenance.archiveRelease,
    catalogHash: initial.witness.catalogHash,
    initialActionId: initial.witness.initialActionId,
    witnesses,
  };
  const repin = (document = collection) => {
    const later = bytes(document);
    const control = JSON.parse(prepared[3]);
    control.laterWitnesses = pin(later);
    return [...prepared.slice(0, 3), bytes(control), table, later];
  };
  return { initial, collection, repin };
}

function placementFixture() {
  const value = fixture(4);
  const raw_text = 'AはPalenqueの歯車に0\nを支払ってワーカーを置いた';
  value.raw.entries.push({ action_id: 8, timestamp_display: '12:01:00', raw_text });
  value.raw.dom_entries.push({
    action_id: 8,
    timestamp_display: '12:01:00',
    messages: [{ raw_text, html: raw_text, icons: [] }],
  });
  value.raw.entry_count++;
  return value;
}

test('pinned later collection reaches ordinary decoder checkpoints without changing initial input', () => {
  const value = placementFixture();
  const retained = { actionId: 8, expected: { round: 1 }, pendingTask: null };
  const { initial, repin } = withLaterWitnesses(value, [retained]);
  const prepared = repin();
  const projected = projectCapturedInitial(...prepared);
  assert.deepEqual(projected.record, initial.record);
  assert.deepEqual(projected.witness.initial, initial.witness.initial);
  assert.deepEqual(projected.witness.witnesses, [retained]);
  assert.equal(projected.provenance.laterWitnesses.sha256, sha(prepared[5]));
  assert.equal(projected.provenance.laterWitnesses.count, 1);
  retained.expected.round = 99;
  assert.equal(projected.witness.witnesses[0].expected.round, 1);

  const move = { type: 'place', gear: 'palenque' };
  const action = { type: 'place', gear: 'palenque', position: 0, cornCost: 0 };
  const first = {
    snapshot: { state: structuredClone(initial.record.initial) },
    observation: { actor: 0, pendingTask: null, legalActions: [{ action, move }] },
  };
  const after = structuredClone(first);
  after.snapshot.state.gears.palenque[0] = { playerId: 0, dummy: false };
  after.snapshot.state.players[0].availableWorkers--;
  after.observation.legalActions = [];
  let finalCheckpoint;
  const dispatch = (request) => {
    if (request.operation === 'publicApply') {
      assert.deepEqual(request.move, move);
      return structuredClone(after);
    }
    assert.equal(request.operation, 'publicReplay');
    assert.deepEqual(request.replay.initial, initial.record.initial);
    const steps = request.replay.steps;
    if (steps.length) finalCheckpoint = structuredClone(steps[0].checkpoint);
    return {
      frames: [structuredClone(first), ...steps.map(() => structuredClone(after))],
      verifiedSteps: steps.length,
      checkpointsVerified: steps.filter((step) => step.checkpoint).length,
      sourceCoverage: { complete: false },
    };
  };
  const decode = (p) => decodeProjectedInitial(...p.slice(0, 4), dispatch, p[4], p[5]);
  const result = decode(prepared);
  assert.equal(result.decoded.manifest.blocked, null);
  assert.equal(result.decoded.record.steps.length, 1);
  assert.deepEqual(finalCheckpoint, {
    source: { reference: value.raw.source_url, actionIds: [8] },
    expected: { round: 1 },
  });
  assert.equal(result.decoded.manifest.checkpointsVerified, 1);
  assert.deepEqual(
    result.decoded.manifest.initialProjection.laterWitnesses,
    projected.provenance.laterWitnesses,
  );
  assert.equal(result.decoded.manifest.trainingReady, false);
  assert.equal(result.projection.provenance.strictHumanQualification, false);

  const contradiction = withLaterWitnesses(value, [{ actionId: 8, expected: { round: 2 } }]);
  const refused = decode(contradiction.repin());
  assert.equal(refused.decoded.manifest.blocked.actionId, 8);
  assert.match(refused.decoded.manifest.blocked.reason, /public checkpoint mismatch/);
  assert.equal(refused.decoded.record.steps.length, 0);

  const unused = withLaterWitnesses(value, [{ actionId: 8, buildings: ['b04'] }]);
  assert.match(decode(unused.repin()).decoded.manifest.blocked.reason, /Unused buildings witness/);

  const fragments = fragmentFixture();
  const fragmentInputs = inputs(fragments);
  const observation = projectCapturedInitial(...fragmentInputs);
  let seenInitialCheckpoint;
  const fragmentDecoded = decodeProjectedInitial(...fragmentInputs, (request) => {
    assert.equal(request.operation, 'publicReplay');
    seenInitialCheckpoint = request.replay.initialCheckpoint;
    return {
      frames: [
        {
          snapshot: { state: request.replay.initial },
          observation: { actor: 0, pendingTask: null, legalActions: [] },
        },
      ],
      verifiedSteps: 0,
      checkpointsVerified: 1,
      sourceCoverage: { initial: false, foodDays: [], terminal: false, complete: false },
    };
  });
  assert.deepEqual(seenInitialCheckpoint, observation.record.initialCheckpoint);
  assert.deepEqual(
    fragmentDecoded.decoded.record.initialCheckpoint.expected,
    observation.witness.initialExpected,
  );
  assert.equal(fragmentDecoded.decoded.manifest.sourceCoverage.complete, false);
  assert.equal(fragmentDecoded.decoded.manifest.trainingReady, false);
});

test('later collection refuses mismatched input bindings and invalid observations before core dispatch', () => {
  const table = bytes({
    schema: 'tzolkin-bga-public-table-ui-v1',
    url: 'https://boardgamearena.com/table?table=123',
    optionsText: 'ゲームモード\nノーマルモード\nウシュマルコーンの制限\n制限なし',
  });
  const value = currentUi(placementFixture());
  // Retain the later entry when converting this synthetic log to current UI schema.
  value.raw.entries.push({
    action_id: 8,
    raw_text: `行動 8 :\n12:01:00\n${value.raw.dom_entries[1].messages[0].raw_text}`,
    timestamp_display: '12:01:00',
  });
  const retained = { actionId: 8, expected: { round: 1 } };
  const { collection, repin } = withLaterWitnesses(value, [retained], table);
  for (const field of Object.keys(collection).filter((key) => key !== 'witnesses')) {
    const altered = structuredClone(collection);
    altered[field] = field === 'initialActionId' ? 8 : 'mismatch';
    const prepared = repin(altered);
    assert.throws(
      () =>
        decodeProjectedInitial(
          ...prepared.slice(0, 4),
          () => assert.fail('Mismatched proof reached core'),
          prepared[4],
          prepared[5],
        ),
      /binding mismatch|Unsupported later witness schema/,
      field,
    );
  }
  for (const [witnesses, pattern] of [
    [[{ actionId: 7, expected: { round: 1 } }], /later witness action/],
    [[{ actionId: 99, expected: { round: 1 } }], /unknown later witness/],
    [[retained, retained], /Duplicate/],
    [[{ actionId: 8 }], /no observation/],
    [[{ ...retained, initial: {} }], /unknown field/],
    [[{ actionId: 8, expected: {} }], /Empty later/],
    [[{ actionId: 8, pendingTask: 1 }], /later pending task/],
    [[{ actionId: 8, buildings: ['missing'] }], /catalog IDs/],
    [[{ actionId: 8, buildingEquivalenceClass: ['b15', 'b18'] }], /different catalog fields/],
    [
      [{ actionId: 8, buildingEquivalenceClass: ['b15', 'b16'], buildings: ['b15'] }],
      /Invalid building equivalence class/,
    ],
    [[{ actionId: 8, monuments: ['b04'] }], /catalog IDs/],
    [[{ actionId: 8, refills: { age2: ['b04'] } }], /catalog IDs/],
    [[{ actionId: 8, refills: { currentAge: ['b04', 'b04'] } }], /catalog IDs/],
    [[{ actionId: 8, refills: { unknown: [] } }], /unknown field/],
    [Array(4001).fill(retained), /invalid list/],
  ]) {
    assert.throws(() => projectCapturedInitial(...repin({ ...collection, witnesses })), pattern);
  }
  const validCards = [
    {
      actionId: 8,
      buildings: ['b04'],
      monuments: ['m01'],
      refills: { currentAge: ['b05'], age2: ['b15'] },
      pendingTask: 'build',
    },
  ];
  assert.deepEqual(
    projectCapturedInitial(...repin({ ...collection, witnesses: validCards })).witness.witnesses,
    validCards,
  );
  const equivalentCards = [{ actionId: 8, buildingEquivalenceClass: ['b16', 'b15'] }];
  assert.deepEqual(
    projectCapturedInitial(...repin({ ...collection, witnesses: equivalentCards })).witness
      .witnesses,
    equivalentCards,
  );
  const prepared = repin();
  assert.throws(() => projectCapturedInitial(...prepared.slice(0, 5)), /Missing declared later/);
  const withoutPin = JSON.parse(prepared[3]);
  delete withoutPin.laterWitnesses;
  assert.throws(
    () => projectCapturedInitial(...prepared.slice(0, 3), bytes(withoutPin), table, prepared[5]),
    /later witnesses pin/,
  );
  assert.throws(
    () => projectCapturedInitial(...prepared.slice(0, 5), Buffer.from('{}')),
    /later witnesses: pin mismatch/,
  );
  const wrongPin = JSON.parse(prepared[3]);
  wrongPin.laterWitnesses.sha256 = '0'.repeat(64);
  assert.throws(
    () => projectCapturedInitial(...prepared.slice(0, 3), bytes(wrongPin), table, prepared[5]),
    /later witnesses: pin mismatch/,
  );
  const oversized = Buffer.alloc(16 * 1024 * 1024 + 1);
  const oversizedControl = JSON.parse(prepared[3]);
  oversizedControl.laterWitnesses = pin(oversized);
  assert.throws(
    () =>
      projectCapturedInitial(...prepared.slice(0, 3), bytes(oversizedControl), table, oversized),
    /later witnesses size/,
  );
  assert.throws(
    () => projectCapturedInitial(...prepared.slice(0, 5), Buffer.alloc(32 * 1024 * 1024)),
    /aggregate byte limit/,
  );
  const duplicated = Buffer.from(prepared[5].toString().replace(/}$/, ',"witnesses":[]}'));
  const duplicateControl = JSON.parse(prepared[3]);
  duplicateControl.laterWitnesses = pin(duplicated);
  assert.throws(
    () =>
      projectCapturedInitial(...prepared.slice(0, 3), bytes(duplicateControl), table, duplicated),
    /duplicate JSON key/,
  );
});

test('file projector carries separately pinned later witnesses and refuses missing or changed files', async () => {
  const { repin } = withLaterWitnesses(placementFixture(), [
    { actionId: 8, expected: { round: 1 } },
  ]);
  const prepared = repin();
  const directory = await mkdtemp(join(tmpdir(), 'tzolkin-playing-later-'));
  try {
    const paths = ['capture', 'atlas', 'raw', 'control', 'later'].map((name) =>
      join(directory, `${name}.json`),
    );
    for (const [i, path] of paths.entries())
      await writeFile(path, prepared[i === 4 ? 5 : i], { flag: 'wx' });
    const output = join(directory, 'projected');
    await assert.rejects(
      projectPlayingFiles(...paths.slice(0, 4), output),
      /Missing declared later/,
    );
    const result = await projectPlayingFiles(...paths.slice(0, 4), output, null, paths[4]);
    const saved = JSON.parse(await readFile(join(output, 'witness.json')));
    assert.deepEqual(saved.witnesses, [{ actionId: 8, expected: { round: 1 } }]);
    assert.equal(result.projection.provenance.laterWitnesses.sha256, sha(prepared[5]));
    const cliOutput = join(directory, 'cli-projected');
    execFileSync(
      process.execPath,
      [
        fileURLToPath(new URL('./bga-playing-project.mjs', import.meta.url)),
        ...paths.slice(0, 4),
        cliOutput,
        '--later-witnesses',
        paths[4],
      ],
      { encoding: 'utf8' },
    );
    assert.deepEqual(
      JSON.parse(await readFile(join(cliOutput, 'witness.json'))).witnesses,
      saved.witnesses,
    );
    await writeFile(paths[4], Buffer.alloc(prepared[5].length, 'x'));
    await assert.rejects(
      projectPlayingFiles(...paths.slice(0, 4), join(directory, 'invalid'), null, paths[4]),
      /pin mismatch/,
    );
    await writeFile(paths[4], Buffer.from('changed'));
    await assert.rejects(
      projectPlayingFiles(...paths.slice(0, 4), join(directory, 'wrong-size'), null, paths[4]),
      /file size mismatch/,
    );
  } finally {
    assert.equal(dirname(resolve(directory)), resolve(tmpdir()));
    await rm(directory, { recursive: true, force: true });
  }
});

test('three/four-player captured projection preserves counters, dummy slots and conditional provenance', () => {
  for (const n of [3, 4]) {
    const projected = projectCapturedInitial(...inputs(fixture(n)));
    const initial = projected.record.initial;
    assert.deepEqual(initial.players[0].resources, {
      corn: 8,
      wood: 0,
      stone: 0,
      gold: 1,
      skull: 0,
    });
    assert.deepEqual(initial.players[0].wealth, ['w01', 'w02']);
    assert.equal(initial.players[0].temples.quetzalcoatl, 2);
    assert.equal(initial.monuments.length, n + 2);
    assert.deepEqual(initial.jungle[3], { corn: n, wood: n });
    assert.equal(Object.values(initial.gears).flat().filter(Boolean).length, (4 - n) * 6);
    if (n === 3) assert.deepEqual(initial.gears.yaxchilan[0], { playerId: -1, dummy: true });
    assert.equal(projected.witness.initialActionId, 7); // Not playerCount+1.
    assert.equal(projected.provenance.policyTrainingReady, false);
    assert.equal(projected.provenance.sourceOptions.additionalBuildings, null);
    assert.equal(Object.hasOwn(initial, 'sourceOptions'), false); // Core DTO stays closed.
  }
  const later = fixture(4);
  later.control.turnOrderEvidence = { kind: 'later-inferred', knownAfterActionId: 12 };
  later.raw.entries.push({ action_id: 12, raw_text: '', timestamp_display: '12:01:00' });
  later.raw.dom_entries.push({ action_id: 12, messages: [], timestamp_display: '12:01:00' });
  later.raw.entry_count++;
  assert.equal(
    projectCapturedInitial(...inputs(later)).provenance.turnOrder.knownAfterActionId,
    12,
  );
});

test('detached HTML, unknown/wrong-release sprites, duplicate JSON and ambiguous dummy geometry block', () => {
  const detached = fixture(4);
  detached.capture.selectedPlayerTiles[0].html =
    detached.capture.selectedPlayerTiles[0].html.replace('-10px', '-999px');
  assert.throws(() => projectCapturedInitial(...inputs(detached)), /detached HTML/);
  const unknown = fixture(4);
  unknown.atlas.faces.shift();
  assert.throws(() => projectCapturedInitial(...inputs(unknown)), /Uncalibrated wealth/);
  const counter = fixture(4);
  const oldCounter = counter.capture.publicCounters[1].html;
  counter.capture.publicCounters[1].html = oldCounter.replace('>8<', '>9<');
  counter.capture.playerBoardsHtml = counter.capture.playerBoardsHtml.replace(
    oldCounter,
    counter.capture.publicCounters[1].html,
  );
  assert.throws(
    () => projectCapturedInitial(...inputs(counter)),
    /does not match initial counters/,
  );
  const release = fixture(4);
  release.atlas.archiveRelease = 'other';
  assert.throws(() => projectCapturedInitial(...inputs(release)), /release\/catalog/);
  const missing = fixture(3);
  missing.capture.skullSpaces.pop();
  assert.throws(() => projectCapturedInitial(...inputs(missing)), /incomplete retained/);
  const ambiguous = fixture(3);
  ambiguous.capture.workerPlaces[12].rect = structuredClone(
    ambiguous.capture.workerPlaces[11].rect,
  );
  assert.throws(() => projectCapturedInitial(...inputs(ambiguous)), /dummy geometry match/);
  const scale = fixture(3);
  scale.capture.boardGeometry.rect.width = 1000;
  assert.throws(() => projectCapturedInitial(...inputs(scale)), /board scale/);
  const unverified = fixture(3);
  unverified.atlas.geometry.evidenceSha256 = 'a'.repeat(64);
  assert.throws(() => projectCapturedInitial(...inputs(unverified)), /geometry calibration/);
  const unknownOrder = fixture(4);
  unknownOrder.control.turnOrderEvidence = { kind: 'later-inferred', knownAfterActionId: 12 };
  assert.throws(() => projectCapturedInitial(...inputs(unknownOrder)), /absent from raw log/);
  const duplicate = inputs(fixture(4));
  duplicate[3] = Buffer.from(
    duplicate[3].toString().replace('"seatOrder":', '"schema":"duplicate","seatOrder":'),
  );
  assert.throws(() => projectCapturedInitial(...duplicate), /duplicate JSON key/);
  for (const tag of ['template', 'canvas', 'object', 'video', 'audio', 'select']) {
    const inert = fixture(4);
    inert.capture.gameAreaHtml += `<${tag}><div class="worker worker_dummy"></div></${tag}>`;
    assert.throws(() => projectCapturedInitial(...inputs(inert)), /unsupported inert/);
  }
  const port = fixture(4);
  port.capture.url = port.capture.url.replace('boardgamearena.com/', 'boardgamearena.com:8443/');
  assert.throws(() => projectCapturedInitial(...inputs(port)), /Unsupported BGA source URL/);
  for (const tag of ['script', 'style']) {
    const hidden = fixture(4);
    const visible = element(
      'div',
      'wealth_tile_1',
      '',
      ' class="wealth_tile" style="background-position: -10px 0px;"',
    );
    const opaque = visible.replace('<div ', `<${tag} `).replace('</div>', `</${tag}>`);
    hidden.capture.gameAreaHtml = hidden.capture.gameAreaHtml.replace(visible, opaque);
    hidden.capture.selectedPlayerTiles[0].html = hidden.capture.selectedPlayerTiles[0].html.replace(
      visible,
      opaque,
    );
    assert.throws(
      () => projectCapturedInitial(...inputs(hidden)),
      /Incomplete player setup fields|Malformed selected wealth|incomplete retained/,
    );
    const hiddenCounter = fixture(4);
    const original = hiddenCounter.capture.publicCounters[0].html;
    const counter = original.replace('<span ', `<${tag} `).replace('</span>', `</${tag}>`);
    hiddenCounter.capture.playerBoardsHtml = hiddenCounter.capture.playerBoardsHtml.replace(
      original,
      counter,
    );
    hiddenCounter.capture.publicCounters[0].html = counter;
    assert.throws(
      () => projectCapturedInitial(...inputs(hiddenCounter)),
      /Incomplete player setup fields|incomplete retained|Missing retained DOM/,
    );
  }
  const misplaced = fixture(4);
  misplaced.capture.gameAreaHtml += '<div id="unowned_card" class="construction"></div>';
  assert.throws(() => projectCapturedInitial(...inputs(misplaced)), /outside initial markets/);
});

test('conditional decode reaches shared core path with warnings while ordinary decode still requires options', () => {
  const prepared = inputs(fixture(4));
  const projected = projectCapturedInitial(...prepared);
  let calls = 0;
  const dispatch = (request) => {
    calls++;
    assert.equal(request.operation, 'publicReplay');
    const frame = { snapshot: { state: request.replay.initial }, observation: { actor: 0 } };
    return {
      frames: [frame],
      verifiedSteps: 0,
      checkpointsVerified: 0,
      sourceCoverage: { complete: false },
    };
  };
  assert.throws(
    () => decodePublicGame(prepared[2], projected.witness, dispatch),
    /explicitly unlimited/,
  );
  assert.equal(calls, 0);
  const result = decodeConditionalPublicGame(
    prepared[2],
    projected.witness,
    dispatch,
    projected.provenance.mechanicalProfile,
  );
  assert.equal(calls, 2);
  assert.equal(result.manifest.trainingReady, false);
  assert.equal(result.manifest.sourceOptionsObserved, false);
  assert.equal(result.manifest.conditionalMechanicalProfile.sourceOptions, 'unknown');
  assert.throws(
    () =>
      decodeConditionalPublicGame(prepared[2], projected.witness, dispatch, {
        ...projected.provenance.mechanicalProfile,
        sourceOptions: 'observed',
      }),
    /Unsupported declared/,
  );

  // These source settings must fail before any core dispatch, in both the pure
  // producer and its direct conditional decoder entry point.
  for (const setting of [
    'ウシュマルコーンの制限\n20',
    '部族\n有効',
    '予言\n有効',
    '追加建物\n有効',
    'クイックアクション\n有効',
  ]) {
    const incompatible = fixture(4);
    incompatible.raw.table_details_text = `ゲーム構成\n${setting}`;
    const pinned = inputs(incompatible);
    assert.throws(() => projectCapturedInitial(...pinned), /contradicts/);
    assert.throws(
      () =>
        decodeConditionalPublicGame(
          pinned[2],
          { ...projected.witness, rawSha256: sha(pinned[2]) },
          dispatch,
          projected.provenance.mechanicalProfile,
        ),
      /contradicts/,
    );
  }
  const limited = fixture(4);
  limited.raw.table_details_text =
    'ゲーム構成\nゲームモード\nアリーナモード\nウシュマルコーンの制限\n20';
  assert.throws(() => projectCapturedInitial(...inputs(limited)), /contradicts/);
  for (const prefix of [
    '部族\n有効',
    'ウシュマルコーンの制限\n20',
    '部族: 有効',
    'Tribes enabled',
    '拡張 有効',
    'Expansion enabled',
    'Extension enabled',
  ]) {
    const prefixed = fixture(4);
    prefixed.raw.table_details_text =
      `${prefix}\nゲーム構成\nウシュマルコーンの制限\n制限なし\n` +
      '部族\n無効\n予言\n無効\n追加建物\n無効\nクイックアクション\n無効';
    const prefixedInputs = inputs(prefixed);
    assert.throws(
      () => projectCapturedInitial(...prefixedInputs),
      /prefix contains option evidence/,
    );
    assert.throws(
      () =>
        decodeConditionalPublicGame(
          prefixedInputs[2],
          { ...projected.witness, rawSha256: sha(prefixedInputs[2]) },
          dispatch,
          projected.provenance.mechanicalProfile,
        ),
      /prefix contains option evidence/,
    );
  }
  const detail = fixture(4);
  for (const skippedDetail of [
    'Tribes enabled',
    '拡張 有効',
    'Expansion enabled',
    'Extension enabled',
  ]) {
    detail.raw.table_details_text =
      `ゲーム構成\nゲームモード\nアリーナモード\nアリーナモード: ${skippedDetail}\n` +
      'ウシュマルコーンの制限\n制限なし';
    const detailInputs = inputs(detail);
    assert.throws(
      () => projectCapturedInitial(...detailInputs),
      /mode detail contains option evidence/,
    );
    assert.throws(
      () =>
        decodeConditionalPublicGame(
          detailInputs[2],
          { ...projected.witness, rawSha256: sha(detailInputs[2]) },
          dispatch,
          projected.provenance.mechanicalProfile,
        ),
      /mode detail contains option evidence/,
    );
  }
  assert.equal(calls, 2);

  const compatible = fixture(4);
  compatible.raw.table_details_text =
    'ゲーム構成\nゲームモード\nアリーナモード\nアリーナモード: Synthetic details\nウシュマルコーンの制限\n制限なし';
  const compatibleInputs = inputs(compatible);
  const partial = projectCapturedInitial(...compatibleInputs);
  assert.equal(partial.provenance.mode, 'arena');
  assert.equal(partial.provenance.sourceOptions.marketLimit, 'unlimited');
  assert.equal(partial.provenance.sourceOptions.tribes, null);
  assert.equal(partial.provenance.observedOptions.complete, false);
  assert.ok(
    partial.provenance.observedOptions.evidence.some(
      ({ source }) => source === 'normalized.context.marketOption',
    ),
  );
  const partialDecode = decodeConditionalPublicGame(
    compatibleInputs[2],
    partial.witness,
    dispatch,
    partial.provenance.mechanicalProfile,
  );
  assert.ok(
    partialDecode.manifest.observedOptions.evidence.some(
      ({ source }) => source === 'normalized.context.mode',
    ),
  );
  compatible.raw.table_details_text += '\n部族\n無効';
  const explicitlyDisabled = projectCapturedInitial(...inputs(compatible));
  assert.equal(explicitlyDisabled.provenance.sourceOptions.tribes, false);
  assert.equal(explicitlyDisabled.provenance.sourceOptions.prophecies, null);

  const current = currentUi(structuredClone(compatible));
  const currentTable = bytes({
    schema: 'tzolkin-bga-public-table-ui-v1',
    url: 'https://boardgamearena.com/table?table=123',
    optionsText: 'ゲームモード\nアリーナモード\nウシュマルコーンの制限\n制限なし\n部族\n無効',
  });
  current.control.table = pin(currentTable);
  const currentInputs = inputs(current);
  const currentProjection = projectCapturedInitial(...currentInputs, currentTable);
  assert.equal(currentProjection.provenance.mode, 'arena');
  assert.equal(currentProjection.provenance.sourceOptions.marketLimit, 'unlimited');
  assert.equal(currentProjection.provenance.sourceOptions.tribes, false);
  assert.equal(currentProjection.provenance.sourceOptions.prophecies, null);
  assert.equal(currentProjection.provenance.policyTrainingReady, false);
  assert.ok(
    currentProjection.provenance.observedOptions.evidence.some(
      ({ source }) => source === 'normalized.context.marketOption',
    ),
  );
  const currentDecode = decodeConditionalPublicGame(
    currentInputs[2],
    currentProjection.witness,
    dispatch,
    currentProjection.provenance.mechanicalProfile,
    currentTable,
  );
  assert.ok(
    currentDecode.manifest.observedOptions.evidence.some(
      ({ source }) => source === 'normalized.context.mode',
    ),
  );
  assert.equal(currentDecode.manifest.trainingReady, false);

  // Supplied table text is inspected as well, never substituted for the context
  // or embedded settings. Unsupported evidence cannot disappear in normalization.
  for (const optionsText of [
    '部族\n有効',
    '部族\n有効\nゲーム構成\n',
    'ウシュマルコーンの制限\n20\nゲーム構成\nウシュマルコーンの制限\n制限なし',
    'ゲームモード\nノーマルモード',
  ]) {
    const supplied = currentUi(structuredClone(compatible));
    const tableBytes = bytes({
      schema: 'tzolkin-bga-public-table-ui-v1',
      url: 'https://boardgamearena.com/table?table=123',
      optionsText,
    });
    supplied.control.table = pin(tableBytes);
    const tableInputs = inputs(supplied);
    assert.throws(
      () => projectCapturedInitial(...tableInputs, tableBytes),
      /contradicts|Ambiguous|Conflicting/,
    );
    assert.throws(
      () =>
        decodeConditionalPublicGame(
          tableInputs[2],
          { ...projected.witness, rawSha256: sha(tableInputs[2]) },
          dispatch,
          projected.provenance.mechanicalProfile,
          tableBytes,
        ),
      /contradicts|Ambiguous|Conflicting/,
    );
  }
  assert.equal(calls, 6);
});

test('pinned parse and create-only file boundaries refuse malformed evidence', async () => {
  const base = currentUi(fixture(3));
  const table = bytes({
    schema: 'tzolkin-bga-public-table-ui-v1',
    url: 'https://boardgamearena.com/table?table=123',
    optionsText: 'ゲームモード\nアリーナモード\nウシュマルコーンの制限\n制限なし',
  });
  base.control.table = pin(table);
  for (const [label, mutate, pattern] of [
    [
      'wrong table pin',
      (v) => {
        v.control.table.sha256 = '0'.repeat(64);
      },
      /table evidence: pin mismatch/,
    ],
    [
      'missing table pin',
      (v) => {
        delete v.control.table;
      },
      /table evidence pin/,
    ],
    [
      'noscript',
      (v) => {
        v.capture.gameAreaHtml += '<noscript></noscript>';
      },
      /unsupported inert/,
    ],
    [
      'incomplete comment',
      (v) => {
        v.capture.gameAreaHtml += '<!-- missing end';
      },
      /incomplete comment/,
    ],
    [
      'duplicate attribute',
      (v) => {
        v.capture.gameAreaHtml += '<div class="a" class="b"></div>';
      },
      /duplicate attribute/,
    ],
    [
      'duplicate ID',
      (v) => {
        v.capture.gameAreaHtml += '<div id="current_round"></div>';
      },
      /duplicate\/empty DOM ID/,
    ],
    [
      'empty ID',
      (v) => {
        v.capture.gameAreaHtml += '<div id=""></div>';
      },
      /duplicate\/empty DOM ID/,
    ],
    [
      'missing geometry',
      (v) => {
        v.atlas.geometry = null;
      },
      /Missing calibrated dummy geometry/,
    ],
    [
      'worker scale',
      (v) => {
        v.capture.gearWorkerNodes[0].rect.width += 1;
      },
      /worker scale/,
    ],
    [
      'noninjective atlas',
      (v) => {
        v.atlas.faces[1].catalogIds = [...v.atlas.faces[0].catalogIds];
      },
      /not injective/,
    ],
    [
      'different building effects',
      (v) => {
        v.atlas.faces.find((f) => f.kind === 'building').catalogIds = ['b01', 'b04'];
      },
      /Different building effects/,
    ],
    [
      'URL userinfo',
      (v) => {
        v.capture.url = v.capture.url.replace('https://', 'https://user@');
      },
      /Unsupported BGA source URL/,
    ],
    [
      'URL fragment',
      (v) => {
        v.capture.url += '#part';
      },
      /Unsupported BGA source URL/,
    ],
    [
      'URL host',
      (v) => {
        v.capture.url = v.capture.url.replace('boardgamearena.com', 'example.invalid');
      },
      /Unsupported BGA source URL/,
    ],
    [
      'duplicate table',
      (v) => {
        v.capture.url += '&table=123';
      },
      /Unsupported BGA source URL/,
    ],
  ]) {
    const v = structuredClone(base);
    mutate(v);
    assert.throws(() => projectCapturedInitial(...inputs(v), table), pattern, label);
  }
  assert.throws(
    () => projectCapturedInitial(...inputs(structuredClone(base))),
    /Missing declared table evidence/,
  );
  const commented = structuredClone(base);
  commented.capture.gameAreaHtml += '<!-- <div id="wealth_tile_999" class="wealth_tile"></div> -->';
  assert.equal(
    projectCapturedInitial(...inputs(commented), table).record.initial.players.length,
    3,
  );

  for (const index of [0, 1, 2, 3, 4]) {
    const prepared = [...inputs(structuredClone(base)), table];
    prepared[index] = Buffer.from(
      prepared[index].toString().replace(/}$/, ',"duplicateProbe":{"key":1,"key":2}}'),
    );
    if (index !== 3) {
      const control = JSON.parse(prepared[3]);
      control[['capture', 'calibration', 'rawLog', null, 'table'][index]] = pin(prepared[index]);
      prepared[3] = bytes(control);
    }
    assert.throws(
      () => projectCapturedInitial(...prepared),
      /duplicate JSON key/,
      `nested duplicate input ${index}`,
    );
  }

  const directory = await mkdtemp(join(tmpdir(), 'tzolkin-playing-boundary-'));
  try {
    const prepared = inputs(structuredClone(base));
    const names = ['capture.json', 'atlas.json', 'raw.json', 'control.json'];
    const paths = names.map((name) => join(directory, name));
    for (const [i, file] of paths.entries()) await writeFile(file, prepared[i], { flag: 'wx' });
    const output = join(directory, 'projected');
    await assert.rejects(
      projectPlayingFiles(...paths, output),
      /Separate table evidence requires the pinned byte API/,
    );
    const withoutTable = JSON.parse(prepared[3]);
    delete withoutTable.table;
    await writeFile(paths[3], bytes(withoutTable));
    const saved = await projectPlayingFiles(...paths, output);
    assert.equal(saved.projection.provenance.coreInitialVerified, false);
    const original = await readFile(join(output, 'projection.json'));
    await assert.rejects(projectPlayingFiles(...paths, output), { code: 'EEXIST' });
    assert.deepEqual(await readFile(join(output, 'projection.json')), original);
    withoutTable.capture.bytes++;
    await writeFile(paths[3], bytes(withoutTable));
    await assert.rejects(
      projectPlayingFiles(...paths, join(directory, 'invalid-output')),
      /size\/pin|size mismatch|pin mismatch/,
    );
  } finally {
    assert.equal(dirname(resolve(directory)), resolve(tmpdir()));
    await rm(directory, { recursive: true, force: true });
  }
});

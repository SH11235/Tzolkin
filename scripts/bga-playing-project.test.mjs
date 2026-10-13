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
  BOUNDARY_CONTROL_SCHEMA,
  BOUNDARY_JUNGLE_TOPS_SCHEMA,
  BOUNDARY_MARKER_CALIBRATION_SCHEMA,
  BOUNDARY_MARKER_CELL_CALIBRATION_SCHEMA,
  BOUNDARY_WORKER_CELL_SCHEMA,
  CONTROL_SCHEMA,
  LATER_WITNESS_SCHEMA,
  decodeProjectedInitial,
  projectCapturedInitial,
  projectRetainedBoundary,
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

function fragmentFixture(n = 4) {
  const value = fixture(n);
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

function boundaryFixture({ split = false, terminal = false, n = 4, jungleTops = false } = {}) {
  const value = fragmentFixture(n),
    capture = value.capture;
  const displayedRound = terminal ? 28 : split ? 22 : 9;
  const actionId = split ? 9 : 8;
  capture.schema = 'bga-retained-visible-replay-board-v1';
  capture.capturedAt = '2000-01-01T00:00:00.000Z';
  capture.moveText = `Coup #${actionId}`;
  capture.roundText = `経過日 ${displayedRound} / 27`;
  capture.boardHtml = capture.boardHtml.replace(
    element('span', 'current_round', '1'),
    element('span', 'current_round', String(displayedRound)),
  );
  capture.renderedReplayControls = 'Next comment';
  capture.prompt = terminal ? 'ゲーム終了\n' : 'Aはワーカーを置くか取り除くかしてください';
  // The three-player worker fixture supplies observed post-rotation dummy positions below.
  if (n === 3)
    for (let index = 1; index <= 6; index++)
      capture.boardHtml = capture.boardHtml.replace(
        element('div', `worker_${index}`, '', ' class="worker worker_dummy"'),
        '',
      );
  const colors = ['008000', 'ff0000', '0000ff', 'ffa500'];
  capture.playerBoards.forEach((player, index) => {
    player.rowHtml = player.rowHtml.replace('color:#008000', `color:#${colors[index]}`);
    player.boardHtml = player.boardHtml.replace(
      element('span', `worker_number_player_${player.playerId}`, '3'),
      element(
        'span',
        `worker_number_player_${player.playerId}`,
        String(index === 0 ? 1 : index === 1 ? 2 : 3),
      ),
    );
    player.boardHtml =
      player.boardHtml.slice(0, -6) +
      `<div class="board_item"><div id="workers_player_${player.playerId}" class="worker worker_${colors[index]}"></div></div></div>`;
  });
  const placed = [0, 0, 1].map((owner, index) => ({
    id: `worker_${index + 1}`,
    className: `worker worker_${colors[owner]}`,
    inlineStyle: `top: 200px; left: ${100 + index * 30}px;`,
    parentId: 'workers',
    computedDisplay: 'block',
    computedVisibility: 'visible',
    rect: { x: 100 + index * 30, y: 200, width: 20, height: 30 },
  }));
  capture.boardHtml =
    capture.boardHtml.slice(0, -6) +
    element(
      'div',
      'workers',
      placed
        .map((worker) =>
          element(
            'div',
            worker.id,
            '',
            ` class="${worker.className}" style="${worker.inlineStyle}"`,
          ),
        )
        .join(''),
    ) +
    '</div>';
  capture.retainedDomGeometry = [
    {
      id: 'board',
      className: '',
      inlineStyle: null,
      parentId: 'board-container',
      computedDisplay: 'block',
      computedVisibility: 'visible',
      rect: { x: 0, y: 0, width: 1000, height: 900 },
    },
    {
      id: 'workers',
      className: null,
      inlineStyle: null,
      parentId: 'board',
      computedDisplay: 'block',
      computedVisibility: 'visible',
      rect: { x: 0, y: 900, width: 1000, height: 0 },
    },
    ...capture.playerBoards.map((player, index) => ({
      id: `workers_player_${player.playerId}`,
      className: `worker worker_${colors[index]}`,
      inlineStyle: null,
      parentId: '',
      computedDisplay: 'inline-block',
      computedVisibility: 'visible',
      rect: { x: 1100, y: index * 100, width: 20, height: 30 },
    })),
    ...placed,
  ];
  if (jungleTops) {
    const corners = [
      null,
      ['bottom', 'right'],
      ['bottom', 'left'],
      ['top', 'right'],
      ['top', 'left'],
    ];
    for (const box of [2, 3, 4, 5]) {
      const tops =
        box === 2
          ? Array.from({ length: n }, (_unused, i) => [i + 1, 'corn'])
          : box === 3
            ? [
                [1, 'wood'],
                [2, 'wood'],
                [3, 'wood'],
              ]
            : box === 4
              ? []
              : [
                  [1, 'wood'],
                  [4, 'corn'],
                ];
      const containerRect = { x: box === 3 ? -2 : box * 100, y: 300, width: 80, height: 80 };
      const geometry = (id, className, inlineStyle, parentId, rect) => ({
        id,
        className,
        inlineStyle,
        parentId,
        computedDisplay: 'block',
        computedVisibility: 'visible',
        rect,
      });
      const tileHtml = tops
        .map(([slot, kind]) => {
          const id = `jungle_${box}_tile_${slot}`,
            className = `tile board_tile ${kind}_tile`,
            [vertical, horizontal] = corners[slot],
            inlineStyle = `${vertical}:0px;${horizontal}:0px`;
          capture.retainedDomGeometry.push(
            geometry(id, className, inlineStyle, `jungle_${box}`, {
              x: containerRect.x + (horizontal === 'right' ? 42 : 0),
              y: containerRect.y + (vertical === 'bottom' ? 42 : 0),
              width: 38,
              height: 38,
            }),
          );
          return element('div', id, '', ` class="${className}" style="${inlineStyle}"`);
        })
        .join('');
      capture.retainedDomGeometry.push(
        geometry(`jungle_${box}`, 'jungle', null, 'board', containerRect),
      );
      const old = element(
        'div',
        `jungle_${box}`,
        Array.from({ length: n }, (_unused, i) =>
          element(
            'div',
            `jungle_${box}_tile_${i + 1}`,
            '',
            ` class="board_tile ${box === 2 ? 'corn_tile' : 'wood_tile'}"`,
          ),
        ).join(''),
      );
      assert.ok(capture.boardHtml.includes(old));
      capture.boardHtml = capture.boardHtml.replace(
        old,
        element('div', `jungle_${box}`, tileHtml, ' class="jungle"'),
      );
    }
    // Rotated top-tile bounding boxes can overlap without sharing a CSS corner.
    capture.retainedDomGeometry.find((item) => item.id === 'jungle_3_tile_1').rect.x = 20;
    capture.retainedDomGeometry.find((item) => item.id === 'jungle_3_tile_2').rect.y = 325;
    const row = capture.playerBoards[0];
    row.rowHtml =
      row.rowHtml.slice(0, -6) +
      element(
        'div',
        `corn_tile_player_${row.playerId}`,
        '',
        ' class="tile player_tile corn_tile"',
      ) +
      '</div>';
  }
  const feeding = {
    raw_text: '食料供給日: Aはワーカー3個で 6\nを支払った',
    html: '食料供給日: Aはワーカー3個で 6<div class="tz_icon resource_corn"></div>を支払った',
    icons: [{ classes: 'tz_icon resource_corn' }],
  };
  const rotation = { raw_text: '歯車が進んだ', html: '歯車が進んだ', icons: [] };
  const end = { raw_text: 'ゲーム終了', html: 'ゲーム終了', icons: [] };
  for (const [id, messages] of split
    ? [
        [8, [feeding]],
        [9, [rotation]],
      ]
    : [[8, [feeding, rotation, ...(terminal ? [end] : [])]]]) {
    value.raw.entries.push({
      action_id: id,
      raw_text: messages.map((message) => message.raw_text).join('\n'),
      timestamp_display: '12:01:00',
    });
    value.raw.dom_entries.push({ action_id: id, messages, timestamp_display: '12:01:00' });
  }
  value.raw.entry_count = value.raw.entries.length;
  const repin = () => {
    const retained = bytes(capture),
      raw = bytes(value.raw);
    return [
      retained,
      raw,
      bytes({
        schema: BOUNDARY_CONTROL_SCHEMA,
        capture: pin(retained),
        rawLog: pin(raw),
        archiveRelease: 'fixture',
        seatOrder: value.control.seatOrder,
        mechanicalProfile: value.control.mechanicalProfile,
        ...(jungleTops
          ? {
              jungleTops: {
                schema: BOUNDARY_JUNGLE_TOPS_SCHEMA,
                archiveRelease: 'fixture',
                stackMechanics: 'basic-corn-under-wood-v1',
              },
            }
          : {}),
      }),
    ];
  };
  return { ...value, repin };
}

function boundaryMarkerFixture(options = {}) {
  const value = boundaryFixture(options);
  const colors = ['008000', 'ff0000', '0000ff', 'ffa500'];
  const templeLevels = [
    [-1, 7, 6],
    [5, 0, 1],
    [0, -1, 2],
    [1, 3, -1],
  ];
  const technologyLevels = [
    [3, 0, 1, 2],
    [0, 3, 2, 1],
    [1, 2, 3, 0],
    [2, 1, 0, 3],
  ];
  const markers = (kind, tracks, levels) =>
    value.control.seatOrder
      .flatMap((id, index) =>
        tracks.map((track, j) =>
          element(
            'div',
            `marker_${id}_${kind}_${track}`,
            '',
            ` class="marker marker_${colors[index]}" style="top: ${kind === 'temple' ? 100 - 10 * levels[index][j] : j * 20}px; left: ${kind === 'temple' ? j * 20 : 10 * levels[index][j]}px;"`,
          ),
        ),
      )
      .join('');
  value.capture.boardHtml =
    value.capture.boardHtml.replace(/<div id="marker_\d+_[^"]+"[^>]*><\/div>/g, '').slice(0, -6) +
    element('div', 'temples', markers('temple', templeNames, templeLevels)) +
    element('div', 'technologies', markers('technology', markerNames, technologyLevels)) +
    '</div>';
  const tracks = JSON.parse(catalogBytes).TEMPLE_TRACKS;
  const positions = (min, max, coordinate) =>
    Array.from({ length: max - min + 1 }, (_v, index) => {
      const level = min + index;
      return { pixels: coordinate(level), level };
    });
  const calibration = {
    schema: BOUNDARY_MARKER_CALIBRATION_SCHEMA,
    archiveRelease: 'fixture',
    catalogSha256: sha(catalogBytes),
    templeTop: Object.fromEntries(
      Object.keys(tracks).map((track) => [
        track,
        positions(-1, tracks[track].points.length - 2, (level) => 100 - 10 * level),
      ]),
    ),
    technologyLeft: Object.fromEntries(
      markerNames.map((track) => [track, positions(0, 3, (level) => 10 * level)]),
    ),
  };
  return {
    ...value,
    templeLevels,
    technologyLevels,
    repinMarkers: (mutate = () => {}) => {
      const prepared = value.repin();
      const declared = { ...structuredClone(calibration), captureSha256: sha(prepared[0]) };
      mutate(declared);
      const calibrated = bytes(declared),
        control = JSON.parse(prepared[2]);
      control.markerCalibration = pin(calibrated);
      return [prepared[0], prepared[1], bytes(control), null, calibrated];
    },
  };
}

function boundaryCellMarkerFixture(options = {}) {
  const value = boundaryMarkerFixture(options);
  value.templeLevels[1][0] = -1;
  value.technologyLevels[1][0] = 3;
  const colors = ['008000', 'ff0000', '0000ff', 'ffa500'];
  const groups = [
    ['temples', 'temple', templeNames, value.templeLevels, { x: 0, y: 0, width: 300, height: 250 }],
    [
      'technologies',
      'technology',
      markerNames,
      value.technologyLevels,
      { x: 400, y: 0, width: 300, height: 250 },
    ],
  ];
  const blocks = groups.map(([group, kind, tracks, levels, containerRect]) => {
    const cells = [],
      markers = [];
    value.capture.retainedDomGeometry.push({
      id: group,
      className: '',
      inlineStyle: null,
      parentId: 'board',
      computedDisplay: 'block',
      computedVisibility: 'visible',
      rect: containerRect,
    });
    const add = (id, className, inlineStyle, rect) => {
      value.capture.retainedDomGeometry.push({
        id,
        className,
        inlineStyle,
        parentId: group,
        computedDisplay: 'block',
        computedVisibility: 'visible',
        rect,
      });
      return element('div', id, '', ` class="${className}" style="${inlineStyle}"`);
    };
    tracks.forEach((track, trackIndex) => {
      const canonical = track === 'kukulcan' ? 'kukulkan' : track;
      const count =
        kind === 'temple' ? JSON.parse(catalogBytes).TEMPLE_TRACKS[canonical].points.length : 5;
      for (let index = 0; index < count; index++) {
        const rect =
          kind === 'temple'
            ? { x: trackIndex * 100, y: 220 - index * 24, width: 80, height: 20 }
            : { x: 400 + index * 60, y: trackIndex * 50, width: 50, height: 40 };
        cells.push(
          add(
            `${kind}_${track}_${kind === 'temple' ? 'step' : 'level'}_${index}`,
            kind === 'temple' ? 'temple_step' : 'technology_level',
            kind === 'temple'
              ? `bottom: ${10 + index * 24}px; left: ${trackIndex * 100}px;`
              : `top: ${trackIndex * 50}px; left: ${index * 60}px; width: 50px;`,
            rect,
          ),
        );
      }
      value.control.seatOrder.forEach((id, playerIndex) => {
        const level = levels[playerIndex][trackIndex];
        const left =
          kind === 'temple'
            ? trackIndex * 100 + 8 + playerIndex * 16
            : level * 60 + 5 + playerIndex * 0.003;
        const top =
          kind === 'temple'
            ? 220 - (level + 1) * 24 + 4 + playerIndex * 0.05
            : trackIndex * 50 + 4 + playerIndex * 8;
        markers.push(
          add(
            `marker_${id}_${kind}_${track}`,
            `marker marker_${colors[playerIndex]}`,
            `top: ${top}px; left: ${left}px;`,
            { x: containerRect.x + left, y: top, width: kind === 'temple' ? 12 : 10, height: 8 },
          ),
        );
      });
    });
    return element('div', group, cells.join('') + markers.join(''));
  });
  value.capture.boardHtml =
    value.capture.boardHtml.slice(0, value.capture.boardHtml.indexOf('<div id="temples">')) +
    blocks.join('') +
    '</div>';
  value.repinMarkers = (mutate = () => {}) => {
    const prepared = value.repin();
    const calibration = {
      schema: BOUNDARY_MARKER_CELL_CALIBRATION_SCHEMA,
      archiveRelease: 'fixture',
      captureSha256: sha(prepared[0]),
      catalogSha256: sha(catalogBytes),
    };
    mutate(calibration);
    const calibrated = bytes(calibration),
      control = JSON.parse(prepared[2]);
    control.markerCalibration = pin(calibrated);
    return [prepared[0], prepared[1], bytes(control), null, calibrated];
  };
  return value;
}

test('retained boundary binds original food/rotation scope and separates available from total workers', () => {
  for (const split of [false, true]) {
    const value = boundaryFixture({ split }),
      scores = [7, -2, 13, 0];
    value.capture.playerBoards.forEach((player, index) => {
      player.boardHtml = player.boardHtml.replace(
        element('span', `player_score_${player.playerId}`, '0'),
        element('span', `player_score_${player.playerId}`, String(scores[index])),
      );
    });
    const projected = projectRetainedBoundary(...value.repin());
    assert.equal(projected.witness.actionId, split ? 9 : 8);
    assert.equal(projected.witness.expected.round, split ? 22 : 9);
    assert.deepEqual(projected.provenance.sourceActionIds, split ? [8, 9] : [8]);
    assert.equal(projected.provenance.foodDay, split ? 21 : 8);
    assert.deepEqual(
      projected.witness.expected.players.map((player) => player.workers),
      [3, 3, 3, 3],
    );
    assert.deepEqual(
      projected.provenance.displayedPlayers.map((player) => player.availableWorkers),
      [1, 2, 3, 3],
    );
    assert.deepEqual(
      projected.witness.expected.players.map((player) => player.score),
      scores,
    );
    assert.deepEqual(
      projected.provenance.displayedPlayers.map((player) => player.scoreDisplay),
      scores,
    );
    assert.equal(projected.provenance.omitted.includes('exact scores'), false);
    assert.equal(projected.provenance.completeSourceBoard, false);
    assert.equal(projected.provenance.humanBcQualified, false);
    assert.equal(projected.provenance.policyTrainingReady, false);
    assert.equal(Object.hasOwn(projected.witness.expected, 'foodDays'), false);
    assert.equal(Object.hasOwn(projected.witness.expected, 'gears'), false);
    assert.equal(Object.hasOwn(projected.provenance, 'markerObservation'), false);
  }
  for (const options of [{}, { split: true }, { terminal: true }]) {
    for (const createFixture of [boundaryMarkerFixture, boundaryCellMarkerFixture]) {
      const value = createFixture(options),
        projected = projectRetainedBoundary(...value.repinMarkers());
      const players = projected.witness.expected.players;
      assert.deepEqual(
        players.map((p) => Object.values(p.temples)),
        value.templeLevels,
      );
      assert.deepEqual(
        players.map((p) => Object.values(p.technologies)),
        value.technologyLevels,
      );
      assert.equal(projected.provenance.markerObservation.calibrationAuthenticated, false);
      assert.equal(projected.provenance.completeSourceBoard, false);
      assert.equal(projected.provenance.humanBcQualified, false);
      assert.equal(projected.provenance.omitted.includes('markers'), false);
      assert.equal(Object.hasOwn(players[0], 'score'), !options.terminal);
      assert.match(projected.provenance.markerObservation.interpretation, /consistency only/);
    }
  }
  const workerFixture = (n = 4) => {
    const value = boundaryFixture({ n });
    const cells = [
      'workerplace_startingPlayer_0',
      ...GEARS.flatMap((gear) =>
        Array.from(
          { length: gear === 'chichenItza' ? 13 : 10 },
          (_unused, index) => `workerplace_${gear}_${index}`,
        ),
      ),
    ].map((id, index) => ({
      id,
      className: 'workerplace',
      inlineStyle: null,
      parentId: 'board',
      computedDisplay: 'block',
      computedVisibility: 'visible',
      rect: {
        x: 500 + (index % 10) * 25,
        y: 400 + Math.floor(index / 10) * 25,
        width: 19,
        height: 19,
      },
    }));
    for (const [index, id] of [
      'workerplace_palenque_1',
      'workerplace_yaxchilan_5',
      'workerplace_tikal_2',
    ].entries())
      cells.find((cell) => cell.id === id).rect = {
        x: 100 + index * 30,
        y: 205,
        width: 19,
        height: 19,
      };
    value.capture.retainedDomGeometry.push(...cells);
    if (n === 3) {
      const dummySlots = [
        'workerplace_palenque_9',
        'workerplace_yaxchilan_0',
        'workerplace_tikal_6',
        'workerplace_uxmal_1',
        'workerplace_uxmal_9',
        'workerplace_chichenItza_12',
      ];
      const dummies = dummySlots.map((id, index) => {
        const rect = cells.find((cell) => cell.id === id).rect;
        return {
          id: `worker_${index + 4}`,
          className: 'worker worker_dummy',
          inlineStyle: `top: ${rect.y + 2}px; left: ${rect.x + 2}px;`,
          parentId: 'workers',
          computedDisplay: 'block',
          computedVisibility: 'visible',
          rect: { x: rect.x + 2, y: rect.y + 2, width: 10, height: 15 },
        };
      });
      const dummyHtml = dummies.map((worker) =>
        element('div', worker.id, '', ` class="${worker.className}" style="${worker.inlineStyle}"`),
      );
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="workers">',
        `<div id="workers">${dummyHtml.join('')}`,
      );
      value.capture.retainedDomGeometry.push(...dummies);
      value.dummySlots = dummySlots;
    }
    value.repinWorkers = () => {
      const prepared = value.repin(),
        control = JSON.parse(prepared[2]);
      control.workerPositions = { schema: BOUNDARY_WORKER_CELL_SCHEMA, archiveRelease: 'fixture' };
      return [prepared[0], prepared[1], bytes(control)];
    };
    return value;
  };
  const positions = workerFixture(),
    beforeOptIn = projectRetainedBoundary(...positions.repin()),
    observed = projectRetainedBoundary(...positions.repinWorkers());
  assert.equal(Object.hasOwn(beforeOptIn.witness.expected, 'gears'), false);
  assert.deepEqual(observed.witness.expected.gears.palenque[1], { playerId: 0, dummy: false });
  assert.deepEqual(observed.witness.expected.gears.yaxchilan[5], { playerId: 0, dummy: false });
  assert.deepEqual(observed.witness.expected.gears.tikal[2], { playerId: 1, dummy: false });
  assert.equal(Object.values(observed.witness.expected.gears).flat().filter(Boolean).length, 3);
  assert.equal(observed.witness.expected.gears.chichenItza.length, 13);
  assert.equal(observed.provenance.omitted.includes('gear positions'), false);
  assert.equal(observed.provenance.workerPositionObservation.sourceAuthenticated, false);
  assert.equal(observed.provenance.completeSourceBoard, false);
  assert.equal(observed.provenance.humanBcQualified, false);
  const setCellClass = (value, id, className) => {
    value.capture.boardHtml = value.capture.boardHtml.replace(
      `id="${id}" class="workerplace"`,
      `id="${id}" class="${className}"`,
    );
    value.capture.retainedDomGeometry.find((item) => item.id === id).className = className;
  };
  const activePositions = workerFixture();
  for (const id of [
    'workerplace_startingPlayer_0',
    ...GEARS.map((gear) => `workerplace_${gear}_0`),
  ])
    setCellClass(activePositions, id, 'workerplace active');
  const activeBeforeOptIn = projectRetainedBoundary(...activePositions.repin()),
    activeObserved = projectRetainedBoundary(...activePositions.repinWorkers());
  assert.deepEqual(activeBeforeOptIn.witness.expected, beforeOptIn.witness.expected);
  assert.equal(Object.hasOwn(activeBeforeOptIn.witness.expected, 'gears'), false);
  assert.deepEqual(activeObserved.witness.expected, observed.witness.expected);
  assert.equal(activeObserved.provenance.workerPositionObservation.sourceAuthenticated, false);
  for (const className of [
    'workerplace active active',
    'workerplace workerplace',
    'workerplace unknown',
    'active',
  ]) {
    const value = workerFixture();
    setCellClass(value, 'workerplace_palenque_0', className);
    assert.throws(
      () => projectRetainedBoundary(...value.repinWorkers()),
      /Incomplete\/unknown boundary worker cells/,
    );
  }
  const detachedActive = workerFixture();
  setCellClass(detachedActive, 'workerplace_palenque_0', 'workerplace active');
  detachedActive.capture.retainedDomGeometry.find(
    (item) => item.id === 'workerplace_palenque_0',
  ).className = 'workerplace';
  assert.throws(
    () => projectRetainedBoundary(...detachedActive.repinWorkers()),
    /Detached consumed boundary geometry/,
  );
  const threePlayers = workerFixture(3),
    threeObserved = projectRetainedBoundary(...threePlayers.repinWorkers());
  assert.equal(threeObserved.witness.expected.players.length, 3);
  assert.equal(
    Object.values(threeObserved.witness.expected.gears)
      .flat()
      .filter((worker) => worker?.dummy).length,
    6,
  );
  for (const id of threePlayers.dummySlots) {
    const [, gear, position] = /^workerplace_(\w+)_(\d+)$/.exec(id);
    assert.deepEqual(threeObserved.witness.expected.gears[gear][Number(position)], {
      playerId: -1,
      dummy: true,
    });
  }
  for (const remaining of [5, 0]) {
    const value = workerFixture(3);
    for (let index = remaining; index < 6; index++) {
      const id = `worker_${index + 4}`,
        worker = value.capture.retainedDomGeometry.find((item) => item.id === id);
      value.capture.boardHtml = value.capture.boardHtml.replace(
        element('div', id, '', ` class="${worker.className}" style="${worker.inlineStyle}"`),
        '',
      );
      value.capture.retainedDomGeometry = value.capture.retainedDomGeometry.filter(
        (item) => item.id !== id,
      );
    }
    assert.throws(
      () => projectRetainedBoundary(...value.repinWorkers()),
      /Incomplete\/excess boundary dummy/,
    );
    assert.equal(
      Object.hasOwn(projectRetainedBoundary(...value.repin()).witness.expected, 'gears'),
      false,
    );
  }
  const startDummy = workerFixture(3),
    startRect = startDummy.capture.retainedDomGeometry.find(
      (item) => item.id === 'workerplace_startingPlayer_0',
    ).rect,
    startWorker = startDummy.capture.retainedDomGeometry.find((item) => item.id === 'worker_4'),
    startStyle = `top: ${startRect.y + 2}px; left: ${startRect.x + 2}px;`;
  startDummy.capture.boardHtml = startDummy.capture.boardHtml.replace(
    element(
      'div',
      startWorker.id,
      '',
      ` class="${startWorker.className}" style="${startWorker.inlineStyle}"`,
    ),
    element('div', startWorker.id, '', ` class="${startWorker.className}" style="${startStyle}"`),
  );
  startWorker.inlineStyle = startStyle;
  startWorker.rect = {
    x: startRect.x + 2,
    y: startRect.y + 2,
    width: 10,
    height: 15,
  };
  assert.throws(
    () => projectRetainedBoundary(...startDummy.repinWorkers()),
    /Dummy worker on starting-player space/,
  );
  for (const mutate of [
    (value) =>
      value.capture.retainedDomGeometry.splice(
        value.capture.retainedDomGeometry.findIndex((g) => g.id === 'workerplace_tikal_2'),
        1,
      ),
    (value) => {
      value.capture.retainedDomGeometry.find(
        (g) => g.id === 'workerplace_tikal_2',
      ).computedVisibility = 'hidden';
    },
    (value) => {
      value.capture.retainedDomGeometry.find((g) => g.id === 'workerplace_tikal_2').parentId =
        'foreign';
    },
    (value) => {
      value.capture.retainedDomGeometry.find((g) => g.id === 'workerplace_tikal_2').rect.x = -1;
    },
    (value) => {
      value.capture.retainedDomGeometry.find((g) => g.id === 'workerplace_tikal_3').rect =
        structuredClone(
          value.capture.retainedDomGeometry.find((g) => g.id === 'workerplace_tikal_2').rect,
        );
    },
    (value) => {
      value.capture.retainedDomGeometry.find((g) => g.id === 'worker_2').rect = structuredClone(
        value.capture.retainedDomGeometry.find((g) => g.id === 'worker_1').rect,
      );
    },
    (value) => {
      const cells = value.capture.retainedDomGeometry;
      [
        cells.find((g) => g.id === 'workerplace_tikal_2').rect,
        cells.find((g) => g.id === 'workerplace_tikal_8').rect,
      ] = [
        cells.find((g) => g.id === 'workerplace_tikal_8').rect,
        cells.find((g) => g.id === 'workerplace_tikal_2').rect,
      ];
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="workerplace_tikal_2" class="workerplace"',
        'id="workerplace_tikal_2" class="workerplace" style="opacity:0%"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'worker worker_ff0000',
        'worker worker_dummy',
      );
      value.capture.retainedDomGeometry.find((g) => g.id === 'worker_3').className =
        'worker worker_dummy';
    },
  ]) {
    const value = workerFixture();
    mutate(value);
    assert.throws(
      () => projectRetainedBoundary(...value.repinWorkers()),
      /boundary|Hidden retained/i,
    );
  }
  const wrongRelease = positions.repinWorkers(),
    workerControl = JSON.parse(wrongRelease[2]);
  workerControl.workerPositions.archiveRelease = 'foreign';
  assert.throws(
    () => projectRetainedBoundary(wrongRelease[0], wrongRelease[1], bytes(workerControl)),
    /release-mismatched/,
  );
  for (const n of [3, 4]) {
    const value = boundaryFixture({ n, jungleTops: true }),
      prepared = value.repin(),
      projected = projectRetainedBoundary(...prepared),
      control = JSON.parse(prepared[2]);
    assert.deepEqual(projected.witness.expected.jungle, {
      2: { corn: n, wood: 0 },
      3: { corn: 3, wood: 3 },
      4: { corn: 0, wood: 0 },
      5: { corn: 2, wood: 1 },
    });
    assert.deepEqual(projected.provenance.jungleObservation.visibleTops[3], {
      exposedCornTops: 0,
      woodTops: 3,
    });
    assert.deepEqual(
      projected.provenance.jungleObservation.conditionalDerivedCounts,
      projected.witness.expected.jungle,
    );
    assert.equal(projected.provenance.jungleObservation.coveredLayersObserved, false);
    assert.equal(projected.provenance.jungleObservation.sourceAuthenticated, false);
    assert.equal(projected.provenance.omitted.includes('covered corn'), true);
    for (const key of [
      'visibilityVerified',
      'geometryVerified',
      'sourceAuthenticated',
      'completeSourceBoard',
      'completeNativeReplayQualified',
      'humanBcQualified',
      'policyTrainingReady',
      'valueTrainingReady',
    ])
      assert.equal(projected.provenance[key], false);
    assert.equal(Object.hasOwn(projected.witness.expected, 'skullSpaces'), false);
    assert.equal(Object.hasOwn(projected.witness.expected, 'buildings'), false);
    delete control.jungleTops;
    const without = projectRetainedBoundary(prepared[0], prepared[1], bytes(control)),
      original = projectRetainedBoundary(...boundaryFixture({ n }).repin());
    assert.deepEqual(without.witness, original.witness);
    const withoutProvenance = { ...without.provenance },
      originalProvenance = { ...original.provenance };
    delete withoutProvenance.captureSha256;
    delete originalProvenance.captureSha256;
    assert.deepEqual(withoutProvenance, originalProvenance);
  }
  const jungleNode = (value, id) =>
    value.capture.retainedDomGeometry.find((item) => item.id === id);
  for (const [axis, size] of [
    ['x', 'width'],
    ['y', 'height'],
  ]) {
    const value = boundaryFixture({ jungleTops: true }),
      original = projectRetainedBoundary(...value.repin()),
      container = jungleNode(value, 'jungle_3').rect,
      tile = jungleNode(value, 'jungle_3_tile_1').rect;
    tile[axis] = container[axis] + container[size] - tile[size] + 0.125;
    assert.ok(tile[axis] + tile[size] > container[axis] + container[size]);
    const projected = projectRetainedBoundary(...value.repin());
    assert.deepEqual(projected.witness.expected, original.witness.expected);
    assert.deepEqual(projected.provenance.jungleObservation, original.provenance.jungleObservation);
    assert.equal(projected.provenance.geometryVerified, false);
    assert.equal(projected.provenance.completeNativeReplayQualified, false);
    assert.equal(projected.provenance.humanBcQualified, false);
  }
  for (const mutate of [
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_2_tile_1" class="tile board_tile corn_tile"',
        'id="jungle_2_tile_1" class="tile board_tile wood_tile"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_3_tile_1" class="tile board_tile wood_tile"',
        'id="jungle_3_tile_1" class="tile board_tile wood_tile corn_tile"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_3_tile_1" class="tile board_tile wood_tile"',
        'id="jungle_3_tile_1" class="tile board_tile unknown"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_3_tile_1"',
        'id="jungle_3_tile_0"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_3_tile_1"',
        'id="jungle_3_tile_1" onclick="void(0)"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="jungle_3_tile_1" class="tile board_tile wood_tile" style="bottom:0px;right:0px"',
        'id="jungle_3_tile_1" class="tile board_tile wood_tile" style="top:0px;right:0px"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="jungle_4" class="jungle"></div>',
        '',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="jungle_4" class="jungle"></div>',
        '<div id="foreign"><div id="jungle_4" class="jungle"></div></div>',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="jungle_3" class="jungle">',
        '<div id="jungle_3" class="jungle" style="opacity:0%">',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="jungle_4" class="jungle"></div>',
        '<div id="jungle_4" class="jungle"><span>unknown</span></div>',
      );
    },
    (value) => {
      value.capture.boardHtml =
        value.capture.boardHtml.slice(0, -6) + '<div id="ghost" class="corn_tile"></div></div>';
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="jungle_4" class="jungle"></div>',
        '<div id="jungle_4" class="jungle"><div id="jungle_3_tile_1" class="tile board_tile wood_tile" style="bottom:0px;right:0px"></div></div>',
      );
    },
    (value) => {
      value.capture.retainedDomGeometry = value.capture.retainedDomGeometry.filter(
        (item) => item.id !== 'jungle_4',
      );
    },
    (value) => {
      jungleNode(value, 'jungle_3').parentId = 'foreign';
    },
    (value) => {
      jungleNode(value, 'jungle_3').computedVisibility = 'hidden';
    },
    (value) => {
      jungleNode(value, 'jungle_3').rect.width = 0;
    },
    (value) => {
      jungleNode(value, 'jungle_3').rect.x = -100;
    },
    (value) => {
      jungleNode(value, 'jungle_3_tile_1').rect.width = Infinity;
    },
    (value) => {
      const tile = jungleNode(value, 'jungle_3_tile_1').rect;
      tile.x = jungleNode(value, 'jungle_3').rect.x - tile.width;
    },
    (value) => {
      const tile = jungleNode(value, 'jungle_3_tile_1').rect,
        container = jungleNode(value, 'jungle_3').rect;
      tile.y = container.y + container.height;
    },
    (value) => {
      const tile = jungleNode(value, 'jungle_3_tile_1').rect,
        container = jungleNode(value, 'jungle_3').rect;
      tile.x = container.x;
      tile.width = container.width + 1;
    },
    (value) => {
      const tile = jungleNode(value, 'jungle_3_tile_1').rect,
        container = jungleNode(value, 'jungle_3').rect;
      tile.y = container.y;
      tile.height = container.height + 1;
    },
    (value) => {
      jungleNode(value, 'jungle_3_tile_1').inlineStyle = 'top:0px;right:0px';
    },
    (value) => {
      value.capture.retainedDomGeometry.push({
        ...structuredClone(jungleNode(value, 'jungle_3_tile_1')),
        id: 'jungle_9_tile_1',
      });
    },
  ]) {
    const value = boundaryFixture({ jungleTops: true });
    mutate(value);
    assert.throws(
      () => projectRetainedBoundary(...value.repin()),
      /boundary|jungle|Hidden retained/i,
    );
  }
  const overfull = boundaryFixture({ n: 3, jungleTops: true });
  overfull.capture.boardHtml = overfull.capture.boardHtml.replace(
    '<div id="jungle_4" class="jungle"></div>',
    '<div id="jungle_4" class="jungle">' +
      Array.from({ length: 4 }, (_unused, i) =>
        element(
          'div',
          `jungle_4_tile_${i + 1}`,
          '',
          ` class="tile board_tile corn_tile" style="${i < 2 ? 'bottom' : 'top'}:0px;${i % 2 ? 'left' : 'right'}:0px"`,
        ),
      ).join('') +
      '</div>',
  );
  assert.throws(() => projectRetainedBoundary(...overfull.repin()), /player-count/);
  for (const mutate of [
    (control) => {
      control.jungleTops.archiveRelease = 'foreign';
    },
    (control) => {
      control.jungleTops.stackMechanics = 'unknown';
    },
    (control) => {
      control.jungleTops.expected = { corn: 4 };
    },
    (control) => {
      delete control.jungleTops.stackMechanics;
    },
  ]) {
    const value = boundaryFixture({ jungleTops: true }),
      prepared = value.repin(),
      control = JSON.parse(prepared[2]);
    mutate(control);
    assert.throws(
      () => projectRetainedBoundary(prepared[0], prepared[1], bytes(control)),
      /boundary jungle/,
    );
  }
  const modern = boundaryFixture();
  Object.assign(modern.raw, {
    ...modern.raw,
    schema_version: 1,
    source_type: 'public_ui_dom',
    export_scope: 'all_rendered_gamereview_log_entries',
    dom_entry_count: modern.raw.dom_entries.length,
    metadata: {
      tableUrl: 'https://boardgamearena.com/table?table=123',
      playerCount: 4,
      players: ['A', 'B', 'C', 'D'].map((name) => ({ name })),
    },
  });
  delete modern.raw.table_details_text;
  modern.raw.dom_entries.forEach((entry) =>
    entry.messages.forEach((message) => {
      message.icons = message.icons.map(({ classes }) => ({ class: classes }));
    }),
  );
  modern.raw.entries = modern.raw.dom_entries.map((entry) => ({
    action_id: entry.action_id,
    timestamp_display: entry.timestamp_display,
    raw_text: `行動 ${entry.action_id} :\n${entry.timestamp_display}\n${entry.messages.map((message) => message.raw_text).join('\n')}`,
  }));
  const prepared = modern.repin(),
    control = JSON.parse(prepared[2]);
  const table = bytes({
    schema: 'tzolkin-bga-public-table-ui-v1',
    url: 'https://boardgamearena.com/table?table=123',
    optionsText: '',
  });
  control.table = pin(table);
  assert.equal(
    projectRetainedBoundary(prepared[0], prepared[1], bytes(control), table).witness.expected.round,
    9,
  );
});

test('retained boundary omits total workers when complete visible geometry is missing', () => {
  for (const mutate of [
    (value) => value.capture.retainedDomGeometry.shift(),
    (value) => value.capture.retainedDomGeometry.pop(),
    (value) => {
      value.capture.retainedDomGeometry.at(-1).computedVisibility = 'hidden';
    },
    (value) => {
      delete value.capture.retainedDomGeometry.at(-1).computedDisplay;
    },
  ]) {
    const value = boundaryFixture();
    mutate(value);
    const projected = projectRetainedBoundary(...value.repin());
    assert.equal(projected.provenance.totalWorkersObserved, false);
    assert.ok(
      projected.witness.expected.players.every((player) => !Object.hasOwn(player, 'workers')),
    );
    assert.equal(projected.witness.expected.players[0].resources.corn, 8);
    assert.equal(projected.provenance.displayedPlayers[0].availableWorkers, 1);
    assert.match(projected.provenance.totalWorkersOmissionReason, /canvas worker geometry/);
  }
  const anonymous = boundaryFixture();
  anonymous.capture.retainedDomGeometry.push(
    ...anonymous.capture.playerBoards.map(() => ({
      id: '',
      parentId: 'tzolkin-interface',
      className: 'whiteblock',
      inlineStyle: null,
    })),
  );
  assert.equal(projectRetainedBoundary(...anonymous.repin()).provenance.totalWorkersObserved, true);
  const unowned = structuredClone(anonymous.capture.retainedDomGeometry.at(-1));
  anonymous.capture.retainedDomGeometry.at(-1).parentId = 'unretained-parent';
  assert.throws(() => projectRetainedBoundary(...anonymous.repin()), /Unknown anonymous/);
  anonymous.capture.retainedDomGeometry[anonymous.capture.retainedDomGeometry.length - 1] = unowned;
  anonymous.capture.retainedDomGeometry.pop();
  assert.throws(
    () => projectRetainedBoundary(...anonymous.repin()),
    /Anonymous boundary row metadata/,
  );
  anonymous.capture.retainedDomGeometry.push(unowned);
  anonymous.capture.retainedDomGeometry.at(-1).parentId = 'workers';
  assert.throws(
    () => projectRetainedBoundary(...anonymous.repin()),
    /Unidentified boundary worker/,
  );
});

test('retained boundary keeps terminal day and integer score displays out of exact native assertions', () => {
  const value = boundaryFixture({ terminal: true });
  value.capture.tableId = 123;
  value.capture.playerBoards[0].boardHtml = value.capture.playerBoards[0].boardHtml.replace(
    element('span', 'player_score_100', '0'),
    element('span', 'player_score_100', '12'),
  );
  const projected = projectRetainedBoundary(...value.repin());
  assert.equal(projected.provenance.kind, 'terminal');
  assert.equal(projected.provenance.displayedRound, 28);
  assert.equal(projected.provenance.displayedPlayers[0].scoreDisplay, 12);
  assert.equal(projected.witness.expected.phase, 'finished');
  assert.equal(Object.hasOwn(projected.witness.expected, 'round'), false);
  assert.ok(projected.witness.expected.players.every((player) => !Object.hasOwn(player, 'score')));
  assert.equal(projected.provenance.omitted.includes('exact scores'), true);
  assert.equal(Object.hasOwn(projected, 'terminalCheckpoint'), false);
  assert.equal(projected.provenance.completeNativeReplayQualified, false);
  const outsideDom = boundaryFixture({ terminal: true });
  outsideDom.raw.dom_entries.at(-1).messages.pop();
  assert.equal(projectRetainedBoundary(...outsideDom.repin()).witness.expected.phase, 'finished');
  const double = boundaryFixture({ terminal: true });
  const message = {
    raw_text: 'Aは歯車を高速化し2段階回した!',
    html: 'Aは歯車を高速化し2段階回した!',
    icons: [],
  };
  double.raw.dom_entries.at(-1).messages.push(message);
  double.raw.entries.at(-1).raw_text += `\n${message.raw_text}`;
  assert.throws(() => projectRetainedBoundary(...double.repin()), /matching settled/);
});

test('retained boundary rejects identity, hidden-counter, ownership and original-source mismatches', () => {
  for (const mutate of [
    (value) => {
      value.capture.sourceUrl = value.capture.sourceUrl.replace('table=123', 'table=124');
    },
    (value) => {
      value.capture.sourceUrl += '&player=100';
    },
    (value) => {
      value.capture.moveText = 'Coup #99';
    },
    (value) => {
      value.capture.completeNativeReplayQualified = true;
    },
    (value) => {
      value.capture.playerBoards[0].boardHtml = value.capture.playerBoards[0].boardHtml.replace(
        '<div ',
        '<div hidden ',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace('worker_008000', 'worker_abcdef');
    },
    (value) => {
      value.capture.playerBoards[0].rowHtml = value.capture.playerBoards[0].rowHtml.replace(
        '<h3',
        `${element('span', 'corn_number_player_100', '0')}<h3`,
      );
    },
    (value) => {
      value.raw.dom_entries.at(-1).messages.pop();
      value.raw.entries.at(-1).raw_text = value.raw.dom_entries
        .at(-1)
        .messages.map((message) => message.raw_text)
        .join('\n');
    },
    (value) => {
      value.capture.retainedDomGeometry.push(structuredClone(value.capture.retainedDomGeometry[0]));
    },
    (value) => {
      value.capture.retainedDomGeometry.at(-1).inlineStyle = 'left: 999px;';
    },
    (value) => {
      value.capture.retainedDomGeometry.push({
        ...structuredClone(value.capture.retainedDomGeometry.at(-1)),
        id: 'ghost',
      });
    },
    (value) => {
      value.capture.retainedDomGeometry.push({
        ...structuredClone(value.capture.retainedDomGeometry.at(-1)),
        id: 'worker_999',
        parentId: 'unretained-parent',
        className: '',
      });
    },
    (value) => {
      value.capture.retainedDomGeometry[2].parentId = 'workers';
    },
    (value) => {
      value.capture.retainedDomGeometry[2].className = 'worker worker_abcdef';
    },
    (value) => {
      value.capture.retainedDomGeometry[2].id = 'workers_player_999';
    },
    (value) => {
      value.capture.playerBoards[0].boardHtml = value.capture.playerBoards[0].boardHtml.replace(
        'workers_player_100',
        'workers_player_200',
      );
    },
    (value) => {
      value.raw.entries.at(-1).raw_text += '\nAは新しいワーカーを獲得した';
    },
    (value) => {
      value.capture.retainedDomGeometry[0].computedDisplay = 'none';
    },
    ...[
      'current_round',
      'skull_left',
      'player_board_100',
      'corn_number_player_100',
      'player_score_100',
    ].map((id) => (value) => {
      value.capture.retainedDomGeometry.push({
        id,
        className: '',
        inlineStyle: null,
        parentId:
          id === 'current_round'
            ? 'rounds_wrap'
            : id === 'skull_left'
              ? 'board'
              : ['corn_number_player_100', 'player_score_100'].includes(id)
                ? 'player_board_100'
                : '',
        computedVisibility: 'hidden',
      });
    }),
    (value) => {
      value.capture.retainedDomGeometry.push({
        id: 'current_round',
        className: 'foreign',
        parentId: 'rounds_wrap',
      });
    },
    (value) => {
      value.capture.retainedDomGeometry.push({
        id: 'rounds_wrap',
        inlineStyle: 'display:none',
        parentId: 'board',
      });
    },
    ...['', ' class="worker_008000"'].map((attributes) => (value) => {
      value.capture.boardHtml =
        value.capture.boardHtml.slice(0, -6) +
        element('div', attributes ? 'canvasghost' : 'worker_999', '', attributes) +
        '</div>';
    }),
    ...['boardHtml', 'rowHtml'].flatMap((field) =>
      [
        element('div', 'worker_999'),
        '<div class="worker worker_ff0000"></div>',
        element('div', 'ghost', '', ' class="worker_008000"'),
        element('div', 'ghost', '', ' class="worker_dummy"'),
        element('div', 'ghost', '', ' class="worker_ABCDEF"'),
      ].map((html) => (value) => {
        const fragment = value.capture.playerBoards[0];
        fragment[field] = fragment[field].slice(0, -6) + html + '</div>';
      }),
    ),
    ...[
      element('div', 'worker_0'),
      element('div', 'worker_007'),
      element('div', 'ghost', '', ' class="worker_ABCDEF"'),
      element('div', 'ghost', '', ' class="worker_dummy"'),
    ].map((html) => (value) => {
      value.capture.boardHtml = value.capture.boardHtml.slice(0, -6) + html + '</div>';
    }),
    ...[
      { id: 'worker_0', className: '' },
      { id: 'worker_007', className: '' },
      { id: 'ghost', className: 'worker_ABCDEF' },
      { id: 'ghost', className: 'worker_dummy' },
    ].map((geometry) => (value) => {
      value.capture.retainedDomGeometry.push({ ...geometry, parentId: 'board' });
    }),
  ]) {
    const value = boundaryFixture();
    mutate(value);
    assert.throws(() => projectRetainedBoundary(...value.repin()));
  }
  const value = boundaryFixture(),
    prepared = value.repin();
  const changed = Buffer.from(prepared[0]);
  changed[0] ^= 1;
  assert.throws(() => projectRetainedBoundary(changed, ...prepared.slice(1)), /pin mismatch/);
  for (const split of [false, true]) {
    for (const caption of ['Aは新しいワーカーを獲得した', '未知の効果']) {
      const value = boundaryFixture({ split });
      value.raw.dom_entries[1].messages.push({ raw_text: caption, html: caption, icons: [] });
      value.raw.entries[1].raw_text += `\n${caption}`;
      assert.throws(() => projectRetainedBoundary(...value.repin()), /Unsupported message/);
    }
  }
  const residualPrior = boundaryFixture({ split: true });
  residualPrior.raw.entries[1].raw_text += '\nAは新しいワーカーを獲得した';
  assert.throws(
    () => projectRetainedBoundary(...residualPrior.repin()),
    /unsupported residual text/,
  );
  // Empty eventless entries are omitted by the same source normalization as the decoder.
  const empty = boundaryFixture({ split: true });
  empty.capture.moveText = 'Coup #10';
  empty.raw.entries.at(-1).action_id = 10;
  empty.raw.dom_entries.at(-1).action_id = 10;
  empty.raw.entries.splice(2, 0, { action_id: 9, raw_text: '', timestamp_display: '12:01:00' });
  empty.raw.dom_entries.splice(2, 0, { action_id: 9, messages: [], timestamp_display: '12:01:00' });
  empty.raw.entry_count++;
  assert.deepEqual(projectRetainedBoundary(...empty.repin()).provenance.sourceActionIds, [8, 10]);
  empty.raw.entries[2].raw_text = '未知の効果';
  empty.raw.dom_entries[2].messages = [{ raw_text: '未知の効果', html: '未知の効果', icons: [] }];
  assert.throws(() => projectRetainedBoundary(...empty.repin()), /matching settled/);
  for (const tableId of [123, 124]) {
    const legacy = boundaryFixture(),
      retained = legacy.repin(),
      control = JSON.parse(retained[2]),
      table = bytes({
        schema: 'tzolkin-bga-public-table-ui-v1',
        url: `https://boardgamearena.com/table?table=${tableId}`,
        optionsText: '',
      });
    control.table = pin(table);
    assert.throws(
      () => projectRetainedBoundary(retained[0], retained[1], bytes(control), table),
      /Legacy input requires embedded table settings/,
    );
  }
  for (const mutate of [
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'marker_100_temple_chaac',
        'marker_999_temple_chaac',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="marker_100_temple_chaac"',
        '<div hidden id="marker_100_temple_chaac"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'class="marker marker_008000"',
        'class="marker marker_ff0000"',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace('top: 110px;', 'top: 111px;');
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="temples">',
        '<div id="temples" inert>',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        /<div id="marker_100_temple_chaac"[^>]*><\/div>/,
        '',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="temples">',
        `<div id="temples">${element('div', 'marker_100_temple_chaac')}`,
      );
    },
    (value) => {
      value.capture.playerBoards[0].rowHtml = value.capture.playerBoards[0].rowHtml.replace(
        '<h3',
        `${element('div', 'marker_100_temple_chaac')}<h3`,
      );
    },
    (value) => {
      value.capture.retainedDomGeometry.push({
        id: 'marker_100_temple_chaac',
        computedVisibility: 'hidden',
      });
    },
  ]) {
    const value = boundaryMarkerFixture();
    mutate(value);
    assert.throws(
      () => projectRetainedBoundary(...value.repinMarkers()),
      /marker|Marker|Hidden|geometry|duplicate/,
    );
  }
  for (const mutate of [
    (c) => {
      c.archiveRelease = 'other';
    },
    (c) => {
      c.captureSha256 = '0'.repeat(64);
    },
    (c) => {
      c.catalogSha256 = '0'.repeat(64);
    },
    (c) => {
      c.templeTop.chaac.pop();
    },
    (c) => {
      c.templeTop.chaac[0].level = 8;
    },
    (c) => {
      c.templeTop.chaac[0].pixels = c.templeTop.chaac[1].pixels;
    },
    (c) => {
      c.technologyLeft.theology[3].level = 4;
    },
    (c) => {
      c.technologyLeft.architecture.reverse();
    },
    (c) => {
      c.expected = {};
    },
  ]) {
    const value = boundaryMarkerFixture();
    assert.throws(
      () => projectRetainedBoundary(...value.repinMarkers(mutate)),
      /calibration|levels|integer/,
    );
  }
  const geometry = (value, id) =>
    value.capture.retainedDomGeometry.find((entry) => entry.id === id);
  for (const mutate of [
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        '<div id="temples">',
        '<div id="temples" inert>',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        'id="temple_chaac_step_0"',
        'id="temple_chaac_step_0" hidden',
      );
    },
    (value) => {
      value.capture.boardHtml = value.capture.boardHtml.replace(
        /<div id="temple_chaac_step_0"[^>]*><\/div>/,
        '',
      );
    },
    (value) => {
      value.capture.playerBoards[0].rowHtml = value.capture.playerBoards[0].rowHtml.replace(
        '<h3',
        `${element('div', 'temple_chaac_step_0', '', ' class="temple_step"')}<h3`,
      );
    },
    (value) => {
      geometry(value, 'temple_chaac_step_0').computedVisibility = 'hidden';
    },
    (value) => {
      geometry(value, 'temple_chaac_step_0').parentId = 'technologies';
    },
    (value) => {
      geometry(value, 'temple_chaac_step_0').rect.width = 0;
    },
    (value) => {
      geometry(value, 'temple_chaac_step_0').rect.extra = 1;
    },
    (value) => {
      geometry(value, 'temple_chaac_step_0').className = 'technology_level';
    },
    (value) => {
      value.capture.retainedDomGeometry = value.capture.retainedDomGeometry.filter(
        (entry) => entry.id !== 'marker_100_temple_chaac',
      );
    },
    (value) => {
      geometry(value, 'technology_agriculture_level_1').rect.x = 420;
    },
    (value) => {
      geometry(value, 'marker_100_technology_agriculture').rect.y = 54;
    },
    (value) => {
      geometry(value, 'marker_100_technology_agriculture').rect.x = 575;
    },
    (value) => {
      geometry(value, 'marker_100_technology_agriculture').rect.x = 645;
    },
    (value) => {
      geometry(value, 'temples').rect.x = -1;
    },
    (value) => {
      const first = /<div id="temple_chaac_step_0"[^>]*><\/div>/.exec(value.capture.boardHtml)[0];
      const second = /<div id="temple_chaac_step_1"[^>]*><\/div>/.exec(value.capture.boardHtml)[0];
      value.capture.boardHtml = value.capture.boardHtml.replace(first + second, second + first);
    },
  ]) {
    const value = boundaryCellMarkerFixture();
    mutate(value);
    assert.throws(
      () => projectRetainedBoundary(...value.repinMarkers()),
      /marker|Marker|Hidden|geometry|rectangle|duplicate/,
    );
  }
  for (const mutate of [
    (declaration) => {
      declaration.captureSha256 = '0'.repeat(64);
    },
    (declaration) => {
      declaration.templeTop = {};
    },
    (declaration) => {
      declaration.expected = {};
    },
  ]) {
    const value = boundaryCellMarkerFixture();
    assert.throws(() => projectRetainedBoundary(...value.repinMarkers(mutate)), /calibration/);
  }
  const calibrated = boundaryMarkerFixture().repinMarkers();
  assert.throws(
    () => projectRetainedBoundary(...calibrated.slice(0, 4)),
    /Missing declared marker/,
  );
  const unbound = [...calibrated],
    declaration = JSON.parse(unbound[2]);
  delete declaration.markerCalibration;
  unbound[2] = bytes(declaration);
  assert.throws(() => projectRetainedBoundary(...unbound), /marker calibration/);
});

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

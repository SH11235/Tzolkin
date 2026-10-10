import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { decodeConditionalPublicGame, decodePublicGame } from './bga-decode.mjs';
import { ATLAS_SCHEMA, CONTROL_SCHEMA, projectCapturedInitial } from './bga-playing-project.mjs';

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
function currentUi(value) {
  value.raw = {
    ...value.raw,
    schema_version: 1,
    source_type: 'public_ui_dom',
    export_scope: 'all_rendered_gamereview_log_entries',
    dom_entry_count: value.raw.dom_entries.length,
    metadata: {
      tableUrl: 'https://boardgamearena.com/table?table=123',
      playerCount: value.raw.metadata.player_count,
      players: value.raw.metadata.players.map(({ text }) => ({ name: text })),
    },
  };
  return value;
}

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
  const inert = fixture(4);
  inert.capture.gameAreaHtml += '<template><div class="worker worker_dummy"></div></template>';
  assert.throws(() => projectCapturedInitial(...inputs(inert)), /unsupported inert/);
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
  assert.equal(calls, 2);

  const compatible = fixture(4);
  compatible.raw.table_details_text =
    'ゲーム構成\nゲームモード\nアリーナモード\nウシュマルコーンの制限\n制限なし';
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
  assert.equal(calls, 4);
});

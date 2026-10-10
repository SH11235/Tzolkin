import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, open, rename, rm, rmdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeGame } from './bga-import.mjs';
import { observedBasicUnlimitedOptions, uniqueJson } from './bga-observed-options.mjs';
import { catalogFingerprint, coreDispatcher } from './bga-replay.mjs';
import { decodeConditionalPublicGame, WITNESS_SCHEMA } from './bga-decode.mjs';

export const PROJECTION_SCHEMA = 'tzolkin-bga-playing-projection-v1';
export const CONTROL_SCHEMA = 'tzolkin-bga-playing-projection-control-v1';
export const ATLAS_SCHEMA = 'tzolkin-bga-playing-atlas-v1';
export const LATER_WITNESS_SCHEMA = 'tzolkin-bga-playing-later-witnesses-v1';
const CAPTURE_SCHEMAS = new Set([
  'tzolkin-bga-public-playing-initial-board-witness-v1',
  'tzolkin-bga-public-setup-complete-playing-board-v1',
]);
const MAX_BYTES = 16 * 1024 * 1024;
const MAX_CONTROL_BYTES = 64 * 1024;
const MAX_ATLAS_BYTES = 256 * 1024;
const catalogBytes = readFileSync(
  new URL('../crates/tzolkin-core/data/catalog.json', import.meta.url),
);
const catalog = JSON.parse(catalogBytes);
const GEARS = ['palenque', 'yaxchilan', 'tikal', 'uxmal', 'chichenItza'];
const RESOURCES = ['corn', 'wood', 'stone', 'gold', 'skull'];
const TEMPLES = ['chaac', 'quetzalcoatl', 'kukulkan'];
const TECHNOLOGIES = ['agriculture', 'extraction', 'architecture', 'theology'];
const VOID = new Set([
  'area',
  'base',
  'br',
  'col',
  'embed',
  'hr',
  'img',
  'input',
  'link',
  'meta',
  'param',
  'source',
  'track',
  'wbr',
]);
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fail = (message) => {
  throw new Error(message);
};
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const canonical = (value) =>
  JSON.stringify(value, (_key, item) =>
    object(item)
      ? Object.fromEntries(
          Object.keys(item)
            .sort()
            .map((key) => [key, item[key]]),
        )
      : item,
  );
const same = (a, b) => canonical(a) === canonical(b);
const token = (node, name) => (node.attrs.class ?? '').split(/\s+/).includes(name);
const list = (value, max, label) => {
  if (!Array.isArray(value) || value.length > max) fail(`${label}: invalid list`);
  return value;
};
function closed(value, keys, label, optional = []) {
  if (
    !object(value) ||
    Object.keys(value).some((key) => !keys.includes(key) && !optional.includes(key)) ||
    keys.some((key) => !Object.hasOwn(value, key))
  )
    fail(`${label}: missing/unknown field`);
}
function integer(value, min, max, label) {
  if (!Number.isSafeInteger(value) || value < min || value > max) fail(`${label}: invalid integer`);
  return value;
}
function number(value, min, max, label) {
  if (!Number.isFinite(value) || value < min || value > max) fail(`${label}: invalid number`);
  return value;
}
function text(value, max, label) {
  if (typeof value !== 'string' || !value.length || value.length > max)
    fail(`${label}: missing/long text`);
  return value;
}
function one(items, predicate, label) {
  const result = items.filter(predicate);
  if (result.length !== 1) fail(`${label}: missing/ambiguous`);
  return result[0];
}

function pin(bytes, declaration, max, label) {
  closed(declaration, ['bytes', 'sha256'], `${label} pin`);
  integer(declaration.bytes, 1, max, `${label} size`);
  if (
    !/^[a-f0-9]{64}$/.test(declaration.sha256) ||
    !Buffer.isBuffer(bytes) ||
    bytes.length !== declaration.bytes ||
    sha(bytes) !== declaration.sha256
  )
    fail(`${label}: pin mismatch`);
}
function entities(value) {
  return value.replace(/&(#x[0-9a-f]+|#\d+|amp|quot|apos|lt|gt|nbsp);/gi, (_match, entity) => {
    if (entity[0] === '#') {
      const code =
        entity[1].toLowerCase() === 'x' ? parseInt(entity.slice(2), 16) : Number(entity.slice(1));
      if (
        !Number.isInteger(code) ||
        code < 1 ||
        code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff)
      )
        fail('Invalid HTML entity');
      return String.fromCodePoint(code);
    }
    return { amp: '&', quot: '"', apos: "'", lt: '<', gt: '>', nbsp: '\u00a0' }[
      entity.toLowerCase()
    ];
  });
}

// Retained outerHTML only: no execution, selectors, network or computed styles.
// Browser-produced balanced markup is required; duplicate IDs/attributes fail closed.
function htmlTree(source, label) {
  text(source, MAX_BYTES, label);
  const root = { tag: '#root', attrs: {}, children: [], start: 0, end: source.length };
  const stack = [root],
    nodes = [],
    ids = new Map();
  let at = 0,
    elements = 0;
  while (at < source.length) {
    if (source.startsWith('<!--', at)) {
      const end = source.indexOf('-->', at + 4);
      if (end < 0) fail(`${label}: incomplete comment`);
      at = end + 3;
      continue;
    }
    if (source[at] !== '<') {
      const end = source.indexOf('<', at);
      const stop = end < 0 ? source.length : end;
      stack.at(-1).children.push(entities(source.slice(at, stop)));
      at = stop;
      continue;
    }
    const start = at;
    const match = /^<(\/)?([A-Za-z][A-Za-z0-9:-]*)/.exec(source.slice(at));
    if (!match) fail(`${label}: unsupported markup`);
    const tag = match[2].toLowerCase();
    if (
      [
        'iframe',
        'template',
        'noscript',
        'textarea',
        'title',
        'xmp',
        'noembed',
        'noframes',
        'plaintext',
        'canvas',
        'object',
        'video',
        'audio',
        'select',
      ].includes(tag)
    )
      fail(`${label}: unsupported inert/embedded container`);
    at += match[0].length;
    if (match[1]) {
      while (/\s/.test(source[at] ?? '') && at < source.length) at++;
      if (source[at++] !== '>' || stack.length === 1 || stack.at(-1).tag !== tag)
        fail(`${label}: unbalanced markup`);
      const node = stack.pop();
      node.innerEnd = start;
      node.end = at;
      continue;
    }
    const attrs = Object.create(null);
    for (;;) {
      while (/\s/.test(source[at] ?? '') && at < source.length) at++;
      if (source[at] === '>' || source.startsWith('/>', at)) break;
      const attr = /^[^\s=/>]+/.exec(source.slice(at));
      if (!attr) fail(`${label}: invalid attribute`);
      const key = attr[0].toLowerCase();
      at += attr[0].length;
      if (Object.hasOwn(attrs, key)) fail(`${label}: duplicate attribute`);
      while (/\s/.test(source[at] ?? '') && at < source.length) at++;
      let value = '';
      if (source[at] === '=') {
        at++;
        while (/\s/.test(source[at] ?? '') && at < source.length) at++;
        const quote = source[at];
        if (quote === '"' || quote === "'") {
          const end = source.indexOf(quote, ++at);
          if (end < 0) fail(`${label}: incomplete attribute`);
          value = source.slice(at, end);
          at = end + 1;
        } else {
          const raw = /^[^\s>]+/.exec(source.slice(at));
          if (!raw) fail(`${label}: missing attribute value`);
          value = raw[0];
          at += value.length;
        }
      }
      attrs[key] = entities(value);
    }
    const selfClosing = source.startsWith('/>', at);
    at += selfClosing ? 2 : 1;
    if (++elements > 20000 || stack.length > 64) fail(`${label}: HTML complexity limit`);
    if (['script', 'style'].includes(tag)) {
      // BGA retains HTML template strings inside scripts. These are raw text,
      // and neither the element nor its body is rendered board evidence.
      if (selfClosing) fail(`${label}: unsupported self-closing raw-text element`);
      const closing = new RegExp(`</${tag}\\s*>`, 'gi');
      closing.lastIndex = at;
      const end = closing.exec(source);
      if (!end) fail(`${label}: incomplete raw-text element`);
      at = closing.lastIndex;
      continue;
    }
    const node = { tag, attrs, children: [], start, innerStart: at, innerEnd: at, end: at };
    stack.at(-1).children.push(node);
    nodes.push(node);
    if (attrs.id !== undefined) {
      if (!attrs.id || ids.has(attrs.id)) fail(`${label}: duplicate/empty DOM ID`);
      ids.set(attrs.id, node);
    }
    if (!selfClosing && !VOID.has(tag)) stack.push(node);
  }
  if (stack.length !== 1) fail(`${label}: incomplete markup`);
  return { source, nodes, ids };
}
const nodeText = (node) =>
  node.children.map((child) => (typeof child === 'string' ? child : nodeText(child))).join('');
const outer = (tree, node) => tree.source.slice(node.start, node.end);
function byId(tree, id) {
  const node = tree.ids.get(id);
  if (!node) fail(`Missing retained DOM node ${id}`);
  return node;
}
function retained(tree, items, predicate, label) {
  const expected = tree.nodes.filter(predicate);
  list(items, 100, label);
  if (items.length !== expected.length) fail(`${label}: incomplete retained node list`);
  const seen = new Set();
  for (const item of items) {
    if (!object(item) || item.visible !== true || seen.has(item.id))
      fail(`${label}: invisible/duplicate node`);
    seen.add(item.id);
    const node = byId(tree, item.id);
    if (!predicate(node) || item.html !== outer(tree, node)) fail(`${label}: detached HTML`);
  }
  return expected;
}
function styles(node) {
  const result = Object.create(null);
  for (const part of (node.attrs.style ?? '').split(';')) {
    if (!part.trim()) continue;
    const at = part.indexOf(':');
    if (at < 1) fail('Invalid retained style');
    const key = part.slice(0, at).trim().toLowerCase();
    if (Object.hasOwn(result, key)) fail('Duplicate retained style');
    result[key] = part.slice(at + 1).trim();
  }
  return result;
}
function pixels(value, label) {
  if (typeof value !== 'string' || !/^-?(?:\d+(?:\.\d+)?|\.\d+)px$/.test(value))
    fail(`${label}: unknown pixels`);
  return number(Number(value.slice(0, -2)), -100000, 100000, label);
}
function sprite(node) {
  const value = styles(node)['background-position'];
  const match = /^(-?(?:\d+(?:\.\d+)?|\.\d+)px)\s+(-?(?:\d+(?:\.\d+)?|\.\d+)px)$/.exec(value ?? '');
  if (!match) fail('Missing/unsupported sprite position');
  return [pixels(match[1], 'sprite x'), pixels(match[2], 'sprite y')];
}
function natural(node, label) {
  const value = nodeText(node).trim();
  if (!/^\d+$/.test(value)) fail(`${label}: nonnumeric DOM counter`);
  return integer(Number(value), 0, 100000, label);
}
function urlIdentity(value) {
  const url = new URL(text(value, 2048, 'source URL'));
  if (
    url.protocol !== 'https:' ||
    !/^(?:(?:en|ja)\.)?boardgamearena\.com$/.test(url.hostname) ||
    url.username ||
    url.password ||
    url.port ||
    url.hash ||
    url.searchParams.getAll('table').length !== 1 ||
    !/^\d+$/.test(url.searchParams.get('table'))
  )
    fail('Unsupported BGA source URL');
  return url;
}
function profile(value) {
  closed(value, ['schema', 'rules', 'market', 'sourceOptions'], 'mechanical profile');
  if (
    value.schema !== 'tzolkin-bga-declared-mechanical-profile-v1' ||
    value.rules !== 'basic' ||
    value.market !== 'unlimited' ||
    value.sourceOptions !== 'unknown'
  )
    fail('Unsupported mechanical profile');
  return structuredClone(value);
}

// An independently pinned collection supplies the ordinary decoder's later
// observations. Identity bindings prove consistency with these inputs, not that
// the observations were authenticated or independently interpreted from BGA.
function laterWitnessesFrom(bytes, binding, game) {
  const value = uniqueJson(bytes, MAX_BYTES, 'later witnesses');
  closed(value, ['schema', ...Object.keys(binding), 'witnesses'], 'later witnesses');
  if (value.schema !== LATER_WITNESS_SCHEMA) fail('Unsupported later witness schema');
  for (const [field, expected] of Object.entries(binding))
    if (value[field] !== expected) fail(`Later witness ${field} binding mismatch`);
  list(value.witnesses, 4000, 'later witnesses');
  const ids = new Set(game.rawEntries.map((entry) => entry.action_id));
  const seen = new Set();
  const buildings = new Map(catalog.BUILDINGS.map((card) => [card.id, card.age]));
  const monuments = new Set(catalog.MONUMENTS.map((card) => card.id));
  for (const witness of value.witnesses) {
    closed(witness, ['actionId'], 'later witness', [
      'expected',
      'refills',
      'buildings',
      'monuments',
      'pendingTask',
    ]);
    integer(witness.actionId, binding.initialActionId + 1, 100000, 'later witness action');
    if (!ids.has(witness.actionId) || seen.has(witness.actionId))
      fail('Duplicate/unknown later witness action ID');
    seen.add(witness.actionId);
    if (Object.keys(witness).length === 1) fail('Later witness has no observation');
    if (
      witness.expected !== undefined &&
      (!object(witness.expected) || !Object.keys(witness.expected).length)
    )
      fail('Empty later witness checkpoint');
    if (witness.pendingTask !== undefined && witness.pendingTask !== null)
      text(witness.pendingTask, 64, 'later pending task');
    const cards = (values, known, label) => {
      list(values, 6, label);
      if (new Set(values).size !== values.length || values.some((id) => !known(id)))
        fail(`Invalid ${label} catalog IDs`);
    };
    if (witness.buildings !== undefined)
      cards(witness.buildings, (id) => buildings.has(id), 'later buildings');
    if (witness.monuments !== undefined)
      cards(witness.monuments, (id) => monuments.has(id), 'later monuments');
    if (witness.refills !== undefined) {
      closed(witness.refills, [], 'later refills', ['currentAge', 'age2']);
      if (witness.refills.currentAge !== undefined)
        cards(witness.refills.currentAge, (id) => buildings.has(id), 'later current-age reveals');
      if (witness.refills.age2 !== undefined)
        cards(witness.refills.age2, (id) => buildings.get(id) === 2, 'later age-II reveals');
    }
  }
  return structuredClone(value.witnesses);
}

function atlasFrom(value, release, captureSha256) {
  closed(
    value,
    ['schema', 'archiveRelease', 'catalogSha256', 'faces', 'markerPositions', 'geometry'],
    'atlas',
  );
  if (
    value.schema !== ATLAS_SCHEMA ||
    value.archiveRelease !== release ||
    value.catalogSha256 !== sha(catalogBytes)
  )
    fail('Atlas release/catalog mismatch');
  const maps = { wealth: new Map(), building: new Map(), monument: new Map() };
  const catalogs = {
    wealth: catalog.STARTING_WEALTH,
    building: catalog.BUILDINGS.filter((c) => c.age === 1),
    monument: catalog.MONUMENTS,
  };
  const identities = new Set();
  for (const face of list(value.faces, 100, 'atlas faces')) {
    closed(face, ['kind', 'position', 'catalogIds'], 'atlas face');
    if (!Object.hasOwn(maps, face.kind)) fail('Unknown atlas face kind');
    list(face.position, 2, 'sprite position');
    if (face.position.length !== 2) fail('Invalid atlas sprite vector');
    face.position.forEach((n) => number(n, -100000, 100000, 'atlas sprite coordinate'));
    const key = face.position.join(',');
    if (maps[face.kind].has(key)) fail('Duplicate atlas sprite');
    list(face.catalogIds, 4, 'atlas identities');
    if (!face.catalogIds.length || (face.kind !== 'building' && face.catalogIds.length !== 1))
      fail('Ambiguous atlas identities');
    const cards = face.catalogIds.map((id) => {
      if (identities.has(`${face.kind}:${id}`)) fail('Atlas catalog identity is not injective');
      identities.add(`${face.kind}:${id}`);
      return one(catalogs[face.kind], (c) => c.id === id, 'atlas catalog identity');
    });
    if (
      cards.length > 1 &&
      cards.some(
        (c) =>
          !same(
            { age: c.age, category: c.category, cost: c.cost, effects: c.effects },
            {
              age: cards[0].age,
              category: cards[0].category,
              cost: cards[0].cost,
              effects: cards[0].effects,
            },
          ),
      )
    )
      fail('Different building effects share one calibrated sprite');
    maps[face.kind].set(key, [...face.catalogIds].sort());
  }
  closed(value.markerPositions, ['templeTop', 'technologyLeft'], 'marker calibration');
  for (const positions of Object.values(value.markerPositions)) {
    list(positions, 8, 'marker positions');
    const levels = new Set(),
      coordinates = new Set();
    for (const entry of positions) {
      closed(entry, ['pixels', 'level'], 'marker position');
      number(entry.pixels, -10000, 10000, 'marker coordinate');
      integer(entry.level, 0, 7, 'marker level');
      if (levels.has(entry.level) || coordinates.has(entry.pixels))
        fail('Ambiguous marker calibration');
      levels.add(entry.level);
      coordinates.add(entry.pixels);
    }
  }
  if (value.geometry !== null) {
    closed(
      value.geometry,
      ['captureSha256', 'coordinateSpace', 'boardSize', 'workerSize', 'offset', 'tolerance'],
      'geometry calibration',
    );
    if (
      value.geometry.captureSha256 !== captureSha256 ||
      value.geometry.coordinateSpace !== 'retained-client-rect'
    )
      fail('Unbound geometry calibration');
    for (const key of ['boardSize', 'workerSize', 'offset']) {
      const fields = key === 'offset' ? ['x', 'y'] : ['width', 'height'];
      closed(value.geometry[key], fields, 'geometry vector');
      for (const field of fields)
        number(
          value.geometry[key][field],
          key === 'offset' ? -100 : 1,
          10000,
          'geometry component',
        );
    }
    number(value.geometry.tolerance, 0, 0.5, 'geometry tolerance');
  }
  return { maps, markerPositions: value.markerPositions, geometry: value.geometry };
}
function faceId(atlas, kind, node) {
  const ids = atlas.maps[kind].get(sprite(node).join(','));
  if (!ids) fail(`Uncalibrated ${kind} sprite`);
  return ids;
}
function markerLevel(atlas, kind, node) {
  const coordinate = pixels(styles(node)[kind === 'temple' ? 'top' : 'left'], 'marker coordinate');
  return one(
    atlas.markerPositions[kind === 'temple' ? 'templeTop' : 'technologyLeft'],
    (position) => position.pixels === coordinate,
    'calibrated marker level',
  ).level;
}
function wealthEffects(ids) {
  const result = {
    resources: Object.fromEntries(RESOURCES.map((r) => [r, 0])),
    workers: 3,
    feedWorkers: 0,
    temples: Object.fromEntries(TEMPLES.map((t) => [t, 0])),
    technologies: Object.fromEntries(TECHNOLOGIES.map((t) => [t, 0])),
  };
  for (const id of ids) {
    const tile = one(catalog.STARTING_WEALTH, (tile) => tile.id === id, 'selected wealth catalog');
    for (const [resource, amount] of Object.entries(tile.resources))
      result.resources[resource] += amount;
    for (const effect of tile.effects) {
      if (effect.type === 'worker') result.workers++;
      else if (effect.type === 'feed' && Number.isInteger(effect.workers))
        result.feedWorkers += effect.workers;
      else if (effect.type === 'technology' && TECHNOLOGIES.includes(effect.technology))
        result.technologies[effect.technology] += effect.steps ?? 1;
      else if (effect.type === 'temple' && TEMPLES.includes(effect.temple))
        result.temples[effect.temple] += effect.steps ?? 1;
      else fail('Unsupported starting wealth effect');
    }
  }
  return result;
}
function dimensions(rect, label) {
  if (!object(rect)) fail(`${label}: missing retained rectangle`);
  for (const key of ['x', 'y', 'width', 'height'])
    number(rect[key], key === 'x' || key === 'y' ? -100000 : 1, 100000, label);
  return rect;
}
function dummyProjection(capture, tree, atlas, n) {
  const gears = Object.fromEntries(
    GEARS.map((gear) => [gear, Array(gear === 'chichenItza' ? 13 : 10).fill(null)]),
  );
  const nodes = retained(
    tree,
    capture.gearWorkerNodes,
    (node) => token(node, 'worker'),
    'gear workers',
  );
  if (
    nodes.length !== (4 - n) * 6 ||
    nodes.some((node) => !token(node, 'worker_dummy') || !/^worker_\d+$/.test(node.attrs.id))
  )
    fail('Initial workers are missing or include prior play');
  const places = tree.nodes.filter((node) => token(node, 'workerplace'));
  const placeIds = new Set(places.map((node) => node.attrs.id));
  const expectedIds = [
    'workerplace_startingPlayer_0',
    ...GEARS.flatMap((gear) => gears[gear].map((_v, i) => `workerplace_${gear}_${i}`)),
  ];
  if (places.length !== expectedIds.length || expectedIds.some((id) => !placeIds.has(id)))
    fail('Initial worker spaces incomplete');
  if (!nodes.length) return gears;
  const geometry = atlas.geometry;
  if (geometry === null) fail('Missing calibrated dummy geometry');
  const board = capture.boardGeometry;
  if (
    !object(board) ||
    board.id !== 'board' ||
    board.visible !== true ||
    board.html !== outer(tree, byId(tree, 'board'))
  )
    fail('Detached retained board geometry');
  const boardRect = dimensions(board.rect, 'board rectangle');
  for (const key of ['width', 'height'])
    if (Math.abs(boardRect[key] - geometry.boardSize[key]) > geometry.tolerance)
      fail('Uncalibrated board scale');
  retained(tree, capture.workerPlaces, (node) => token(node, 'workerplace'), 'worker spaces');
  for (const worker of capture.gearWorkerNodes) {
    const rect = dimensions(worker.rect, 'dummy rectangle');
    for (const key of ['width', 'height'])
      if (Math.abs(rect[key] - geometry.workerSize[key]) > geometry.tolerance)
        fail('Uncalibrated worker scale');
    const candidates = capture.workerPlaces.filter((place) => {
      const p = dimensions(place.rect, 'worker space rectangle');
      return (
        Math.abs(rect.x - p.x - geometry.offset.x) <= geometry.tolerance &&
        Math.abs(rect.y - p.y - geometry.offset.y) <= geometry.tolerance
      );
    });
    const place = one(candidates, () => true, 'dummy geometry match');
    const match = /^workerplace_(palenque|yaxchilan|tikal|uxmal|chichenItza)_(\d+)$/.exec(place.id);
    if (!match || gears[match[1]][Number(match[2])] !== null) fail('Duplicate/invalid dummy space');
    gears[match[1]][Number(match[2])] = { playerId: -1, dummy: true };
  }
  return gears;
}

/** Pure source-pinned projector. Caller pins prove byte consistency, not BGA
 * authenticity or independent image review. Unknown sprites always block. */
export function projectCapturedInitial(
  captureBytes,
  calibrationBytes,
  rawLogBytes,
  controlBytes,
  tableBytes = null,
  laterWitnessBytes = null,
) {
  if (
    [
      captureBytes,
      calibrationBytes,
      rawLogBytes,
      controlBytes,
      ...(tableBytes === null ? [] : [tableBytes]),
      ...(laterWitnessBytes === null ? [] : [laterWitnessBytes]),
    ].some((input) => !Buffer.isBuffer(input)) ||
    captureBytes.length +
      calibrationBytes.length +
      rawLogBytes.length +
      controlBytes.length +
      (tableBytes?.length ?? 0) +
      (laterWitnessBytes?.length ?? 0) >
      32 * 1024 * 1024
  )
    fail('Projection input aggregate byte limit');
  const control = uniqueJson(controlBytes, MAX_CONTROL_BYTES, 'control');
  closed(
    control,
    [
      'schema',
      'capture',
      'calibration',
      'rawLog',
      'mechanicalProfile',
      'seatOrder',
      'turnOrderEvidence',
    ],
    'control',
    ['table', 'laterWitnesses'],
  );
  if (control.schema !== CONTROL_SCHEMA) fail('Unsupported projection control');
  pin(captureBytes, control.capture, MAX_BYTES, 'capture');
  pin(calibrationBytes, control.calibration, MAX_ATLAS_BYTES, 'calibration');
  pin(rawLogBytes, control.rawLog, MAX_BYTES, 'raw log');
  if (tableBytes !== null) pin(tableBytes, control.table, MAX_BYTES, 'table evidence');
  else if (control.table !== undefined && control.table !== null)
    fail('Missing declared table evidence');
  if (laterWitnessBytes !== null)
    pin(laterWitnessBytes, control.laterWitnesses, MAX_BYTES, 'later witnesses');
  else if (control.laterWitnesses !== undefined && control.laterWitnesses !== null)
    fail('Missing declared later witnesses');
  const capture = uniqueJson(captureBytes, MAX_BYTES, 'capture');
  const calibration = uniqueJson(calibrationBytes, MAX_ATLAS_BYTES, 'calibration');
  const raw = uniqueJson(rawLogBytes, MAX_BYTES, 'raw log');
  const table = tableBytes === null ? null : uniqueJson(tableBytes, MAX_BYTES, 'table evidence');
  observedBasicUnlimitedOptions(raw, table);
  const mechanicalProfile = profile(control.mechanicalProfile);
  if (!CAPTURE_SCHEMAS.has(capture.schema)) fail('Unsupported Playing capture schema');
  const url = urlIdentity(capture.url);
  const release = /^\/archive\/replay\/([A-Za-z0-9_-]{1,64})\/$/.exec(url.pathname)?.[1];
  if (
    !release ||
    url.searchParams.getAll('player').length !== 1 ||
    url.searchParams.get('player') !== String(capture.viewPlayerId) ||
    url.searchParams.get('table') !== String(capture.tableId)
  )
    fail('Capture archive identity mismatch');
  const match = /^Move #([1-9]\d*)$/.exec(capture.move ?? '');
  const actionId = integer(match ? Number(match[1]) : null, 1, 100000, 'capture move');
  const game = normalizeGame(raw, sha(rawLogBytes), tableBytes);
  const observedOptions = observedBasicUnlimitedOptions(raw, table, game.context);
  if (tableBytes !== null && game.source.adapter !== 'bga-public-ui-dom-v1')
    fail('Legacy input requires embedded table settings, not separate table evidence');
  if (
    game.tableId !== String(capture.tableId) ||
    urlIdentity(game.source.reference).searchParams.get('table') !== game.tableId ||
    !game.rawEntries.some((entry) => entry.action_id === actionId)
  )
    fail('Raw log/capture boundary mismatch');
  const tree = htmlTree(capture.gameAreaHtml, 'game area');
  const boards = htmlTree(capture.playerBoardsHtml, 'player boards');
  const atlas = atlasFrom(calibration, release, control.capture.sha256);
  if (
    capture.offerNodesRemaining !== 0 ||
    tree.nodes.some(
      (node) =>
        token(node, 'wealth_tile') && tree.source.slice(node.start, node.end).includes('active'),
    )
  )
    fail('Starting offers remain active');
  if (
    natural(byId(tree, 'current_round'), 'round') !== 1 ||
    !/^経過日\s+1\s*\/\s*27$/.test(capture.roundsText ?? '')
  )
    fail('Capture is not day one');
  const playerNodes = boards.nodes.filter((node) =>
    /^overall_player_board_\d+$/.test(node.attrs.id ?? ''),
  );
  const n = playerNodes.length;
  if (![3, 4].includes(n) || game.context.playerCount !== n)
    fail('Only complete three/four-player captures are supported');
  const sourceIds = playerNodes.map((node) => node.attrs.id.slice('overall_player_board_'.length));
  list(control.seatOrder, 4, 'seat order');
  if (
    control.seatOrder.length !== n ||
    new Set(control.seatOrder).size !== n ||
    control.seatOrder.some((id) => typeof id !== 'string' || !sourceIds.includes(id))
  )
    fail('Seat order differs from captured roster');
  closed(control.turnOrderEvidence, ['kind', 'knownAfterActionId'], 'turn order provenance');
  if (
    !['declared', 'later-inferred'].includes(control.turnOrderEvidence.kind) ||
    (control.turnOrderEvidence.kind === 'declared' &&
      control.turnOrderEvidence.knownAfterActionId !== null)
  )
    fail('Invalid noncausal turn order provenance');
  if (control.turnOrderEvidence.kind === 'later-inferred') {
    integer(
      control.turnOrderEvidence.knownAfterActionId,
      actionId + 1,
      100000,
      'later order boundary',
    );
    if (
      !game.rawEntries.some(
        (entry) => entry.action_id === control.turnOrderEvidence.knownAfterActionId,
      )
    )
      fail('Later order boundary is absent from raw log');
  }
  retained(
    boards,
    capture.publicCounters,
    (node) => /^(worker|corn|wood|stone|gold|skull)_number_player_\d+$/.test(node.attrs.id ?? ''),
    'public counters',
  );
  retained(
    tree,
    capture.selectedPlayerTiles,
    (node) => /^player_wealthtiles_\d+$/.test(node.attrs.id ?? ''),
    'selected wealth panels',
  );
  retained(
    tree,
    capture.templeMarkers,
    (node) => /^marker_\d+_temple_/.test(node.attrs.id ?? ''),
    'temple markers',
  );
  retained(
    tree,
    capture.technologyMarkers,
    (node) => /^marker_\d+_technology_/.test(node.attrs.id ?? ''),
    'technology markers',
  );
  if (
    capture.publicCounters.length !== n * 6 ||
    capture.selectedPlayerTiles.length !== n ||
    capture.templeMarkers.length !== n * 3 ||
    capture.technologyMarkers.length !== n * 4 ||
    tree.nodes.filter((node) => token(node, 'wealth_tile')).length !== n * 2
  )
    fail('Incomplete player setup fields');
  const usedWealth = new Set();
  const players = control.seatOrder.map((id, index) => {
    const nameNode = byId(boards, `player_name_${id}`);
    const link = one(
      nameNode.children.filter(object),
      (node) => node.tag === 'a',
      'player name link',
    );
    const name = text(nodeText(link).trim(), 128, 'player name');
    const color = styles(link).color;
    if (!/^#[a-f0-9]{6}$/i.test(color ?? '')) fail('Unknown player color');
    const wealthPanel = byId(tree, `player_wealthtiles_${id}`);
    const tiles = wealthPanel.children.filter(object);
    if (
      tiles.length !== 2 ||
      tiles.some(
        (node) => !token(node, 'wealth_tile') || !/^wealth_tile_\d+$/.test(node.attrs.id ?? ''),
      )
    )
      fail('Malformed selected wealth tiles');
    const wealth = tiles.map((node) => faceId(atlas, 'wealth', node)[0]).sort();
    for (const tile of wealth) {
      if (usedWealth.has(tile)) fail('Duplicated selected wealth');
      usedWealth.add(tile);
    }
    const resources = Object.fromEntries(
      RESOURCES.map((resource) => [
        resource,
        natural(byId(boards, `${resource}_number_player_${id}`), 'resource'),
      ]),
    );
    const workers = natural(byId(boards, `worker_number_player_${id}`), 'workers');
    const temples = Object.fromEntries(
      TEMPLES.map((temple) => [
        temple,
        markerLevel(
          atlas,
          'temple',
          byId(tree, `marker_${id}_temple_${temple === 'kukulkan' ? 'kukulcan' : temple}`),
        ),
      ]),
    );
    const technologies = Object.fromEntries(
      TECHNOLOGIES.map((technology) => [
        technology,
        markerLevel(atlas, 'technology', byId(tree, `marker_${id}_technology_${technology}`)),
      ]),
    );
    const effects = wealthEffects(wealth);
    if (
      !same(resources, effects.resources) ||
      !same(temples, effects.temples) ||
      !same(technologies, effects.technologies) ||
      workers !== effects.workers ||
      natural(byId(boards, `player_score_${id}`), 'score') !== 0
    )
      fail('Selected artwork does not match initial counters/markers');
    return {
      id: index,
      name,
      color,
      resources,
      score: 0,
      workers,
      temples,
      technologies,
      buildings: [],
      monuments: [],
      wealth,
      feedWorkers: effects.feedWorkers,
      feedAll: false,
      feedDiscount: 0,
      cornTiles: 0,
      woodTiles: 0,
      skullsPlaced: 0,
      buildingSkulls: 0,
      doubleAdvanceAvailable: true,
      templePoints: 0,
    };
  });
  if (
    new Set(players.map((p) => p.name)).size !== n ||
    !same([...players.map((p) => p.name)].sort(), [...game.context.players].sort())
  )
    fail('Captured names differ from raw log roster');
  const firstToken = byId(boards, 'firstplayer');
  const firstParent = one(
    boards.nodes,
    (node) => node.children.includes(firstToken),
    'first-player token parent',
  );
  const firstId = /^firstplayer_(\d+)$/.exec(firstParent.attrs.id ?? '')?.[1];
  const firstPlayer = control.seatOrder.indexOf(firstId);
  const suffix = 'はワーカーを置くか取り除くかしてください';
  const promptName = players[firstPlayer]?.name;
  if (
    firstPlayer < 0 ||
    (capture.header !== `${promptName}${suffix}` &&
      !(firstId === String(capture.viewPlayerId) && capture.header === `あなた${suffix}`))
  )
    fail('Initial prompt differs from first-player marker');
  retained(
    tree,
    capture.gears,
    (node) =>
      /^gear_(central|palenque|yaxchilan|tikal|uxmal|chichenItza)$/.test(node.attrs.id ?? ''),
    'unrotated gears',
  );
  if (
    capture.gears.length !== 6 ||
    capture.gears.some((item) => styles(byId(tree, item.id)).transform !== 'rotate(0deg)')
  )
    fail('Initial gears are missing/rotated');
  const gears = dummyProjection(capture, tree, atlas, n);
  const jungle = {};
  for (let box = 2; box <= 5; box++) {
    const tiles = byId(tree, `jungle_${box}`).children.filter(object);
    if (
      tiles.length !== n ||
      tiles.some(
        (tile) =>
          !token(tile, 'board_tile') ||
          !token(tile, box === 2 ? 'corn_tile' : 'wood_tile') ||
          !new RegExp(`^jungle_${box}_tile_[1-9]\\d*$`).test(tile.attrs.id ?? ''),
      )
    )
      fail('Initial jungle tiles incomplete');
    jungle[box] = { corn: n, wood: box === 2 ? 0 : n };
  }
  const skulls = retained(
    tree,
    capture.skullSpaces,
    (node) => /^skullplace_\d+$/.test(node.attrs.id ?? ''),
    'skull spaces',
  );
  if (
    skulls.length !== 9 ||
    Array.from({ length: 9 }, (_v, i) => `skullplace_${i + 1}`).some((id) => !tree.ids.has(id)) ||
    skulls.some((node) => node.children.some((child) => typeof child !== 'string' || child.trim()))
  )
    fail('Skull spaces incomplete/occupied');
  if (capture.skullRemainingHtml !== outer(tree, byId(tree, 'skull_left_wrap')))
    fail('Detached skull supply HTML');
  const skullSupply = natural(byId(tree, 'skull_left'), 'skull supply');
  if (skullSupply + players.reduce((sum, player) => sum + player.resources.skull, 0) !== 13)
    fail('Initial skull conservation mismatch');
  const construction = (kind, html, containerId, count) => {
    if (html !== outer(tree, byId(tree, containerId))) fail(`Detached ${kind} market HTML`);
    const container = htmlTree(html, `${kind} market`);
    const nodes = container.nodes.filter((node) => token(node, 'construction'));
    const pattern = kind === 'building' ? /^building_age_1_(\d+)$/ : /^monument_(\d+)$/;
    if (nodes.length !== count || nodes.some((node) => !pattern.test(node.attrs.id ?? '')))
      fail(`Incomplete ${kind} market`);
    nodes.sort((a, b) => Number(pattern.exec(a.attrs.id)[1]) - Number(pattern.exec(b.attrs.id)[1]));
    const used = new Set();
    return nodes.map((node) => {
      const id = faceId(atlas, kind, node).find((id) => !used.has(id));
      if (!id) fail(`Overused ${kind} sprite copies`);
      used.add(id);
      return id;
    });
  };
  const buildings = construction('building', capture.buildingsHtml, 'buildings', 6);
  const monuments = construction('monument', capture.monumentsHtml, 'monuments', n + 2);
  if (tree.nodes.filter((node) => token(node, 'construction')).length !== 6 + n + 2)
    fail('Unexpected construction outside initial markets');
  const initial = {
    version: 1,
    additionalBuildings: false,
    phase: 'playing',
    round: 1,
    age: 1,
    players,
    currentPlayer: firstPlayer,
    firstPlayer,
    turnOrder: Array.from({ length: n }, (_v, i) => (firstPlayer + i) % n),
    turnIndex: 0,
    turn: { mode: 'none', count: 0, begged: false },
    gears,
    jungle,
    skullSupply,
    skullSpaces: Array(10).fill(null),
    firstPlayerClaimed: null,
    accumulatedCorn: 0,
    buildings,
    buildingDeckCount: catalog.BUILDINGS.filter((c) => c.age === 1).length - 6,
    age2DeckCount: catalog.BUILDINGS.filter((c) => c.age === 2).length,
    monuments,
    pending: null,
    log: [],
    foodDays: [],
    finalScores: [],
    hidden: { seed: 'unknown', setupOffers: 'unknown', deckOrder: 'unknown' },
  };
  const witness = {
    schema: WITNESS_SCHEMA,
    tableId: game.tableId,
    rawSha256: sha(rawLogBytes),
    catalogHash: catalogFingerprint(catalogBytes),
    initialActionId: actionId,
    initial: structuredClone(initial),
    witnesses:
      laterWitnessBytes === null
        ? []
        : laterWitnessesFrom(
            laterWitnessBytes,
            {
              tableId: game.tableId,
              rawLogReference: game.source.reference,
              initialCaptureReference: capture.url,
              rawLogSha256: sha(rawLogBytes),
              initialCaptureSha256: sha(captureBytes),
              calibrationSha256: sha(calibrationBytes),
              tableEvidenceSha256: tableBytes === null ? null : sha(tableBytes),
              archiveRelease: release,
              catalogHash: catalogFingerprint(catalogBytes),
              initialActionId: actionId,
            },
            game,
          ),
  };
  const record = {
    schema: 'tzolkin-public-replay-v1',
    rulesVersion: 1,
    catalogHash: witness.catalogHash,
    market: 'unlimited',
    source: { reference: game.source.reference, actionIds: [actionId] },
    initial,
    initialCheckpoint: null,
    steps: [],
    terminalCheckpoint: null,
  };
  return {
    schema: PROJECTION_SCHEMA,
    record,
    witness,
    provenance: {
      captureSha256: sha(captureBytes),
      calibrationSha256: sha(calibrationBytes),
      rawLogSha256: sha(rawLogBytes),
      tableEvidenceSha256: tableBytes === null ? null : sha(tableBytes),
      ...(laterWitnessBytes === null
        ? {}
        : {
            laterWitnesses: {
              schema: LATER_WITNESS_SCHEMA,
              sha256: sha(laterWitnessBytes),
              bytes: laterWitnessBytes.length,
              count: witness.witnesses.length,
            },
          }),
      archiveRelease: release,
      initialActionId: actionId,
      playerCount: n,
      mechanicalProfile,
      turnOrder: structuredClone(control.turnOrderEvidence),
      sourceOptions: {
        tribes: observedOptions.tribes,
        prophecies: observedOptions.prophecies,
        additionalBuildings: observedOptions.additionalBuildings,
        quickActions: observedOptions.quickActions,
        marketLimit: observedOptions.marketLimit,
      },
      observedOptions,
      mode: observedOptions.mode,
      derivations: [
        'Basic setup declares covered corn below visible wood tiles.',
        'Feeding benefit is derived only from calibrated selected artwork.',
        'Remaining deck counts derive from the pinned current basic catalog.',
        'Market order follows DOM ID suffixes; identical-card sprites use the lowest unused catalog ID mechanically.',
      ],
      limits: [
        'Caller pins establish byte consistency, not source authentication or independent image review.',
        'Image bytes and independent artwork or geometry interpretation are not verified by this producer.',
        'Seat cycle is an explicit noncausal mechanical declaration.',
        'A standalone replay record cannot carry this separate conditional provenance in the current app.',
      ],
      sourceAuthenticated: false,
      coreInitialVerified: false,
      fullGameReplayed: false,
      strictHumanQualification: false,
      policyTrainingReady: false,
      valueTrainingReady: false,
    },
  };
}

/** Run the ordinary legal decoder under the separately declared conditions.
 * Core verification remains authoritative; projector success alone is insufficient. */
export function decodeProjectedInitial(
  captureBytes,
  calibrationBytes,
  rawLogBytes,
  controlBytes,
  dispatch,
  tableBytes = null,
  laterWitnessBytes = null,
) {
  const projection = projectCapturedInitial(
    captureBytes,
    calibrationBytes,
    rawLogBytes,
    controlBytes,
    tableBytes,
    laterWitnessBytes,
  );
  const decoded = decodeConditionalPublicGame(
    rawLogBytes,
    projection.witness,
    dispatch,
    projection.provenance.mechanicalProfile,
    tableBytes,
  );
  projection.provenance.coreInitialVerified = true;
  decoded.manifest.initialProjection = projection.provenance;
  return { projection, decoded };
}

async function boundedPinnedRead(path, declaration, max, label) {
  closed(declaration, ['bytes', 'sha256'], `${label} pin`);
  integer(declaration.bytes, 1, max, `${label} size`);
  if (!/^[a-f0-9]{64}$/.test(declaration.sha256)) fail(`${label}: invalid SHA pin`);
  const file = await open(path, 'r');
  try {
    const stat = await file.stat();
    if (!stat.isFile() || stat.size !== declaration.bytes) fail(`${label}: file size mismatch`);
    const bytes = Buffer.alloc(declaration.bytes + 1);
    let used = 0;
    while (used < bytes.length) {
      const { bytesRead } = await file.read(bytes, used, bytes.length - used, null);
      if (!bytesRead) break;
      used += bytesRead;
    }
    const result = bytes.subarray(0, used);
    pin(result, declaration, max, label);
    return result;
  } finally {
    await file.close();
  }
}

export async function projectPlayingFiles(
  capturePath,
  calibrationPath,
  rawLogPath,
  controlPath,
  outputPath,
  cliPath = null,
  laterWitnessPath = null,
) {
  // Read the small control through a bounded handle before parsing its file declarations.
  const file = await open(controlPath, 'r');
  let controlBytes;
  try {
    const stat = await file.stat();
    if (!stat.isFile() || stat.size < 1 || stat.size > MAX_CONTROL_BYTES)
      fail('Control file size limit');
    const buffer = Buffer.alloc(stat.size + 1);
    let used = 0;
    while (used < buffer.length) {
      const { bytesRead } = await file.read(buffer, used, buffer.length - used, null);
      if (!bytesRead) break;
      used += bytesRead;
    }
    if (used !== stat.size) fail('Control changed during read');
    controlBytes = buffer.subarray(0, used);
  } finally {
    await file.close();
  }
  const control = uniqueJson(controlBytes, MAX_CONTROL_BYTES, 'control');
  if (control.table !== undefined && control.table !== null)
    fail('Separate table evidence requires the pinned byte API');
  if (laterWitnessPath === null && control.laterWitnesses != null)
    fail('Missing declared later witnesses');
  const laterSize = laterWitnessPath === null ? 0 : control.laterWitnesses?.bytes;
  if (
    !object(control.capture) ||
    !object(control.calibration) ||
    !object(control.rawLog) ||
    !Number.isSafeInteger(
      control.capture.bytes + control.calibration.bytes + control.rawLog.bytes + laterSize,
    ) ||
    control.capture.bytes +
      control.calibration.bytes +
      control.rawLog.bytes +
      laterSize +
      controlBytes.length >
      32 * 1024 * 1024
  )
    fail('Projection input aggregate byte limit');
  const inputs = await Promise.all([
    boundedPinnedRead(capturePath, control.capture, MAX_BYTES, 'capture'),
    boundedPinnedRead(calibrationPath, control.calibration, MAX_ATLAS_BYTES, 'calibration'),
    boundedPinnedRead(rawLogPath, control.rawLog, MAX_BYTES, 'raw log'),
  ]);
  const laterBytes =
    laterWitnessPath === null
      ? null
      : await boundedPinnedRead(
          laterWitnessPath,
          control.laterWitnesses,
          MAX_BYTES,
          'later witnesses',
        );
  const result =
    cliPath === null
      ? { projection: projectCapturedInitial(...inputs, controlBytes, null, laterBytes) }
      : decodeProjectedInitial(...inputs, controlBytes, coreDispatcher(cliPath), null, laterBytes);
  // Publish together using the same create-only reservation as replay exports.
  // Sidecar provenance is mandatory in this artifact, even though the app's
  // existing standalone PublicReplayRecord format cannot carry it itself.
  const output = resolve(outputPath);
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  let temporary = null,
    published = false;
  try {
    temporary = await mkdtemp(`${output}.tmp-`);
    if (dirname(temporary) !== dirname(output)) fail('Invalid projection staging path');
    await writeFile(
      join(temporary, 'projection.json'),
      `${JSON.stringify(result.projection, null, 2)}\n`,
      { flag: 'wx' },
    );
    await writeFile(
      join(temporary, 'replay.json'),
      `${JSON.stringify(result.decoded?.record ?? result.projection.record, null, 2)}\n`,
      { flag: 'wx' },
    );
    await writeFile(
      join(temporary, 'witness.json'),
      `${JSON.stringify(result.projection.witness, null, 2)}\n`,
      { flag: 'wx' },
    );
    if (result.decoded)
      await writeFile(
        join(temporary, 'decode-manifest.json'),
        `${JSON.stringify(result.decoded.manifest, null, 2)}\n`,
        { flag: 'wx' },
      );
    await rmdir(output); // Remove only this call's empty reservation.
    await rename(temporary, output);
    published = true;
  } finally {
    if (!published) {
      if (temporary !== null && dirname(temporary) === dirname(output))
        await rm(temporary, { recursive: true, force: true });
      await rmdir(output).catch((error) => {
        if (!['ENOENT', 'ENOTEMPTY', 'EEXIST'].includes(error.code)) throw error;
      });
    }
  }
  return result;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let laterWitnessPath = null;
  if (args.at(-2) === '--later-witnesses') {
    laterWitnessPath = args.pop();
    args.pop();
  }
  if (
    ![5, 6].includes(args.length) ||
    args.some((arg) => !arg || arg.startsWith('--')) ||
    laterWitnessPath === '' ||
    laterWitnessPath?.startsWith('--')
  ) {
    console.error(
      'Usage: node scripts/bga-playing-project.mjs capture.json atlas.json raw-log.json control.json new-output-directory [core-cli] [--later-witnesses later.json]',
    );
    process.exitCode = 1;
  } else {
    try {
      const result = await projectPlayingFiles(
        ...args.slice(0, 5),
        args[5] ?? null,
        laterWitnessPath,
      );
      console.log(
        JSON.stringify({
          schema: PROJECTION_SCHEMA,
          initialActionId: result.projection.provenance.initialActionId,
          coreInitialVerified: result.decoded !== undefined,
          trainingReady: false,
          conditionalMechanicalProfile: result.projection.provenance.mechanicalProfile,
        }),
      );
    } catch (error) {
      console.error(error.message);
      process.exitCode = 1;
    }
  }
}

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, mkdtemp, open, rename, rm, rmdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeGame } from './bga-import.mjs';
import { observedBasicUnlimitedOptions, uniqueJson } from './bga-observed-options.mjs';
import { catalogFingerprint, coreDispatcher } from './bga-replay.mjs';
import {
  decodeConditionalPublicGame,
  validateBuildingEquivalenceClass,
  WITNESS_SCHEMA,
} from './bga-decode.mjs';

export const PROJECTION_SCHEMA = 'tzolkin-bga-playing-projection-v1';
export const CONTROL_SCHEMA = 'tzolkin-bga-playing-projection-control-v1';
export const ATLAS_SCHEMA = 'tzolkin-bga-playing-atlas-v1';
export const LATER_WITNESS_SCHEMA = 'tzolkin-bga-playing-later-witnesses-v1';
export const BOUNDARY_CONTROL_SCHEMA = 'tzolkin-bga-retained-boundary-control-v1';
const BOUNDARY_CAPTURE_SCHEMA = 'bga-retained-visible-replay-board-v1';
const LEGACY_CAPTURE_SCHEMA = 'tzolkin-bga-public-initial-selection-observation-v1';
const LEGACY_CAPTURE_ADAPTER = 'tzolkin-bga-legacy-initial-capture-adapter-v1';
const FRAGMENT_CAPTURE_SCHEMA = 'bga-retained-visible-initial-dom-v1';
const FRAGMENT_CAPTURE_ADAPTER = 'tzolkin-bga-retained-initial-fragment-adapter-v1';
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
function htmlTree(source, label, fragmentOnly = false) {
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
    if (fragmentOnly && ['style', 'svg', 'math', 'head'].includes(tag))
      fail(`${label}: unsupported retained initial fragment container`);
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
function retained(tree, items, predicate, label, domOnly = false) {
  const expected = tree.nodes.filter(predicate);
  list(items, 100, label);
  if (items.length !== expected.length) fail(`${label}: incomplete retained node list`);
  const seen = new Set();
  for (const item of items) {
    if (!object(item) || (!domOnly && item.visible !== true) || seen.has(item.id))
      fail(`${label}: invisible/duplicate node`);
    seen.add(item.id);
    const node = byId(tree, item.id);
    if (!predicate(node) || item.html !== outer(tree, node)) fail(`${label}: detached HTML`);
    if (domOnly) unhiddenFragmentNode(tree, node);
  }
  return expected;
}

function unhiddenFragmentNode(tree, node) {
  for (const ancestor of tree.nodes.filter(
    (item) => item.start <= node.start && item.end >= node.end,
  )) {
    if (/\/\*|\\|[{}@]/.test(ancestor.attrs.style ?? ''))
      fail('Unsupported retained initial fragment style');
    const css = styles(ancestor);
    if (Object.keys(css).some((key) => key === 'all' || key.startsWith('--')))
      fail('Unsupported retained initial fragment style');
    const value = (key) =>
      (css[key] ?? '')
        .replace(/\s*!\s*important\s*$/i, '')
        .trim()
        .toLowerCase();
    const supported = {
      display: [
        'none',
        'block',
        'inline',
        'inline-block',
        'flex',
        'inline-flex',
        'grid',
        'inline-grid',
        'flow-root',
        'contents',
        'list-item',
      ],
      visibility: ['visible', 'hidden', 'collapse'],
      'content-visibility': ['visible', 'hidden'],
    };
    for (const [property, keywords] of Object.entries(supported))
      if (value(property) !== '' && !keywords.includes(value(property)))
        fail('Unsupported retained initial fragment visibility');
    const opacityText = value('opacity');
    const opacity = Number(opacityText.replace(/%$/, ''));
    if (
      opacityText !== '' &&
      (!/^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?%?$/i.test(opacityText) ||
        !Number.isFinite(opacity))
    )
      fail('Unsupported retained initial fragment opacity');
    if (
      ['datalist', 'details', 'meter', 'progress', 'rp'].includes(ancestor.tag) ||
      (ancestor.tag === 'dialog' && !Object.hasOwn(ancestor.attrs, 'open')) ||
      Object.hasOwn(ancestor.attrs, 'popover') ||
      Object.hasOwn(ancestor.attrs, 'hidden') ||
      Object.hasOwn(ancestor.attrs, 'inert') ||
      ancestor.attrs['aria-hidden']?.trim().toLowerCase() === 'true' ||
      value('display') === 'none' ||
      ['hidden', 'collapse'].includes(value('visibility')) ||
      value('content-visibility') === 'hidden' ||
      (opacityText !== '' && opacity <= 0)
    )
      fail('Hidden retained initial fragment node');
  }
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
function natural(node, label, plain = false) {
  if (plain && node.children.some(object)) fail(`${label}: nonscalar DOM counter`);
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

function retainedInitialFragments(capture) {
  closed(
    capture,
    [
      'schema',
      'sourceKind',
      'tableId',
      'sourceUrl',
      'moveText',
      'roundText',
      'prompt',
      'boardHtml',
      'playerBoards',
      'renderedReplayControls',
      'completeNativeReplayQualified',
      'humanBcQualified',
    ],
    'retained initial fragments',
  );
  const url = urlIdentity(capture.sourceUrl);
  if (
    capture.sourceKind !== 'public_ui_dom_only' ||
    capture.completeNativeReplayQualified !== false ||
    capture.humanBcQualified !== false ||
    url.searchParams.get('table') !== capture.tableId ||
    !/^[1-9]\d*$/.test(capture.moveText) ||
    capture.roundText !== '1'
  )
    fail('Retained initial fragment identity/boundary mismatch');
  for (const control of list(capture.renderedReplayControls, 3, 'replay controls')) {
    closed(control, ['id', 'html'], 'replay control');
    if (!['archive_restart', 'archive_next', 'archive_end_game'].includes(control.id))
      fail('Unknown retained replay control');
    text(control.html, 4096, 'replay control HTML');
  }
  const board = htmlTree(capture.boardHtml, 'retained board', true);
  if (
    byId(board, 'board').tag !== 'div' ||
    outer(board, byId(board, 'board')) !== capture.boardHtml
  )
    fail('Detached retained board fragment');
  unhiddenFragmentNode(board, byId(board, 'board'));
  for (const id of [
    'rounds_wrap',
    'current_round',
    'skull_left_wrap',
    'skull_left',
    'buildings',
    'monuments',
    'centralgearcorns',
    'workerplace_startingPlayer_0',
    ...[2, 3, 4, 5].map((box) => `jungle_${box}`),
  ])
    byId(board, id);
  const rounds = byId(board, 'rounds_wrap');
  const round = byId(board, 'current_round');
  if (round.start <= rounds.start || round.end >= rounds.end)
    fail('Round counter outside retained rounds wrapper');
  for (const node of board.nodes.filter(
    (node) => node.start >= rounds.start && node.end <= rounds.end,
  ))
    unhiddenFragmentNode(board, node);
  const identities = new Map(),
    panels = [],
    rows = [],
    selected = [];
  for (const item of list(capture.playerBoards, 4, 'retained player fragments')) {
    closed(
      item,
      ['playerId', 'boardHtml', 'rowHtml', 'buildingsHtml', 'wealthTilesHtml'],
      'player fragments',
    );
    const id = item.playerId;
    if (typeof id !== 'string' || !/^\d+$/.test(id) || identities.has(id))
      fail('Duplicate/invalid retained player identity');
    const panel = htmlTree(item.boardHtml, 'retained player board', true);
    const owner = byId(panel, `player_board_${id}`);
    if (owner.tag !== 'div' || outer(panel, owner) !== item.boardHtml)
      fail('Detached retained player board');
    if (byId(panel, `player_score_${id}`).tag !== 'span') fail('Malformed retained score counter');
    const row = htmlTree(item.rowHtml, 'retained player row', true);
    const root = row.nodes[0];
    if (
      !root ||
      root.tag !== 'div' ||
      !token(root, 'whiteblock') ||
      outer(row, root) !== item.rowHtml
    )
      fail('Detached retained player row');
    for (const [tree, own] of [
      [panel, owner],
      [row, root],
    ]) {
      unhiddenFragmentNode(tree, own);
      for (const node of tree.nodes) {
        const key = node.attrs.id ?? '';
        const other =
          /^(?:(?:worker|corn|wood|stone|gold|skull)_number_player_|player_(?:board|name|score|buildings|wealthtiles)_|firstplayer_|marker_)(\d+)(?:_|$)/.exec(
            key,
          )?.[1];
        if (
          (other !== undefined && other !== id) ||
          (node !== owner && /^(?:overall_player_board_|player_board_)/.test(key))
        )
          fail('Retained player fragment crosses owner identity');
      }
    }
    const heading = one(
      root.children.filter(object),
      (node) => node.tag === 'h3',
      'retained player heading',
    );
    unhiddenFragmentNode(row, heading);
    if (heading.children.some(object)) fail('Malformed retained player heading');
    const label = nodeText(heading).trim();
    if (!label.endsWith(':')) fail('Malformed retained player heading');
    identities.set(id, {
      name: text(label.slice(0, -1), 128, 'player name'),
      color: styles(heading).color,
    });
    const wealth = byId(row, `player_wealthtiles_${id}`);
    const buildings = byId(row, `player_buildings_${id}`);
    if (
      wealth.tag !== 'div' ||
      buildings.tag !== 'div' ||
      !root.children.includes(wealth) ||
      !root.children.includes(buildings) ||
      outer(row, wealth) !== item.wealthTilesHtml ||
      outer(row, buildings) !== item.buildingsHtml ||
      buildings.children.some((child) => typeof child !== 'string' || child.trim())
    )
      fail('Detached/nonempty retained player cards');
    unhiddenFragmentNode(row, buildings);
    panels.push(item.boardHtml);
    rows.push(item.rowHtml);
    selected.push({ id: wealth.attrs.id, html: item.wealthTilesHtml });
  }
  if (![3, 4].includes(identities.size) || !identities.has(url.searchParams.get('player')))
    fail('Incomplete retained roster/view identity');
  const gameAreaHtml = capture.boardHtml + rows.join('');
  const playerBoardsHtml = panels.join('');
  const tree = htmlTree(gameAreaHtml, 'retained game fragments', true);
  const boards = htmlTree(playerBoardsHtml, 'retained counter fragments', true);
  if ([...tree.ids.keys()].some((id) => boards.ids.has(id)))
    fail('Conflicting retained sibling node identities');
  const boardRoot = byId(tree, 'board');
  for (const node of tree.nodes) {
    const boardField =
      /^(?:rounds_wrap$|current_round$|skull_left(?:_wrap)?$|buildings$|monuments$|centralgearcorns$|jungle_|workerplace_|gear_|skullplace_|marker_\d+_(?:temple|technology)_)/.test(
        node.attrs.id ?? '',
      ) || ['workerplace', 'board_tile', 'construction'].some((kind) => token(node, kind));
    if (boardField && (node.start <= boardRoot.start || node.end >= boardRoot.end))
      fail('Retained board field outside original board fragment');
  }
  if (tree.nodes.some((node) => token(node, 'worker')))
    fail('Retained fragments lack placed-worker geometry');
  const nodes = (source, predicate, tag) =>
    source.nodes.filter(predicate).map((node) => {
      if (node.tag !== tag) fail('Malformed retained initial field node');
      return { id: node.attrs.id, html: outer(source, node) };
    });
  return {
    ...capture,
    url: capture.sourceUrl,
    viewPlayerId: url.searchParams.get('player'),
    move: `Move #${capture.moveText}`,
    header: capture.prompt,
    roundsText: nodeText(rounds).trim().replace(/\s+/g, ' '),
    gameAreaHtml,
    playerBoardsHtml,
    identities,
    offerNodesRemaining: tree.nodes.filter(
      (node) =>
        token(node, 'wealth_tile') &&
        !selected.some((item) => {
          const panel = byId(tree, item.id);
          return node.start > panel.start && node.end < panel.end;
        }),
    ).length,
    publicCounters: nodes(
      boards,
      (node) => /^(worker|corn|wood|stone|gold|skull)_number_player_\d+$/.test(node.attrs.id ?? ''),
      'span',
    ),
    selectedPlayerTiles: selected,
    templeMarkers: nodes(tree, (node) => /^marker_\d+_temple_/.test(node.attrs.id ?? ''), 'div'),
    technologyMarkers: nodes(
      tree,
      (node) => /^marker_\d+_technology_/.test(node.attrs.id ?? ''),
      'div',
    ),
    gears: nodes(
      tree,
      (node) =>
        /^gear_(central|palenque|yaxchilan|tikal|uxmal|chichenItza)$/.test(node.attrs.id ?? ''),
      'img',
    ),
    gearWorkerNodes: [],
    skullSpaces: nodes(tree, (node) => /^skullplace_\d+$/.test(node.attrs.id ?? ''), 'div'),
    skullRemainingHtml: outer(tree, byId(tree, 'skull_left_wrap')),
    buildingsHtml: outer(tree, byId(tree, 'buildings')),
    monumentsHtml: outer(tree, byId(tree, 'monuments')),
  };
}

/** Project only independently retained counters at a settled food/terminal boundary.
 * The returned witness fits the ordinary later-witness collection; its separate
 * provenance must travel with it. This does not establish full board coverage. */
export function projectRetainedBoundary(
  captureBytes,
  rawLogBytes,
  controlBytes,
  tableBytes = null,
) {
  const inputs = [
    captureBytes,
    rawLogBytes,
    controlBytes,
    ...(tableBytes === null ? [] : [tableBytes]),
  ];
  if (
    inputs.some((input) => !Buffer.isBuffer(input)) ||
    inputs.reduce((total, input) => total + input.length, 0) > 32 * 1024 * 1024
  )
    fail('Boundary projection input aggregate byte limit');
  const control = uniqueJson(controlBytes, MAX_CONTROL_BYTES, 'boundary control');
  closed(
    control,
    ['schema', 'capture', 'rawLog', 'archiveRelease', 'seatOrder', 'mechanicalProfile'],
    'boundary control',
    ['table'],
  );
  if (control.schema !== BOUNDARY_CONTROL_SCHEMA) fail('Unsupported boundary control');
  pin(captureBytes, control.capture, MAX_BYTES, 'boundary capture');
  pin(rawLogBytes, control.rawLog, MAX_BYTES, 'raw log');
  if (tableBytes !== null) pin(tableBytes, control.table, MAX_BYTES, 'table evidence');
  else if (control.table !== undefined && control.table !== null)
    fail('Missing declared table evidence');
  const capture = uniqueJson(captureBytes, MAX_BYTES, 'boundary capture');
  const raw = uniqueJson(rawLogBytes, MAX_BYTES, 'raw log');
  const table = tableBytes === null ? null : uniqueJson(tableBytes, MAX_BYTES, 'table evidence');
  const mechanicalProfile = profile(control.mechanicalProfile);
  observedBasicUnlimitedOptions(raw, table);
  const game = normalizeGame(raw, sha(rawLogBytes), tableBytes);
  if (tableBytes !== null && game.source.adapter !== 'bga-public-ui-dom-v1')
    fail('Legacy input requires embedded table settings, not separate table evidence');
  closed(
    capture,
    [
      'schema',
      'sourceKind',
      'tableId',
      'sourceUrl',
      'capturedAt',
      'boardHtml',
      'moveText',
      'playerBoards',
      'prompt',
      'renderedReplayControls',
      'retainedDomGeometry',
      'roundText',
      'completeNativeReplayQualified',
      'humanBcQualified',
    ],
    'retained boundary',
  );
  const url = urlIdentity(capture.sourceUrl);
  const release = /^\/archive\/replay\/([A-Za-z0-9_-]{1,64})\/$/.exec(url.pathname)?.[1];
  const tableId =
    typeof capture.tableId === 'number'
      ? String(integer(capture.tableId, 1, Number.MAX_SAFE_INTEGER, 'boundary table'))
      : capture.tableId;
  const move = /^(?:Coup|Move) #([1-9]\d*)$/.exec(capture.moveText ?? '');
  const actionId = integer(move ? Number(move[1]) : null, 1, 100000, 'boundary move');
  if (
    capture.schema !== BOUNDARY_CAPTURE_SCHEMA ||
    capture.sourceKind !== 'public_ui_dom_only' ||
    capture.completeNativeReplayQualified !== false ||
    capture.humanBcQualified !== false ||
    !release ||
    release !== control.archiveRelease ||
    url.searchParams.get('table') !== tableId ||
    game.tableId !== tableId ||
    url.searchParams.getAll('player').length !== 1 ||
    !/^\d+$/.test(url.searchParams.get('player') ?? '')
  )
    fail('Retained boundary identity mismatch');
  text(capture.capturedAt, 64, 'boundary capture time');
  text(capture.renderedReplayControls, 4096, 'retained replay controls');
  text(capture.roundText, 160, 'boundary round text');
  text(capture.prompt, 256, 'boundary prompt');
  const sourceIndex = game.rawEntries.findIndex((entry) => entry.action_id === actionId);
  if (sourceIndex < 0) fail('Retained boundary action is absent from raw log');
  const events = game.events.filter((event) => event.actionId === actionId);
  const priorId = game.rawEntries[sourceIndex - 1]?.action_id;
  const prior = game.events.filter((event) => event.actionId === priorId);
  const board = htmlTree(capture.boardHtml, 'retained boundary board', true);
  const root = byId(board, 'board');
  if (root.tag !== 'div' || outer(board, root) !== capture.boardHtml)
    fail('Detached retained boundary board');
  unhiddenFragmentNode(board, root);
  const rounds = byId(board, 'rounds_wrap'),
    round = byId(board, 'current_round');
  if (round.start <= rounds.start || round.end >= rounds.end)
    fail('Boundary round counter outside rounds wrapper');
  unhiddenFragmentNode(board, round);
  const displayedRound = natural(round, 'boundary round', true);
  if (
    nodeText(rounds).trim().replace(/\s+/g, ' ') !==
      capture.roundText.trim().replace(/\s+/g, ' ') ||
    !new RegExp(`^経過日\\s*${displayedRound}\\s*/\\s*27$`).test(capture.roundText.trim())
  )
    fail('Boundary round text mismatch');
  const terminal = capture.prompt.trim() === 'ゲーム終了';
  // A split food/rotation log is admitted only across adjacent original entries.
  // Rotation messages without actor gameplay cannot inherit a more distant food day.
  const split =
    !events.some((event) => event.kind === 'feeding') &&
    events.every(
      (event) =>
        event.kind === 'gearAdvanced' ||
        (event.kind === 'unparsed' &&
          /^スタートプレイヤースペースに置かれているワーカーは.+の手元に戻った$/.test(
            event.rawText,
          )),
    ) &&
    !prior.some((event) => ['gearAdvanced', 'gameEnd'].includes(event.kind)) &&
    prior.some((event) => event.kind === 'feeding');
  const envelope = split ? [...prior, ...events] : events;
  const rotationDays = envelope.some((event) => event.kind === 'doubleAdvance') ? 2 : 1;
  const foodDay = terminal ? 27 : displayedRound - rotationDays;
  if (
    events.filter((event) => event.kind === 'gearAdvanced').length !== 1 ||
    envelope.filter((event) => event.kind === 'doubleAdvance').length > 1 ||
    !envelope.some((event) => event.kind === 'feeding') ||
    envelope.some((event) =>
      [
        'place',
        'gearAction',
        'technology',
        'trade',
        'build',
        'monument',
        'beg',
        'cancellation',
      ].includes(event.kind),
    ) ||
    (terminal
      ? displayedRound !== 28 ||
        game.quality.status !== 'normalEnd' ||
        !events.some((event) => event.kind === 'gameEnd')
      : ![8, 14, 21].includes(foodDay) || events.some((event) => event.kind === 'gameEnd'))
  )
    fail('Capture has no matching settled food/terminal source boundary');
  const ids = list(control.seatOrder, 4, 'boundary seat order');
  const fragments = list(capture.playerBoards, 4, 'boundary player fragments');
  if (
    ![3, 4].includes(ids.length) ||
    fragments.length !== ids.length ||
    game.context.playerCount !== ids.length ||
    new Set(ids).size !== ids.length ||
    ids.some((id) => typeof id !== 'string' || !/^\d+$/.test(id)) ||
    !ids.includes(url.searchParams.get('player'))
  )
    fail('Incomplete retained boundary roster/view');
  const trees = [board],
    identities = new Map(),
    displayedPlayers = [],
    anonymousRows = [];
  const players = ids.map((id, index) => {
    const item = one(fragments, (item) => item.playerId === id, 'boundary player identity');
    closed(
      item,
      ['playerId', 'boardHtml', 'rowHtml', 'buildingsHtml', 'wealthTilesHtml'],
      'boundary player',
    );
    const panel = htmlTree(item.boardHtml, 'boundary player board', true);
    const owner = byId(panel, `player_board_${id}`);
    const row = htmlTree(item.rowHtml, 'boundary player row', true),
      rowRoot = row.nodes[0];
    if (
      owner.tag !== 'div' ||
      outer(panel, owner) !== item.boardHtml ||
      !rowRoot ||
      rowRoot.tag !== 'div' ||
      !token(rowRoot, 'whiteblock') ||
      outer(row, rowRoot) !== item.rowHtml
    )
      fail('Detached retained boundary player fragment');
    for (const tree of [panel, row])
      for (const node of tree.nodes) {
        const key = node.attrs.id ?? '';
        const other =
          /^(?:(?:worker|corn|wood|stone|gold|skull)_number_player_|player_(?:board|name|score|buildings|wealthtiles)_|firstplayer_|marker_)(\d+)(?:_|$)/.exec(
            key,
          )?.[1];
        if (
          (other !== undefined && other !== id) ||
          (token(node, 'worker') && /^worker_\d+$/.test(key)) ||
          (node !== owner && /^(?:overall_player_board_|player_board_)/.test(key))
        )
          fail('Boundary fragment crosses original owner/board scope');
      }
    unhiddenFragmentNode(panel, owner);
    unhiddenFragmentNode(row, rowRoot);
    if (
      !rowRoot.attrs.id &&
      rowRoot.attrs.class === 'whiteblock' &&
      rowRoot.attrs.style === undefined
    )
      anonymousRows.push(rowRoot);
    const heading = one(
      rowRoot.children.filter(object),
      (node) => node.tag === 'h3',
      'boundary player heading',
    );
    unhiddenFragmentNode(row, heading);
    const label = nodeText(heading).trim(),
      color = styles(heading).color;
    if (
      heading.children.some(object) ||
      !label.endsWith(':') ||
      !/^#[a-f0-9]{6}$/i.test(color ?? '')
    )
      fail('Malformed retained boundary player heading');
    const name = text(label.slice(0, -1), 128, 'boundary player name');
    identities.set(id, { name, color: color.slice(1).toLowerCase() });
    for (const [key, html] of [
      [`player_buildings_${id}`, item.buildingsHtml],
      [`player_wealthtiles_${id}`, item.wealthTilesHtml],
    ]) {
      const node = byId(row, key);
      if (node.tag !== 'div' || !rowRoot.children.includes(node) || outer(row, node) !== html)
        fail('Detached retained boundary card fragment');
    }
    const counter = (key, signed = false) => {
      const node = byId(panel, key);
      unhiddenFragmentNode(panel, node);
      if (node.tag !== 'span' || node.children.some(object)) fail('Malformed boundary counter');
      const value = nodeText(node).trim();
      if (!(signed ? /^-?\d+$/ : /^\d+$/).test(value)) fail('Nonnumeric boundary counter');
      return integer(Number(value), signed ? -100000 : 0, 100000, 'boundary counter');
    };
    const resources = Object.fromEntries(
      RESOURCES.map((resource) => [resource, counter(`${resource}_number_player_${id}`)]),
    );
    const availableWorkers = integer(
      counter(`worker_number_player_${id}`),
      0,
      6,
      'available workers',
    );
    displayedPlayers.push({
      playerId: index,
      sourcePlayerId: id,
      name,
      availableWorkers,
      scoreDisplay: counter(`player_score_${id}`, true),
    });
    trees.push(panel, row);
    return { id: index, resources };
  });
  const names = [...identities.values()].map((player) => player.name);
  const colors = [...identities.values()].map((player) => player.color);
  if (
    new Set(names).size !== ids.length ||
    new Set(colors).size !== ids.length ||
    !same([...names].sort(), [...game.context.players].sort())
  )
    fail('Boundary roster differs from raw log');
  const prompt = capture.prompt.trim(),
    suffix = 'はワーカーを置くか取り除くかしてください';
  if (!terminal && !names.some((name) => prompt === name + suffix) && prompt !== `あなた${suffix}`)
    fail('Boundary prompt is not a settled Playing decision');
  const nodeIds = new Set();
  for (const tree of trees)
    for (const id of tree.ids.keys()) {
      if (nodeIds.has(id)) fail('Conflicting retained boundary sibling identities');
      nodeIds.add(id);
    }
  const workerRoot = byId(board, 'workers');
  if (workerRoot.tag !== 'div' || !root.children.includes(workerRoot))
    fail('Detached boundary worker layer');
  unhiddenFragmentNode(board, workerRoot);
  const placed = board.nodes.filter((node) => token(node, 'worker'));
  if (
    placed.some(
      (node) =>
        node.tag !== 'div' ||
        !/^worker_[1-9]\d*$/.test(node.attrs.id ?? '') ||
        !workerRoot.children.includes(node) ||
        node.children.some((child) => typeof child !== 'string' || child.trim()),
    ) ||
    workerRoot.children.some((child) => (object(child) ? !placed.includes(child) : child.trim()))
  )
    fail('Unknown retained boundary worker ownership');
  const counts = Array(ids.length).fill(0);
  for (const node of placed) {
    unhiddenFragmentNode(board, node);
    const classes = (node.attrs.class ?? '').split(/\s+/).filter(Boolean);
    const owner = classes.find((value) => /^worker_[a-f0-9]{6}$/.test(value));
    if (
      classes.length !== 2 ||
      (owner ? !colors.includes(owner.slice(7)) : !classes.includes('worker_dummy'))
    )
      fail('Unknown/ambiguous boundary worker color');
    if (owner) counts[colors.indexOf(owner.slice(7))]++;
  }
  const geometry = new Map();
  let anonymousRowMetadata = 0;
  for (const item of list(capture.retainedDomGeometry, 20000, 'boundary geometry')) {
    if (!object(item) || typeof item.id !== 'string')
      fail('Invalid/duplicate boundary geometry identity');
    const declaresWorker =
      item.parentId === 'workers' ||
      /^worker_[1-9]\d*$/.test(item.id) ||
      (typeof item.className === 'string' && item.className.split(/\s+/).includes('worker'));
    // The retained anonymous row wrappers are separately owner-validated above.
    // Their external parent and rect are metadata, never canvas worker evidence.
    if (item.id === '') {
      if (declaresWorker) fail('Unidentified boundary worker geometry');
      if (
        item.parentId !== 'tzolkin-interface' ||
        item.className !== 'whiteblock' ||
        item.inlineStyle != null
      )
        fail('Unknown anonymous boundary geometry');
      anonymousRowMetadata++;
      continue;
    }
    if (geometry.has(item.id)) fail('Invalid/duplicate boundary geometry identity');
    if (declaresWorker) {
      const node = placed.find((node) => node.attrs.id === item.id);
      if (
        !node ||
        item.parentId !== 'workers' ||
        (item.className ?? '') !== (node.attrs.class ?? '') ||
        (item.inlineStyle ?? '') !== (node.attrs.style ?? '')
      )
        fail('Unbound boundary worker geometry');
    }
    geometry.set(item.id, item);
  }
  if (anonymousRowMetadata !== 0 && anonymousRowMetadata !== anonymousRows.length)
    fail('Anonymous boundary row metadata differs from retained owned rows');
  const visibleRect = (node) => {
    const item = geometry.get(node.attrs.id);
    if (!item) return null;
    if (
      (item.className ?? '') !== (node.attrs.class ?? '') ||
      (item.inlineStyle ?? '') !== (node.attrs.style ?? '')
    )
      fail('Detached boundary geometry');
    const rect = item.rect;
    if (
      item.computedDisplay !== 'block' ||
      item.computedVisibility !== 'visible' ||
      !object(rect) ||
      !['x', 'y', 'width', 'height'].every((key) => Number.isFinite(rect[key])) ||
      rect.width < 0 ||
      rect.height < 0
    )
      return null;
    return rect;
  };
  const boardRect = visibleRect(root),
    layerRect = visibleRect(workerRoot);
  const geometricWorkers = [...geometry.values()].filter((item) =>
    /^worker_[1-9]\d*$/.test(item.id),
  );
  const completeWorkers =
    boardRect?.width > 0 &&
    boardRect.height > 0 &&
    layerRect !== null &&
    geometry.get('workers').parentId === 'board' &&
    layerRect.x === boardRect.x &&
    layerRect.width === boardRect.width &&
    geometricWorkers.length === placed.length &&
    geometricWorkers.every((item) => placed.some((node) => node.attrs.id === item.id)) &&
    placed.every((node) => {
      const rect = visibleRect(node);
      return (
        rect !== null &&
        rect.width > 0 &&
        rect.height > 0 &&
        geometry.get(node.attrs.id).parentId === 'workers' &&
        rect.x >= boardRect.x &&
        rect.y >= boardRect.y &&
        rect.x + rect.width <= boardRect.x + boardRect.width &&
        rect.y + rect.height <= boardRect.y + boardRect.height
      );
    });
  if (completeWorkers)
    players.forEach((player, index) => {
      player.workers = integer(
        displayedPlayers[index].availableWorkers + counts[index],
        3,
        6,
        'total workers',
      );
    });
  const skull = byId(board, 'skull_left');
  unhiddenFragmentNode(board, skull);
  if (skull.tag !== 'span') fail('Malformed boundary skull counter');
  const expected = {
    players,
    skullSupply: natural(skull, 'boundary skull supply', true),
    phase: terminal ? 'finished' : 'playing',
  };
  // The source displays day 28 and integral scores after finalization. Neither
  // is an exact native round/score assertion; preserve those display facts apart.
  if (!terminal) expected.round = displayedRound;
  return {
    witness: { actionId, expected },
    provenance: {
      schema: 'tzolkin-bga-retained-boundary-projection-v1',
      sourceSchema: capture.schema,
      captureSha256: sha(captureBytes),
      rawLogSha256: sha(rawLogBytes),
      tableEvidenceSha256: tableBytes === null ? null : sha(tableBytes),
      archiveRelease: release,
      mechanicalProfile,
      sourceActionIds: split ? [priorId, actionId] : [actionId],
      foodDay,
      kind: terminal ? 'terminal' : 'post-food',
      displayedRound,
      displayedPlayers,
      totalWorkersObserved: Boolean(completeWorkers),
      completeSourceBoard: false,
      omitted: [
        'foodDays',
        'turnOrder',
        'turnIndex',
        'markers',
        'card identities',
        'gear positions',
        'covered corn',
        'exact scores',
        ...(completeWorkers ? [] : ['total workers']),
      ],
      visibilityVerified: false,
      geometryVerified: false,
      sourceAuthenticated: false,
      completeNativeReplayQualified: false,
      humanBcQualified: false,
      policyTrainingReady: false,
      valueTrainingReady: false,
    },
  };
}

// This older capture retained nested sidebar panels rather than one board and
// explicit node lists. Extract only the same captured markup; no rectangles,
// option values or artwork identities are reconstructed.
function legacyInitialCapture(capture) {
  closed(
    capture,
    [
      'schema',
      'url',
      'move',
      'header',
      'roundsText',
      'gameAreaHtml',
      'playerPanels',
      'selectedPlayerTiles',
    ],
    'legacy capture',
    ['capturedAt', 'title', 'wealthTilesHtml', 'normalNextClicksSinceRestart', 'trainingReady'],
  );
  const url = urlIdentity(capture.url);
  if (
    !/^\/archive\/replay\/[A-Za-z0-9_-]{1,64}\/$/.test(url.pathname) ||
    url.searchParams.getAll('player').length !== 1 ||
    !/^\d+$/.test(url.searchParams.get('player') ?? '') ||
    (capture.trainingReady !== undefined && capture.trainingReady !== false)
  )
    fail('Legacy capture archive identity/readiness mismatch');
  const panels = new Map();
  for (const item of list(capture.playerPanels, 8, 'legacy player panels')) {
    closed(item, ['id', 'html'], 'legacy player panel');
    const tree = htmlTree(item.html, 'legacy player panel');
    const root = byId(tree, item.id);
    if (outer(tree, root) !== item.html) fail('Detached legacy player panel');
    const board = one(
      tree.nodes,
      (node) => /^player_board_\d+$/.test(node.attrs.id ?? ''),
      'legacy board',
    );
    const id = board.attrs.id.slice('player_board_'.length);
    if (item.id !== board.attrs.id && !/^player_board_inner_[A-Za-z0-9]+$/.test(item.id))
      fail('Unsupported legacy player panel ID');
    const name = tree.ids.get(`player_name_${id}`);
    for (const node of tree.nodes) {
      const owner =
        /^(?:(?:worker|corn|wood|stone|gold|skull)_number_player_|player_(?:name|score)_|firstplayer_|overall_player_board_)(\d+)$/.exec(
          node.attrs.id ?? '',
        )?.[1];
      if (owner !== undefined && owner !== id) fail('Legacy player panel crosses player identity');
      if (
        (owner !== undefined || node.attrs.id === 'firstplayer') &&
        ![board, name].some(
          (range) => range !== undefined && node.start >= range.start && node.end <= range.end,
        )
      )
        fail('Legacy setup node outside retained board/name');
    }
    const boardHtml = outer(tree, board);
    const nameHtml = name === undefined ? null : outer(tree, name);
    const prior = panels.get(id);
    if (
      prior &&
      (prior.board !== boardHtml ||
        (prior.name !== null && nameHtml !== null && prior.name !== nameHtml))
    )
      fail('Conflicting duplicate legacy player panel');
    panels.set(id, { board: boardHtml, name: nameHtml ?? prior?.name ?? null });
  }
  if (![3, 4].includes(panels.size) || !panels.has(url.searchParams.get('player')))
    fail('Legacy capture roster/view mismatch');
  const tree = htmlTree(capture.gameAreaHtml, 'legacy game area');
  const selected = list(capture.selectedPlayerTiles, 4, 'legacy selected panels');
  const selectedIds = new Set();
  for (const item of selected) {
    closed(item, ['id', 'html'], 'legacy selected panel');
    const id = /^player_wealthtiles_(\d+)$/.exec(item.id)?.[1];
    if (!panels.has(id) || selectedIds.has(id) || item.html !== outer(tree, byId(tree, item.id)))
      fail('Missing/conflicting/detached legacy selected panel');
    selectedIds.add(id);
  }
  if (selectedIds.size !== panels.size) fail('Incomplete legacy selected panels');
  for (const node of tree.nodes) {
    const id = /^(?:marker_(\d+)_(?:temple|technology)_|player_wealthtiles_(\d+)$)/.exec(
      node.attrs.id ?? '',
    );
    if (id && !panels.has(id[1] ?? id[2])) fail('Legacy board crosses player identity');
  }
  if (tree.nodes.some((node) => token(node, 'worker')))
    fail('Legacy capture lacks retained worker geometry');
  if (
    capture.wealthTilesHtml != null &&
    capture.wealthTilesHtml !== outer(tree, byId(tree, 'wealth_tiles'))
  )
    fail('Detached legacy starting offers');
  // Generated list markers mean parsed retained elements, not observed CSS
  // visibility. Conditional provenance below records that distinction.
  const nodes = (predicate) =>
    tree.nodes
      .filter(predicate)
      .map((node) => ({ id: node.attrs.id, html: outer(tree, node), visible: true }));
  const playerBoardsHtml = [...panels]
    .map(([id, panel]) => {
      if (panel.name === null) fail('Missing legacy player name');
      const boardTree = htmlTree(panel.board, 'legacy board');
      return `<div id="overall_player_board_${id}">${boardTree.ids.has(`player_name_${id}`) ? '' : panel.name}${panel.board}</div>`;
    })
    .join('');
  const boards = htmlTree(playerBoardsHtml, 'legacy player boards');
  return {
    ...capture,
    tableId: url.searchParams.get('table'),
    viewPlayerId: url.searchParams.get('player'),
    playerBoardsHtml,
    offerNodesRemaining: tree.nodes.filter(
      (node) =>
        token(node, 'wealth_tile') &&
        !selected.some((item) => {
          const panel = byId(tree, item.id);
          return node.start > panel.start && node.end < panel.end;
        }),
    ).length,
    publicCounters: boards.nodes
      .filter((node) =>
        /^(worker|corn|wood|stone|gold|skull)_number_player_\d+$/.test(node.attrs.id ?? ''),
      )
      .map((node) => ({ id: node.attrs.id, html: outer(boards, node), visible: true })),
    selectedPlayerTiles: selected.map((item) => ({ ...item, visible: true })),
    templeMarkers: nodes((node) => /^marker_\d+_temple_/.test(node.attrs.id ?? '')),
    technologyMarkers: nodes((node) => /^marker_\d+_technology_/.test(node.attrs.id ?? '')),
    gears: nodes((node) =>
      /^gear_(central|palenque|yaxchilan|tikal|uxmal|chichenItza)$/.test(node.attrs.id ?? ''),
    ),
    gearWorkerNodes: [],
    skullSpaces: nodes((node) => /^skullplace_\d+$/.test(node.attrs.id ?? '')),
    skullRemainingHtml: outer(tree, byId(tree, 'skull_left_wrap')),
    buildingsHtml: outer(tree, byId(tree, 'buildings')),
    monumentsHtml: outer(tree, byId(tree, 'monuments')),
  };
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
      'buildingEquivalenceClass',
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
    validateBuildingEquivalenceClass(witness);
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
function dummyProjection(capture, tree, atlas, n, domOnly = false) {
  const gears = Object.fromEntries(
    GEARS.map((gear) => [gear, Array(gear === 'chichenItza' ? 13 : 10).fill(null)]),
  );
  const nodes = retained(
    tree,
    capture.gearWorkerNodes,
    (node) => token(node, 'worker'),
    'gear workers',
    domOnly,
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
  if (domOnly) places.forEach((node) => unhiddenFragmentNode(tree, node));
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
  let capture = uniqueJson(captureBytes, MAX_BYTES, 'capture');
  const calibration = uniqueJson(calibrationBytes, MAX_ATLAS_BYTES, 'calibration');
  const raw = uniqueJson(rawLogBytes, MAX_BYTES, 'raw log');
  const table = tableBytes === null ? null : uniqueJson(tableBytes, MAX_BYTES, 'table evidence');
  observedBasicUnlimitedOptions(raw, table);
  const mechanicalProfile = profile(control.mechanicalProfile);
  const legacy = capture.schema === LEGACY_CAPTURE_SCHEMA;
  const fragments = capture.schema === FRAGMENT_CAPTURE_SCHEMA;
  if (legacy) capture = legacyInitialCapture(capture);
  else if (fragments) capture = retainedInitialFragments(capture);
  else if (!CAPTURE_SCHEMAS.has(capture.schema)) fail('Unsupported Playing capture schema');
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
  const tree = htmlTree(capture.gameAreaHtml, 'game area', fragments);
  const boards = htmlTree(capture.playerBoardsHtml, 'player boards', fragments);
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
    natural(byId(tree, 'current_round'), 'round', fragments) !== 1 ||
    !/^経過日\s+1\s*\/\s*27$/.test(capture.roundsText ?? '')
  )
    fail('Capture is not day one');
  if (fragments) unhiddenFragmentNode(tree, byId(tree, 'current_round'));
  const playerNodes = boards.nodes.filter((node) =>
    (fragments ? /^player_board_\d+$/ : /^overall_player_board_\d+$/).test(node.attrs.id ?? ''),
  );
  const n = playerNodes.length;
  if (![3, 4].includes(n) || game.context.playerCount !== n)
    fail('Only complete three/four-player captures are supported');
  const sourceIds = playerNodes.map((node) =>
    node.attrs.id.slice((fragments ? 'player_board_' : 'overall_player_board_').length),
  );
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
    fragments,
  );
  retained(
    tree,
    capture.selectedPlayerTiles,
    (node) => /^player_wealthtiles_\d+$/.test(node.attrs.id ?? ''),
    'selected wealth panels',
    fragments,
  );
  retained(
    tree,
    capture.templeMarkers,
    (node) => /^marker_\d+_temple_/.test(node.attrs.id ?? ''),
    'temple markers',
    fragments,
  );
  retained(
    tree,
    capture.technologyMarkers,
    (node) => /^marker_\d+_technology_/.test(node.attrs.id ?? ''),
    'technology markers',
    fragments,
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
    let identity = fragments ? capture.identities.get(id) : null;
    if (!fragments) {
      const nameNode = byId(boards, `player_name_${id}`);
      const link = one(
        nameNode.children.filter(object),
        (node) => node.tag === 'a',
        'player name link',
      );
      identity = {
        name: text(nodeText(link).trim(), 128, 'player name'),
        color: styles(link).color,
      };
    }
    const { name, color } = identity;
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
        natural(byId(boards, `${resource}_number_player_${id}`), 'resource', fragments),
      ]),
    );
    const workers = natural(byId(boards, `worker_number_player_${id}`), 'workers', fragments);
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
    if (fragments) {
      unhiddenFragmentNode(boards, byId(boards, `player_score_${id}`));
      tiles.forEach((node) => unhiddenFragmentNode(tree, node));
    }
    if (
      !same(resources, effects.resources) ||
      !same(temples, effects.temples) ||
      !same(technologies, effects.technologies) ||
      workers !== effects.workers ||
      natural(byId(boards, `player_score_${id}`), 'score', fragments) !== 0
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
  if (fragments) unhiddenFragmentNode(boards, firstToken);
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
    fragments,
  );
  if (
    capture.gears.length !== 6 ||
    capture.gears.some((item) => styles(byId(tree, item.id)).transform !== 'rotate(0deg)')
  )
    fail('Initial gears are missing/rotated');
  const gears = dummyProjection(capture, tree, atlas, n, fragments);
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
    if (fragments) tiles.forEach((node) => unhiddenFragmentNode(tree, node));
    jungle[box] = { corn: n, wood: box === 2 ? 0 : n };
  }
  const skulls = retained(
    tree,
    capture.skullSpaces,
    (node) => /^skullplace_\d+$/.test(node.attrs.id ?? ''),
    'skull spaces',
    fragments,
  );
  if (
    skulls.length !== 9 ||
    Array.from({ length: 9 }, (_v, i) => `skullplace_${i + 1}`).some((id) => !tree.ids.has(id)) ||
    skulls.some((node) => node.children.some((child) => typeof child !== 'string' || child.trim()))
  )
    fail('Skull spaces incomplete/occupied');
  if (capture.skullRemainingHtml !== outer(tree, byId(tree, 'skull_left_wrap')))
    fail('Detached skull supply HTML');
  const skullSupply = natural(byId(tree, 'skull_left'), 'skull supply', fragments);
  if (fragments) unhiddenFragmentNode(tree, byId(tree, 'skull_left'));
  if (skullSupply + players.reduce((sum, player) => sum + player.resources.skull, 0) !== 13)
    fail('Initial skull conservation mismatch');
  const construction = (kind, html, containerId, count) => {
    if (html !== outer(tree, byId(tree, containerId))) fail(`Detached ${kind} market HTML`);
    const container = htmlTree(html, `${kind} market`);
    const nodes = container.nodes.filter((node) => token(node, 'construction'));
    const pattern = kind === 'building' ? /^building_age_1_(\d+)$/ : /^monument_(\d+)$/;
    if (nodes.length !== count || nodes.some((node) => !pattern.test(node.attrs.id ?? '')))
      fail(`Incomplete ${kind} market`);
    if (fragments) nodes.forEach((node) => unhiddenFragmentNode(tree, byId(tree, node.attrs.id)));
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
  let initialExpected = null;
  if (fragments) {
    for (const id of ['workerplace_startingPlayer_0', 'centralgearcorns']) {
      const node = byId(tree, id);
      unhiddenFragmentNode(tree, node);
      if (node.children.some((child) => typeof child !== 'string' || child.trim()))
        fail('Retained initial starting area is nonempty');
    }
    // Select only fields checked against retained nodes. Covered corn, seat
    // order, age and food history remain mechanical declarations, not witnesses.
    initialExpected = {
      round: 1,
      currentPlayer: firstPlayer,
      firstPlayer,
      players: players.map(({ id, resources, score, workers, temples, technologies, wealth }) => ({
        id,
        resources,
        score,
        workers,
        temples,
        technologies,
        wealth,
        buildings: [],
        monuments: [],
      })),
      gears,
      jungle: Object.fromEntries(
        [2, 3, 4, 5].map((box) => [box, { [box === 2 ? 'corn' : 'wood']: n }]),
      ),
      skullSupply,
      skullSpaces: Array(10).fill(null),
      firstPlayerClaimed: null,
      accumulatedCorn: 0,
      monuments,
    };
    const market = htmlTree(capture.buildingsHtml, 'retained building identities');
    if (
      market.nodes
        .filter((node) => token(node, 'construction'))
        .every((node) => faceId(atlas, 'building', node).length === 1)
    )
      initialExpected.buildings = buildings;
  }
  const witness = {
    schema: WITNESS_SCHEMA,
    tableId: game.tableId,
    rawSha256: sha(rawLogBytes),
    catalogHash: catalogFingerprint(catalogBytes),
    initialActionId: actionId,
    initial: structuredClone(initial),
    ...(initialExpected === null ? {} : { initialExpected: structuredClone(initialExpected) }),
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
    initialCheckpoint:
      initialExpected === null
        ? null
        : {
            source: { reference: game.source.reference, actionIds: [actionId] },
            expected: structuredClone(initialExpected),
          },
    steps: [],
    terminalCheckpoint: null,
  };
  return {
    schema: PROJECTION_SCHEMA,
    record,
    witness,
    provenance: {
      captureSha256: sha(captureBytes),
      ...(legacy
        ? {
            captureAdapter: {
              schema: LEGACY_CAPTURE_ADAPTER,
              sourceSchema: LEGACY_CAPTURE_SCHEMA,
              visibilityVerified: false,
              geometryVerified: false,
            },
          }
        : {}),
      ...(fragments
        ? {
            captureAdapter: {
              schema: FRAGMENT_CAPTURE_ADAPTER,
              sourceSchema: FRAGMENT_CAPTURE_SCHEMA,
              visibilityVerified: false,
              geometryVerified: false,
            },
            initialObservation: {
              kind: 'partial-retained-dom',
              playerCounters: 'playerBoards[].boardHtml: exact owner spans and score',
              playerCards:
                'playerBoards[].rowHtml: exact selected and empty acquired-card fragments',
              playerIdentity:
                'playerBoards[].rowHtml: own direct h3 text/color; IDs bind the sibling board',
              markersAndFaces:
                'boardHtml/rowHtml: exact nodes interpreted by the separately pinned atlas',
              startingArea: 'boardHtml: empty workerplace_startingPlayer_0 and centralgearcorns',
              omitted: [
                'age',
                'foodDays',
                'turnOrder',
                'turnIndex',
                'covered corn under wood',
                ...(initialExpected.buildings === undefined
                  ? ['ambiguous physical building identities']
                  : []),
              ],
              completeSourceBoard: false,
            },
          }
        : {}),
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

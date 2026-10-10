const fail = (message) => {
  throw new Error(message);
};
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const FLAGS = ['tribes', 'prophecies', 'additionalBuildings', 'quickActions'];
const LABELS = [
  [/^(?:ゲームモード|Game mode)$/i, 'mode'],
  [/^(?:ウシュマルコーンの制限|Uxmal corn limit)$/i, 'marketLimit'],
  [/^(?:部族|Tribes?)$/i, 'tribes'],
  [/^(?:予言|預言|Prophec(?:y|ies))$/i, 'prophecies'],
  [/^(?:追加建物|追加建築|Additional buildings?)$/i, 'additionalBuildings'],
  [/^(?:クイックアクション|Quick actions?)$/i, 'quickActions'],
];
const LABEL_SUBSTRING = new RegExp(
  [
    ...LABELS.map(([pattern]) => pattern.source.slice(1, -1)),
    '拡張',
    'expansion',
    'extension',
  ].join('|'),
  'i',
);

// Shared duplicate-key guard; validation precedes JSON.parse so an overwritten
// option cannot suppress explicit contradictory source evidence.
export function uniqueJson(bytes, max, label) {
  if (!Buffer.isBuffer(bytes) || bytes.length > max) fail(`${label}: byte limit`);
  const source = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  let at = 0,
    tokens = 0;
  const whitespace = () => {
    while (/\s/.test(source[at] ?? '') && at < source.length) at++;
  };
  function string() {
    const start = at++;
    while (at < source.length) {
      const char = source[at++];
      if (char === '"') return JSON.parse(source.slice(start, at));
      if (char === '\\') at++;
    }
    fail(`${label}: unterminated JSON string`);
  }
  function value(depth) {
    if (depth > 64 || ++tokens > 250000) fail(`${label}: JSON complexity limit`);
    whitespace();
    if (source[at] === '{' || source[at] === '[') {
      const isObject = source[at++] === '{';
      const end = isObject ? '}' : ']';
      const keys = new Set();
      whitespace();
      if (source[at] === end) {
        at++;
        return;
      }
      for (;;) {
        if (isObject) {
          whitespace();
          if (source[at] !== '"') fail(`${label}: invalid JSON key`);
          const key = string();
          if (keys.has(key)) fail(`${label}: duplicate JSON key`);
          keys.add(key);
          whitespace();
          if (source[at++] !== ':') fail(`${label}: invalid JSON colon`);
        }
        value(depth + 1);
        whitespace();
        const next = source[at++];
        if (next === end) break;
        if (next !== ',') fail(`${label}: invalid JSON separator`);
      }
    } else if (source[at] === '"') string();
    else {
      const match = /^(?:true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(
        source.slice(at),
      );
      if (!match) fail(`${label}: invalid JSON value`);
      at += match[0].length;
    }
  }
  value(0);
  whitespace();
  if (at !== source.length) fail(`${label}: trailing JSON`);
  return JSON.parse(source);
}

/** Only concrete observed scalar context fields and explicit option label/value
 * text are decoded. Missing expansion switches never become disabled switches.
 * This is compatibility evidence, not source authentication or BC admission. */
export function observedBasicUnlimitedOptions(raw, table = null, normalizedContext = null) {
  if (!object(raw)) fail('Invalid option source');
  const result = {
    mode: null,
    marketLimit: null,
    tribes: null,
    prophecies: null,
    additionalBuildings: null,
    quickActions: null,
    evidence: [],
    complete: false,
  };
  function scalar(value, label) {
    if (typeof value !== 'string' || value.length > 128) fail(`Invalid observed ${label}`);
    return value.trim();
  }
  function set(key, rawValue, source) {
    if (rawValue === null || rawValue === undefined || rawValue === '') return;
    const value = scalar(rawValue, key);
    let known;
    if (key === 'mode') {
      if (/^(?:アリーナモード|Arena mode|Arena)$/i.test(value)) known = 'arena';
      else if (/^(?:ノーマルモード|Normal mode|Normal)$/i.test(value)) known = 'normal';
      else fail('Unsupported observed game mode');
    } else if (key === 'marketLimit') {
      if (!/^(?:制限なし|Unlimited)$/i.test(value))
        fail('Observed market contradicts declared unlimited profile');
      known = 'unlimited';
    } else {
      if (!FLAGS.includes(key) || !/^(?:無効|なし|false|disabled|off|no|0)$/i.test(value))
        fail(`Observed ${key} contradicts or does not establish the declared basic profile`);
      known = false;
    }
    if (result[key] !== null && result[key] !== known) fail(`Conflicting observed ${key}`);
    result[key] = known;
    result.evidence.push({ source, key, rawValue, value: known });
  }
  function optionText(value, source, allowConfigurationPrefix = false) {
    if (value === undefined || value === null || value === '') return;
    if (typeof value !== 'string' || value.length > 65536) fail('Invalid observed option text');
    const sections = value.replaceAll('\r\n', '\n').split('ゲーム構成\n');
    if (sections.length > 2 || (!allowConfigurationPrefix && sections.length !== 1))
      fail('Ambiguous observed game configuration');
    if (sections.length === 2 && LABEL_SUBSTRING.test(sections[0]))
      fail('Observed configuration prefix contains option evidence');
    const lines = sections
      .at(-1)
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    for (let i = 0; i < lines.length;) {
      const label = lines[i++];
      if (label === 'ゲームの速度') {
        if (
          !/^ターンベース • 1日あたり[1-9]\d*手番$/.test(lines[i++] ?? '') ||
          !/^毎手番ごとに\+\d+h[0-5]\d\(最大[1-9]\d* 日\)$/.test(lines[i++] ?? '')
        )
          fail('Unsupported observed speed text');
        continue;
      }
      const key = LABELS.find(([pattern]) => pattern.test(label))?.[1];
      if (!key || i === lines.length) fail('Unsupported/partial observed option text');
      const observed = lines[i++];
      set(key, observed, source);
      if (key === 'mode' && lines[i]?.startsWith(`${observed}: `)) {
        if (LABEL_SUBSTRING.test(lines[i])) fail('Observed mode detail contains option evidence');
        i++;
      }
    }
  }
  // Embedded text is inspected even for the newer UI adapter, whose normalizer
  // otherwise substitutes separate table evidence and would discard it.
  optionText(raw.table_details_text, 'raw.table_details_text', true);
  if (normalizedContext !== null) {
    if (!object(normalizedContext)) fail('Invalid normalized context');
    set('mode', normalizedContext.mode, 'normalized.context.mode');
    set('marketLimit', normalizedContext.marketOption, 'normalized.context.marketOption');
    if (
      normalizedContext.extensionOptions !== null &&
      normalizedContext.extensionOptions !== undefined
    )
      fail('Unsupported observed extension options');
  }
  if (table !== null) {
    if (
      !object(table) ||
      table.schema !== 'tzolkin-bga-public-table-ui-v1' ||
      typeof table.optionsText !== 'string'
    )
      fail('Unsupported observed table option evidence');
    optionText(table.optionsText, 'table.optionsText');
  }
  result.complete =
    result.marketLimit === 'unlimited' && FLAGS.every((key) => result[key] === false);
  return result;
}

import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, rename, rm, rmdir, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const SCHEMA = 'tzolkin-bga-partial-v1';
const gears = {
  Palenque: 'palenque',
  Yaxchilan: 'yaxchilan',
  Tikal: 'tikal',
  Uxmal: 'uxmal',
  'Chichen Itza': 'chichenItza',
};
const technologies = {
  農業: 'agriculture',
  収集: 'extraction',
  建築: 'architecture',
  神学: 'theology',
};
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const increment = (counts, key) => {
  counts[key] = (counts[key] ?? 0) + 1;
};

// Only explicit text patterns are interpreted. Raw text and icon classes remain the evidence.
export function parseMessage(message, players, { englishTerminal = false } = {}) {
  const raw = message.raw_text;
  const actor =
    [...players]
      .sort((a, b) => b.length - a.length)
      .find((p) => raw.startsWith(`${p}は`) || raw.startsWith(`${p}が`)) ?? null;
  const text = actor ? raw.slice(actor.length) : raw;
  const iconResources = (message.icons ?? []).map(
    (icon) =>
      icon.classes
        .split(/\s+/)
        .find((c) => c.startsWith('resource_'))
        ?.slice(9) ?? null,
  );
  const result = { actor, kind: 'unparsed', detail: {}, iconResources };
  const match = (pattern) => text.match(pattern);
  let m;
  if (actor && text === 'は行動をキャンセルした') result.kind = 'cancellation';
  else if (raw === '歯車が進んだ') result.kind = 'gearAdvanced';
  else if (
    raw === 'ゲーム終了' ||
    (englishTerminal && players.some((player) => raw === `End of game : ${player} wins!`))
  )
    result.kind = 'gameEnd';
  else if (actor && text === 'は新しいワーカーを獲得した') result.kind = 'workerGain';
  else if (
    actor &&
    (m = match(
      /^は(Palenque|Yaxchilan|Tikal|Uxmal|Chichen Itza)の歯車に(\d+)\s*を支払ってワーカーを置いた$/,
    ))
  ) {
    result.kind = 'place';
    result.detail = { gear: gears[m[1]], cornCost: Number(m[2]) };
  } else if (
    actor &&
    (m = match(
      /^は(Palenque|Yaxchilan|Tikal|Uxmal|Chichen Itza) (\d+)のワーカーを取り除き、[\s\S]*?n°(\d+)\s*の?アクションを行った$/,
    ))
  ) {
    result.kind = 'gearAction';
    result.detail = {
      gear: gears[m[1]],
      workerPosition: Number(m[2]),
      actionPosition: Number(m[3]),
      source: 'removal',
    };
  } else if (
    actor &&
    (m = match(
      /^はコーン(\d+)個を支払って(Palenque|Yaxchilan|Tikal|Uxmal|Chichen Itza) n°(\d+)のアクションを行った$/,
    ))
  ) {
    result.kind = 'gearAction';
    result.detail = {
      gear: gears[m[2]],
      actionPosition: Number(m[3]),
      cornCost: Number(m[1]),
      source: 'selectedAction',
    };
  } else if (actor && (m = match(/^は(農業|収集|建築|神学)を(\d+)レベル進めた$/))) {
    result.kind = 'technology';
    result.detail = { technology: technologies[m[1]], steps: Number(m[2]) };
  } else if (actor && (m = match(/^は(\d+)\s*を(\d+)\s*に交換した$/))) {
    result.kind = 'trade';
    if (
      iconResources.length === 2 &&
      iconResources.every((r) => ['corn', 'wood', 'stone', 'gold', 'skull'].includes(r))
    ) {
      result.detail = {
        from: { resource: iconResources[0], amount: Number(m[1]) },
        to: { resource: iconResources[1], amount: Number(m[2]) },
      };
    }
  } else if (actor && /^は\d*\s*を(獲得した|支払った)$/.test(text) && iconResources.length > 0) {
    result.kind = text.endsWith('獲得した') ? 'resourceGain' : 'resourcePayment';
    result.detail = {
      explicitAmount: text.match(/^は(\d+)/)?.[1] ? Number(text.match(/^は(\d+)/)[1]) : null,
      iconResources,
    };
  } else if (actor && text === 'は建物を建てた') result.kind = 'build';
  else if (actor && /記念碑.*(建てた|獲得した)/.test(text)) result.kind = 'monument';
  else if (actor && text.includes('の信仰を') && text.includes('コーン3個を受け取った'))
    result.kind = 'beg';
  else if (actor && text === 'は歯車を高速化し2段階回した!') result.kind = 'doubleAdvance';
  else if (actor && /がこのラウンドのスタートプレイヤーです$/.test(text))
    result.kind = 'firstPlayer';
  else if (raw.startsWith('食料供給日: ')) {
    m = raw.match(/^食料供給日: (.+?)はワーカー(\d+)個で/);
    if (m && players.includes(m[1])) {
      result.actor = m[1];
      result.kind = 'feeding';
      result.detail = { workers: Number(m[2]) };
    }
  }
  return result;
}

function validate(raw) {
  if (
    raw.schema_version !== '1.0' ||
    raw.source_type !== 'public_ui_visible_log' ||
    raw.export_scope !== 'all_observed_log_entries'
  )
    throw new Error('Unsupported UI export schema/source/scope');
  if (
    !/^\d+$/.test(raw.table_id) ||
    !Array.isArray(raw.entries) ||
    !raw.entries.length ||
    !Array.isArray(raw.dom_entries) ||
    raw.entry_count !== raw.entries.length ||
    raw.entries.length !== raw.dom_entries.length
  )
    throw new Error('Invalid table ID or entry counts');
  const meta = raw.metadata;
  const players = meta?.players?.map((p) => p.text);
  if (
    !players ||
    ![3, 4].includes(players.length) ||
    meta.player_count !== players.length ||
    new Set(players).size !== players.length ||
    players.some((p) => typeof p !== 'string' || !p.length)
  )
    throw new Error('Invalid player metadata');
  if (meta.table_id !== raw.table_id) throw new Error('Metadata table ID mismatch');
  const url = new URL(raw.source_url);
  if (
    !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
      url.hostname,
    ) ||
    url.protocol !== 'https:' ||
    url.pathname !== '/gamereview' ||
    url.searchParams.get('table') !== raw.table_id
  )
    throw new Error('Source URL/table mismatch');
  let previous = 0;
  for (let i = 0; i < raw.entries.length; i++) {
    const entry = raw.entries[i];
    const dom = raw.dom_entries[i];
    if (
      !Number.isSafeInteger(entry.action_id) ||
      entry.action_id <= previous ||
      entry.action_id !== dom.action_id ||
      typeof entry.raw_text !== 'string' ||
      typeof entry.timestamp_display !== 'string' ||
      entry.timestamp_display !== dom.timestamp_display ||
      !Array.isArray(dom.messages)
    )
      throw new Error('Action ID/timestamp/text/DOM mismatch');
    for (const m of dom.messages)
      if (
        typeof m.raw_text !== 'string' ||
        typeof m.html !== 'string' ||
        !Array.isArray(m.icons) ||
        m.icons.some((icon) => typeof icon.classes !== 'string')
      )
        throw new Error('Invalid message/icon evidence');
    previous = entry.action_id;
  }
  if (typeof meta.history_text !== 'string' || typeof raw.table_details_text !== 'string')
    throw new Error('Missing public history/table evidence');
  return players;
}

// The current collector preserves complete entry text (including its UI header)
// and literal icon `class` attributes. This view is transient: original raw
// entries and their file checksum remain the exported source evidence.
function publicUiDomView(raw, tableEvidenceBytes) {
  if (
    raw.schema_version !== 1 ||
    raw.source_type !== 'public_ui_dom' ||
    raw.export_scope !== 'all_rendered_gamereview_log_entries'
  )
    throw new Error('Unsupported UI export schema/source/scope');
  if (
    !Array.isArray(raw.entries) ||
    !Array.isArray(raw.dom_entries) ||
    raw.entries.length !== raw.dom_entries.length ||
    raw.entry_count !== raw.entries.length ||
    raw.dom_entry_count !== raw.dom_entries.length ||
    !Array.isArray(raw.metadata?.players)
  )
    throw new Error('Invalid current UI entry counts/metadata');
  const tableUrl = new URL(raw.metadata.tableUrl);
  if (
    tableUrl.protocol !== 'https:' ||
    !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
      tableUrl.hostname,
    ) ||
    tableUrl.pathname !== '/table' ||
    tableUrl.searchParams.get('table') !== raw.table_id
  )
    throw new Error('Current metadata table URL mismatch');
  const entries = raw.entries.map((entry, index) => {
    const dom = raw.dom_entries[index];
    if (!Array.isArray(dom.messages) || typeof entry.raw_text !== 'string')
      throw new Error('Invalid current UI message evidence');
    const prefix = `行動 ${entry.action_id} :\n${entry.timestamp_display}\n`;
    const text = entry.raw_text.replaceAll('\r\n', '\n');
    if (!text.startsWith(prefix)) throw new Error('Current UI action header/timestamp mismatch');
    const body = text.slice(prefix.length);
    if (
      body !==
      dom.messages
        .map((message) => message.raw_text)
        .join('\n')
        .replaceAll('\r\n', '\n')
    )
      throw new Error('Current UI raw/DOM message mismatch');
    return { ...entry, raw_text: body };
  });
  const dom_entries = raw.dom_entries.map((entry) => ({
    ...entry,
    messages: entry.messages.map((message) => ({
      ...message,
      icons: message.icons.map((icon) => {
        if (typeof icon.class !== 'string') throw new Error('Missing current UI icon class');
        return { ...icon, classes: icon.class };
      }),
    })),
  }));
  const history = [];
  if (raw.metadata.date != null) {
    if (typeof raw.metadata.date !== 'string') throw new Error('Invalid current UI date');
    history.push(raw.metadata.date);
  }
  for (const player of raw.metadata.players) {
    if (player.rank == null && player.score == null) continue;
    if (!/^\d+位$/.test(player.rank) || !/^-?\d+\s*$/.test(player.score))
      throw new Error('Invalid current UI displayed result');
    history.push(player.rank, player.name, player.score.trim() + ' ');
  }
  let table_details_text = '';
  let tableEvidence;
  let terminationText = raw.metadata.historyRowText ?? null;
  if (terminationText != null && typeof terminationText !== 'string')
    throw new Error('Invalid current UI history row text');
  terminationText = terminationText?.trim() || null;
  if (tableEvidenceBytes != null) {
    if (!Buffer.isBuffer(tableEvidenceBytes)) throw new Error('Expected raw table evidence bytes');
    const table = JSON.parse(tableEvidenceBytes);
    if (
      table.schema !== 'tzolkin-bga-public-table-ui-v1' ||
      table.url !== raw.metadata.tableUrl ||
      typeof table.optionsText !== 'string'
    )
      throw new Error('Table evidence schema/identity/options mismatch');
    table_details_text = `ゲーム構成\n${table.optionsText}`;
    tableEvidence = { reference: table.url, sha256: hash(tableEvidenceBytes) };
    if (table.resultText != null) {
      if (typeof table.resultText !== 'string') throw new Error('Invalid visible result text');
      terminationText =
        [terminationText, table.resultText.trim()].filter(Boolean).join('\n') || null;
    }
  }
  return {
    raw: {
      ...raw,
      schema_version: '1.0',
      source_type: 'public_ui_visible_log',
      export_scope: 'all_observed_log_entries',
      entries,
      dom_entries,
      metadata: {
        ...raw.metadata,
        table_id: raw.table_id,
        player_count: raw.metadata.playerCount,
        players: raw.metadata.players.map((player) => ({ text: player.name })),
        history_text: history.join('\n'),
      },
      table_details_text,
    },
    tableEvidence,
    terminationText,
  };
}

export function normalizeGame(original, sourceSha256, tableEvidenceBytes = null) {
  const adapted =
    original.schema_version === 1 ? publicUiDomView(original, tableEvidenceBytes) : null;
  const raw = adapted?.raw ?? original;
  const players = validate(raw);
  const events = [];
  let rotationMarkersBefore = 0;
  for (let i = 0; i < raw.entries.length; i++) {
    const entry = raw.entries[i];
    for (const [messageIndex, message] of raw.dom_entries[i].messages.entries()) {
      const parsed = parseMessage(message, players, { englishTerminal: !!adapted });
      events.push({
        actionId: entry.action_id,
        messageIndex,
        timestampDisplay: entry.timestamp_display,
        rotationMarkersBefore,
        ...parsed,
        rawText: message.raw_text,
      });
      if (parsed.kind === 'gearAdvanced') rotationMarkersBefore++;
    }
  }
  const cancellationCount = events.filter((e) => e.kind === 'cancellation').length;
  const terminationText = adapted ? adapted.terminationText : raw.metadata.history_text;
  const abandoned =
    terminationText?.includes('放棄されたテーブル') ||
    (adapted &&
      /^\s*(?:Abandoned table|Table (?:was )?abandoned)\s*$/im.test(terminationText ?? ''));
  const forfeit =
    terminationText?.includes('投了') ||
    (adapted && /^\s*(?:Conceded|Game conceded)\s*$/im.test(terminationText ?? ''));
  const normalEndEvidence =
    !adapted || /^\s*(?:ゲーム終了|Game ended|Game finished)\s*$/im.test(terminationText ?? '');
  const status = abandoned
    ? 'abandoned'
    : forfeit
      ? 'forfeit'
      : !normalEndEvidence
        ? 'unknown'
        : events.some((event) => event.kind === 'gameEnd') ||
            raw.entries.some((entry) => entry.raw_text.split('\n').includes('ゲーム終了'))
          ? 'normalEnd'
          : 'unknown';
  const config = raw.table_details_text.split('ゲーム構成\n')[1] ?? '';
  const results = [...raw.metadata.history_text.matchAll(/(\d+)位\n([^\n]+)\n(-?\d+)\s/g)].map(
    (m) => ({ player: m[2].trim(), rank: Number(m[1]), scoreDisplay: Number(m[3]) }),
  );
  const splitByte = parseInt(hash(raw.table_id).slice(0, 8), 16) % 10;
  return {
    schema: SCHEMA,
    tableId: raw.table_id,
    source: {
      provider: 'BGA',
      reference: raw.source_url,
      sha256: sourceSha256,
      capturedOn: raw.captured_on,
      ...(adapted
        ? {
            adapter: 'bga-public-ui-dom-v1',
            ...(adapted.tableEvidence ? { tableEvidence: adapted.tableEvidence } : {}),
          }
        : {}),
    },
    split: splitByte < 8 ? 'train' : splitByte === 8 ? 'validation' : 'test',
    context: {
      players,
      playerCount: players.length,
      endDateDisplay:
        raw.metadata.history_text.match(/\d{4}年\d{2}月\d{2}日 \d{2}:\d{2}/)?.[0] ?? null,
      mode: config.match(/ゲームモード\n([^\n]+)/)?.[1] ?? null,
      marketOption: config.match(/ウシュマルコーンの制限\n([^\n]+)/)?.[1] ?? null,
      extensionOptions: null,
      initialResources: null,
      initialSeatOrder: null,
      seed: null,
    },
    quality: {
      status,
      ...(adapted ? { terminalLogObserved: events.some((event) => event.kind === 'gameEnd') } : {}),
      cancellationCount,
      rollbackResolved: cancellationCount === 0,
      verifiedComplete: false,
      policyTrainingReady: false,
      valueTrainingReady: false,
      rawActionCount: raw.entry_count,
      parsedMessages: events.filter((e) => e.kind !== 'unparsed').length,
      totalMessages: events.length,
      missing: [
        'initialResources',
        'initialSeatOrder',
        'boardSetup',
        'legalActions',
        'decisionObservations',
        'validatedTransitions',
        'extensionOptions',
      ],
      warnings: [
        'Observed occurrence counts are not net state changes.',
        'Rotation markers are not day or turn numbers.',
        'Train/validation/test split is game-level only; seed-family independence is unverified.',
      ],
    },
    resultsDisplay: results,
    // Preserve every action, including terminal text outside message divs.
    rawEntries: original.entries,
    events,
  };
}

export function profileGame(game) {
  return game.context.players.map((player) => {
    const events = game.events.filter((e) => e.actor === player);
    const observedCounts = {};
    for (const e of events) {
      increment(observedCounts, e.kind);
      if (e.kind === 'gearAction')
        increment(observedCounts, `action:${e.detail.gear}:${e.detail.actionPosition}`);
      if (e.kind === 'technology') increment(observedCounts, `technology:${e.detail.technology}`);
      if (
        e.kind === 'trade' &&
        e.detail.from?.resource === 'corn' &&
        e.detail.to?.resource !== 'corn'
      )
        increment(observedCounts, `trade:${e.detail.to.resource}:buy`);
      if (
        e.kind === 'trade' &&
        e.detail.to?.resource === 'corn' &&
        e.detail.from?.resource !== 'corn'
      )
        increment(observedCounts, `trade:${e.detail.from.resource}:sell`);
    }
    return {
      tableId: game.tableId,
      player,
      split: game.split,
      context: game.context,
      status: game.quality.status,
      cancellationRisk: game.quality.cancellationCount > 0,
      observedCounts,
      firstWorkerGain: events.find((e) => e.kind === 'workerGain') ?? null,
      feedingWorkerCounts: events
        .filter((e) => e.kind === 'feeding')
        .map((e) => ({ actionId: e.actionId, workers: e.detail.workers })),
      finalWorkers: null,
      resultDisplay: game.resultsDisplay.find((r) => r.player === player) ?? null,
    };
  });
}

export async function importCorpus(input, output) {
  const names = (await readdir(join(input, 'games'))).filter((n) => /^\d+\.json$/.test(n)).sort();
  if (!names.length) throw new Error('No exported games found');
  const games = [];
  const sources = [];
  const seen = new Set();
  for (const name of names) {
    const bytes = await readFile(join(input, 'games', name));
    const raw = JSON.parse(bytes);
    if (`${raw.table_id}.json` !== name || seen.has(raw.table_id))
      throw new Error('Duplicate or misnamed table');
    seen.add(raw.table_id);
    const sha256 = hash(bytes);
    let tableEvidenceBytes = null;
    if (raw.schema_version === 1) {
      try {
        tableEvidenceBytes = await readFile(join(input, 'table-evidence', name));
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
      }
    }
    const game = normalizeGame(raw, sha256, tableEvidenceBytes);
    games.push(game);
    sources.push({
      tableId: game.tableId,
      file: `games/${name}`,
      bytes: bytes.length,
      sha256,
      ...(tableEvidenceBytes
        ? {
            tableEvidence: {
              file: `table-evidence/${name}`,
              bytes: tableEvidenceBytes.length,
              sha256: hash(tableEvidenceBytes),
            },
          }
        : {}),
    });
  }
  const profiles = games.flatMap(profileGame);
  const summary = {
    games: games.length,
    actionEntries: games.reduce((n, g) => n + g.quality.rawActionCount, 0),
    messages: games.reduce((n, g) => n + g.events.length, 0),
    parsedMessages: games.reduce((n, g) => n + g.quality.parsedMessages, 0),
    playerGames: profiles.length,
    statuses: {},
    cancellations: games.reduce((n, g) => n + g.quality.cancellationCount, 0),
    unknownKinds: {},
  };
  for (const g of games) increment(summary.statuses, g.quality.status);
  for (const g of games)
    for (const e of g.events)
      if (e.kind === 'unparsed')
        increment(summary.unknownKinds, e.rawText.replaceAll(/\d+/g, '#').slice(0, 100));
  const manifest = {
    schema: SCHEMA,
    importer: 'bga-ui-ja-v1',
    createdAt: new Date().toISOString(),
    sourcePolicy: 'Public UI exports; no fetched credentials or network calls.',
    verifiedComplete: false,
    policyTrainingRows: 0,
    valueTrainingRows: 0,
    sources,
    summary,
  };
  // Refuse an existing target. Publish only after every source and write succeeds.
  await mkdir(dirname(output), { recursive: true });
  const temporary = `${output}.tmp-${process.pid}`;
  if (
    resolve(temporary) === resolve(output) ||
    dirname(resolve(temporary)) !== dirname(resolve(output))
  )
    throw new Error('Invalid staging directory');
  await mkdir(temporary);
  let published = false;
  try {
    await mkdir(output); // Reserve destination; never merge with an existing dataset.
    await mkdir(join(temporary, 'games'));
    for (const g of games)
      await writeFile(join(temporary, 'games', `${g.tableId}.json`), JSON.stringify(g));
    await writeFile(
      join(temporary, 'events.jsonl'),
      games
        .flatMap((g) =>
          g.events.map((e) =>
            JSON.stringify({ schema: SCHEMA, tableId: g.tableId, split: g.split, ...e }),
          ),
        )
        .join('\n') + '\n',
    );
    await writeFile(join(temporary, 'profiles.json'), JSON.stringify(profiles, null, 2));
    await writeFile(join(temporary, 'manifest.json'), JSON.stringify(manifest, null, 2));
    await writeFile(
      join(temporary, 'review-queue.json'),
      JSON.stringify(
        games.map((g) => ({
          tableId: g.tableId,
          url: g.source.reference,
          split: g.split,
          context: g.context,
          status: g.quality.status,
          missing: g.quality.missing,
          cancellationCount: g.quality.cancellationCount,
        })),
        null,
        2,
      ),
    );
    await rmdir(output); // Remove only our empty reservation, never its contents.
    await rename(temporary, output);
    published = true;
  } finally {
    if (!published) {
      await rm(temporary, { recursive: true, force: true });
    }
  }
  return manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [input, output] = process.argv.slice(2);
  if (!input || !output || process.argv.length !== 4) {
    console.error('Usage: node scripts/bga-import.mjs RAW_CORPUS_DIR NEW_OUTPUT_DIR');
    process.exitCode = 1;
  } else
    importCorpus(resolve(input), resolve(output))
      .then((manifest) =>
        console.log(
          JSON.stringify({
            ...manifest.summary,
            unknownKinds: Object.keys(manifest.summary.unknownKinds).length,
          }),
        ),
      )
      .catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
      });
}

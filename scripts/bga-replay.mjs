import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rename, rm, rmdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizeGame } from './bga-import.mjs';

export const RECONSTRUCTION_SCHEMA = 'tzolkin-bga-reconstruction-v1';
export const MAX_INPUT_BYTES = 16 * 1024 * 1024;
export const MAX_CORE_OUTPUT_BYTES = 64 * 1024 * 1024;
export const MAX_TOTAL_EVIDENCE_BYTES = 32 * 1024 * 1024;
const MAX_STEPS = 4000;
const MAX_EVENTS = 100000;
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fail = (message) => {
  throw new Error(message);
};
const object = (value) => !!value && typeof value === 'object' && !Array.isArray(value);
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
const stateHash = (state) => sha256(canonical(state));

export function assertBasicUnlimitedOptions(config) {
  if (typeof config !== 'string') fail('Missing observed BGA table settings');
  if (
    /部族|予言|預言|拡張|追加(?:建物|建築)|クイックアクション|Tribes?|Prophec(?:y|ies)|Expansion|Additional buildings?|Quick actions?/i.test(
      config,
    )
  )
    fail('BGA table has unsupported/unknown settings');
  const lines = config
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
  const settings = new Map();
  for (let index = 0; index < lines.length;) {
    const key = lines[index++];
    const value = lines[index++];
    if (settings.has(key)) fail('BGA table has unsupported/unknown settings');
    if (key === 'ゲームモード') {
      if (
        settings.size ||
        !['ノーマルモード', 'アリーナモード', 'Normal mode', 'Arena mode'].includes(value)
      )
        fail('BGA table has unsupported/unknown settings');
      const prefix = `${value}: `;
      if (lines[index]?.startsWith(prefix) && lines[index].length > prefix.length) index++;
    } else if (key === 'ゲームの速度') {
      if (
        !settings.has('ゲームモード') ||
        settings.has('ウシュマルコーンの制限') ||
        !/^ターンベース • 1日あたり[1-9]\d*手番$/.test(value) ||
        !/^毎手番ごとに\+\d+h[0-5]\d\(最大[1-9]\d* 日\)$/.test(lines[index++])
      )
        fail('BGA table has unsupported/unknown settings');
    } else if (key !== 'ウシュマルコーンの制限' || !['制限なし', 'Unlimited'].includes(value))
      fail('BGA table has unsupported/unknown settings');
    settings.set(key, value);
  }
  if (!settings.has('ウシュマルコーンの制限'))
    fail('Only explicitly unlimited BGA markets are supported');
}

function keys(value, allowed, context) {
  if (!object(value) || Object.keys(value).some((key) => !allowed.includes(key)))
    fail(`${context}: unexpected object/field`);
}
function array(value, maximum, context) {
  if (!Array.isArray(value) || value.length > maximum) fail(`${context}: invalid/big array`);
  return value;
}
function id(value, context) {
  if (!Number.isSafeInteger(value) || value < 1) fail(`${context}: invalid action ID`);
  return value;
}
function identifier(value, context) {
  if (typeof value !== 'string' || !/^[a-zA-Z0-9_-]{1,80}$/.test(value))
    fail(`${context}: invalid identifier`);
  return value;
}
function checksum(value, context) {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value))
    fail(`${context}: invalid SHA-256`);
}
function tableUrl(reference, tableId) {
  let url;
  try {
    url = new URL(reference);
  } catch {
    fail('Evidence requires a BGA table URL');
  }
  if (
    url.protocol !== 'https:' ||
    !['boardgamearena.com', 'ja.boardgamearena.com', 'en.boardgamearena.com'].includes(
      url.hostname,
    ) ||
    url.username ||
    url.password ||
    url.port ||
    url.hash ||
    !(
      /^\/archive\/replay\/[a-zA-Z0-9_-]+\/$/.test(url.pathname) || url.pathname === '/gamereview'
    ) ||
    url.searchParams.getAll('table').length !== 1 ||
    url.searchParams.get('table') !== tableId
  )
    fail('Evidence URL/table mismatch');
}
function evidence(source, tableId, maximum, knownIds) {
  keys(source, ['reference', 'actionIds'], 'source');
  tableUrl(source.reference, tableId);
  const ids = array(source.actionIds, 64, 'source.actionIds');
  if (!ids.length || new Set(ids).size !== ids.length) fail('Evidence needs unique action IDs');
  for (const actionId of ids)
    if (id(actionId, 'source') > maximum || !knownIds.has(actionId))
      fail('Future/unknown evidence action ID');
}
function contains(actual, expected, context) {
  if (object(expected)) {
    if (!object(actual) || !Object.keys(expected).length)
      fail(`${context}: empty/invalid checkpoint`);
    for (const [key, value] of Object.entries(expected))
      contains(actual[key], value, `${context}.${key}`);
  } else if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length)
      fail(`${context}: array mismatch`);
    expected.forEach((value, index) => contains(actual[index], value, `${context}[${index}]`));
  } else if (actual !== expected) fail(`${context}: checkpoint mismatch`);
}

// The counter projection is deliberately small. Card identities, marker styles
// and worker coordinates remain manual evidence; they are never guessed here.
export function counterCheckpoint(board, names) {
  if (
    !object(board) ||
    !Number.isSafeInteger(board.round) ||
    !Number.isSafeInteger(board.skullSupply)
  )
    fail('Board witness has no public round/skull counters');
  if (!Array.isArray(board.players) || board.players.length !== names.length)
    fail('Board witness player count mismatch');
  if (
    !Array.isArray(board.workers) ||
    !Array.isArray(board.workerPlaces) ||
    board.workerPlaces.length !== 54 ||
    !Array.isArray(board.jungle) ||
    board.jungle.length !== 4 ||
    !Array.isArray(board.markers) ||
    board.markers.length !== names.length * 7 ||
    !Array.isArray(board.skullSpaces) ||
    board.skullSpaces.length !== 9 ||
    !Array.isArray(board.wealth) ||
    board.wealth.length !== names.length ||
    board.wealth.some((wealth) => !Array.isArray(wealth.tiles) || wealth.tiles.length !== 2)
  )
    fail('Board witness lacks complete worker/marker/jungle/skull/wealth captures');
  const players = names.map((name) => {
    const matching = board.players.filter((player) => player.name === name);
    if (matching.length !== 1) fail('Board witness player identity mismatch');
    const player = matching[0];
    if (typeof player.score !== 'number' || !Number.isFinite(player.score))
      fail('Invalid board score');
    const resources = Object.fromEntries(
      ['corn', 'gold', 'skull', 'stone', 'wood'].map((resource) => {
        const amount = player.counters?.[resource];
        if (!Number.isSafeInteger(amount) || amount < 0) fail('Invalid public resource counter');
        return [resource, amount];
      }),
    );
    return { name, resources, score: player.score };
  });
  return { round: board.round, skullSupply: board.skullSupply, players };
}

function boardHash(board) {
  const visible = { ...board };
  delete visible.sourceActionId;
  delete visible.title;
  // Keep all recorded board fields, including workers, market, markers and
  // geometry. A no-op cannot be established from unchanged corn alone.
  return stateHash(visible);
}

export function sourceEvents(raw, game, cutoff) {
  const events = game.events.filter((event) => event.actionId <= cutoff);
  const messagesByAction = new Map();
  for (const event of events) {
    if (!messagesByAction.has(event.actionId)) messagesByAction.set(event.actionId, []);
    messagesByAction.get(event.actionId).push(event);
  }
  for (const entry of raw.entries.filter((entry) => entry.action_id <= cutoff)) {
    const messages = messagesByAction.get(entry.action_id) ?? [];
    // Exported entry bodies exclude the separate timestamp/header controls.
    // Normalize line endings only; never discard arbitrary unmatched UI text.
    const normalized = (value) => value.replace(/\r\n/g, '\n');
    const body = normalized(entry.raw_text);
    const joined = messages.map((message) => normalized(message.rawText)).join('\n');
    const terminalSuffix = joined ? `${joined}\nゲーム終了` : 'ゲーム終了';
    if (body === terminalSuffix && !messages.some((event) => event.kind === 'gameEnd'))
      events.push({
        actionId: entry.action_id,
        messageIndex: -1,
        kind: 'gameEnd',
        rawText: 'ゲーム終了',
      });
    else if (body !== joined || !messages.length)
      fail(`Raw/DOM message text mismatch at action ${entry.action_id}: unsupported residual text`);
  }
  return events.sort((a, b) => a.actionId - b.actionId || a.messageIndex - b.messageIndex);
}

export function coreDispatcher(cliPath) {
  const cli = resolve(cliPath);
  return (request) => {
    const input = JSON.stringify(request);
    if (Buffer.byteLength(input) > MAX_INPUT_BYTES) fail('Core request exceeds 16 MiB');
    const output = execFileSync(cli, ['dispatch'], {
      input,
      encoding: 'utf8',
      maxBuffer: MAX_CORE_OUTPUT_BYTES,
      timeout: 60000,
      windowsHide: true,
    });
    return JSON.parse(output);
  };
}

/** Audit explicit, reviewed moves. No text-to-move generation happens here. */
export function auditReconstruction(rawBytes, companion, documents, dispatch) {
  if (rawBytes.length > MAX_INPUT_BYTES) fail('Raw game exceeds 16 MiB');
  keys(
    companion,
    [
      'schema',
      'tableId',
      'rawSha256',
      'cutoffActionId',
      'status',
      'unsupportedRules',
      'record',
      'evidenceFiles',
      'witnesses',
      'initialWitnessId',
      'cardMappings',
      'timeline',
      'coverage',
    ],
    'companion',
  );
  if (companion.schema !== RECONSTRUCTION_SCHEMA) fail('Unsupported reconstruction schema');
  checksum(companion.rawSha256, 'rawSha256');
  if (sha256(rawBytes) !== companion.rawSha256) fail('Raw source SHA-256 mismatch');
  const raw = JSON.parse(rawBytes);
  const game = normalizeGame(raw, companion.rawSha256);
  if (companion.tableId !== game.tableId) fail('Companion table mismatch');
  const cutoff = id(companion.cutoffActionId, 'cutoffActionId');
  const events = sourceEvents(raw, game, cutoff);
  if (events.length > MAX_EVENTS) fail('Prefix exceeds event limit');
  if (!['partial', 'complete'].includes(companion.status))
    fail('Explicit partial/complete status required');
  if (!Array.isArray(companion.unsupportedRules) || companion.unsupportedRules.length)
    fail('Unsupported/unknown rules must be resolved before reconstruction');
  const config = raw.table_details_text.split('ゲーム構成\n').slice(1).join('ゲーム構成\n');
  assertBasicUnlimitedOptions(config);
  const record = companion.record;
  if (
    !object(record) ||
    record.schema !== 'tzolkin-public-replay-v1' ||
    record.market !== 'unlimited'
  )
    fail('Expected a basic unlimited public replay record');
  const steps = array(record.steps, MAX_STEPS, 'record.steps');
  if (!Array.isArray(record.initial?.players)) fail('Missing initial public players');
  const names = record.initial.players.map((player) => player.name);
  if (!same([...names].sort(), [...game.context.players].sort()))
    fail('Replay player identities mismatch');
  if (
    !same(record.initial.hidden, {
      seed: 'unknown',
      setupOffers: 'unknown',
      deckOrder: 'unknown',
    }) ||
    record.initial.additionalBuildings !== false ||
    !same(record.initial.log, [])
  )
    fail('Initial record contains hidden, unsupported or unobserved metadata');

  const fileDefinitions = array(companion.evidenceFiles, 32, 'evidenceFiles');
  const files = new Map();
  let evidenceBytes = 0;
  for (const definition of fileDefinitions) {
    keys(definition, ['id', 'path', 'sha256', 'reference'], 'evidenceFile');
    identifier(definition.id, 'evidenceFile.id');
    checksum(definition.sha256, 'evidenceFile.sha256');
    tableUrl(definition.reference, companion.tableId);
    if (files.has(definition.id)) fail('Duplicate evidence file ID');
    const bytes = documents.get(definition.id);
    if (
      !Buffer.isBuffer(bytes) ||
      bytes.length > MAX_INPUT_BYTES ||
      sha256(bytes) !== definition.sha256
    )
      fail('Evidence file SHA-256/size mismatch');
    evidenceBytes += bytes.length;
    if (evidenceBytes > MAX_TOTAL_EVIDENCE_BYTES) fail('Evidence files exceed aggregate 32 MiB');
    const parsed = JSON.parse(bytes);
    if (!Array.isArray(parsed) || parsed.length > MAX_EVENTS)
      fail('Evidence must be a bounded board-witness array');
    files.set(definition.id, { definition, parsed, bytes: bytes.length });
  }
  const witnesses = new Map();
  const knownIds = new Set(
    raw.entries.filter((entry) => entry.action_id <= cutoff).map((entry) => entry.action_id),
  );
  for (const witness of array(companion.witnesses, MAX_STEPS * 3 + 1, 'witnesses')) {
    keys(witness, ['id', 'actionId', 'fileId', 'index', 'expected'], 'witness');
    identifier(witness.id, 'witness.id');
    if (witnesses.has(witness.id)) fail('Duplicate witness ID');
    if (id(witness.actionId, 'witness.actionId') > cutoff) fail('Future board witness');
    const file = files.get(witness.fileId);
    if (!file || !Number.isSafeInteger(witness.index) || witness.index < 0)
      fail('Missing witness file/index');
    const board = file.parsed[witness.index];
    if (!object(board) || board.sourceActionId !== witness.actionId)
      fail('Witness action/index mismatch');
    const expected = counterCheckpoint(board, names);
    if (!same(witness.expected, expected))
      fail('Witness checkpoint differs from captured public counters');
    witnesses.set(witness.id, { ...witness, board, reference: file.definition.reference });
    knownIds.add(witness.actionId);
  }
  if (!knownIds.has(cutoff)) fail('Cutoff has no log/board witness');
  const witnessFor = (witnessId) => witnesses.get(witnessId) ?? fail('Unknown board witness ID');
  const cardMappings = array(companion.cardMappings, 100, 'cardMappings');
  const cardMap = new Map();
  for (const mapping of cardMappings) {
    keys(mapping, ['uiId', 'catalogId', 'sprite'], 'cardMapping');
    if (
      typeof mapping.uiId !== 'string' ||
      !/^(building_age_[12]_\d+|monument_\d+)$/.test(mapping.uiId) ||
      typeof mapping.catalogId !== 'string' ||
      !/^[bm]\d{2}$/.test(mapping.catalogId) ||
      !Array.isArray(mapping.sprite) ||
      mapping.sprite.length !== 2 ||
      !mapping.sprite.every(Number.isSafeInteger) ||
      cardMap.has(mapping.uiId)
    )
      fail('Invalid/duplicate captured-card mapping');
    cardMap.set(mapping.uiId, mapping);
  }
  const checkMarket = (witness, publicState) => {
    if (!Array.isArray(witness.board.market)) fail('Witness has no captured public market');
    const found = { buildings: [], monuments: [] };
    for (const card of witness.board.market) {
      const mapping = cardMap.get(card.id);
      const sprite = card.style?.match(/background-position:\s*(-?\d+)px\s+(-?\d+)px\s*;/);
      if (!mapping || !sprite || !same(mapping.sprite, [Number(sprite[1]), Number(sprite[2])]))
        fail('Unknown/unmatched public card identity: explicit sprite mapping required');
      const kind = card.id.startsWith('monument_') ? 'monuments' : 'buildings';
      if (!mapping.catalogId.startsWith(kind === 'monuments' ? 'm' : 'b'))
        fail('Captured card kind mismatch');
      found[kind].push(mapping.catalogId);
    }
    if (
      !same(found.buildings, publicState.buildings) ||
      !same(found.monuments, publicState.monuments)
    )
      fail('Public market differs from captured cards or uses future reveals');
  };
  const initialWitness = witnessFor(companion.initialWitnessId);
  evidence(record.source, companion.tableId, initialWitness.actionId, knownIds);
  if (!record.initialCheckpoint) fail('Initial public checkpoint is mandatory');
  evidence(record.initialCheckpoint.source, companion.tableId, initialWitness.actionId, knownIds);
  if (
    !same(record.initialCheckpoint.expected, initialWitness.expected) ||
    !record.initialCheckpoint.source.actionIds.includes(initialWitness.actionId)
  )
    fail('Initial checkpoint/provenance mismatch');
  const initialReport = dispatch({
    operation: 'publicReplay',
    replay: {
      ...record,
      steps: [],
      terminalCheckpoint: null,
      terminalDisplayCheckpoint: null,
    },
  });
  if (
    !Array.isArray(initialReport.frames) ||
    initialReport.frames.length !== 1 ||
    initialReport.verifiedSteps !== 0
  )
    fail('Invalid core initial verification report');
  let state = initialReport.frames[0].snapshot.state;
  contains(state, initialWitness.expected, 'initial witness');
  checkMarket(initialWitness, state);
  const markerPlayerId = initialWitness.board.firstPlayer?.match(/^firstplayer_(\d+)$/)?.[1];
  const markerPlayer = initialWitness.board.players.find((player) => player.id === markerPlayerId);
  if (!markerPlayer || names[state.firstPlayer] !== markerPlayer.name)
    fail('Initial seat/first-player witness mismatch');
  const initialState = state;
  let previousWitness = initialWitness;
  let lastActionId = initialWitness.actionId;
  const live = [];
  const operations = new Map();
  const cancellations = new Map();
  const audit = [];
  const timeline = array(companion.timeline, MAX_STEPS * 2, 'timeline');
  for (const node of timeline) {
    if (node.kind === 'move') {
      keys(node, ['kind', 'id', 'step', 'witnessId'], 'timeline move');
      identifier(node.id, 'move.id');
      if (operations.has(node.id)) fail('Duplicate timeline operation ID');
      const step = node.step;
      if (!object(step)) fail('Missing explicit operation');
      const ids = array(step.sourceActionIds, 64, 'step.sourceActionIds');
      if (!ids.length || new Set(ids).size !== ids.length)
        fail('Operation needs unique provenance IDs');
      for (const actionId of ids)
        if (
          id(actionId, 'step action') > cutoff ||
          !knownIds.has(actionId) ||
          actionId < lastActionId
        )
          fail('Future/unknown/nonmonotonic operation provenance');
      const witness = witnessFor(node.witnessId);
      if (witness.actionId !== Math.max(...ids))
        fail('Operation witness must be at its action boundary');
      if (!step.checkpoint) fail('Every reconstructed operation needs a captured checkpoint');
      evidence(step.checkpoint.source, companion.tableId, witness.actionId, knownIds);
      if (
        !same(step.checkpoint.expected, witness.expected) ||
        !step.checkpoint.source.actionIds.includes(witness.actionId)
      )
        fail('Operation checkpoint/provenance mismatch');
      const before = dispatch({ operation: 'publicInspect', state });
      const legal = before.observation?.legalActions?.find((action) =>
        same(action.move, step.move),
      );
      if (!legal) fail('Reconstructed move absent from authoritative legal actions');
      const next = dispatch({
        operation: 'publicApply',
        state,
        actor: step.actor,
        move: step.move,
        refills: step.refills ?? { currentAge: [], age2: [] },
      });
      state = next.snapshot.state;
      contains(state, witness.expected, `operation ${node.id}`);
      checkMarket(witness, state);
      const item = {
        ...node,
        state,
        beforeState: before.snapshot.state,
        witness,
        action: legal.action,
      };
      operations.set(node.id, item);
      live.push(item);
      audit.push({
        kind: 'move',
        id: node.id,
        actionIds: ids,
        stateSha256: stateHash(state),
        witnessId: witness.id,
      });
      previousWitness = witness;
      lastActionId = witness.actionId;
    } else if (node.kind === 'cancel') {
      keys(
        node,
        ['kind', 'actionId', 'rollbackTo', 'removedMoveIds', 'beforeWitnessId', 'afterWitnessId'],
        'timeline cancel',
      );
      const actionId = id(node.actionId, 'cancel.actionId');
      if (actionId <= lastActionId || actionId > cutoff || cancellations.has(actionId))
        fail('Invalid cancellation order/ID');
      const cancelEvents = game.events.filter(
        (event) => event.actionId === actionId && event.kind === 'cancellation',
      );
      if (cancelEvents.length !== 1)
        fail('Cancellation needs exactly one source cancellation event');
      const actor = names.indexOf(cancelEvents[0].actor);
      if (actor < 0 || state.currentPlayer !== actor) fail('Cancellation actor mismatch');
      const before = witnessFor(node.beforeWitnessId);
      const after = witnessFor(node.afterWitnessId);
      if (before.id !== previousWitness.id || after.actionId !== actionId)
        fail('Cancellation witness boundary mismatch');
      const targetIndex =
        node.rollbackTo === 'initial'
          ? -1
          : live.findIndex((operation) => operation.id === node.rollbackTo);
      if (targetIndex === -1 && node.rollbackTo !== 'initial')
        fail('Cancellation target is not an active operation');
      const removed = live.slice(targetIndex + 1);
      if (
        !same(
          node.removedMoveIds,
          removed.map((operation) => operation.id),
        )
      )
        fail('Cancellation suffix/targets mismatch');
      if (removed.some((operation) => operation.step.actor !== actor))
        fail('Cancellation crosses another actor');
      const target = targetIndex < 0 ? initialWitness : live[targetIndex].witness;
      if (boardHash(after.board) !== boardHash(target.board))
        fail('Cancellation board does not restore explicit target');
      if (!removed.length && boardHash(before.board) !== boardHash(after.board))
        fail('No-op cancellation changes recorded board');
      state = targetIndex < 0 ? initialState : live[targetIndex].state;
      const restored = dispatch({ operation: 'publicInspect', state });
      contains(restored.snapshot.state, after.expected, `cancel ${actionId}`);
      if (restored.snapshot.state.currentPlayer !== actor)
        fail('Cancellation crosses a turn boundary');
      live.splice(targetIndex + 1);
      const resolution = {
        actionId,
        rollbackTo: node.rollbackTo,
        removedMoveIds: node.removedMoveIds,
        noOp: removed.length === 0,
        beforeWitnessId: before.id,
        afterWitnessId: after.id,
        restoredStateSha256: stateHash(restored.snapshot.state),
      };
      cancellations.set(actionId, resolution);
      audit.push({ kind: 'cancel', ...resolution });
      previousWitness = after;
      lastActionId = actionId;
    } else fail('Unsupported timeline node');
  }
  if (
    !same(
      live.map((operation) => operation.step),
      steps,
    )
  )
    fail('Effective rollback operation sequence differs from record.steps');
  if (lastActionId !== cutoff) fail('Timeline must finish exactly at cutoff');

  const eventMap = new Map(
    events.map((event) => [`${event.actionId}:${event.messageIndex}`, event]),
  );
  const covered = new Set();
  const operationCoverage = new Map();
  for (const row of array(companion.coverage, MAX_EVENTS, 'coverage')) {
    keys(row, ['actionId', 'messageIndex', 'kind', 'moveIds', 'reason'], 'coverage');
    const key = `${row.actionId}:${row.messageIndex}`;
    const event = eventMap.get(key);
    if (!event || covered.has(key)) fail('Unknown/duplicate/future source coverage');
    covered.add(key);
    if (row.kind === 'initial') {
      if (
        event.actionId > initialWitness.actionId ||
        event.rawText !== '各プレイヤーは対応する資源を手に入れた'
      )
        fail('Invalid initial-event coverage');
    } else if (row.kind === 'nonGame') {
      const suffix = 'のプレイヤーカラーは設定で選ばれました。設定を変更する';
      const colorNames = event.rawText.endsWith(suffix)
        ? event.rawText
            .slice(0, -suffix.length)
            .split(',')
            .map((name) => name.trim())
        : [];
      if (
        row.reason !== 'playerColors' ||
        /[\r\n]/.test(event.rawText) ||
        !colorNames.length ||
        colorNames.some((name) => !names.includes(name))
      )
        fail('Game/unknown event cannot be ignored');
    } else if (row.kind === 'cancel') {
      if (event.kind !== 'cancellation' || !cancellations.has(event.actionId))
        fail('Unresolved source cancellation');
    } else if (row.kind === 'terminal') {
      if (
        event.kind !== 'gameEnd' ||
        !(record.terminalCheckpoint || record.terminalDisplayCheckpoint)
      )
        fail('Invalid terminal coverage');
    } else if (row.kind === 'move') {
      const moveIds = array(row.moveIds, 64, 'coverage.moveIds');
      if (!moveIds.length || new Set(moveIds).size !== moveIds.length)
        fail('Missing unique operation coverage');
      const matched = moveIds.map(
        (moveId) => operations.get(moveId) ?? fail('Unknown coverage operation'),
      );
      if (matched.some((operation) => !operation.step.sourceActionIds.includes(event.actionId)))
        fail('Source event/operation action ID mismatch');
      if (event.kind === 'cancellation' || event.kind === 'gameEnd')
        fail('Cancellation/terminal cannot be an operation effect');
      if (event.kind === 'unparsed' && row.reason !== 'reviewedEffect')
        fail('Unparsed event needs explicit reviewedEffect annotation');
      const actorOperations = matched.filter(
        (operation) => names[operation.step.actor] === event.actor,
      );
      if (event.actor !== null && actorOperations.length !== matched.length)
        fail('Source group contains an operation by a different actor');
      if (event.kind === 'gearAction' && event.detail.source === 'removal') {
        if (
          matched.length !== 2 ||
          matched[0].action.type !== 'remove' ||
          matched[1].action.type !== 'useAction' ||
          !same(matched[0].state, matched[1].beforeState)
        )
          fail('Source removal requires only its consecutive remove/useAction micro-operations');
      } else if (matched.length !== 1) fail('Source event supports exactly one explicit operation');
      const supported = [
        'place',
        'gearAction',
        'technology',
        'workerGain',
        'build',
        'monument',
        'firstPlayer',
        'gearAdvanced',
        'unparsed',
      ];
      if (!supported.includes(event.kind))
        fail(`Source effect ${event.kind} has no reviewed semantic matcher`);
      if (
        event.kind === 'place' &&
        !actorOperations.some(
          (operation) =>
            operation.step.move.type === 'place' &&
            operation.step.move.gear === event.detail.gear &&
            operation.beforeState.players[operation.step.actor].resources.corn -
              operation.state.players[operation.step.actor].resources.corn ===
              event.detail.cornCost,
        )
      )
        fail('Source placement/gear/cost differs from explicit operation');
      if (
        event.kind === 'gearAction' &&
        !actorOperations.some(
          (operation) =>
            operation.action.type === 'useAction' &&
            operation.action.gear === event.detail.gear &&
            operation.action.position === event.detail.actionPosition,
        )
      )
        fail('Source gear action differs from explicit selected action');
      if (
        event.kind === 'gearAction' &&
        event.detail.source === 'removal' &&
        !actorOperations.some(
          (operation) =>
            operation.action.type === 'remove' &&
            operation.action.gear === event.detail.gear &&
            operation.action.position === event.detail.workerPosition,
        )
      )
        fail('Source worker removal differs from explicit removed position');
      if (
        event.kind === 'gearAction' &&
        event.detail.source === 'selectedAction' &&
        !actorOperations.some(
          (operation) =>
            operation.action.type === 'useAction' &&
            operation.action.gear === event.detail.gear &&
            operation.action.position === event.detail.actionPosition &&
            operation.action.cornCost === event.detail.cornCost,
        )
      )
        fail('Source selected action differs from explicit corn cost');
      if (
        event.kind === 'technology' &&
        !actorOperations.some(
          (operation) =>
            operation.action.type === 'technology' &&
            operation.action.technology === event.detail.technology,
        )
      )
        fail('Source technology differs from explicit operation');
      if (
        event.kind === 'workerGain' &&
        !actorOperations.some(
          (operation) =>
            operation.state.players[operation.step.actor].workers -
              operation.beforeState.players[operation.step.actor].workers ===
            1,
        )
      )
        fail('Source worker gain differs from actor/worker transition');
      for (const kind of ['build', 'monument', 'firstPlayer'])
        if (
          event.kind === kind &&
          !actorOperations.some((operation) => operation.action.type === kind)
        )
          fail(`Source ${kind} differs from actor/typed operation`);
      if (
        event.kind === 'gearAdvanced' &&
        !matched.some(
          (operation) =>
            ((operation.action.type === 'endTurn' && operation.action.doubleAdvance !== true) ||
              (operation.action.type === 'rotate' && operation.action.days === 1)) &&
            (operation.state.round - operation.beforeState.round === 1 ||
              (operation.beforeState.phase === 'playing' && operation.state.phase === 'finished')),
        )
      )
        fail('Source gear advance differs from explicit round transition');
      matched.forEach((operation) => {
        if (!operationCoverage.has(operation.id)) operationCoverage.set(operation.id, new Set());
        operationCoverage.get(operation.id).add(event.actionId);
      });
    } else fail('Unsupported coverage kind');
  }
  if (covered.size !== events.length) fail('Uncovered source prefix events');
  for (const operation of operations.values()) {
    for (const actionId of operation.step.sourceActionIds)
      if (
        !operationCoverage.get(operation.id)?.has(actionId) &&
        !(
          operation.step.move.type === 'endTurn' &&
          !raw.entries.some((entry) => entry.action_id === actionId)
        )
      )
        fail(
          'Operation provenance ID has no matching move coverage or board-only endTurn boundary',
        );
  }
  if (
    events
      .filter((event) => event.kind === 'cancellation')
      .some((event) => !cancellations.has(event.actionId))
  )
    fail('Unresolved prefix cancellation');
  const hasTerminalSource = events.some((event) => event.kind === 'gameEnd');
  const terminal = record.terminalCheckpoint ?? record.terminalDisplayCheckpoint;
  if (companion.status === 'complete' || hasTerminalSource || terminal) {
    if (
      cutoff !== raw.entries.at(-1).action_id ||
      !hasTerminalSource ||
      !terminal ||
      (companion.status === 'complete' && !record.terminalCheckpoint) ||
      (companion.status === 'partial' &&
        (record.terminalCheckpoint || record.terminalDisplayCheckpoint?.mode !== 'floorTotal'))
    )
      fail('Terminal reconstruction requires explicit exact or display-only evidence');
    evidence(terminal.source, companion.tableId, cutoff, knownIds);
    if (!terminal.source.actionIds.includes(cutoff)) fail('Terminal checkpoint must cite cutoff');
    if (
      game.resultsDisplay.length !== names.length ||
      !Array.isArray(terminal.scores) ||
      terminal.scores.length !== names.length ||
      new Set(terminal.scores.map((score) => score.playerId)).size !== names.length ||
      terminal.scores.some((score) => {
        const shown = game.resultsDisplay.find((result) => result.player === names[score.playerId]);
        return !shown || shown.rank !== score.rank || shown.scoreDisplay !== score.total;
      })
    )
      fail('Terminal scores/ranks differ from preserved BGA results');
  }

  const report = dispatch({ operation: 'publicReplay', replay: record });
  if (
    report.status !== companion.status ||
    report.verifiedSteps !== steps.length ||
    report.checkpointsVerified !== steps.length + 1 + (record.terminalCheckpoint ? 1 : 0) ||
    !Array.isArray(report.frames) ||
    report.frames.length !== steps.length + 1 ||
    report.frames.length > MAX_STEPS + 1 ||
    report.trainingReady !== false ||
    (companion.status === 'complete' && (!report.verifiedComplete || !report.terminalMatched)) ||
    (companion.status === 'partial' && (report.verifiedComplete || report.terminalMatched)) ||
    (record.terminalDisplayCheckpoint &&
      (!report.terminalDisplayComparison ||
        report.frames.at(-1)?.snapshot.state.phase !== 'finished'))
  )
    fail('Core replay status/frame/verification mismatch');
  if (!same(report.frames.at(-1).snapshot.state, state))
    fail('Core effective replay differs from rollback trace');
  return {
    record,
    manifest: {
      schema: 'tzolkin-bga-replay-export-v1',
      tableId: companion.tableId,
      cutoffActionId: cutoff,
      rawSha256: companion.rawSha256,
      reconstructionSha256: sha256(canonical(companion)),
      recordSha256: sha256(JSON.stringify(record)),
      evidenceFiles: fileDefinitions.map((definition) => ({
        ...definition,
        bytes: files.get(definition.id).bytes,
      })),
      verifiedSteps: report.verifiedSteps,
      checkpointsVerified: report.checkpointsVerified,
      status: report.status,
      verifiedComplete: report.verifiedComplete,
      terminalMatched: report.terminalMatched,
      mechanicallyFinished: report.frames.at(-1).snapshot.state.phase === 'finished',
      terminalDisplayComparison: report.terminalDisplayComparison ?? null,
      trainingReady: false,
      sourceEventsCovered: covered.size,
      timelineOperations: operations.size,
      cancellationAuditResolved: true,
      cancellations: [...cancellations.values()],
      trace: audit,
      missingReasons: report.missingReasons,
      limitations: [
        'Explicit human reconstruction; the tool does not infer unobserved moves or card identities.',
        'Counter checkpoints and captured cancellation boards are audited; witness provenance is locally supplied, not authenticated by BGA.',
        'Core replay legality and checkpoints do not independently validate every manual icon/geometry interpretation.',
        'No policy/value training approval is granted by this export.',
      ],
    },
  };
}

async function boundedRead(path) {
  if ((await stat(path)).size > MAX_INPUT_BYTES) fail('Input/evidence exceeds 16 MiB');
  const bytes = await readFile(path);
  if (bytes.length > MAX_INPUT_BYTES) fail('Input/evidence exceeds 16 MiB');
  return bytes;
}

// Same byte fingerprint as tzolkin-core, including whitespace in the catalog.
export function catalogFingerprint(bytes) {
  let hash = 0xcbf29ce484222325n;
  for (const byte of bytes) hash = ((hash ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
  return hash.toString(16).padStart(16, '0');
}

/**
 * Recompute an existing public record with the current engine. This does not
 * reconstruct raw logs, authenticate the source, or bypass the strict exporter.
 * A catalog update is safe here only when the entire difference concerns
 * unselected starting tiles: their resources/effects are no longer used after
 * setup. Their dummy placement geometry must remain identical as well.
 */
export function revalidateRecord(recordBytes, previousCatalogBytes, currentCatalogBytes, dispatch) {
  for (const bytes of [recordBytes, previousCatalogBytes, currentCatalogBytes])
    if (!Buffer.isBuffer(bytes) || bytes.length > MAX_INPUT_BYTES)
      fail('Record/catalog exceeds 16 MiB');
  const original = JSON.parse(recordBytes);
  if (
    !object(original) ||
    original.schema !== 'tzolkin-public-replay-v1' ||
    original.rulesVersion !== 1 ||
    original.market !== 'unlimited' ||
    !Array.isArray(original.initial?.players) ||
    ![3, 4].includes(original.initial.players.length) ||
    original.initial.phase !== 'playing'
  )
    fail('Expected a basic 3/4-player post-setup public replay');
  array(original.steps, MAX_STEPS, 'record.steps');
  if (original.catalogHash !== catalogFingerprint(previousCatalogBytes))
    fail('Previous catalog fingerprint differs from original record');
  const previous = JSON.parse(previousCatalogBytes);
  const current = JSON.parse(currentCatalogBytes);
  const selected = new Set();
  for (const player of original.initial.players) {
    if (!Array.isArray(player.wealth) || player.wealth.length !== 2)
      fail('Expected two observed starting tiles per player');
    for (const id of player.wealth) selected.add(id);
  }
  const oldTiles = array(previous.STARTING_WEALTH, 100, 'previous STARTING_WEALTH');
  const newTiles = array(current.STARTING_WEALTH, 100, 'current STARTING_WEALTH');
  if (
    !same({ ...previous, STARTING_WEALTH: null }, { ...current, STARTING_WEALTH: null }) ||
    !same(
      oldTiles.map((tile) => tile.id),
      newTiles.map((tile) => tile.id),
    ) ||
    new Set(oldTiles.map((tile) => tile.id)).size !== oldTiles.length
  )
    fail('Catalog change affects playing rules or starting-tile inventory');
  const changedUnusedTiles = [];
  oldTiles.forEach((tile, index) => {
    const next = newTiles[index];
    if (same(tile, next)) return;
    if (selected.has(tile.id) || tile.gear !== next.gear || tile.position !== next.position)
      fail('Catalog change affects selected wealth or dummy placement');
    changedUnusedTiles.push(tile.id);
  });
  if ([...selected].some((id) => !oldTiles.some((tile) => tile.id === id)))
    fail('Selected starting tile is absent from previous catalog');
  const record = { ...original, catalogHash: catalogFingerprint(currentCatalogBytes) };
  // All initial counters, dummy reachability, actions, refill streams and
  // observed checkpoints are independently checked by the current Rust core.
  const report = dispatch({ operation: 'publicReplay', replay: record });
  const last = report.frames?.at(-1)?.snapshot?.state;
  if (
    !Array.isArray(report.frames) ||
    report.frames.length !== record.steps.length + 1 ||
    report.verifiedSteps !== record.steps.length ||
    report.checkpointsVerified !==
      Number(!!record.initialCheckpoint) +
        record.steps.filter((step) => step.checkpoint != null).length +
        Number(!!record.terminalCheckpoint) ||
    !['partial', 'complete'].includes(report.status) ||
    report.trainingReady !== false ||
    (report.status === 'complete' && (!report.verifiedComplete || !report.terminalMatched)) ||
    (report.status === 'partial' && (report.verifiedComplete || report.terminalMatched)) ||
    (record.terminalDisplayCheckpoint &&
      (last?.phase !== 'finished' || !report.terminalDisplayComparison))
  )
    fail('Current core replay verification mismatch');
  return {
    record,
    manifest: {
      schema: 'tzolkin-public-replay-revalidation-v1',
      originalRecordSha256: sha256(recordBytes),
      recordSha256: sha256(JSON.stringify(record)),
      previousCatalogSha256: sha256(previousCatalogBytes),
      currentCatalogSha256: sha256(currentCatalogBytes),
      previousCatalogHash: original.catalogHash,
      catalogHash: record.catalogHash,
      changedUnusedStartingTiles: changedUnusedTiles,
      preservedSourceAndOperations: true,
      verifiedSteps: report.verifiedSteps,
      checkpointsVerified: report.checkpointsVerified,
      frames: report.frames.length,
      mechanicallyFinished: last?.phase === 'finished',
      status: report.status,
      verifiedComplete: report.verifiedComplete,
      terminalMatched: report.terminalMatched,
      terminalDisplayComparison: report.terminalDisplayComparison ?? null,
      sourceCoverage: report.sourceCoverage,
      trainingReady: false,
      strictExporterAccepted: false,
      sourceAuthenticityIndependentlyAudited: false,
      missingReasons: report.missingReasons,
      limitations: [
        'Existing explicit public actions are recomputed; this is not raw-log conversion or a source/cancellation audit.',
        'Omitted checkpoints and unknown setup offers, seed and deck order remain unknown.',
        'Display-only terminal comparison preserves official fractional scores and does not grant exact result or training approval.',
      ],
    },
  };
}

export async function exportVerifiedRecord(
  recordPath,
  previousCatalogPath,
  currentCatalogPath,
  cliPath,
  outputPath,
) {
  const recordBytes = await boundedRead(recordPath);
  const previous = await boundedRead(previousCatalogPath);
  const current = await boundedRead(currentCatalogPath);
  const cliSha256 = sha256(await boundedRead(cliPath));
  const result = revalidateRecord(recordBytes, previous, current, coreDispatcher(cliPath));
  if (sha256(await boundedRead(cliPath)) !== cliSha256)
    fail('Core executable changed during verification');
  result.manifest.coreCliSha256 = cliSha256;
  return publishReplay(result, outputPath);
}

export async function exportReconstruction(rawPath, companionPath, cliPath, outputPath) {
  const rawBytes = await boundedRead(rawPath);
  const companionBytes = await boundedRead(companionPath);
  const companion = JSON.parse(companionBytes);
  const documents = new Map();
  let evidenceBytes = 0;
  for (const file of array(companion.evidenceFiles, 32, 'evidenceFiles')) {
    if (
      typeof file.path !== 'string' ||
      !file.path.length ||
      file.path.includes('\0') ||
      /^\w+:\/\//.test(file.path)
    )
      fail('Evidence requires an explicit local file path');
    const bytes = await boundedRead(resolve(dirname(companionPath), file.path));
    evidenceBytes += bytes.length;
    if (evidenceBytes > MAX_TOTAL_EVIDENCE_BYTES) fail('Evidence files exceed aggregate 32 MiB');
    documents.set(file.id, bytes);
  }
  const result = auditReconstruction(rawBytes, companion, documents, coreDispatcher(cliPath));
  result.manifest.reconstructionFileSha256 = sha256(companionBytes);
  result.manifest.inputs = {
    rawPath: resolve(rawPath),
    reconstructionPath: resolve(companionPath),
    coreCliPath: resolve(cliPath),
    outputPath: resolve(outputPath),
  };
  return publishReplay(result, outputPath);
}

export async function publishReplay(result, outputPath) {
  const output = resolve(outputPath);
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output); // An existing output, including an empty one, is rejected.
  const temporary = await mkdtemp(`${output}.tmp-`);
  if (dirname(temporary) !== dirname(output)) fail('Invalid staging path');
  let published = false;
  try {
    await writeFile(join(temporary, 'record.json'), JSON.stringify(result.record));
    await writeFile(join(temporary, 'manifest.json'), JSON.stringify(result.manifest, null, 2));
    await rmdir(output); // Only remove the empty reservation this call created.
    await rename(temporary, output);
    published = true;
  } finally {
    if (!published) {
      await rm(temporary, { recursive: true, force: true });
      await rmdir(output).catch((error) => {
        if (!['ENOENT', 'ENOTEMPTY', 'EEXIST'].includes(error.code)) throw error;
      });
    }
  }
  return result.manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const verify = args[0] === 'verify-record';
  if (args.length !== (verify ? 6 : 4) || args.some((arg) => !arg)) {
    console.error(
      'Usage: node scripts/bga-replay.mjs RAW_GAME_JSON RECONSTRUCTION_JSON RUST_CLI NEW_OUTPUT_DIR\n' +
        '   or: node scripts/bga-replay.mjs verify-record RECORD_JSON PREVIOUS_CATALOG CURRENT_CATALOG RUST_CLI NEW_OUTPUT_DIR',
    );
    process.exitCode = 1;
  } else {
    (verify
      ? exportVerifiedRecord(...args.slice(1).map((path) => resolve(path)))
      : exportReconstruction(...args.map((path) => resolve(path)))
    )
      .then((manifest) =>
        console.log(
          JSON.stringify({
            tableId: manifest.tableId,
            status: manifest.status,
            verifiedSteps: manifest.verifiedSteps,
            checkpointsVerified: manifest.checkpointsVerified,
            mechanicallyFinished: manifest.mechanicallyFinished,
            terminalMatched: manifest.terminalMatched,
            terminalDisplayComparison: manifest.terminalDisplayComparison,
            cancellations: manifest.cancellations?.length ?? null,
            trainingReady: false,
          }),
        ),
      )
      .catch((error) => {
        console.error(error.message);
        process.exitCode = 1;
      });
  }
}

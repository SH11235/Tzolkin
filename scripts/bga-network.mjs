import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { uniqueJson } from './bga-observed-options.mjs';

const MAX_BYTES = 16 * 1024 * 1024;
// New client revisions require independent review before they can authorize decoding.
const REVIEWED_CLIENT_PAIRS = new Set([
  'bb384177730e85653781f3b193f65b1df88c5624439d12d3dae6a22473f19334:e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
  '8cec9fefd73a8abdfbd539ad1c547e1e92ee8db4ebc0acabeeb8ee87ebdb0b41:e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
  'baefe213893a46474a278e315243e16daaf2cc0ffe46396e6c0782d6f38faf47:e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
  'b020a816c96ad050c9b96ff29a6b8e14efdda44132f2daa0c29893ecf596b5eb:e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
  '8e4f493ad56c2a4b9e79d7c5d5c853cca4fe48f2e0c76547bd6d03f62cdca3f1:e00b9901089706b391093157ec8b315e1bba13b2843d7a9333d38f523ce67326',
]);
const validatedContracts = new WeakSet();
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const object = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
const fail = (message) => {
  throw new Error(`Network replay: ${message}`);
};
function integer(value, name, minimum = 0) {
  if (typeof value === 'string' && !/^(0|[1-9]\d*)$/.test(value)) fail(`invalid ${name}`);
  const n = Number(value);
  if (!['number', 'string'].includes(typeof value) || !Number.isSafeInteger(n) || n < minimum)
    fail(`invalid ${name}`);
  return n;
}
function keys(x, allowed, name) {
  if (!object(x) || Object.keys(x).some((k) => !allowed.includes(k))) fail(`unknown ${name} field`);
}

// Classification is explicit: notification counts are never human-decision counts.
const CLASSIFICATION = {
  gameStateChange: 'state',
  gameStateMultipleActiveUpdate: 'state',
  chooseWealthTiles: 'privateSetup',
  wealthTiles: 'setupEffect',
  takeGoods: 'resourceEffect',
  payGoods: 'resourceEffect',
  increaseTechnology: 'technologyEffect',
  decreaseTechnology: 'technologyEffect',
  alterTemple: 'templeEffect',
  placeWorker: 'placement',
  pickUpWorker: 'removal',
  newWorker: 'workerEffect',
  takeTile: 'harvestEffect',
  selectAction: 'actionSelection',
  firstPlayer: 'calendarEffect',
  firstPlayerCorn: 'calendarCaption',
  cornOnWheel: 'calendarEffect',
  rotateWheel: 'rotation',
  undoRestoreDISAB: 'archiveCancellationCaption',
  trade: 'tradeSelection',
  foodDayCost: 'feedingCaption',
  updateScore: 'scoreEffect',
  construct: 'construction',
  pickBuilding: 'refill',
  resetAllBuildings: 'ageRefill',
  skullplace: 'skullEffect',
  extractionMax: 'technologyCaption',
  agricultureMax: 'technologyCaption',
  architectureMax: 'technologyCaption',
  theologyMax: 'technologyCaption',
  workerLimit: 'workerCapCaption',
  middleOfAgeRewards: 'awardCaption',
  templeLeader: 'awardCaption',
  templeScore: 'awardCaption',
  finalScoring: 'awardCaption',
  simpleNode: 'terminalCaption',
  simpleNote: 'initialColorPreferenceCaption',
  flipBoard: 'doubleAdvanceAvailability',
  updateReflexionTime: 'clock',
  wakeupPlayers: 'administrativeWakeup',
};

/** Verify only the observed client's archive dispatch contract, not server undo semantics. */
export function inspectArchiveClient(studioBytes, gameBytes) {
  for (const b of [studioBytes, gameBytes])
    if (!Buffer.isBuffer(b) || b.length > 4 * MAX_BYTES) fail('invalid client source bytes');
  const studioSha256 = sha(studioBytes);
  const gameSha256 = sha(gameBytes);
  if (!REVIEWED_CLIENT_PAIRS.has(`${studioSha256}:${gameSha256}`))
    fail('unreviewed archive client version');
  const studio = studioBytes.toString('utf8');
  const game = gameBytes.toString('utf8');
  if (/subscribe\("wakeupPlayers"|ntf_wakeupPlayers/.test(studio) || game.includes('wakeupPlayers'))
    fail('unreviewed wakeup state handler');
  const requirements = [
    'initArchiveIndex:function()',
    '.packet_id)>toint(this.notifqueue.last_packet_id)',
    'e.publish(t.type,[t])',
    'onMove:function(){},onNextMove:function(e)',
  ];
  // The archived client uses loose comparison in one of its view filters.
  const filter = 't.channel!=this.private_channel&&t.channel!="/player/p"+this.player_id';
  if (!requirements.slice(0, 4).every((x) => studio.includes(x)) || !studio.includes(filter))
    fail('unsupported archive dispatch client');
  if (
    studio.includes('undoRestoreDISAB') ||
    game.includes('undoRestoreDISAB') ||
    /onNextMove|onMove|undoRestore|resetInterfaceWithAllDatas/.test(game)
  )
    fail('client overrides observed cancellation contract');
  if (
    !studio.includes('e.subscribe("undoRestorePoint",this,"ntf_undoRestorePoint")') ||
    !studio.includes('ntf_undoRestorePoint:function(t){globalThis.g_archive_mode||')
  )
    fail('unsupported undo dispatch contract');
  const contract = Object.freeze({
    schema: 'tzolkin-bga-archive-client-contract-v1',
    studioSha256,
    gameSha256,
    cancellationType: 'undoRestoreDISAB',
    scope: 'unchanged archive start-turn captions only; no server deletion claim',
  });
  validatedContracts.add(contract);
  return contract;
}

/** Read a saved response extract; no browser requests, credential loading or code evaluation. */
export function parseNetworkReplay(bytes) {
  if (!Buffer.isBuffer(bytes) || bytes.length > MAX_BYTES) fail('invalid saved response bytes');
  const raw = uniqueJson(bytes, MAX_BYTES, 'network extract');
  keys(raw, ['schema', 'tableId', 'source', 'initialState', 'gameLogs'], 'extract');
  if (raw.schema !== 'tzolkin-bga-replay-network-extract-v1') fail('unsupported extract schema');
  const tableId = String(integer(raw.tableId, 'table ID', 1));
  keys(raw.source, ['type', 'url', 'httpStatus', 'extraction', 'capturedAt'], 'source');
  if (raw.source.type !== 'normal_replay_document_http_response' || raw.source.httpStatus !== 200)
    fail('unsupported source');
  const url = new URL(raw.source.url);
  if (
    url.protocol !== 'https:' ||
    !/^(?:[a-z]{2}\.)?boardgamearena\.com$/.test(url.hostname) ||
    url.username ||
    url.password ||
    !/^\/archive\/replay\/[\w-]+\/$/.test(url.pathname) ||
    url.searchParams.getAll('table').length !== 1 ||
    url.searchParams.get('table') !== tableId ||
    url.searchParams.getAll('player').length !== 1
  )
    fail('source replay URL mismatch');
  if (
    [...url.searchParams.keys()].some((k) => !['table', 'player', 'comments', 'goto'].includes(k))
  )
    fail('unsupported replay URL query');
  const viewer = String(integer(url.searchParams.get('player'), 'view player', 1));
  const initial = raw.initialState;
  if (
    !object(initial) ||
    !object(initial.players) ||
    !Array.isArray(initial.workers) ||
    !Array.isArray(initial.playerorder) ||
    !object(initial.gamestate) ||
    !object(initial.notifications) ||
    !object(initial.gamestates)
  )
    fail('invalid initial state');
  const playerIds = Object.keys(initial.players);
  if (
    ![3, 4].includes(playerIds.length) ||
    !playerIds.includes(viewer) ||
    initial.playerorder.length !== playerIds.length ||
    new Set(initial.playerorder.map(String)).size !== playerIds.length ||
    initial.playerorder.some((id) => !playerIds.includes(String(id)))
  )
    fail('invalid player order/view');
  if (!playerIds.includes(String(initial.firstPlayer))) fail('invalid first player');
  for (const [id, p] of Object.entries(initial.players)) {
    integer(id, 'source player ID', 1);
    if (
      String(p.id) !== id ||
      typeof p.name !== 'string' ||
      !p.name ||
      playerIds.some((other) => other !== id && initial.players[other].name === p.name)
    )
      fail('invalid player identity');
    for (const k of [
      'corn',
      'wood',
      'stone',
      'gold',
      'skull',
      'agriculture',
      'extraction',
      'architecture',
      'theology',
      'chaac',
      'quetzalcoatl',
      'kukulcan',
    ])
      integer(p[k], `initial ${k}`);
    if (!Number.isFinite(Number(p.score))) fail('invalid initial score');
  }
  const checkpoint = integer(initial.notifications.last_packet_id, 'checkpoint packet', 1);
  const checkpointMove = integer(initial.notifications.move_nbr, 'checkpoint move', 1);
  keys(raw.gameLogs, ['status', 'data'], 'gameLogs');
  if (raw.gameLogs.status !== 1) fail('unsuccessful gameLogs status');
  keys(raw.gameLogs.data, ['valid', 'data'], 'gameLogs.data');
  const packets = raw.gameLogs.data.data;
  if (
    raw.gameLogs.data.valid !== 1 ||
    !Array.isArray(packets) ||
    !packets.length ||
    packets.length > 4000
  )
    fail('invalid packet list');
  const seen = new Set();
  const entries = [];
  let previousPacket = 0;
  let previousMove = 0;
  const counts = {
    totalPackets: packets.length,
    visiblePackets: 0,
    hiddenPackets: 0,
    checkpointPackets: 0,
    visibleNotifications: 0,
    hiddenNotifications: 0,
    administrativeNotifications: 0,
  };
  for (const [pi, packet] of packets.entries()) {
    keys(
      packet,
      ['channel', 'table_id', 'packet_id', 'packet_type', 'move_id', 'time', 'data'],
      'packet',
    );
    const packetId = integer(packet.packet_id, 'packet ID', 1);
    const administrative =
      packet.move_id === null &&
      Array.isArray(packet.data) &&
      packet.data.length === 1 &&
      packet.data[0]?.type === 'wakeupPlayers';
    const moveId = administrative ? null : integer(packet.move_id, 'move ID', 1);
    if (
      String(packet.table_id) !== tableId ||
      packetId <= previousPacket ||
      (moveId !== null && moveId < previousMove) ||
      !Array.isArray(packet.data) ||
      !packet.data.length ||
      packet.data.length > 2000
    )
      fail('invalid packet identity/order');
    previousPacket = packetId;
    if (moveId !== null) previousMove = moveId;
    if (
      packet.channel !== `/table/t${tableId}` &&
      !playerIds.some((id) => packet.channel === `/player/p${id}`)
    )
      fail('unknown packet channel');
    const visible =
      packet.channel === `/table/t${tableId}` || packet.channel === `/player/p${viewer}`;
    counts[visible ? 'visiblePackets' : 'hiddenPackets']++;
    if (packetId <= checkpoint) counts.checkpointPackets++;
    for (const [ni, n] of packet.data.entries()) {
      keys(n, ['uid', 'type', 'log', 'args', 'synchro', 'lock_uuid', 'h'], 'notification');
      if (
        typeof n.uid !== 'string' ||
        !n.uid ||
        seen.has(n.uid) ||
        !Object.hasOwn(CLASSIFICATION, n.type) ||
        typeof n.log !== 'string' ||
        !(object(n.args) || Array.isArray(n.args))
      )
        fail('unknown/invalid/duplicate notification');
      seen.add(n.uid);
      if (n.type === 'wakeupPlayers') {
        if (
          !administrative ||
          packet.channel !== `/table/t${tableId}` ||
          n.log !== '' ||
          !Array.isArray(n.args) ||
          n.args.length !== 0
        )
          fail('unsupported administrative wakeup packet');
        counts.administrativeNotifications++;
      }
      if (n.type === 'theologyMax') {
        keys(n.args, ['player_name'], 'theology caption');
        if (
          n.log !== 'Theology bonus: ${player_name} takes a skull' ||
          !Object.values(initial.players).some((p) => p.name === n.args.player_name)
        )
          fail('unsupported theology caption');
      }
      if (n.type === 'workerLimit') {
        keys(n.args, ['player_name'], 'worker cap caption');
        if (
          n.log !== '${player_name} already have 6 workers' ||
          !Object.values(initial.players).some((p) => p.name === n.args.player_name)
        )
          fail('unsupported worker cap caption');
      }
      if (n.type === 'simpleNote') {
        keys(n.args, ['players', 'change_preferences'], 'color preference caption');
        const link = n.args.change_preferences;
        if (
          packetId > checkpoint ||
          ![
            'Color of ${players} has been chosen according to his/her preferences. ${change_preferences}',
            'Colors of ${players} have been chosen according to their preferences. ${change_preferences}',
          ].includes(n.log) ||
          typeof n.args.players !== 'string' ||
          !object(link) ||
          link.log !==
            '<a href="https://boardgamearena.com/preferences" target="_blank">${label}</a>' ||
          link.args?.label !== 'Change my preferences.'
        )
          fail('unsupported simpleNote; only checkpoint color-preference caption is reviewed');
      }
      if (n.type === 'flipBoard') {
        keys(n.args, ['player_id', 'side', 'player_name'], 'board flip');
        if (!playerIds.includes(String(n.args.player_id)) || ![0, 1].includes(n.args.side))
          fail('invalid double-advance availability notification');
        if (
          n.args.side === 0 &&
          (n.log !== '${player_name} flips his board lighter side up' ||
            n.args.player_name !== initial.players[String(n.args.player_id)].name)
        )
          fail('unsupported speed-up renewal caption');
      }
      counts[visible ? 'visibleNotifications' : 'hiddenNotifications']++;
      entries.push({
        packetIndex: pi,
        notificationIndex: ni,
        packetId,
        moveId,
        administrative,
        uid: n.uid,
        type: n.type,
        classification: CLASSIFICATION[n.type],
        args: structuredClone(n.args),
        visible,
        beforeCheckpoint: packetId <= checkpoint,
        channel: packet.channel,
      });
      if (entries.length > 20000) fail('notification limit');
    }
  }
  const boundary = packets.find((p) => Number(p.packet_id) === checkpoint);
  if (!boundary || Number(boundary.move_id) !== checkpointMove) fail('missing checkpoint packet');
  // This first contract is deliberately restricted to a pre-distribution initial snapshot.
  const before = entries.filter((n) => n.beforeCheckpoint);
  const activeUpdate = before.filter((n) => n.type === 'gameStateMultipleActiveUpdate').at(-1);
  if (
    integer(initial.day, 'initial day') !== 0 ||
    integer(initial.gamestate.id, 'initial state ID') !== 2 ||
    before.some(
      (n) =>
        ![
          'gameStateChange',
          'gameStateMultipleActiveUpdate',
          'chooseWealthTiles',
          'simpleNote',
        ].includes(n.type),
    ) ||
    !activeUpdate ||
    !isDeepStrictEqual(activeUpdate.args, initial.gamestate.multiactive) ||
    initial.workers.some((w) => w.place !== null || w.position !== null) ||
    Object.values(initial.players).some((p) =>
      ['corn', 'wood', 'stone', 'gold', 'skull'].some((k) => Number(p[k]) !== 0),
    )
  )
    fail('unsupported initial checkpoint contents');
  const visibleEntries = entries.filter(
    (n) => n.visible && !n.beforeCheckpoint && !n.administrative,
  );
  const groups = [];
  for (const n of visibleEntries) {
    if (groups.at(-1)?.moveId !== n.moveId) groups.push({ moveId: n.moveId, entries: [] });
    groups.at(-1).entries.push(n);
  }
  return {
    schema: 'tzolkin-bga-network-index-v1',
    rawSha256: sha(bytes),
    tableId,
    viewer,
    sourceUrl: url.href,
    initial,
    checkpoint,
    checkpointMove,
    counts,
    entries,
    administrativeEntries: entries.filter((n) => n.administrative),
    groups,
  };
}

/** Check client-bound display-only cancellation clusters; never reconstruct missing trials. */
export function inspectCancellationClusters(index, contract) {
  if (
    !object(contract) ||
    !validatedContracts.has(contract) ||
    contract?.schema !== 'tzolkin-bga-archive-client-contract-v1' ||
    contract.cancellationType !== 'undoRestoreDISAB' ||
    !/^[a-f\d]{64}$/.test(contract.studioSha256) ||
    !/^[a-f\d]{64}$/.test(contract.gameSha256)
  )
    fail('cancellation needs an observed archive client contract');
  return inspectCancellationSequence(index).map((cluster) => ({ ...cluster, client: contract }));
}

/** Structural diagnostics only; this does not authorize display-only cancellation handling. */
export function inspectCancellationSequence(index) {
  const clusters = [];
  let state = null;
  let previousWasCancellation = false;
  let previousActor = null;
  for (const group of index.groups) {
    const cancelled = group.entries.filter((n) => n.type === 'undoRestoreDISAB');
    if (cancelled.length) {
      const n = cancelled[0];
      keys(n.args, ['player_name'], 'cancellation args');
      const actor = Object.values(index.initial.players).find(
        (p) => p.name === n.args.player_name,
      )?.id;
      if (
        cancelled.length !== 1 ||
        group.entries.length !== 1 ||
        !actor ||
        Number(state?.id) !== 4 ||
        String(state?.active_player) !== String(actor) ||
        (previousWasCancellation && previousActor !== actor)
      )
        fail('unsupported cancellation boundary/actor');
      if (!previousWasCancellation)
        clusters.push({
          actorPlayerId: actor,
          startStateId: 4,
          moves: [],
          notifications: [],
        });
      clusters.at(-1).moves.push(group.moveId);
      clusters.at(-1).notifications.push({
        packetId: n.packetId,
        packetIndex: n.packetIndex,
        notificationIndex: n.notificationIndex,
        uid: n.uid,
      });
      previousActor = actor;
      previousWasCancellation = true;
    } else {
      for (const n of group.entries) if (n.type === 'gameStateChange') state = n.args;
      previousWasCancellation = false;
      previousActor = null;
    }
  }
  return clusters;
}

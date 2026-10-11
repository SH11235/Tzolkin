import { useEffect, useRef, useState } from 'react';
import { applyMove, createGame, inspectGame, observeGame, type GameSnapshot } from './game/engine';
import {
  chooseCpu,
  discardCpuGame,
  getActiveCpuGame,
  getLoadedCpuModel,
  loadCpuModel,
  notifyCpuCallback,
  startCpuGame,
  type CpuGameMetadata,
  type CpuModelMetadata,
  type GuardedCpuDecision,
  type Observation,
} from './game/cpu';
import { NnDecisionTiming } from './game/cpuTiming';
import type { GameMove } from './game/types';
import { CalendarArt, Icon } from './ui/Icons';
import { PlayersSidebar } from './ui/PlayersSidebar';
import { GameBoardView, type View } from './ui/GameBoardView';
import { ActionsSidebar } from './ui/ActionsSidebar';
import {
  SAVE_KEY,
  controllersFor,
  parseSession,
  readSession,
  requiresNnCpuGuard,
  supportsNnCpu,
  type Controller,
  type Session,
} from './game/storage';
import { ResetDialog } from './ui/ResetDialog';
import { ReplayViewer } from './ui/ReplayViewer';
import { saveGameFile } from './game/files';
import './App.css';

function App() {
  const [saved, setSaved] = useState<Session | null>(null);
  const [active, setActive] = useState<{ session: Session; snapshot: GameSnapshot } | null>(null);
  const [loading, setLoading] = useState(true);
  const [startupError, setStartupError] = useState(false);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const actionFocus = useRef<HTMLElement | null>(null);
  const [view, setView] = useState<View>('board');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [autosaveAvailable, setAutosaveAvailable] = useState(true);
  const [count, setCount] = useState(2);
  const [additionalBuildings, setAdditionalBuildings] = useState(false);
  const [tribes, setTribes] = useState(false);
  const [prophecies, setProphecies] = useState(false);
  const [quickActions, setQuickActions] = useState(false);
  const [cpuMode, setCpuMode] = useState<'heuristic' | 'nn'>('heuristic');
  const [modelMetadata, setModelMetadata] = useState<CpuModelMetadata>();
  const modelRef = useRef<{ bytes: Uint8Array; metadata: CpuModelMetadata } | null>(null);
  const modelOperationRef = useRef<AbortController | null>(null);
  const cpuAbortRef = useRef<AbortController | null>(null);
  const actionAbortRef = useRef<AbortController | null>(null);
  const cpuGameRef = useRef<CpuGameMetadata | null>(null);
  const [guardMetadata, setGuardMetadata] = useState<CpuGameMetadata>();
  const [preparingCpuModel, setPreparingCpuModel] = useState(false);
  const [saving, setSaving] = useState(false);
  const [names, setNames] = useState(['翡翠の民', '黄金の民', '珊瑚の民', '藍の民', '紫水晶の民']);
  const [controllers, setControllers] = useState<Controller[]>([
    'human',
    'human',
    'human',
    'human',
    'human',
  ]);
  const [cpuPaused, setCpuPaused] = useState(false);
  const [confirmReset, setConfirmReset] = useState(false);
  const [replayOpen, setReplayOpen] = useState(false);
  const replayOpenRef = useRef(false);
  const pausedBeforeReplayRef = useRef(true);
  const importInput = useRef<HTMLInputElement>(null);
  const modelInput = useRef<HTMLInputElement>(null);
  const choicesRef = useRef<HTMLHeadingElement>(null);
  const [resetOpener, setResetOpener] = useState<HTMLElement | null>(null);
  const session = active?.session;
  const game = active?.snapshot.state;
  const actor = game?.players[game.currentPlayer];
  const choices = active?.snapshot.choices ?? [];
  const moves = game?.phase === 'playing' ? (active?.snapshot.moves ?? []) : [];
  const activeRef = useRef(active);
  useEffect(
    () => () => {
      cpuAbortRef.current?.abort();
      modelOperationRef.current?.abort();
      actionAbortRef.current?.abort();
      discardCpuGame(cpuGameRef.current ?? undefined);
      modelRef.current = null;
    },
    [],
  );
  const cpuTurn =
    !!session &&
    !!game &&
    game.phase !== 'finished' &&
    controllersFor(session)[game.currentPlayer] === 'cpu';
  useEffect(() => {
    let cancelled = false;
    void readSession()
      .then((data) => {
        if (!cancelled) {
          setSaved(data);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setStartupError(true);
          setError('保存した対局を読み込めませんでした。再試行してください。');
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);
  function retryRead() {
    setStartupError(false);
    setError('');
    void readSession()
      .then((data) => {
        setSaved(data);
        setLoading(false);
      })
      .catch(() => {
        setStartupError(true);
        setError('保存した対局を読み込めませんでした。再試行してください。');
      });
  }
  async function run(action: (signal: AbortSignal) => Promise<void>) {
    if (busyRef.current || loading) return;
    cpuAbortRef.current?.abort();
    actionFocus.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    busyRef.current = true;
    setBusy(true);
    const controller = new AbortController();
    actionAbortRef.current = controller;
    try {
      await action(controller.signal);
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : '操作を実行できませんでした。');
    } finally {
      if (actionAbortRef.current === controller) actionAbortRef.current = null;
      busyRef.current = false;
      setBusy(false);
    }
  }
  function openReset() {
    if (busyRef.current) return;
    cpuAbortRef.current?.abort();
    setResetOpener(document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setCpuPaused(true);
    setConfirmReset(true);
  }
  function openReplay() {
    if (busyRef.current) return;
    pausedBeforeReplayRef.current = cpuPaused;
    cpuAbortRef.current?.abort();
    setCpuPaused(true);
    replayOpenRef.current = true;
    setReplayOpen(true);
  }
  function closeReplay() {
    replayOpenRef.current = false;
    setReplayOpen(false);
    if (cpuGameRef.current && getActiveCpuGame() !== cpuGameRef.current)
      discardGuard(cpuGameRef.current);
    const guardRequired = session && requiresNnCpuGuard(session);
    if (
      !pausedBeforeReplayRef.current &&
      (!guardRequired || (cpuGameRef.current && getActiveCpuGame() === cpuGameRef.current))
    )
      setCpuPaused(false);
  }
  function updateSession(next: Session, snapshot: GameSnapshot) {
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(next));
      setSaved(next);
      setAutosaveAvailable(true);
    } catch {
      setAutosaveAvailable(false);
      setNotice('自動保存を利用できません。保存ファイルを書き出して対局を残せます。');
    }
    const nextActive = { session: next, snapshot };
    activeRef.current = nextActive;
    setActive(nextActive);
  }
  function missingModel(next: Session): string {
    return requiresNnCpuGuard(next) &&
      modelRef.current?.metadata.checksum !== next.cpuPolicy?.modelChecksum
      ? 'このNN対局と同じモデルを読み込んでから、CPUを再開してください。'
      : '';
  }
  function discardGuard(expected: CpuGameMetadata) {
    discardCpuGame(expected);
    if (cpuGameRef.current === expected) {
      cpuGameRef.current = null;
      setGuardMetadata(undefined);
      setCpuPaused(true);
    }
  }
  async function initializeGuard(nextSession: Session, signal: AbortSignal, fresh = true) {
    if (!requiresNnCpuGuard(nextSession)) {
      if (cpuGameRef.current) discardGuard(cpuGameRef.current);
      return;
    }
    const payload = modelRef.current;
    if (!payload || payload.metadata.checksum !== nextSession.cpuPolicy?.modelChecksum)
      throw new Error('このNN対局と同じモデルを読み込んでください。');
    const existing = cpuGameRef.current;
    if (
      !fresh &&
      existing &&
      getActiveCpuGame() === existing &&
      existing.modelChecksum === payload.metadata.checksum
    )
      return;
    if (getLoadedCpuModel()?.checksum !== payload.metadata.checksum)
      await loadCpuModel(payload.bytes, signal, payload.metadata.checksum);
    const next = await startCpuGame(signal, payload.metadata.checksum);
    if (signal.aborted) {
      discardCpuGame(next);
      return;
    }
    cpuGameRef.current = next;
    setGuardMetadata(next);
  }
  async function notifyGuard(
    nextSession: Session,
    observation: Observation,
    signal: AbortSignal,
    timing?: NnDecisionTiming,
  ) {
    if (!requiresNnCpuGuard(nextSession)) return undefined;
    const current = cpuGameRef.current;
    if (current) return notifyCpuCallback(observation, signal, current, timing);
    return undefined;
  }
  function toggleCpu() {
    if (busyRef.current || !session) return;
    if (!cpuPaused) {
      cpuAbortRef.current?.abort();
      setCpuPaused(true);
      setError('');
      return;
    }
    const missing = missingModel(session);
    if (missing) {
      setError(missing);
      return;
    }
    void run(async (signal) => {
      const previous = cpuGameRef.current;
      await initializeGuard(session, signal, false);
      if (signal.aborted) return;
      if (previous !== cpuGameRef.current && cpuGameRef.current)
        setNotice('この局面からNN CPUの履歴を新しく開始します。');
      setCpuPaused(false);
    });
  }
  async function importModel(file?: File) {
    if (!file) return;
    await run(async () => {
      const expected = activeRef.current;
      const controller = new AbortController();
      modelOperationRef.current = controller;
      try {
        if (expected && requiresNnCpuGuard(expected.session)) setCpuPaused(true);
        if (file.size > 1024 * 1024) throw new Error('NNモデルが大きすぎます（上限1MiB）。');
        if (cpuGameRef.current) discardGuard(cpuGameRef.current);
        // Keep one payload in memory; neither its body nor its filename is saved.
        modelRef.current = null;
        setModelMetadata(undefined);
        const bytes = new Uint8Array(await file.arrayBuffer());
        if (controller.signal.aborted || modelOperationRef.current !== controller) return;
        const metadata = await loadCpuModel(
          bytes,
          controller.signal,
          expected && requiresNnCpuGuard(expected.session)
            ? expected.session.cpuPolicy?.modelChecksum
            : undefined,
        );
        if (
          controller.signal.aborted ||
          modelOperationRef.current !== controller ||
          activeRef.current !== expected
        )
          return;
        modelRef.current = { bytes, metadata };
        setModelMetadata(metadata);
        if (expected) await initializeGuard(expected.session, controller.signal);
        setNotice(
          expected && requiresNnCpuGuard(expected.session)
            ? 'NNモデルを読み込みました。CPUを再開できます。'
            : 'NNモデルを読み込みました。',
        );
      } catch (failure) {
        if (!controller.signal.aborted) throw failure;
      } finally {
        if (modelOperationRef.current === controller) modelOperationRef.current = null;
        if (modelInput.current) modelInput.current.value = '';
      }
    });
  }
  useEffect(() => {
    if (busy) return;
    const opener = actionFocus.current;
    actionFocus.current = null;
    if (opener && document.activeElement !== document.body && document.activeElement !== opener)
      return;
    if (game?.pending || game?.phase === 'setup') {
      choicesRef.current?.focus();
    } else if (opener) {
      if (opener.isConnected) opener.focus();
      if (document.activeElement === document.body) choicesRef.current?.focus();
    }
  }, [busy, game?.pending, game?.phase, game?.currentPlayer]);
  function play(move: GameMove) {
    if (!session || cpuTurn) return;
    void run(async (signal) => {
      const expected = activeRef.current;
      if (!expected || expected.session !== session) return;
      const guarded = requiresNnCpuGuard(session) ? cpuGameRef.current : null;
      if (requiresNnCpuGuard(session) && (!guarded || getActiveCpuGame() !== guarded)) {
        if (guarded) discardGuard(guarded);
        else setCpuPaused(true);
        throw new Error('この局面からNN CPUの履歴を明示的に再開してください。');
      }
      let complete = false;
      try {
        if (guarded) {
          const observation = await observeGame(session.state, session.state.currentPlayer);
          if (signal.aborted || activeRef.current !== expected) return;
          await notifyGuard(session, observation, signal);
        }
        if (signal.aborted || activeRef.current !== expected) return;
        const snapshot = await applyMove(session.state, move);
        if (signal.aborted || activeRef.current !== expected) return;
        complete = true;
        updateSession(
          {
            ...session,
            state: snapshot.state,
            history: [...session.history, session.state].slice(-60),
          },
          snapshot,
        );
      } finally {
        if (!complete && guarded) discardGuard(guarded);
      }
    });
  }
  useEffect(() => {
    if (!active || !cpuTurn || cpuPaused || busy || confirmReset || replayOpen) return;
    const controller = new AbortController();
    cpuAbortRef.current = controller;
    const expected = active;
    let issuedGame: CpuGameMetadata | undefined;
    let timing: NnDecisionTiming | undefined;
    let complete = false;
    // Keep each decision on its own task so pause, undo and reset stay responsive.
    const timer = setTimeout(() => {
      void (async () => {
        try {
          const selection = expected.session.cpuPolicy ?? { mode: 'heuristic' as const };
          if (selection.mode === 'nn') {
            const payload = modelRef.current;
            if (!payload || payload.metadata.checksum !== selection.modelChecksum)
              throw new Error(missingModel(expected.session));
            if (getActiveCpuGame() !== cpuGameRef.current || !cpuGameRef.current) {
              if (cpuGameRef.current) discardGuard(cpuGameRef.current);
              throw new Error('この局面からNN CPUを明示的に再開してください。');
            }
            timing = new NnDecisionTiming(cpuGameRef.current, (failure) => {
              controller.abort();
              if (issuedGame) discardGuard(issuedGame);
              if (activeRef.current === expected && !replayOpenRef.current)
                setError(failure.message);
            });
          }
          issuedGame = requiresNnCpuGuard(expected.session)
            ? (cpuGameRef.current ?? undefined)
            : undefined;
          const observation = await observeGame(
            expected.session.state,
            expected.session.state.currentPlayer,
          );
          timing?.inputReady();
          if (controller.signal.aborted || replayOpenRef.current || activeRef.current !== expected)
            return;
          if (
            requiresNnCpuGuard(expected.session) &&
            (!cpuGameRef.current || getActiveCpuGame() !== cpuGameRef.current)
          ) {
            if (cpuGameRef.current) discardGuard(cpuGameRef.current);
            throw new Error('この局面からNN CPUの履歴を明示的に再開してください。');
          }
          const callback = await notifyGuard(
            expected.session,
            observation,
            controller.signal,
            timing,
          );
          if (controller.signal.aborted || replayOpenRef.current || activeRef.current !== expected)
            return;
          const decision = await chooseCpu(observation, controller.signal, selection, callback);
          if (controller.signal.aborted || replayOpenRef.current || activeRef.current !== expected)
            return;
          const legal = [...expected.snapshot.choices, ...expected.snapshot.moves].some(
            (choice) =>
              !choice.disabled && JSON.stringify(choice.move) === JSON.stringify(decision.move),
          );
          if (!legal) throw new Error('CPUが合法でない操作を返しました。');
          timing?.legal();
          const snapshot = await applyMove(expected.session.state, decision.move);
          if (controller.signal.aborted || replayOpenRef.current || activeRef.current !== expected)
            return;
          complete = true;
          if (selection.mode === 'nn' && (decision as GuardedCpuDecision).trade?.overridden)
            setNotice('Tradeの反復を避けて終了しました。');
          updateSession(
            {
              ...expected.session,
              state: snapshot.state,
              history: [...expected.session.history, expected.session.state].slice(-60),
            },
            snapshot,
          );
        } catch (failure) {
          timing?.failed(failure);
          if (controller.signal.aborted || replayOpenRef.current || activeRef.current !== expected)
            return;
          setCpuPaused(true);
          setError(
            failure instanceof Error ? failure.message : 'CPUの操作を実行できませんでした。',
          );
        } finally {
          timing?.failed(
            controller.signal.aborted ? new DOMException('Cancelled', 'AbortError') : undefined,
          );
          if (!complete && issuedGame) discardGuard(issuedGame);
          if (cpuAbortRef.current === controller) setPreparingCpuModel(false);
        }
      })();
    }, 20);
    return () => {
      clearTimeout(timer);
      timing?.failed(new DOMException('Cancelled', 'AbortError'));
      controller.abort();
      if (!complete && issuedGame) discardGuard(issuedGame);
      if (cpuAbortRef.current === controller) {
        cpuAbortRef.current = null;
        setPreparingCpuModel(false);
      }
    };
  }, [active, cpuTurn, cpuPaused, busy, confirmReset, replayOpen]);
  function start() {
    void run(async (signal) => {
      const payload = modelRef.current;
      if (cpuMode === 'nn') {
        if (
          (count !== 3 && count !== 4) ||
          additionalBuildings ||
          tribes ||
          prophecies ||
          quickActions
        )
          throw new Error('NN CPUは拡張なしの基本3・4人対局で利用できます。');
        if (!payload) throw new Error('NNモデルを読み込んでください。');
      }
      const snapshot = await createGame(
        names.slice(0, count).map((n, i) => n.trim() || `プレイヤー ${i + 1}`),
        undefined,
        { additionalBuildings, tribes, prophecies, quickActions: quickActions || count === 5 },
      );
      if (cpuMode === 'nn' && !supportsNnCpu(snapshot.state))
        throw new Error('NN CPUの対象外の対局です。');
      const nextSession: Session = {
        state: snapshot.state,
        history: [],
        controllers: controllers.slice(0, count),
        ...(cpuMode === 'nn' && payload
          ? { cpuPolicy: { mode: 'nn' as const, modelChecksum: payload.metadata.checksum } }
          : {}),
      };
      await initializeGuard(nextSession, signal);
      if (signal.aborted) return;
      updateSession(nextSession, snapshot);
      setCpuPaused(false);
      setView('board');
    });
  }
  function undo() {
    if (!session?.history.length) return;
    cpuAbortRef.current?.abort();
    setCpuPaused(true);
    const state = session.history.at(-1);
    if (state)
      void run(async (signal) => {
        const snapshot = await inspectGame(state);
        const nextSession = {
          ...session,
          state: snapshot.state,
          history: session.history.slice(0, -1),
        };
        const missing = missingModel(nextSession);
        if (!missing) await initializeGuard(nextSession, signal);
        else if (cpuGameRef.current) discardGuard(cpuGameRef.current);
        if (signal.aborted) return;
        updateSession(nextSession, snapshot);
      });
  }
  function resume() {
    if (!saved) return;
    void run(async (signal) => {
      const snapshot = await inspectGame(saved.state);
      const nextSession = { ...saved, state: snapshot.state };
      const missing = missingModel(nextSession);
      if (!missing) await initializeGuard(nextSession, signal);
      else if (cpuGameRef.current) discardGuard(cpuGameRef.current);
      if (signal.aborted) return;
      const nextActive = { session: nextSession, snapshot };
      activeRef.current = nextActive;
      setActive(nextActive);
      setCpuPaused(!!missing);
      if (missing) setNotice(missing);
      setView('board');
    });
  }
  async function exportSave() {
    const data = session ?? saved;
    if (!data || saving) return;
    setSaving(true);
    try {
      const written = await saveGameFile(
        JSON.stringify(data, null, 2),
        `tzolkin-day-${data.state.round}.json`,
      );
      if (written) setNotice('対局の保存ファイルを書き出しました。');
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存ファイルを書き出せませんでした。');
    } finally {
      setSaving(false);
    }
  }
  async function importSave(file?: File) {
    if (!file) return;
    await run(async (signal) => {
      try {
        if (file.size > 5_000_000) throw new Error('保存ファイルが大きすぎます（上限5MB）。');
        const imported = await parseSession(await file.text());
        const snapshot = await inspectGame(imported.state);
        const nextSession = { ...imported, state: snapshot.state };
        const missing = missingModel(nextSession);
        if (!missing) await initializeGuard(nextSession, signal);
        else if (cpuGameRef.current) discardGuard(cpuGameRef.current);
        if (signal.aborted) return;
        updateSession(nextSession, snapshot);
        setCpuPaused(!!missing);
        setView('board');
        setNotice(missing || '保存した対局を読み込みました。');
      } finally {
        if (importInput.current) importInput.current.value = '';
      }
    });
  }
  const common = (
    <>
      <input
        ref={modelInput}
        type="file"
        accept="application/json,.json"
        aria-label="NNモデルファイル"
        className="visually-hidden"
        tabIndex={-1}
        onChange={(e) => void importModel(e.target.files?.[0])}
      />
      <input
        ref={importInput}
        type="file"
        accept="application/json,.json"
        aria-label="対局の保存ファイル"
        className="visually-hidden"
        tabIndex={-1}
        onChange={(e) => void importSave(e.target.files?.[0])}
      />
      {error && (
        <div className="error-message" role="alert">
          {error}
          <button aria-label="エラーを閉じる" onClick={() => setError('')}>
            ×
          </button>
        </div>
      )}
      {notice && (
        <div className="notice-message" role="status">
          {notice}
          <button aria-label="通知を閉じる" onClick={() => setNotice('')}>
            ×
          </button>
        </div>
      )}
    </>
  );
  if (replayOpen) return <ReplayViewer onClose={closeReplay} />;
  if (!game || !actor)
    return (
      <main className="welcome">
        {common}
        <section className="welcome-art">
          <span className="brand-mark">
            TZOLK’IN <span>THE MAYAN CALENDAR</span>
          </span>
          <CalendarArt />
          <div className="welcome-caption">
            <span className="eyebrow">時を育て、文明を築く</span>
            <h1>
              歯車に、
              <br />
              未来を託す。
            </h1>
            <p>
              時が進むたび、できることが増えていく。
              <br />
              マヤの暦を巡る、27日間の戦略。
            </p>
          </div>
          <span className="welcome-footnote">
            2–5 PLAYERS <i /> LOCAL MULTIPLAYER
          </span>
        </section>
        <section className="welcome-form">
          <div className="welcome-form-inner">
            <span className="eyebrow">新しい暦をはじめる</span>
            <h2>
              ツォルキン<span>マヤの暦</span>
            </h2>
            <p className="muted">一つの画面で交代して遊ぶか、CPUと対戦できます。</p>
            <fieldset className="player-count">
              <legend>プレイヤー数</legend>
              {[2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  className={count === n ? 'selected' : ''}
                  aria-pressed={count === n}
                  disabled={n === 5 && !quickActions}
                  title={
                    n === 5 && !quickActions
                      ? '5人対局はクイックアクションを選ぶと遊べます'
                      : undefined
                  }
                  aria-describedby={n === 5 ? 'five-player-note' : undefined}
                  onClick={() => setCount(n)}
                >
                  {n}人
                </button>
              ))}
            </fieldset>
            <p className="player-count-note" id="five-player-note">
              5人で遊ぶときは、クイックアクションを選んでください。
            </p>
            <div className="name-fields">
              {names.slice(0, count).map((name, i) => (
                <div className="seat-fields" key={i}>
                  <label key={i}>
                    <span className={`player-dot player-dot-${i}`} />
                    <span>プレイヤー {i + 1}</span>
                    <input
                      value={name}
                      maxLength={24}
                      onChange={(e) =>
                        setNames(names.map((old, j) => (j === i ? e.target.value : old)))
                      }
                    />
                  </label>
                  <label className="controller-field">
                    <span>操作</span>
                    <select
                      aria-label={`プレイヤー ${i + 1}の操作`}
                      value={controllers[i]}
                      onChange={(event) =>
                        setControllers(
                          controllers.map((old, index) =>
                            index === i ? (event.target.value as Controller) : old,
                          ),
                        )
                      }
                    >
                      <option value="human">人間</option>
                      <option value="cpu">CPU</option>
                    </select>
                  </label>
                </div>
              ))}
            </div>
            <fieldset className="nn-model-controls" disabled={loading || busy}>
              <legend>CPUの種類</legend>
              <label>
                <span>CPUの判断方法</span>
                <select
                  aria-label="CPUの判断方法"
                  value={cpuMode}
                  onChange={(e) => setCpuMode(e.target.value as 'heuristic' | 'nn')}
                >
                  <option value="heuristic">標準CPU</option>
                  <option value="nn">NNモデル</option>
                </select>
              </label>
              <button className="text-button" onClick={() => modelInput.current?.click()}>
                NNモデルを読み込む
              </button>
              <p>{modelMetadata ? 'NNモデルを選択済み' : 'NNモデルは未選択です。'}</p>
              {modelMetadata && (
                <code title={modelMetadata.checksum}>{modelMetadata.checksum.slice(0, 12)}</code>
              )}
              {cpuMode === 'nn' && (
                <p>拡張なしの基本3・4人対局で使えます。モデル本体は保存されません。</p>
              )}
            </fieldset>
            <fieldset className="expansion-options">
              <legend>
                拡張ルール <span>好きな組み合わせで追加できます</span>
              </legend>
              <label>
                <input
                  type="checkbox"
                  checked={tribes}
                  onChange={(e) => setTribes(e.target.checked)}
                />
                <span>
                  部族<small>2枚から1枚を選び、それぞれの特殊能力を使います。</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={prophecies}
                  onChange={(e) => setProphecies(e.target.checked)}
                />
                <span>
                  予言<small>公開される3つの災厄に備え、食料日に追加得点を狙います。</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={quickActions}
                  onChange={(e) => {
                    setQuickActions(e.target.checked);
                    if (!e.target.checked && count === 5) setCount(4);
                  }}
                />
                <span>
                  クイックアクション・5人対局
                  <small>その手番で実行できる配置先を追加します。2〜4人でも使えます。</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={additionalBuildings}
                  onChange={(e) => setAdditionalBuildings(e.target.checked)}
                />
                <span>
                  追加建物8枚を混ぜる<small>各時代に4枚ずつ、新しい効果の建物を加えます。</small>
                </span>
              </label>
            </fieldset>
            <button
              className="primary-button start-button"
              onClick={start}
              disabled={
                loading ||
                busy ||
                (cpuMode === 'nn' &&
                  (!modelMetadata ||
                    (count !== 3 && count !== 4) ||
                    additionalBuildings ||
                    tribes ||
                    prophecies ||
                    quickActions))
              }
            >
              対局をはじめる
              <Icon name="arrow" />
            </button>
            {startupError && (
              <button className="resume-button" onClick={retryRead}>
                保存した対局の読み込みを再試行
              </button>
            )}
            {saved && (
              <button className="resume-button" onClick={resume} disabled={loading || busy}>
                保存した対局を続ける
                <span>
                  第{saved.state.round}日 · {saved.state.players.length}人
                  {saved.cpuPolicy ? ' · NN CPU' : ''}
                </span>
              </button>
            )}
            <button
              className="text-button import-start"
              disabled={loading || busy}
              onClick={() => importInput.current?.click()}
            >
              <Icon name="save" size={16} />
              保存ファイルを読み込む
            </button>
            <button
              className="text-button import-start"
              disabled={loading || busy}
              onClick={openReplay}
            >
              公開リプレイを読み込む
            </button>
            <div className="welcome-notes">
              <p>拡張を選ばなければ基本ゲームで遊べます。進行は自動保存されます。</p>
              <p>Daniele Tascini & Simone Lucianiによるボードゲームの非公式実装。</p>
            </div>
          </div>
        </section>
      </main>
    );
  const nextFood = [8, 14, 21, 27].find((day) => !game.foodDays.includes(day));
  return (
    <div className="game-app" aria-busy={busy}>
      {common}
      <header className="app-header">
        <div className="brand">
          <Icon name="gear" size={29} />
          <div>
            TZOLK’IN<span>マヤの暦</span>
          </div>
        </div>
        <div className="header-day">
          <span className="eyebrow">{game.age === 1 ? '第一の時代' : '第二の時代'}</span>
          <b>
            第 {game.round} 日 <small>/ 27</small>
          </b>
        </div>
        <div className="header-food">
          <Icon name="corn" />
          <span>
            {nextFood ? (
              <>
                {nextFood <= game.round ? (
                  '今日は食料日'
                ) : (
                  <>
                    食料日まで <b>{nextFood - game.round}</b> 日
                  </>
                )}
              </>
            ) : (
              '最終得点を集計済み'
            )}
          </span>
        </div>
        <div className="header-tools">
          <button
            title="公開リプレイを読み込む"
            aria-label="公開リプレイを読み込む"
            onClick={openReplay}
            disabled={busy}
          >
            ▷
          </button>
          {session && controllersFor(session).includes('cpu') && game.phase !== 'finished' && (
            <button
              className="cpu-toggle"
              aria-label={cpuPaused ? 'CPUを再開' : 'CPUを一時停止'}
              title={cpuPaused ? 'CPUを再開' : 'CPUを一時停止'}
              onClick={toggleCpu}
              disabled={busy}
            >
              {cpuPaused ? '▶' : 'Ⅱ'}
            </button>
          )}
          <button
            title="1つ戻す"
            aria-label="1つ戻す"
            disabled={!session?.history.length || busy}
            onClick={undo}
          >
            <Icon name="undo" size={18} />
          </button>
          <button
            title="保存ファイルを書き出す"
            aria-label="保存ファイルを書き出す"
            onClick={() => void exportSave()}
            disabled={saving || busy}
          >
            <Icon name="save" size={18} />
          </button>
          <button title="新しい対局" aria-label="新しい対局" onClick={openReset} disabled={busy}>
            <Icon name="sun" size={18} />
          </button>
        </div>
      </header>
      {session && modelMetadata && requiresNnCpuGuard(session) && (
        <section className="nn-session-panel" aria-label="NN CPUのモデル">
          <div>
            <b>NN CPU</b>
            <span>
              {guardMetadata
                ? 'Tradeの反復を避ける判断を使用します。'
                : 'この局面からの再開が必要です。'}
            </span>
            <span>
              {preparingCpuModel
                ? 'モデルを準備しています。'
                : modelMetadata.checksum ===
                    (session.cpuPolicy?.modelChecksum ?? modelMetadata.checksum)
                  ? 'この対局のモデルを選択済み'
                  : '同じモデルの再読み込みが必要です。'}
            </span>
            <code title={session.cpuPolicy?.modelChecksum ?? modelMetadata.checksum}>
              {(session.cpuPolicy?.modelChecksum ?? modelMetadata.checksum).slice(0, 12)}
            </code>
          </div>
          <button
            className="text-button"
            onClick={() => modelInput.current?.click()}
            disabled={busy}
          >
            同じNNモデルを読み込む
          </button>
          {!guardMetadata && (
            <button className="text-button" onClick={toggleCpu} disabled={busy}>
              この局面からCPUを再開
            </button>
          )}
        </section>
      )}
      <div className="day-track" aria-label="27日間の暦">
        {Array.from({ length: 27 }, (_, i) => i + 1).map((day) => (
          <span
            key={day}
            className={`${day < game.round ? 'past' : ''} ${day === game.round ? 'today' : ''} ${[8, 14, 21, 27].includes(day) ? 'food-day' : ''}`}
            title={`${day}日${[8, 14, 21, 27].includes(day) ? '・食料日' : ''}`}
          >
            {day === game.round || [8, 14, 21, 27].includes(day) ? day : '·'}
          </span>
        ))}
      </div>
      <div className="game-layout" inert={busy || (cpuTurn && !cpuPaused)}>
        <PlayersSidebar
          game={game}
          availableWorkers={active!.snapshot.availableWorkers}
          expansionCatalog={active!.snapshot.expansionCatalog}
          controllers={controllersFor(session!)}
        />
        <GameBoardView
          game={game}
          costs={active!.snapshot.placementCosts}
          moves={moves}
          play={play}
          view={view}
          setView={setView}
          expansionCatalog={active!.snapshot.expansionCatalog}
        />
        <ActionsSidebar
          key={game.phase === 'setup' ? `setup-${game.currentPlayer}` : 'playing'}
          game={game}
          choices={choices}
          moves={moves}
          play={play}
          autosaveAvailable={autosaveAvailable}
          choicesRef={choicesRef}
          showRules={() => setView('rules')}
          newGame={openReset}
          cpuTurn={cpuTurn}
          cpuPaused={cpuPaused}
        />
      </div>
      {confirmReset && (
        <ResetDialog
          returnFocus={resetOpener}
          onCancel={() => setConfirmReset(false)}
          onConfirm={() => {
            cpuAbortRef.current?.abort();
            if (cpuGameRef.current) discardGuard(cpuGameRef.current);
            activeRef.current = null;
            setActive(null);
            setSaved(null);
            try {
              localStorage.removeItem(SAVE_KEY);
            } catch {
              /* Storage is optional. */
            }
            setConfirmReset(false);
            setError('');
            setNotice('');
          }}
        />
      )}
    </div>
  );
}

export default App;

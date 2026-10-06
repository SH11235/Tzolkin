import { useEffect, useRef, useState } from 'react';
import { applyMove, createGame, getAvailableMoves, getChoices } from './game/engine';
import type { GameMove } from './game/types';
import { CalendarArt, Icon } from './ui/Icons';
import { PlayersSidebar } from './ui/PlayersSidebar';
import { GameBoardView, type View } from './ui/GameBoardView';
import { ActionsSidebar } from './ui/ActionsSidebar';
import { SAVE_KEY, parseSession, readSession, type Session } from './game/storage';
import { ResetDialog } from './ui/ResetDialog';
import { saveGameFile } from './game/files';
import './App.css';

function App() {
  const [saved, setSaved] = useState<Session | null>(readSession);
  const [session, setSession] = useState<Session | null>(null);
  const [view, setView] = useState<View>('board');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [autosaveAvailable, setAutosaveAvailable] = useState(true);
  const [count, setCount] = useState(2);
  const [additionalBuildings, setAdditionalBuildings] = useState(false);
  const [saving, setSaving] = useState(false);
  const [names, setNames] = useState(['翡翠の民', '黄金の民', '珊瑚の民', '藍の民']);
  const [confirmReset, setConfirmReset] = useState(false);
  const importInput = useRef<HTMLInputElement>(null);
  const choicesRef = useRef<HTMLHeadingElement>(null);
  const [resetOpener, setResetOpener] = useState<HTMLElement | null>(null);
  const game = session?.state;
  const actor = game?.players[game.currentPlayer];
  const choices = game ? getChoices(game) : [];
  const moves = game?.phase === 'playing' && !game.pending ? getAvailableMoves(game) : [];
  function openReset() {
    setResetOpener(document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setConfirmReset(true);
  }
  function updateSession(next: Session) {
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(next));
      setSaved(next);
      setAutosaveAvailable(true);
    } catch {
      setAutosaveAvailable(false);
      setNotice('自動保存を利用できません。保存ファイルを書き出して対局を残せます。');
    }
    setSession(next);
  }
  useEffect(() => {
    if (game?.pending || game?.phase === 'setup') choicesRef.current?.focus();
  }, [game?.pending, game?.phase, game?.currentPlayer]);
  function play(move: GameMove) {
    if (!session) return;
    try {
      updateSession({
        state: applyMove(session.state, move),
        history: [...session.history, session.state].slice(-60),
      });
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'アクションを実行できませんでした。');
    }
  }
  function start() {
    try {
      updateSession({
        state: createGame(
          names.slice(0, count).map((n, i) => n.trim() || `プレイヤー ${i + 1}`),
          undefined,
          { additionalBuildings },
        ),
        history: [],
      });
      setView('board');
      setError('');
    } catch (e) {
      setError(e instanceof Error ? e.message : '対局を開始できませんでした。');
    }
  }
  function undo() {
    if (!session?.history.length) return;
    const state = session.history.at(-1);
    if (state) {
      updateSession({ state, history: session.history.slice(0, -1) });
      setError('');
    }
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
    try {
      if (file.size > 5_000_000) throw new Error('保存ファイルが大きすぎます（上限5MB）。');
      updateSession(parseSession(await file.text()));
      setView('board');
      setError('');
      setNotice('保存した対局を読み込みました。');
    } catch (e) {
      setError(e instanceof Error ? e.message : '保存ファイルを読み込めませんでした。');
    } finally {
      if (importInput.current) importInput.current.value = '';
    }
  }
  const common = (
    <>
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
            2–4 PLAYERS <i /> LOCAL MULTIPLAYER
          </span>
        </section>
        <section className="welcome-form">
          <div className="welcome-form-inner">
            <span className="eyebrow">新しい暦をはじめる</span>
            <h2>
              ツォルキン<span>マヤの暦</span>
            </h2>
            <p className="muted">一つの画面を囲んで、交代で遊べます。</p>
            <fieldset className="player-count">
              <legend>プレイヤー数</legend>
              {[2, 3, 4].map((n) => (
                <button
                  key={n}
                  className={count === n ? 'selected' : ''}
                  aria-pressed={count === n}
                  onClick={() => setCount(n)}
                >
                  {n}人
                </button>
              ))}
            </fieldset>
            <div className="name-fields">
              {names.slice(0, count).map((name, i) => (
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
              ))}
            </div>
            <label className="extra-buildings-option">
              <input
                type="checkbox"
                checked={additionalBuildings}
                onChange={(e) => setAdditionalBuildings(e.target.checked)}
              />
              追加建物8枚を混ぜる
            </label>
            <button className="primary-button start-button" onClick={start}>
              対局をはじめる
              <Icon name="arrow" />
            </button>
            {saved && (
              <button
                className="resume-button"
                onClick={() => {
                  setSession(saved);
                  setView('board');
                }}
              >
                保存した対局を続ける
                <span>
                  第{saved.state.round}日 · {saved.state.players.length}人
                </span>
              </button>
            )}
            <button
              className="text-button import-start"
              onClick={() => importInput.current?.click()}
            >
              <Icon name="save" size={16} />
              保存ファイルを読み込む
            </button>
            <div className="welcome-notes">
              <p>基本ゲームのルールで遊びます。進行は自動保存されます。</p>
              <p>Daniele Tascini & Simone Lucianiによるボードゲームの非公式実装。</p>
            </div>
          </div>
        </section>
      </main>
    );
  const nextFood = [8, 14, 21, 27].find((day) => !game.foodDays.includes(day));
  return (
    <div className="game-app">
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
            title="1つ戻す"
            aria-label="1つ戻す"
            disabled={!session?.history.length}
            onClick={undo}
          >
            <Icon name="undo" size={18} />
          </button>
          <button
            title="保存ファイルを書き出す"
            aria-label="保存ファイルを書き出す"
            onClick={() => void exportSave()}
            disabled={saving}
          >
            <Icon name="save" size={18} />
          </button>
          <button title="新しい対局" aria-label="新しい対局" onClick={openReset}>
            <Icon name="sun" size={18} />
          </button>
        </div>
      </header>
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
      <div className="game-layout">
        <PlayersSidebar game={game} />
        <GameBoardView game={game} moves={moves} play={play} view={view} setView={setView} />
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
        />
      </div>
      {confirmReset && (
        <ResetDialog
          returnFocus={resetOpener}
          onCancel={() => setConfirmReset(false)}
          onConfirm={() => {
            setSession(null);
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

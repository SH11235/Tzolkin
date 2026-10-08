import { useEffect, useRef, useState } from 'react';
import { loadPublicReplay, type PublicReplayReport } from '../game/publicReplay';
import { chooseCpu, type CpuDecision } from '../game/cpu';
import { GameBoardView, type View } from './GameBoardView';
import { PlayersSidebar } from './PlayersSidebar';
import { formatScore } from './content';
import './ReplayViewer.css';

const missingReasonLabels: Record<string, string> = {
  'record ends before the game finished': '記録が終局より前で終わっています。',
  'observed terminal scores/ranks have not been matched':
    '元対局の最終得点・順位を照合できていません。',
  'full observed initial/food-day board checkpoints are not all covered':
    '初期盤面と各食料日の盤面に、照合できていない項目があります。',
};

export function ReplayViewer({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<PublicReplayReport | null>(null);
  const [filename, setFilename] = useState('');
  const [position, setPosition] = useState(0);
  const [view, setView] = useState<View>('board');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [cpuBusy, setCpuBusy] = useState(false);
  const [cpuError, setCpuError] = useState('');
  const [cpuCandidate, setCpuCandidate] = useState<{ decision: CpuDecision; label: string } | null>(
    null,
  );
  const input = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  const cpuGeneration = useRef(0);
  const cpuController = useRef<AbortController | null>(null);
  useEffect(
    () => () => {
      generation.current++;
      cpuGeneration.current++;
      cpuController.current?.abort();
    },
    [],
  );

  function cancelCpu() {
    cpuGeneration.current++;
    cpuController.current?.abort();
    cpuController.current = null;
    setCpuBusy(false);
    setCpuCandidate(null);
    setCpuError('');
  }
  function navigate(next: number) {
    cancelCpu();
    setPosition(next);
  }
  function close() {
    generation.current++;
    cancelCpu();
    onClose();
  }

  async function importReplay(file?: File) {
    if (!file) return;
    cancelCpu();
    const token = ++generation.current;
    setBusy(true);
    setError('');
    try {
      const next = await loadPublicReplay(file);
      if (token !== generation.current) return;
      setReport(next);
      setFilename(file.name);
      setPosition(0);
      setView('board');
    } catch (failure) {
      if (token !== generation.current) return;
      setError(failure instanceof Error ? failure.message : 'リプレイを読み込めませんでした。');
    } finally {
      if (token === generation.current) {
        setBusy(false);
        if (input.current) input.current.value = '';
      }
    }
  }
  const frame = report?.frames[position];
  const last = (report?.frames.length ?? 1) - 1;
  const game = frame?.snapshot.state;
  const calculatedToEnd = report?.frames.at(-1)?.snapshot.state.phase === 'finished';
  async function analyzeCpu() {
    const observation = frame?.observation;
    if (!frame || !observation || busy || cpuBusy || cpuController.current) return;
    const token = ++cpuGeneration.current;
    const controller = new AbortController();
    cpuController.current = controller;
    setCpuBusy(true);
    setCpuError('');
    setCpuCandidate(null);
    try {
      // The worker gets this frame's allowlisted observation, never the record or later frames.
      const decision = await chooseCpu(observation, controller.signal);
      if (token !== cpuGeneration.current || controller.signal.aborted) return;
      const choice = [...frame.snapshot.choices, ...frame.snapshot.moves].find(
        (option) =>
          !option.disabled && JSON.stringify(option.move) === JSON.stringify(decision.move),
      );
      if (
        decision.actor !== observation.actor ||
        decision.observationKey !== observation.observationKey ||
        !choice
      )
        throw new Error('CPUの候補がこの局面の合法手と一致しません。');
      setCpuCandidate({ decision, label: choice.label });
    } catch (failure) {
      if (token !== cpuGeneration.current || controller.signal.aborted) return;
      setCpuError(failure instanceof Error ? failure.message : 'CPUの候補を確認できませんでした。');
    } finally {
      if (token === cpuGeneration.current) {
        cpuController.current = null;
        setCpuBusy(false);
      }
    }
  }
  return (
    <div className="game-app replay-app" aria-busy={busy}>
      <header className="replay-header">
        <div>
          <span className="eyebrow">公開対局の研究</span>
          <h1>リプレイ</h1>
        </div>
        <div className="replay-header-tools">
          <button disabled={busy} onClick={() => input.current?.click()}>
            {report ? '別のリプレイを読み込む' : '公開リプレイを読み込む'}
          </button>
          <button onClick={close}>対局画面に戻る</button>
        </div>
      </header>
      <input
        ref={input}
        type="file"
        accept="application/json,.json"
        className="visually-hidden"
        aria-label="公開リプレイのJSONファイル"
        tabIndex={-1}
        disabled={busy}
        onChange={(event) => void importReplay(event.target.files?.[0])}
      />
      {error && (
        <p className="error-message replay-error" role="alert">
          {error}
        </p>
      )}
      {busy && (
        <p className="replay-loading" role="status">
          全操作をルールエンジンで検証しています。
        </p>
      )}
      {!report && !busy && (
        <main className="replay-intro">
          <h2>検証済みの盤面を順に見る</h2>
          <p>
            初期盤面と有効操作を記録した公開リプレイJSONを選んでください。合法手と状態遷移を検証してから表示します。
          </p>
          <p>
            途中までの記録は、その範囲だけ再生できます。取得したテキストログには初期盤面などの補足が必要です。
          </p>
          <p>再生中は盤面を操作できません。保存した対局はこの画面の操作では変更されません。</p>
        </main>
      )}
      {report && frame && game && (
        <>
          <section className="replay-controls" aria-label="リプレイの操作">
            <div className="replay-summary">
              <b>{filename}</b>
              <span className={`replay-status replay-status-${report.status}`}>
                {report.status === 'complete'
                  ? '終局まで検証済み・最終結果一致'
                  : calculatedToEnd
                    ? report.terminalDisplayComparison
                      ? '終局まで計算済み・表示得点のみ照合'
                      : '終局まで計算済み・元対局の結果は未照合'
                    : '途中までの記録・終局未検証'}
              </span>
              <p>
                合法手・状態遷移 {report.verifiedSteps}手検証 / 盤面照合{' '}
                {report.checkpointsVerified}件
              </p>
              <p>
                原対局の盤面照合：
                {report.sourceCoverage?.complete === true
                  ? '必須の盤面をすべて照合済み'
                  : '追加の照合が必要'}
              </p>
              <p>学習用データ：{report.trainingReady === true ? '使用可' : '未承認'}</p>
            </div>
            <div className="replay-navigation">
              <button disabled={busy || position === 0} onClick={() => navigate(0)}>
                先頭
              </button>
              <button
                disabled={busy || position === 0}
                onClick={() => navigate(Math.max(0, position - 1))}
              >
                前
              </button>
              <label>
                再生位置{' '}
                <output>
                  {position} / {last}
                </output>
                <input
                  type="range"
                  aria-label="リプレイの再生位置"
                  min={0}
                  max={last}
                  value={position}
                  disabled={busy || last === 0}
                  onChange={(event) => navigate(Number(event.target.value))}
                />
              </label>
              <button
                disabled={busy || position === last}
                onClick={() => navigate(Math.min(last, position + 1))}
              >
                次
              </button>
              <button disabled={busy || position === last} onClick={() => navigate(last)}>
                {calculatedToEnd ? '終局' : '記録の末尾'}
              </button>
            </div>
            <p className="replay-position" aria-live="polite">
              第{game.round}日 ·{' '}
              {game.phase === 'finished' ? '終局' : game.players[game.currentPlayer]?.name}
              {frame.sourceActionIds.length > 0 &&
                ` · 元ログの行動 ${frame.sourceActionIds.join('・')}`}
            </p>
            <p className="replay-readonly">
              閲覧専用です。未公開の初期財産候補と山札の順序は不明のまま扱います。
            </p>
            {game.phase === 'finished' &&
              !report.terminalMatched &&
              !report.terminalDisplayComparison && (
                <p className="replay-readonly">
                  以下の得点・順位はルールエンジンの計算結果です。元対局の終局結果とは照合できていません。
                </p>
              )}
            {game.phase === 'finished' && report.terminalDisplayComparison && (
              <section className="replay-display-comparison" aria-label="終局の表示得点との比較">
                <h2>公式得点と元対局の表示</h2>
                <p>
                  公式得点の小数部を切り捨てた値と、元対局の表示得点・順位を照合しました。
                  公式得点は小数部を保持します。元対局の内部計算や丸め方を確認したものではありません。
                </p>
                {!report.terminalMatched && (
                  <p>
                    得点そのものの厳密な一致は未確認です。記録は部分検証のままで、学習用データは未承認です。
                  </p>
                )}
                <div className="table-scroll">
                  <table aria-label="公式得点と表示得点">
                    <thead>
                      <tr>
                        <th>プレイヤー</th>
                        <th>公式得点</th>
                        <th>元対局の表示</th>
                        <th>差</th>
                        <th>順位（公式／元対局）</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.terminalDisplayComparison.scores.map((score) => (
                        <tr key={score.playerId}>
                          <th scope="row">{game.players[score.playerId]?.name}</th>
                          <td>{formatScore(score.nativeTotal)}</td>
                          <td>{formatScore(score.sourceTotal)}</td>
                          <td>{formatScore(score.difference)}</td>
                          <td>
                            {score.nativeRank}／{score.sourceRank}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <details>
                  <summary>表示得点の証拠</summary>
                  <p>{report.terminalDisplayComparison.source.reference}</p>
                  {report.terminalDisplayComparison.source.actionIds.length > 0 && (
                    <p>
                      元ログの行動 {report.terminalDisplayComparison.source.actionIds.join('・')}
                    </p>
                  )}
                </details>
              </section>
            )}
            <p className="replay-readonly">
              山札の残り：時代{game.age} {game.buildingDeckCount}枚
              {game.age === 1 && ` / 時代2 ${game.age2DeckCount}枚`}
            </p>
            {report.missingReasons.length > 0 && (
              <details className="replay-missing">
                <summary>検証できていない範囲</summary>
                <ul>
                  {report.missingReasons.map((reason, index) => (
                    <li key={index}>
                      {Object.hasOwn(missingReasonLabels, reason)
                        ? missingReasonLabels[reason]
                        : reason}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <section className="replay-cpu" aria-label="この局面のCPU候補">
              <button
                disabled={busy || cpuBusy || !frame.observation || game.phase === 'finished'}
                aria-label="この局面のCPU候補を確認"
                onClick={() => void analyzeCpu()}
              >
                {cpuBusy ? 'CPUが候補を検討中' : 'この局面のCPU候補を確認'}
              </button>
              {cpuCandidate && (
                <p className="replay-cpu-result" role="status">
                  CPU候補：{cpuCandidate.label}
                  <small>方策 {cpuCandidate.decision.policyVersion}</small>
                </p>
              )}
              {cpuError && <p role="alert">{cpuError}</p>}
            </section>
          </section>
          <div className="game-layout replay-layout">
            <PlayersSidebar game={game} availableWorkers={frame.snapshot.availableWorkers} />
            <GameBoardView
              game={game}
              costs={frame.snapshot.placementCosts}
              moves={[]}
              play={() => {}}
              view={view}
              setView={setView}
            />
          </div>
        </>
      )}
    </div>
  );
}

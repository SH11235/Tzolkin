import { useState, type CSSProperties, type RefObject } from 'react';
import { STARTING_WEALTH } from '../game/catalog';
import type { Choice, GameMove, GameState } from '../game/types';
import { Icon } from './Icons';
import { effectText, resourceText } from './content';
export function ActionsSidebar({
  game,
  choices,
  moves,
  play,
  choicesRef,
  autosaveAvailable,
  showRules,
  newGame,
  cpuTurn = false,
  cpuPaused = false,
}: {
  game: GameState;
  choices: Choice[];
  moves: Choice[];
  play: (move: GameMove) => void;
  autosaveAvailable: boolean;
  choicesRef: RefObject<HTMLHeadingElement | null>;
  showRules: () => void;
  newGame: () => void;
  cpuTurn?: boolean;
  cpuPaused?: boolean;
}) {
  const [revealed, setRevealed] = useState(false);
  const actor = game.players[game.currentPlayer]!;
  if (cpuTurn)
    return (
      <aside className="actions-sidebar" aria-label="手番のアクション">
        <div className="turn-heading" style={{ '--player-color': actor.color } as CSSProperties}>
          <span className="eyebrow">CPUの操作</span>
          <h2 ref={choicesRef} tabIndex={-1}>
            {actor.name}の番です
          </h2>
          <p role="status">
            {cpuPaused
              ? 'CPUを一時停止しています。上の再開ボタンで続けられます。'
              : 'CPUが考えています…'}
          </p>
        </div>
      </aside>
    );
  const choosingTribe = game.phase === 'setup' && !actor.tribe && !!actor.tribeOffer?.length;
  const setupHasTribes = !!actor.tribeOffer?.length;
  const actionTitle =
    game.phase === 'setup'
      ? (game.pending?.title ?? `${actor.name}の${choosingTribe ? '部族' : '初期資源'}`)
      : game.phase === 'finished'
        ? '暦が巡りました'
        : (game.pending?.title ?? `${actor.name}の手番`);
  if (game.phase === 'setup' && !revealed)
    return (
      <aside className="actions-sidebar" aria-label="手番のアクション">
        <div className="turn-heading" style={{ '--player-color': actor.color } as CSSProperties}>
          <span className="eyebrow">ゲームの準備</span>
          <h2 ref={choicesRef} tabIndex={-1}>
            {actor.name}の番です
          </h2>
          <p>
            画面を渡してください。{setupHasTribes ? '部族と初期資源' : '初期資源'}
            は、全員が選び終えるまで他の人に見せずに選びます。
          </p>
        </div>
        <button className="primary-button handoff-button" onClick={() => setRevealed(true)}>
          {setupHasTribes ? '自分の部族と初期資源を見る' : '自分の初期資源を見る'}
          <Icon name="arrow" />
        </button>
      </aside>
    );
  return (
    <aside className="actions-sidebar" aria-label="手番のアクション">
      <div className="turn-heading" style={{ '--player-color': actor.color } as CSSProperties}>
        <span className="eyebrow">
          {game.phase === 'setup'
            ? 'ゲームの準備'
            : game.phase === 'finished'
              ? '最終結果'
              : 'ただいまの手番'}
        </span>
        <h2 ref={choicesRef} tabIndex={-1}>
          {actionTitle}
        </h2>
        <p>
          {game.phase === 'setup'
            ? choosingTribe
              ? '2つの部族から1つを選びます。公開されている盤面と、下の初期資源を見て決めてください。'
              : game.pending
                ? '初期資源や部族の効果を選んでください。'
                : setupHasTribes
                  ? '配られた初期資源から選びます。部族の能力と組み合わせを確認して決定してください。'
                  : '4枚のうち2枚を選びます。組み合わせを確認して決定してください。'
            : game.pending
              ? '内容を確認して選択してください。'
              : game.turn.mode === 'none'
                ? actor.tribe
                  ? '配置か回収、または部族の能力を選びます。使える操作が表示されます。'
                  : '配置か回収を選びます。同じ手番で両方はできません。'
                : game.turn.mode === 'place'
                  ? actor.tribe
                    ? `${game.turn.count}人配置しました。配置先ごとの費用は盤面に表示されます。`
                    : `${game.turn.count}人配置しました。次の追加コストはコーン${game.turn.count}です。`
                  : `${game.turn.count}人回収しました。続けて回収できます。`}
        </p>
      </div>
      {game.phase === 'setup' && (
        <div className="wealth-offer">
          {actor.wealthOffer.map((id) => {
            const w = STARTING_WEALTH.find((t) => t.id === id);
            return (
              w && (
                <article key={id}>
                  <span>{w.name}</span>
                  <p>{resourceText(w.resources)}</p>
                  {w.effects.map((e, i) => (
                    <small key={i}>{effectText(e)}</small>
                  ))}
                </article>
              )
            );
          })}
        </div>
      )}
      <div className={`choices-list ${choosingTribe ? 'tribe-choices' : ''}`}>
        {choices.map((c) => (
          <ChoiceButton key={c.id} choice={c} play={play} />
        ))}
      </div>
      {game.pending && game.phase === 'playing' && (
        <div className="turn-actions">
          {moves
            .filter((choice) => choice.move.type === 'tribeAbility')
            .map((choice) => (
              <ChoiceButton key={choice.id} choice={choice} play={play} />
            ))}
        </div>
      )}
      {!game.pending && game.phase === 'playing' && (
        <>
          <div className="action-instruction">
            <Icon name="worker" size={27} />
            <p>配置は歯車の下のボタンから。回収は色のついたワーカーを選びます。</p>
          </div>
          <div className="turn-actions">
            {moves
              .filter((c) => !['place', 'remove', 'quickAction'].includes(c.move.type))
              .map((c) => (
                <ChoiceButton key={c.id} choice={c} play={play} />
              ))}
          </div>
        </>
      )}
      {game.phase === 'finished' && (
        <button className="primary-button" onClick={() => newGame()}>
          新しい対局をはじめる
          <Icon name="arrow" />
        </button>
      )}
      <div className="sidebar-tip">
        <span className="eyebrow">暦の知恵</span>
        <p>
          {game.phase === 'setup'
            ? 'コーンは配置にも食料にも必要です。初期資源の選び方が、最初の戦略になります。'
            : game.pending
              ? '技術の費用や建物の効果を確認してから選びましょう。最初の選択で、回収アクションの実行を見送ることもできます。'
              : 'ワーカーは毎日ひとつ先へ進みます。長く待つほど強いアクションを使えますが、歯車の端から落ちると戻されます。'}
        </p>
        <button className="text-button" onClick={() => showRules()}>
          <Icon name="book" size={16} />
          遊び方を見る
        </button>
      </div>
      <div className={`autosave ${autosaveAvailable ? '' : 'unavailable'}`}>
        <span />
        {autosaveAvailable ? '自動保存' : '端末への自動保存を利用できません'}
      </div>
    </aside>
  );
}
function ChoiceButton({ choice, play }: { choice: Choice; play: (move: GameMove) => void }) {
  return (
    <button
      className={`choice-button ${choice.move.type === 'endTurn' ? 'end-turn' : ''}`}
      disabled={choice.disabled}
      onClick={() => play(choice.move)}
    >
      <span>
        {choice.label}
        {choice.description && <small>{choice.description}</small>}
      </span>
      <Icon name="arrow" size={16} />
    </button>
  );
}

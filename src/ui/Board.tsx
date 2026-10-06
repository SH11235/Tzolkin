import type { CSSProperties } from 'react';
import { GEAR_IDS, type Choice, type GameMove, type GameState, type GearId } from '../game/types';
import { CalendarArt, Icon } from './Icons';
import { gearActions, gearNames, gearSubtitles } from './content';

export function Board({
  game,
  moves,
  play,
  costs,
}: {
  game: GameState;
  moves: Choice[];
  play: (move: GameMove) => void;
  costs: Record<GearId, number | null>;
}) {
  const actor = game.players[game.currentPlayer];
  return (
    <div className="board-grid">
      {GEAR_IDS.map((gear) => {
        const slots = game.gears[gear];
        const visible = gear === 'chichenItza' ? 11 : 8;
        const place = moves.find((c) => c.move.type === 'place' && c.move.gear === gear);
        return (
          <section className={`gear-card gear-${gear}`} key={gear} aria-label={gearNames[gear]}>
            <header>
              <span className="eyebrow">{gearSubtitles[gear]}</span>
              <h3>{gearNames[gear]}</h3>
            </header>
            <div className="small-gear">
              <svg
                viewBox="0 0 240 240"
                className="small-gear-ring"
                aria-hidden="true"
                style={
                  { '--angle': `${((game.round - 1) * 360) / slots.length}deg` } as CSSProperties
                }
              >
                <g fill="currentColor">
                  {slots.map((_, i) => (
                    <rect
                      key={i}
                      x="110"
                      y="1"
                      width="20"
                      height="27"
                      rx="2"
                      transform={`rotate(${(i * 360) / slots.length} 120 120)`}
                    />
                  ))}
                  <circle cx="120" cy="120" r="104" />
                </g>
                <circle cx="120" cy="120" r="93" className="gear-fill" />
                <circle
                  cx="120"
                  cy="120"
                  r="55"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1"
                />
              </svg>
              <div className="gear-center">
                <Icon
                  name={
                    gear === 'palenque'
                      ? 'corn'
                      : gear === 'yaxchilan'
                        ? 'stone'
                        : gear === 'tikal'
                          ? 'temple'
                          : gear === 'uxmal'
                            ? 'gold'
                            : 'skull'
                  }
                  size={32}
                />
                <span>{gearSubtitles[gear]}</span>
              </div>
              {slots.map((worker, position) => {
                const angle = (position / slots.length) * Math.PI * 2 - Math.PI / 2;
                const remove = moves.find(
                  (c) =>
                    c.move.type === 'remove' &&
                    c.move.gear === gear &&
                    c.move.position === position,
                );
                const owner = worker && !worker.dummy ? game.players[worker.playerId] : null;
                const action = gearActions[gear][position] ?? '歯車の裏側';
                const label = `${gearNames[gear]} ${position}：${action}${worker ? `、${worker.dummy ? 'ダミー' : owner?.name}のワーカー` : '、空き'}`;
                return (
                  <button
                    key={position}
                    className={`gear-slot ${position >= visible ? 'hidden-slot' : ''} ${worker ? 'occupied' : ''} ${remove ? 'can-remove' : ''} ${worker?.dummy ? 'dummy' : ''}`}
                    aria-label={label}
                    title={label}
                    disabled={!remove || remove.disabled}
                    onClick={() => remove && play(remove.move)}
                    style={
                      {
                        left: `${50 + Math.cos(angle) * 39}%`,
                        top: `${50 + Math.sin(angle) * 39}%`,
                        '--player-color': owner?.color ?? '#817d71',
                      } as CSSProperties
                    }
                  >
                    {worker ? <Icon name="worker" size={17} /> : position}
                    <span className="slot-number">{worker ? position : ''}</span>
                  </button>
                );
              })}
            </div>
            {gear === 'palenque' && (
              <div className="jungle-mini" aria-label="ジャングルの残りタイル">
                {[2, 3, 4, 5].map((p) => (
                  <span
                    key={p}
                    title={`${p}：コーン${game.jungle[p]?.corn ?? 0}枚・木材${game.jungle[p]?.wood ?? 0}枚`}
                  >
                    {p}
                    <Icon name="corn" size={12} />
                    {game.jungle[p]?.corn ?? 0}
                    {p > 2 && (
                      <>
                        <Icon name="wood" size={12} />
                        {game.jungle[p]?.wood ?? 0}
                      </>
                    )}
                  </span>
                ))}
              </div>
            )}
            {gear === 'chichenItza' && (
              <div className="skull-mini" aria-label="髑髏が置かれた場所">
                {game.skullSpaces.slice(1, 10).map((player, i) => (
                  <span
                    key={i}
                    className={player === null ? '' : 'filled'}
                    title={`${i + 1}：${player === null ? '空き' : game.players[player]?.name}`}
                  >
                    <Icon name="skull" size={12} />
                    {i + 1}
                  </span>
                ))}
              </div>
            )}
            <button
              className="place-button"
              disabled={!place || place.disabled}
              onClick={() => place && play(place.move)}
            >
              <Icon name="worker" size={15} />
              配置する{' '}
              <span>
                <Icon name="corn" size={13} />
                {costs[gear] !== null ? costs[gear] : '—'}
              </span>
            </button>
            <details className="action-reference">
              <summary>アクションを見る</summary>
              <ol>
                {gearActions[gear].slice(1, gear === 'chichenItza' ? 11 : 8).map((label, i) => (
                  <li key={i}>
                    <b>{i + 1}</b>
                    {label}
                  </li>
                ))}
              </ol>
            </details>
          </section>
        );
      })}
      <section className="calendar-card">
        <CalendarArt round={game.round} />
        <div className="calendar-round">
          <span className="eyebrow">マヤの暦</span>
          <strong>{game.round.toString().padStart(2, '0')}</strong>
          <span> / 27日</span>
        </div>
        <div className="first-player-space">
          <span className="eyebrow">スタートプレイヤー</span>
          <p>{game.players[game.firstPlayer]?.name}</p>
          <span>
            <Icon name="corn" size={16} /> {game.accumulatedCorn}
          </span>
          <button
            className="text-button"
            disabled={!moves.some((c) => c.move.type === 'firstPlayer' && !c.disabled)}
            onClick={() => play({ type: 'firstPlayer' })}
          >
            場所を取る
            <Icon name="arrow" size={15} />
          </button>
        </div>
        <p className="board-hint">
          {game.pending
            ? '右の選択を完了してください'
            : game.turn.mode === 'remove'
              ? `${actor?.name}のワーカーを選んで回収`
              : game.turn.mode === 'place'
                ? '続けて配置するか、手番を終了'
                : 'ワーカーを配置、または自分のワーカーを回収'}
        </p>
      </section>
    </div>
  );
}

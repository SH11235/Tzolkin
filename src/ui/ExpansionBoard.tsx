import type { CSSProperties } from 'react';
import type { Choice, ExpansionCatalog, GameMove, GameViewState } from '../game/types';
import { Icon } from './Icons';
import { formatScore } from './content';

export function ExpansionBoard({
  game,
  catalog,
  moves,
  play,
}: {
  game: GameViewState;
  catalog?: ExpansionCatalog;
  moves: Choice[];
  play: (move: GameMove) => void;
}) {
  const expansion = game.expansion;
  if (!expansion) return null;
  const quick = expansion.quickActions;
  const currentAction = catalog?.quickActions.find((action) => action.id === quick?.current);
  const placement = moves.find((choice) => choice.move.type === 'quickAction');
  const schedule = quick
    ? [
        { age: 1, tiles: quick.age1, firstDay: 1 },
        ...(game.foodDays.includes(8) ? [{ age: 2, tiles: quick.age2, firstDay: 15 }] : []),
      ]
    : [];
  return (
    <div className="expansion-board">
      {expansion.prophecies.length > 0 && (
        <section className="prophecy-board" aria-label="3つの予言">
          <header className="expansion-heading">
            <div>
              <span className="eyebrow">災厄に備える</span>
              <h2>三つの予言</h2>
            </div>
            <p>
              最初の食料日までは災厄はありません。予言は食料日の翌日から、次の食料日の終了まで続きます。
            </p>
          </header>
          <div className="prophecy-cards">
            {expansion.prophecies.map((id, index) => {
              const prophecy = catalog?.prophecies.find((item) => item.id === id);
              const active = game.phase !== 'finished' && expansion.activeProphecy === index;
              const completed = !active && game.foodDays.length > index + 1;
              return (
                <article key={id} className={`prophecy-card ${active ? 'active' : ''}`}>
                  <div className="prophecy-status">
                    <span>予言 {index + 1}</span>
                    <b>{completed ? '得点済み' : active ? '災厄が有効' : 'これから'}</b>
                  </div>
                  <h3>{prophecy?.name ?? id}</h3>
                  <p>{prophecy?.description}</p>
                  <div className="prophecy-scoring">
                    <span>食料日での得点条件</span>
                    <p>{prophecy?.scoring}</p>
                  </div>
                  {prophecy && (
                    <dl className="prophecy-bands" aria-label={`${prophecy.name}の得点表`}>
                      {prophecy.bands.map((band, bandIndex) => (
                        <div key={bandIndex}>
                          <dt>
                            {band.minimum === null
                              ? `${band.maximum}以下`
                              : band.maximum === null
                                ? `${band.minimum}以上`
                                : band.minimum === band.maximum
                                  ? band.minimum
                                  : `${band.minimum}〜${band.maximum}`}
                          </dt>
                          <dd>
                            {formatScore(band.points)}
                            <small>点</small>
                          </dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </article>
              );
            })}
          </div>
        </section>
      )}
      {quick && (
        <section className="quick-action-board" aria-label="クイックアクション">
          <div className="quick-action-intro">
            <span className="eyebrow">その手番で、すぐに実行</span>
            <h2>クイックアクション</h2>
            <h3>{currentAction?.name ?? quick.current}</h3>
            <p>{currentAction?.description}</p>
            <small>
              配置を終えたら実行します。技術の効果は適用されず、ワーカーは一日の終わりに手元へ戻ります。
            </small>
          </div>
          <div className="quick-action-controls">
            <div className="quick-action-spaces">
              {quick.spaces.map((playerId, index) => {
                const dummy = playerId !== null && playerId < 0;
                const player = playerId !== null && !dummy ? game.players[playerId] : null;
                return (
                  <div
                    key={index}
                    className={`quick-action-space ${playerId !== null ? 'occupied' : ''}`}
                    style={{ '--player-color': player?.color ?? '#817d71' } as CSSProperties}
                  >
                    <Icon name="worker" size={21} />
                    <b>場所 {index + 1}</b>
                    <span>{playerId === null ? '空き' : dummy ? 'ダミー' : player?.name}</span>
                    <small>
                      <Icon name="corn" size={12} /> 基本費用 1
                    </small>
                  </div>
                );
              })}
            </div>
            <button
              className="place-button"
              disabled={!placement || placement.disabled}
              onClick={() => placement && play(placement.move)}
            >
              <Icon name="worker" size={15} />
              <span>
                {placement?.label ?? 'クイックアクションに配置する'}
                <small>
                  {placement?.description ?? '配置する手番で、空いている場所を使います'}
                </small>
              </span>
              <Icon name="arrow" size={15} />
            </button>
          </div>
          <div className="quick-action-schedule" aria-label="公開されたクイックアクションの予定">
            {schedule.map(({ age, tiles, firstDay }) => (
              <section key={age} aria-label={`時代${age}のクイックアクション予定`}>
                <h3>時代 {age} の予定</h3>
                <ol>
                  {tiles.map((id, index) => {
                    const start = firstDay + index * 2;
                    const end = age === 2 && index === tiles.length - 1 ? 27 : start + 1;
                    const current = game.round >= start && game.round <= end;
                    return (
                      <li key={index} aria-current={current ? 'step' : undefined}>
                        <span>
                          {start}〜{end} 日
                        </span>
                        <b>
                          {catalog?.quickActions.find((action) => action.id === id)?.name ?? id}
                        </b>
                      </li>
                    );
                  })}
                </ol>
              </section>
            ))}
            {!game.foodDays.includes(8) && (
              <p>時代 2 の予定は最初の食料日の給食後に公開されます。</p>
            )}
          </div>
        </section>
      )}
    </div>
  );
}

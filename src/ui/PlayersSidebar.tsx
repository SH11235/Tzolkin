import type { CSSProperties } from 'react';
import { ALL_BUILDINGS as BUILDINGS, MONUMENTS, MONUMENT_DESCRIPTIONS } from '../game/catalog';
import {
  RESOURCE_IDS,
  TECHNOLOGY_IDS,
  TEMPLE_IDS,
  type ExpansionCatalog,
  type GameState,
} from '../game/types';
import { Icon } from './Icons';
import { effectText, formatScore, resourceNames, technologyNames, templeNames } from './content';
import type { Controller } from '../game/storage';
export function PlayersSidebar({
  game,
  availableWorkers,
  expansionCatalog,
  controllers,
}: {
  game: GameState;
  availableWorkers: number[];
  expansionCatalog?: ExpansionCatalog;
  controllers?: Controller[];
}) {
  return (
    <aside className="players-sidebar" aria-label="プレイヤーの状況">
      <div className="section-label">
        文明の記録<span>{game.players.length}人</span>
      </div>
      {game.players.map((player) => (
        <section
          key={player.id}
          className={`player-panel ${player.id === game.currentPlayer ? 'active' : ''}`}
          style={{ '--player-color': player.color } as CSSProperties}
        >
          <header>
            <span className="player-dot" />
            <h2>
              {player.name}
              {controllers?.[player.id] === 'cpu' && <small className="cpu-badge">CPU</small>}
            </h2>
            <span className="player-score">
              {game.phase === 'setup' ? '—' : formatScore(player.score)}
              <small>点</small>
            </span>
          </header>
          {game.phase === 'setup' ? (
            <p className="setup-player-note">
              {player.wealth.length > 0
                ? '選択を終えました'
                : player.tribeOffer?.length
                  ? '部族と初期資源を選びます'
                  : '初期資源を選びます'}
            </p>
          ) : (
            <>
              {player.tribe && (
                <details className="player-tribe">
                  <summary>
                    {expansionCatalog?.tribes.find((tribe) => tribe.id === player.tribe)?.name ??
                      player.tribe}
                  </summary>
                  <p>
                    {
                      expansionCatalog?.tribes.find((tribe) => tribe.id === player.tribe)
                        ?.description
                    }
                  </p>
                </details>
              )}
              <div className="resource-grid">
                {RESOURCE_IDS.map((r) => (
                  <div key={r} className={`resource resource-${r}`} title={resourceNames[r]}>
                    <Icon name={r} size={18} />
                    <b>{player.resources[r]}</b>
                    <span>{resourceNames[r]}</span>
                  </div>
                ))}
              </div>
              <div className="worker-count">
                <Icon name="worker" size={16} />
                <span>
                  手元 <b>{availableWorkers[player.id]}</b> / {player.workers}人
                </span>
                <span className="double-token" title="2日進める権利">
                  {player.doubleAdvanceAvailable ? '☀' : '○'}
                </span>
              </div>
              <div className="player-tech">
                {TECHNOLOGY_IDS.map((t) => (
                  <div key={t} title={`${technologyNames[t]}レベル${player.technologies[t]}`}>
                    <span>{technologyNames[t]}</span>
                    <div>
                      {[1, 2, 3].map((n) => (
                        <i key={n} className={player.technologies[t] >= n ? 'lit' : ''} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="player-temples">
                {TEMPLE_IDS.map((t) => (
                  <span key={t} title={`${templeNames[t]}ランク${player.temples[t]}`}>
                    {templeNames[t].slice(0, 2)} <b>{player.temples[t]}</b>
                  </span>
                ))}
              </div>
              <details className="player-holdings">
                <summary>
                  建物 {player.buildings.length} · 記念碑 {player.monuments.length}
                </summary>
                {player.buildings.map((id) => (
                  <p key={id}>
                    {BUILDINGS.find((b) => b.id === id)?.name ?? id}
                    <small>
                      {BUILDINGS.find((b) => b.id === id)
                        ?.effects.map(effectText)
                        .join(' / ')}
                    </small>
                  </p>
                ))}
                {player.monuments.map((id) => (
                  <p key={id}>
                    {MONUMENTS.find((m) => m.id === id)?.name ?? id}
                    <small>
                      {MONUMENT_DESCRIPTIONS[MONUMENTS.find((m) => m.id === id)?.scoreKey ?? '']}
                    </small>
                  </p>
                ))}
              </details>
            </>
          )}
        </section>
      ))}
      <div className="supply-note">
        <Icon name="skull" size={17} />
        髑髏の共通ストック <b>{game.phase === 'setup' ? '—' : game.skullSupply}</b>
      </div>
    </aside>
  );
}

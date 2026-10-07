import {
  ALL_BUILDINGS as BUILDINGS,
  MONUMENTS,
  MONUMENT_DESCRIPTIONS,
  TEMPLE_TRACKS,
  TECHNOLOGY_DESCRIPTIONS,
  TECHNOLOGY_LABELS,
} from '../game/catalog';
import {
  TEMPLE_IDS,
  TECHNOLOGY_IDS,
  type Choice,
  type GameMove,
  type GameState,
  type GearId,
} from '../game/types';
import { Board } from './Board';
import { Icon } from './Icons';
import {
  effectText,
  formatScore,
  resourceText,
  templeMax,
  templeNames,
  templePoints,
} from './content';
export type View = 'board' | 'temples' | 'buildings' | 'log' | 'rules';
export function GameBoardView({
  game,
  costs,
  moves,
  play,
  view: requestedView,
  setView,
}: {
  game: GameState;
  costs: Record<GearId, number | null>;
  moves: Choice[];
  play: (move: GameMove) => void;
  view: View;
  setView: (view: View) => void;
}) {
  const view =
    game.phase === 'setup' && (requestedView === 'log' || requestedView === 'temples')
      ? 'board'
      : requestedView;
  return (
    <main className="main-board">
      <nav className="view-tabs" aria-label="ゲームの表示">
        {(
          [
            { id: 'board', label: '歯車' },
            { id: 'temples', label: '神殿' },
            { id: 'buildings', label: '建物・記念碑' },
            { id: 'log', label: '対局記録' },
            { id: 'rules', label: '遊び方' },
          ] as const
        ).map((tab) => (
          <button
            key={tab.id}
            className={view === tab.id ? 'active' : ''}
            aria-current={view === tab.id ? 'page' : undefined}
            disabled={game.phase === 'setup' && (tab.id === 'temples' || tab.id === 'log')}
            onClick={() => setView(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>
      {game.phase === 'finished' && (
        <section className="results-panel">
          <span className="eyebrow">27日間の旅の終わり</span>
          <h1>
            {game.finalScores
              .filter((s) => s.rank === 1)
              .map((s) => game.players[s.playerId]?.name)
              .join('・')}
            の勝利
          </h1>
          <p>同点の場合は、歯車に残したワーカー数で順位を決めます。</p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>順位</th>
                  <th>プレイヤー</th>
                  <th>獲得点</th>
                  <th>資源</th>
                  <th>髑髏</th>
                  <th>記念碑</th>
                  <th>合計</th>
                </tr>
              </thead>
              <tbody>
                {[...game.finalScores]
                  .sort((a, b) => a.rank - b.rank)
                  .map((s) => (
                    <tr key={s.playerId}>
                      <td>{s.rank}</td>
                      <th>{game.players[s.playerId]?.name}</th>
                      <td>{formatScore(s.pointsBeforeFinal)}</td>
                      <td>{formatScore(s.resourcePoints)}</td>
                      <td>{s.skullPoints}</td>
                      <td>{s.monumentPoints}</td>
                      <td>
                        <b>{formatScore(s.total)}</b>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {view === 'board' && <Board game={game} moves={moves} play={play} costs={costs} />}
      {view === 'temples' && (
        <section className="temples-view">
          <div className="page-intro">
            <span className="eyebrow">神々への捧げもの</span>
            <h1>三つの神殿</h1>
            <p>食料日に資源や勝利点を得ます。最上段に到達できるのは各神殿で1人です。</p>
          </div>
          <div className="temples-grid">
            {TEMPLE_IDS.map((t) => (
              <section className={`temple-track temple-${t}`} key={t}>
                <Icon name="temple" size={38} />
                <h2>{templeNames[t]}</h2>
                {Array.from({ length: templeMax[t] + 2 }, (_, i) => templeMax[t] - i).map(
                  (rank) => (
                    <div className={`temple-step ${rank === 0 ? 'start' : ''}`} key={rank}>
                      <span>{rank === 0 ? 'START' : rank}</span>
                      <div>
                        {game.players
                          .filter((p) => p.temples[t] === rank)
                          .map((p) => (
                            <span
                              className="temple-player"
                              key={p.id}
                              title={p.name}
                              style={{ backgroundColor: p.color }}
                            >
                              <Icon name="worker" size={16} />
                            </span>
                          ))}
                      </div>
                      <b>
                        {templePoints[t][rank + 1]}
                        <small>点</small>
                      </b>
                      <small className="temple-resource">
                        {resourceText(TEMPLE_TRACKS[t].resourceRewards[rank + 1] ?? {}) === 'なし'
                          ? ''
                          : resourceText(TEMPLE_TRACKS[t].resourceRewards[rank + 1] ?? {})}
                      </small>
                    </div>
                  ),
                )}
                <p className="temple-bonus">
                  首位ボーナス：時代1 {TEMPLE_TRACKS[t].age1Bonus}点 / 時代2{' '}
                  {TEMPLE_TRACKS[t].age2Bonus}点
                </p>
              </section>
            ))}
          </div>
          <div className="technology-reference">
            <h2>技術の発展</h2>
            <div>
              {TECHNOLOGY_IDS.map((t) => (
                <section key={t}>
                  <h3>{TECHNOLOGY_LABELS[t]}</h3>
                  <ol>
                    {TECHNOLOGY_DESCRIPTIONS[t].map((text, i) => (
                      <li key={i}>
                        {i === 3 ? '上限後の発展：' : ''}
                        {text}
                      </li>
                    ))}
                  </ol>
                </section>
              ))}
            </div>
          </div>
        </section>
      )}
      {view === 'buildings' && (
        <section className="market-view">
          <div className="page-intro">
            <span className="eyebrow">都市を築く</span>
            <h1>建物と記念碑</h1>
            <p>ティカルやウシュマルの建築アクションで購入します。手番終了時に補充されます。</p>
          </div>
          <h2 className="market-heading">
            公開中の建物<span>時代 {game.age}</span>
          </h2>
          <div className="market-grid">
            {game.buildings.map((id) => {
              const b = BUILDINGS.find((b) => b.id === id);
              return (
                b && (
                  <article className={`building-card category-${b.category}`} key={id}>
                    <span className="eyebrow">
                      {b.id} · 時代{b.age}
                    </span>
                    <Icon name="temple" size={31} />
                    <h3>{b.name}</h3>
                    <p className="card-cost">{resourceText(b.cost)}</p>
                    <ul>
                      {b.effects.map((e, i) => (
                        <li key={i}>{effectText(e)}</li>
                      ))}
                    </ul>
                  </article>
                )
              );
            })}
          </div>
          <h2 className="market-heading">
            記念碑<span>ゲーム終了時に得点</span>
          </h2>
          <div className="market-grid">
            {game.monuments.map((id) => {
              const m = MONUMENTS.find((m) => m.id === id);
              return (
                m && (
                  <article className="monument-card" key={id}>
                    <span className="eyebrow">{m.id}</span>
                    <Icon name="temple" size={36} />
                    <h3>{m.name}</h3>
                    <p className="card-cost">{resourceText(m.cost)}</p>
                    <p className="monument-description">{MONUMENT_DESCRIPTIONS[m.scoreKey]}</p>
                  </article>
                )
              );
            })}
          </div>
        </section>
      )}
      {view === 'log' && (
        <section className="log-view">
          <div className="page-intro">
            <span className="eyebrow">暦の足跡</span>
            <h1>対局記録</h1>
          </div>
          <ol>
            {[...game.log].reverse().map((entry, i) => (
              <li key={`${i}-${entry}`}>
                <span>{String(game.log.length - i).padStart(2, '0')}</span>
                {entry}
              </li>
            ))}
          </ol>
        </section>
      )}
      {view === 'rules' && <Rules />}
    </main>
  );
}
function Rules() {
  const rules = [
    [
      '初期資源を選ぶ',
      '各プレイヤーは配られた4枚から2枚を選びます。ワーカーは3人で開始し、最大6人まで増やせます。2〜3人対局のダミーワーカーは配置場所をふさぎます。',
    ],
    [
      '配置するか、回収するか',
      '手番では1人以上を配置、または1人以上を回収します。配置と回収は同じ手番でできません。配置は歯車で最も低い空き位置に入り、位置の数字と、同じ手番で配置する人数による追加コストをコーンで支払います。回収するとその位置のアクションを使います。より低い位置を使う場合は差分のコーンを支払い、実行を見送ることもできます。',
    ],
    [
      '歯車が回る',
      '全員の手番が終わると1日進み、ワーカーは次の位置へ移動します。スタートプレイヤーの場所を取ると、たまったコーンを受け取り、次の日の最初の手番になります。権利が残っていれば、条件を満たす日に2日進められます。',
    ],
    [
      '食料日と神殿',
      '8・14・21・27日には、ワーカー1人につきコーン2を支払います。農場は必要量を減らします。食べさせられないワーカーは1人につき3点を失います。8・21日は神殿から資源、14・27日は神殿の得点と順位ボーナスを得ます。',
    ],
    [
      '27日目、文明を比べる',
      '最終食料日後、資源をコーンへ換算し、コーン4につき1点、残った髑髏1につき3点、記念碑の得点を加えます。同点は歯車に残ったワーカーの数で比較します。',
    ],
    [
      'このアプリでの操作',
      '歯車のワーカーを押すと回収できます。技術・建築・交易などの選択はアクション欄に表示されます。「1つ戻す」で操作を取り消せます。進行は端末内へ自動保存され、保存ファイルを使って別のブラウザでも続きを遊べます。',
    ],
  ];
  return (
    <article className="rules-view">
      <div className="page-intro">
        <span className="eyebrow">時を味方にする</span>
        <h1>遊び方</h1>
        <p>2〜4人で同じ画面を使う、基本ゲームの対局です。</p>
      </div>
      {rules.map(([title, text], i) => (
        <section key={title}>
          <span className="rule-number">{String(i + 1).padStart(2, '0')}</span>
          <div>
            <h2>{title}</h2>
            <p>{text}</p>
          </div>
        </section>
      ))}
      <p className="source-note">
        公式ルールの日本語・英語PDFと、補助資料のWikiはリポジトリの docs/rules
        に保存しています。部族・予言などの拡張ルールは含みません。
      </p>
    </article>
  );
}

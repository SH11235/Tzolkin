import { useState } from "react";
import "./App.css";
import { Temple } from "./components/temple";
import { ChaacBonus, QuetzalcoatlBonus, KukulkanBonus } from "./constant";
import { Players } from "./components/players";
import { PalenqueSpace } from "./components/PalenqueSpace";
import { GamePlayers } from "./types/GamePlayer";
import { FirstResourcesState } from "./types/FirstResource";
import { FirstResourceModal } from "./components/FirstResourceModal";

function App() {
  const [players, setPlayers] = useState<GamePlayers>([]);
  const [firstResources, setFirstResources] = useState<FirstResourcesState>([]);
  const [round, setRound] = useState(1);
  const [palenqueChips, setPalenqueChips] = useState<
    {
      wood: number;
      corn: number;
    }[]
  >([
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
  ]);
  const [palenqueWorkers, setPalenqueWorkers] = useState<(null | string)[]>([
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
  ]);

  return (
    <>
      <div className="container">
        <div className="row">
          <a
            href="https://ja.boardgamearena.com/gamepanel?game=tzolkin"
            target="_blank"
          >
            <img src="/gear_logo.png" className="logo gear" alt="Gear logo" />
          </a>
        </div>

        <div className="row">
          <Players
            setPlayers={setPlayers}
            setPalenqueChips={setPalenqueChips}
            setFirstResources={setFirstResources}
          />
        </div>

        <div className="row">ラウンド：{round}</div>

        <div className="row">
          <div className="field-container">
            <PalenqueSpace
              palenqueChips={palenqueChips}
              palenqueWorkers={palenqueWorkers}
            />
            <div className="yaxchilan-space">
              <span className="space-name">Yaxchilan</span>
              <span className="space-container">
                <span className="space">
                  0<div className="worker-space"></div>
                </span>
                <span className="space">
                  1<div className="worker-space"></div>
                </span>
                <span className="space">
                  2<div className="worker-space"></div>
                </span>
                <span className="space">
                  3<div className="worker-space"></div>
                </span>
                <span className="space">
                  4<div className="worker-space"></div>
                </span>
                <span className="space">
                  5<div className="worker-space"></div>
                </span>
                <span className="space">
                  6<div className="worker-space"></div>
                </span>
                <span className="space">
                  7<div className="worker-space"></div>
                </span>
              </span>
            </div>
            <div className="tikal-space">
              <span className="space-name">Tikal</span>
              <span className="space-container">
                <span className="space">
                  0<div className="worker-space"></div>
                </span>
                <span className="space">
                  1<div className="worker-space"></div>
                </span>
                <span className="space">
                  2<div className="worker-space"></div>
                </span>
                <span className="space">
                  3<div className="worker-space"></div>
                </span>
                <span className="space">
                  4<div className="worker-space"></div>
                </span>
                <span className="space">
                  5<div className="worker-space"></div>
                </span>
                <span className="space">
                  6<div className="worker-space"></div>
                </span>
                <span className="space">
                  7<div className="worker-space"></div>
                </span>
              </span>
            </div>
            <div className="uxmal-space">
              <span className="space-name">Uxmal</span>
              <span className="space-container">
                <span className="space">
                  0<div className="worker-space"></div>
                </span>
                <span className="space">
                  1<div className="worker-space"></div>
                </span>
                <span className="space">
                  2<div className="worker-space"></div>
                </span>
                <span className="space">
                  3<div className="worker-space"></div>
                </span>
                <span className="space">
                  4<div className="worker-space"></div>
                </span>
                <span className="space">
                  5<div className="worker-space"></div>
                </span>
                <span className="space">
                  6<div className="worker-space"></div>
                </span>
                <span className="space">
                  7<div className="worker-space"></div>
                </span>
              </span>
            </div>
            <div className="chichen-itza-space">
              <span className="space-name">Chichen Itza</span>
              <span className="space">
                0<div className="worker-space"></div>
              </span>
              <span className="space">
                1<div className="worker-space"></div>
              </span>
              <span className="space">
                2<div className="worker-space"></div>
              </span>
              <span className="space">
                3<div className="worker-space"></div>
              </span>
              <span className="space">
                4<div className="worker-space"></div>
              </span>
              <span className="space">
                5<div className="worker-space"></div>
              </span>
              <span className="space">
                6<div className="worker-space"></div>
              </span>
              <span className="space">
                7<div className="worker-space"></div>
              </span>
              <span className="space">
                8<div className="worker-space"></div>
              </span>
              <span className="space">
                9<div className="worker-space"></div>
              </span>
              <span className="space">
                10<div className="worker-space"></div>
              </span>
              <span className="space">
                11<div className="worker-space"></div>
              </span>
            </div>
          </div>
          <span className="temples">
            <Temple
              name="Chaac"
              playerScores={
                players.map((player) => {
                  return {
                    color: player.color,
                    index: player.temple_faith.chaac,
                  };
                }) || []
              }
              templeBonus={ChaacBonus}
              templeColor="brown"
            />
            <Temple
              name="Quetzalcoatl"
              playerScores={
                players.map((player) => {
                  return {
                    color: player.color,
                    index: player.temple_faith.quetzalcoatl,
                  };
                }) || []
              }
              templeBonus={QuetzalcoatlBonus}
              templeColor="yellow"
            />
            <Temple
              name="Kukulkan"
              playerScores={
                players.map((player) => {
                  return {
                    color: player.color,
                    index: player.temple_faith.kukulkan,
                  };
                }) || []
              }
              templeBonus={KukulkanBonus}
              templeColor="green"
            />
          </span>
        </div>

        <div className="row">
          <span>
            ターンプレイヤー: {players.length > 0 ? players[0].name : ""}
          </span>
        </div>
      </div>
      <FirstResourceModal
        firstResources={firstResources}
        setFirstResources={setFirstResources}
        players={players}
        setPlayers={setPlayers}
      />
    </>
  );
}

export default App;

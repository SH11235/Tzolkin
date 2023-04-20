import { useState } from "react";
import "./App.css";
import { PlayersNumber } from "./components/PlayersNumber";
import { FieldContainer } from "./components/FieldContainer";
import { GamePlayers } from "./types/GamePlayer";
import { FirstResourcesState } from "./types/FirstResource";
import { FirstResourceModal } from "./components/FirstResourceModal";
import { TechnologyLevel } from "./components/TechnologyLevel";
import { PlayersTable } from "./components/PlayerStatus";
import { TemplesContainer } from "./components/TemplesContainer";

function App() {
  const [players, setPlayers] = useState<GamePlayers>([]);
  const [fieldSkulls, setFieldSkulls] = useState(13);
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
          <PlayersNumber
            setPlayers={setPlayers}
            setPalenqueChips={setPalenqueChips}
            setFirstResources={setFirstResources}
          />
        </div>

        <div className="row">
          <span className="game-status">Round：{round}</span>
          <span className="game-status">Skull：{fieldSkulls}</span>
        </div>

        <div className="row">
          <FieldContainer
            palenqueChips={palenqueChips}
            palenqueWorkers={palenqueWorkers}
          />
        </div>

        <div className="row">
          <div className="turn-player">
            Turn Player: {players.length > 0 ? players[0].name : ""}
          </div>
        </div>

        <div className="row">
          <TemplesContainer players={players} />
          <TechnologyLevel players={players} />
        </div>

        <div className="row">
          <PlayersTable players={players}></PlayersTable>
        </div>
      </div>
      <FirstResourceModal
        firstResources={firstResources}
        setFirstResources={setFirstResources}
        players={players}
        setPlayers={setPlayers}
        setFieldSkulls={setFieldSkulls}
      />
    </>
  );
}

export default App;

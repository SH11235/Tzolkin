import { useState } from "react";
import "./App.css";
import { GameCondition } from "./components/GameCondition";
import { FieldContainer } from "./components/FieldContainer";
import { GamePlayers } from "./types/GamePlayer";
import { FirstResourcesState } from "./types/FirstResource";
import { FirstResourceModal } from "./components/FirstResourceModal";
import { TechnologyLevel } from "./components/TechnologyLevel";
import { PlayersTable } from "./components/PlayerStatus";
import { TemplesContainer } from "./components/TemplesContainer";
import { TurnPlayer } from "./components/TurnPlayer";
import { PassButton } from "./components/PassButton";

function App() {
  const [players, setPlayers] = useState<GamePlayers>([]);
  const [turnPlayerIndex, setTurnPlayerIndex] = useState(0);
  const [fieldSkulls, setFieldSkulls] = useState(13);
  const [firstResources, setFirstResources] = useState<FirstResourcesState>([]);
  const [round, setRound] = useState(1);
  const [isStartIconHidden, setIsStartIconHidden] = useState(false);
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
        {!isStartIconHidden && (
          <div className="row">
            <a
              href="https://ja.boardgamearena.com/gamepanel?game=tzolkin"
              target="_blank"
            >
              <img src="/gear_logo.png" className="logo gear" alt="Gear logo" />
            </a>
          </div>
        )}

        {!isStartIconHidden && (
          <div className="row">
            <GameCondition
              setPlayers={setPlayers}
              setPalenqueChips={setPalenqueChips}
              setFirstResources={setFirstResources}
              hideIcon={() => setIsStartIconHidden(true)}
            />
          </div>
        )}

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

        <TurnPlayer players={players} turnPlayerIndex={turnPlayerIndex} />
        <PassButton
          playersNumber={players.length}
          playerIndex={turnPlayerIndex}
          setTurnPlayerIndex={setTurnPlayerIndex}
          round={round}
          setRound={setRound}
        />

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

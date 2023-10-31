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
import { MAX_SKULL_COUNT, START_ROUND } from "./constant";

function App() {
  const [players, setPlayers] = useState<GamePlayers>([]);
  const [turnPlayerIndex, setTurnPlayerIndex] = useState(0);
  const [fieldSkulls, setFieldSkulls] = useState(MAX_SKULL_COUNT);
  const [firstResources, setFirstResources] = useState<FirstResourcesState>([]);
  const [round, setRound] = useState(START_ROUND);
  const [isGotFirstPlayer, setIsGotFirstPlayer] = useState(false);
  const [skipNextRound, setSkipNextRound] = useState(false);
  const [passButtonDisabled, setPassButtonDisabled] = useState(true);
  const [nextFirstPlayerIndex, setNextFirstPlayerIndex] = useState(0);
  const [boardCorns, setBoardCorns] = useState(0);
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
          passButtonDisabled={passButtonDisabled}
          playersNumber={players.length}
          playerIndex={turnPlayerIndex}
          isGotFirstPlayer={isGotFirstPlayer}
          setIsGotFirstPlayer={setIsGotFirstPlayer}
          skipNextRound={skipNextRound}
          setSkipNextRound={setSkipNextRound}
          nextFirstPlayerIndex={nextFirstPlayerIndex}
          boardCorns={boardCorns}
          setBoardCorns={setBoardCorns}
          setTurnPlayerIndex={setTurnPlayerIndex}
          setPlayers={setPlayers}
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

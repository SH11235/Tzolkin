import { invoke } from "@tauri-apps/api";
import React from "react";
import { FirstResources, FirstResourcesState } from "../types/FirstResource";
import { GamePlayers } from "../types/GamePlayer";

interface gameConditionProps {
  setPlayers: React.Dispatch<React.SetStateAction<GamePlayers>>;
  setPalenqueChips: React.Dispatch<
    React.SetStateAction<
      {
        wood: number;
        corn: number;
      }[]
    >
  >;
  setFirstResources: React.Dispatch<React.SetStateAction<FirstResourcesState>>;
}

export const GameCondition = ({
  setPlayers,
  setPalenqueChips,
  setFirstResources,
}: gameConditionProps) => {
  const [playersNumber, setPlayersNumber] = React.useState(0);
  const [isExpansion, setIsExpansion] = React.useState(false);
  const [isFormDisabled, setIsFormDisabled] = React.useState(false);

  const numberOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    if (/^[1-4]$/.test(newValue)) {
      const number = parseInt(newValue, 10);
      setPlayersNumber(number);
    }
  };

  const onRadioChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setIsExpansion(event.target.value === "yes");
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const gamePlayers: GamePlayers = await invoke("set_players", {
        number: playersNumber,
      });
      setPlayers(gamePlayers);
      setPalenqueChips([
        {
          wood: 0,
          corn: playersNumber,
        },
        ...Array(3).fill({}).map(() => ({
          wood: playersNumber,
          corn: playersNumber,
        })),
    ]);
      
      setIsFormDisabled(true);

      const firstResourceTiles: FirstResources = await invoke(
        "get_first_resource_tiles"
      );
      const firstResourceTilesState = firstResourceTiles.map((resourceTiles) => {
        return resourceTiles.map((resourceTile) => {
          return {
            ...resourceTile,
            selected: false,
          };
        });
      });
      setFirstResources(firstResourceTilesState);
    } catch (e) {
      // TODO error modal
      console.error(e);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      Number of players: 
      <input
        id="number-of-players-input"
        type="number"
        value={playersNumber}
        onChange={numberOnChange}
        disabled={isFormDisabled}
      />
      <div>
        <label>
          <input
            type="radio"
            value="yes"
            checked={isExpansion}
            onChange={onRadioChange}
            disabled={isFormDisabled}
          />
          Expansion
        </label>
        <label>
          <input
            type="radio"
            value="no"
            checked={!isExpansion}
            onChange={onRadioChange}
            disabled={isFormDisabled}
          />
          No Expansion
        </label>
      </div>
      <button id="submit-button" type="submit" disabled={isFormDisabled}>
        OK
      </button>
    </form>
  );
};

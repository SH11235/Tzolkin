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

  const numberOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    if (/^[1-4]$/.test(newValue)) {
      const number = parseInt(newValue);
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
        {
          wood: playersNumber,
          corn: playersNumber,
        },
        {
          wood: playersNumber,
          corn: playersNumber,
        },
        {
          wood: playersNumber,
          corn: playersNumber,
        },
      ]);
      const input = document.getElementById(
        "number-of-players-input"
      ) as HTMLInputElement;
      if (input) {
        input.disabled = true;
      }
      const button = document.getElementById(
        "submit-button"
      ) as HTMLButtonElement;
      if (button) {
        button.disabled = true;
      }

      const firstResourceTiles: FirstResources = await invoke(
        "get_first_resource_tiles"
      );
      const firstResourceTilesState = firstResourceTiles.map(
        (resourceTiles) => {
          return resourceTiles.map((resourceTile) => {
            return {
              ...resourceTile,
              selected: false,
            };
          });
        }
      );
      setFirstResources(firstResourceTilesState);
    } catch (e) {
      // TODO error modal
      console.error(e);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      Number of players{"： "}
      <input
        id="number-of-players-input"
        type="number"
        value={playersNumber}
        onChange={numberOnChange}
      />
      <div>
        <label>
          <input
            type="radio"
            value="yes"
            checked={isExpansion}
            onChange={onRadioChange}
          />
          Expansion
        </label>
        <label>
          <input
            type="radio"
            value="no"
            checked={!isExpansion}
            onChange={onRadioChange}
          />
          No Expansion
        </label>
      </div>
      <button id="submit-button" type="submit">
        OK
      </button>
    </form>
  );
};

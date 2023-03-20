import { invoke } from "@tauri-apps/api";
import React from "react";
import { GamePlayers } from "../types/GamePlayer";

interface playersProps {
  setPlayers: React.Dispatch<React.SetStateAction<GamePlayers>>;
  setPalenqueChips: React.Dispatch<
    React.SetStateAction<
      {
        wood: number;
        corn: number;
      }[]
    >
  >;
}

export const Players = ({ setPlayers, setPalenqueChips }: playersProps) => {
  const [playersNumber, setplayersNumber] = React.useState(0);
  const numberOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    if (/^[1-4]$/.test(newValue)) {
      const number = parseInt(newValue);
      setplayersNumber(number);
    }
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
    } catch {
      // ユーザーに警告を出す
      // alert("プレイ人数の設定に失敗しました");
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      プレイ人数{"： "}
      <input
        id="number-of-players-input"
        type="number"
        value={playersNumber}
        onChange={numberOnChange}
      />
      <button id="submit-button" type="submit">
        決定
      </button>
    </form>
  );
};

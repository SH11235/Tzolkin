import React from "react";

interface playersProps {
  players: {
    name: string;
    index: number;
    color: string;
  }[];
  setPlayers: React.Dispatch<
    React.SetStateAction<
      {
        name: string;
        index: number;
        color: string;
      }[]
    >
  >;
  onSubmit: (numberOfPlayers: number) => void;
}

export const Players = ({
  players,
  setPlayers,
  onSubmit,
}: playersProps) => {
  const playerColors = ["red", "blue", "green", "yellow"];
  const numberOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    if (/^[1-4]$/.test(newValue)) {
      const number = parseInt(newValue);
      setPlayers(
        Array(number)
          .fill(0)
          .map((_, index) => {
            return {
              name: `player${index + 1}`,
              index: index + 1,
              color: playerColors[index],
            };
          })
      );
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit(players.length);
  };

  return (
    <form onSubmit={handleSubmit}>
      プレイ人数{"： "}
      <input
        id="number-of-players-input"
        type="number"
        value={players.length}
        onChange={numberOnChange}
      />
      <button id="submit-button" type="submit">
        決定
      </button>
    </form>
  );
};

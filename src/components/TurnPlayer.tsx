import { GamePlayers } from "../types/GamePlayer";

interface turnPlayerProps {
  players: GamePlayers;
  turnPlayerIndex: number;
}

export const TurnPlayer = ({ players, turnPlayerIndex }: turnPlayerProps) => {
  return (
    <div className="row">
      <div className="turn-player">
        Turn Player:{" "}
        <span
          style={
            players.length > 0 ? { color: players[turnPlayerIndex].color } : {}
          }
        >
          {players.length > 0 ? players[turnPlayerIndex].name : ""}
        </span>
        <span></span>
      </div>
    </div>
  );
};

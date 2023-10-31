import { Button } from "@mui/material";
import { invoke } from "@tauri-apps/api";
import { Player } from "../types/GamePlayer";

type PassButtonProps = {
  playersNumber: number;
  playerIndex: number;
  isGetFirstPlayer: boolean;
  nextFirstPlayerIndex: number;
  boardCorns: number;
  setBoardCorns: (corns: number) => void;
  setTurnPlayerIndex: (index: number) => void;
  setPlayers: (players: Player[]) => void;
  round: number;
  setRound: (round: number) => void;
};

export const PassButton = ({
  playersNumber,
  playerIndex,
  isGetFirstPlayer,
  nextFirstPlayerIndex,
  boardCorns,
  setBoardCorns,
  setTurnPlayerIndex,
  setPlayers,
  round,
  setRound,
}: PassButtonProps) => {
  const handleClick = async () => {
    // 最後手番のプレイヤーの場合
    if (playerIndex === playersNumber - 1) {
      if (isGetFirstPlayer) {
        // first playerを取ったプレイヤーにcornを追加
        await invoke("add_resource", {
          player_id: nextFirstPlayerIndex,
          resource_type: "corn",
          amount: boardCorns,
        });
        const corns: number = await invoke("reset_board_corns");
        setBoardCorns(corns);
        const players: Player[] = await invoke("set_first_player", {
          index: nextFirstPlayerIndex,
        });
        setPlayers(players);
      } else {
        const corns: number = await invoke("add_board_corns");
        setBoardCorns(corns);
      }
      setTurnPlayerIndex(0);
      const round: number = await invoke("next_round");
      setRound(round);
    } else { // 最後意外の手番のプレイヤーの場合
      setTurnPlayerIndex(playerIndex + 1);
    }
  };

  return (
    <Button variant="contained" color="primary" onClick={handleClick}>
      Pass
    </Button>
  );
};

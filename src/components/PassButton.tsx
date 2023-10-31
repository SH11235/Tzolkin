import { Button } from "@mui/material";
import { invoke } from "@tauri-apps/api";
import { Player } from "../types/GamePlayer";
import { FOURTH_FOOD_DAY } from "../constant";

type PassButtonProps = {
  passButtonDisabled: boolean;
  playersNumber: number;
  playerIndex: number;
  isGotFirstPlayer: boolean;
  setIsGotFirstPlayer: (isGotFirstPlayer: boolean) => void;
  skipNextRound: boolean;
  setSkipNextRound: (skipNextRound: boolean) => void;
  nextFirstPlayerIndex: number;
  boardCorns: number;
  setBoardCorns: (corns: number) => void;
  setTurnPlayerIndex: (index: number) => void;
  setPlayers: (players: Player[]) => void;
  setRound: (round: number) => void;
};

export const PassButton = ({
  passButtonDisabled,
  playersNumber,
  playerIndex,
  isGotFirstPlayer,
  setIsGotFirstPlayer,
  skipNextRound,
  setSkipNextRound,
  nextFirstPlayerIndex,
  boardCorns,
  setBoardCorns,
  setTurnPlayerIndex,
  setPlayers,
  setRound,
}: PassButtonProps) => {
  const handleClick = async () => {
    // 最後手番のプレイヤーの場合
    if (playerIndex === playersNumber - 1) {
      const this_round: number = await invoke("get_round");
      if (this_round >= FOURTH_FOOD_DAY) {
        await invoke("finish_game"); // TODO: ゲーム終了処理
        // プレイヤーに神殿の判定、モニュメントの得点、残りの資源を得点に換算
        return;
      }
      if (isGotFirstPlayer) {
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
        setIsGotFirstPlayer(false);
      } else {
        const corns: number = await invoke("add_board_corns");
        setBoardCorns(corns);
      }
      setTurnPlayerIndex(0);
      await invoke("next_round");
      if (skipNextRound) {
        await invoke("next_round");
        setSkipNextRound(false);
      }
      const round: number = await invoke("get_round");
      setRound(round);
    } else {
      // 最後意外の手番のプレイヤーの場合
      setTurnPlayerIndex(playerIndex + 1);
    }
  };

  return (
    <Button
      variant="contained"
      color="primary"
      onClick={handleClick}
      disabled={passButtonDisabled}
    >
      Pass
    </Button>
  );
};

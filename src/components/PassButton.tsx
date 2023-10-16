import { Button } from "@mui/material";

type PassButtonProps = {
  playersNumber: number;
  playerIndex: number;
  setTurnPlayerIndex: (index: number) => void;
  round: number;
  setRound: (round: number) => void;
};

export const PassButton = ({
  playersNumber,
  playerIndex,
  setTurnPlayerIndex,
  round,
  setRound,
}: PassButtonProps) => {
  const handleClick = () => {
    if (playerIndex === playersNumber - 1) {
      // TODO start player変更時の処理
      setTurnPlayerIndex(0);
      setRound(round + 1);
    } else  {
      setTurnPlayerIndex(playerIndex + 1);
    }
  };

  return (
    <Button variant="contained" color="primary" onClick={handleClick}>
      Pass
    </Button>
  );
};

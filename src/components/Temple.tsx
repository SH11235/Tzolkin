import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { PlayerColor } from "../types/GamePlayer";

export type TempleBonus = {
  resource: "stone" | "gold" | "wood" | "skull" | null;
  point: number;
}[];

type TempleColor = "brown" | "yellow" | "green";

type Props = {
  name: string;
  playerScores: {
    color: PlayerColor;
    index: number;
  }[];
  templeBonus: TempleBonus;
  templePoints: number[];
  templeColor: TempleColor;
};

export const Temple: React.FC<Props> = ({
  name,
  playerScores,
  templeBonus,
  templePoints,
  templeColor,
}) => {
  const ResourceCell = styled(TableCell)({
    height: "25px",
    width: "61px",
    padding: "2px 5px",
    border: "1px solid #000",
    textAlign: "center",
    verticalAlign: "middle",
    position: "relative",
  });

  const PlayerCell = styled(TableCell)({
    height: "25px",
    width: "120px",
    padding: "2px 5px",
    border: "1px solid #000",
    textAlign: "center",
    verticalAlign: "middle",
    position: "relative",
  });

  const ScoreCell = styled(TableCell)({
    height: "25px",
    width: "32px",
    padding: "2px 5px",
    border: "1px solid #000",
    textAlign: "center",
    verticalAlign: "middle",
    position: "relative",
  });

  const templeBackGroundColor =
    templeColor === "brown"
      ? "#B67A48"
      : templeColor === "yellow"
      ? "#F1C614"
      : "#8BBF3D";
  const TempleTable = styled(Table)({
    width: "210px",
    backgroundColor: templeBackGroundColor,
  });

  const offset = 9 - templeBonus.length;
  const array = Array(offset).fill(0);
  const offsetRows = array.map((_, index) => {
    return (
      <TableRow key={`offsetRows-${index}`}>
        <ResourceCell>✕</ResourceCell>
        <PlayerCell>✕</PlayerCell>
        <ScoreCell>✕</ScoreCell>
      </TableRow>
    );
  });

  const rows = templeBonus.map((row, rowIndex) => {
    const playerScoreElements = playerScores.map((playerScore) => {
      const offset = templeBonus.length - 2;
      if (playerScore.index + rowIndex === offset) {
        return (
          <Typography
            sx={{
              display: "inline-block",
              width: "20px",
              height: "20px",
              borderRadius: "50%",
              verticalAlign: "middle",
              textAlign: "center",
              backgroundColor: playerScore.color,
            }}
            key={`${playerScore.color}-${rowIndex}`}
          ></Typography>
        );
      }
      return null;
    });

    return (
      <TableRow key={rowIndex}>
        <ResourceCell>{row.resource}</ResourceCell>
        <PlayerCell>{playerScoreElements}</PlayerCell>
        <ScoreCell>{row.point}</ScoreCell>
      </TableRow>
    );
  });

  return (
    <Box sx={{ m: 1 }}>
      <Typography sx={{ display: "inline" }} variant="h6">
        {name}
      </Typography>
      <Typography
        sx={{
          display: "inline",
          border: "1px solid #000",
          padding: "2px 5px",
          marginLeft: "5px",
        }}
      >
        {templePoints[0]}
      </Typography>
      <Typography
        sx={{
          display: "inline",
          border: "1px solid #000",
          padding: "2px 5px",
          marginLeft: "5px",
        }}
      >
        {templePoints[1]}
      </Typography>
      <TableContainer sx={{ mt: 2, marginTop: "4px" }}>
        <TempleTable border={1} size="small">
          <TableHead>
            <TableRow>
              <ResourceCell>Resource</ResourceCell>
              <PlayerCell>Player</PlayerCell>
              <ScoreCell>Point</ScoreCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {offsetRows}
            {rows}
          </TableBody>
        </TempleTable>
      </TableContainer>
    </Box>
  );
};

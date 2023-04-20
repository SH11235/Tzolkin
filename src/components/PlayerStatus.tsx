import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from "@mui/material";
import { GamePlayers } from "../types/GamePlayer";

type Props = {
  players: GamePlayers;
};

export const PlayersTable = ({ players }: Props) => {
  return (
    <TableContainer component={Paper}>
      <Table aria-label="Players Table">
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell align="center">Color</TableCell>
            <TableCell align="center">Turn Order</TableCell>
            <TableCell align="center">Acceleration</TableCell>
            <TableCell align="center">
              Corn Saving
              <br />
              (Single, Triple, All)
            </TableCell>
            <TableCell align="center">Workers</TableCell>
            <TableCell align="center">Corns</TableCell>
            <TableCell align="center">Woods</TableCell>
            <TableCell align="center">Stones</TableCell>
            <TableCell align="center">Golds</TableCell>
            <TableCell align="center">Skulls</TableCell>
            <TableCell align="center">Corn Tiles</TableCell>
            <TableCell align="center">Wood Tiles</TableCell>
            <TableCell align="center">Points</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {players.map((player) => (
            <TableRow key={player.name}>
              <TableCell component="th" scope="row">
                {player.name}
              </TableCell>
              <TableCell align="center">{player.color}</TableCell>
              <TableCell align="center">{player.order}</TableCell>
              <TableCell align="center">
                {player.accelerating_ability ? "OK" : "NG"}
              </TableCell>
              <TableCell align="center">
                {player.corn_save.single}, {player.corn_save.triple},{" "}
                {player.corn_save.all}
              </TableCell>
              <TableCell align="center">
                {
                  player.workers.filter(
                    (worker) => worker.position !== "Locked"
                  ).length
                }
              </TableCell>
              <TableCell align="center">{player.corns}</TableCell>
              <TableCell align="center">{player.resource.woods}</TableCell>
              <TableCell align="center">{player.resource.stones}</TableCell>
              <TableCell align="center">{player.resource.golds}</TableCell>
              <TableCell align="center">{player.resource.skulls}</TableCell>
              <TableCell align="center">{player.corn_tiles}</TableCell>
              <TableCell align="center">{player.wood_tiles}</TableCell>
              <TableCell align="center">{player.points}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

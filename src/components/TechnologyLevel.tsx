import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import { AGRICULTURE, CONSTRUCTION, RESOURCE, TEMPLE } from "../constant";
import { Player } from "../types/GamePlayer";

interface Props {
  players: Player[];
}

export const TechnologyLevel: React.FC<Props> = ({ players }) => {
  const technologyTypes = [AGRICULTURE, RESOURCE, CONSTRUCTION, TEMPLE];
  return (
    <TableContainer sx={{ maxWidth: 600 }}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell sx={{ borderRight: "1px solid #ddd" }}>
              Technology
            </TableCell>
            <TableCell align="center" sx={{ borderRight: "1px solid #ddd" }}>
              Level 0
            </TableCell>
            <TableCell align="center" sx={{ borderRight: "1px solid #ddd" }}>
              Level 1
            </TableCell>
            <TableCell align="center" sx={{ borderRight: "1px solid #ddd" }}>
              Level 2
            </TableCell>
            <TableCell align="center" sx={{ borderRight: "1px solid #ddd" }}>
              Level 3
            </TableCell>
            <TableCell align="center" sx={{ borderRight: "1px solid #ddd" }}>
              Level 4
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {technologyTypes.map((technologyType) => (
            <TableRow key={technologyType}>
              <TableCell sx={{ borderRight: "1px solid #ddd" }}>{technologyType}</TableCell>
              {[0, 1, 2, 3, 4].map((level) => (
                <TableCell
                  key={level}
                  align="center"
                  sx={{ borderRight: "1px solid #ddd" }}
                >
                  {players.map((player) => {
                    switch (technologyType) {
                      case AGRICULTURE:
                        if (player.technology.agriculture === level) {
                          return (
                            <span
                              key={`${player.color}-${player.id}`}
                              className={`circle-${player.color}`}
                            ></span>
                          );
                        }
                        break;
                      case RESOURCE:
                        if (player.technology.resource === level) {
                          return (
                            <span
                              key={`${player.color}-${player.id}`}
                              className={`circle-${player.color}`}
                            ></span>
                          );
                        }
                        break;
                      case CONSTRUCTION:
                        if (player.technology.construction === level) {
                          return (
                            <span
                              key={`${player.color}-${player.id}`}
                              className={`circle-${player.color}`}
                            ></span>
                          );
                        }
                        break;
                      case TEMPLE:
                        if (player.technology.temple === level) {
                          return (
                            <span
                              key={`${player.color}-${player.id}`}
                              className={`circle-${player.color}`}
                            ></span>
                          );
                        }
                        break;
                    }
                    return null;
                  })}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

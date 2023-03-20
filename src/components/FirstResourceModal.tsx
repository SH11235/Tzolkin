import styled from "@emotion/styled";
import { Card, CardContent, CardHeader } from "@mui/material";
import { FirstResources } from "../types/FirstResource";
import { GamePlayers } from "../types/GamePlayer";
import { PlayerCard } from "./PlayerCard";

type FirstResourceProps = {
  firstResources: FirstResources;
  players: GamePlayers;
  setPlayers: React.Dispatch<React.SetStateAction<GamePlayers>>;
};

const StyledCard = styled(Card)`
  max-width: 90%;
  width: 100%;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
`;

const StyledCardHeader = styled(CardHeader)`
  background-color: #f8bbd0;
`;

const StyledCardContent = styled(CardContent)`
  display: flex;
  flex-direction: row;
  justify-content: center;
  flex-wrap: wrap;
  gap: 8px;
  padding: 16px;
`;

export const FirstResourceModal = ({
  firstResources,
  players,
}: FirstResourceProps) => {
  if (firstResources.length > 0) {
    return (
      <StyledCard>
        <StyledCardHeader title="初期資源" />
        <StyledCardContent>
          {players.map((player, index) => (
            <PlayerCard
              key={player.name}
              player={player}
              resourceTiles={firstResources[index]}
            />
          ))}
        </StyledCardContent>
      </StyledCard>
    );
  } else {
    return null;
  }
};

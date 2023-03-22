import styled from "@emotion/styled";
import { Button, Card, CardContent, CardHeader } from "@mui/material";
import { FirstResourcesState } from "../types/FirstResource";
import { GamePlayers } from "../types/GamePlayer";
import { PlayerCard } from "./PlayerCard";

type FirstResourceProps = {
  firstResources: FirstResourcesState;
  setFirstResources: React.Dispatch<React.SetStateAction<FirstResourcesState>>;
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
  setFirstResources,
  players,
  setPlayers,
}: FirstResourceProps) => {
  const handleConfirm = () => {};

  if (firstResources.length > 0) {
    return (
      <StyledCard>
        <StyledCardHeader title="初期資源" />
        <StyledCardContent>
          {players.map((player, index) => {
            return (
              <PlayerCard
                key={player.name}
                player={player}
                playerIndex={index}
                firstResource={firstResources[index]}
                setFirstResources={setFirstResources}
              />
            );
          })}
        </StyledCardContent>
        <Button
          disabled={
            // 全てのプレイヤーが2枚選択したらdisabledを解除
            firstResources.every((firstResource) => {
              return (
                firstResource.filter((resourceTile) => {
                  return resourceTile.selected;
                }).length === 2
              );
            })
              ? false
              : true
          }
          onClick={handleConfirm}
        >
          決定
        </Button>
      </StyledCard>
    );
  } else {
    return null;
  }
};

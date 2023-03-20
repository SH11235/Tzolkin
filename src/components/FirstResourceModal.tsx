import styled from "@emotion/styled";
import { Card, CardContent, CardHeader, Typography } from "@mui/material";
import {
  FirstResources,
  ResourceTile,
  WorkSpace,
} from "../types/FirstResource";
import { GamePlayers, Player } from "../types/GamePlayer";

type ResourceTileCardProps = {
  resourceTiles: [string, number | boolean | WorkSpace | null][];
};

const StyledResourceCardContent = styled(CardContent)`
  white-space: pre-line;
`;

const ResourceTileCard = ({ resourceTiles }: ResourceTileCardProps) => {
  return (
    <Card>
      <StyledResourceCardContent>
        {resourceTiles.map(([key, value]) => {
          if (key === "work_space") {
            // 例：
            // workSpace: {
            //   "Yaxchilan": 7
            // }
            return (
              <div key={key}>
                {`${key}: ${Object.entries(value as WorkSpace).map(
                  ([key, value]) => `${key} ${value}`
                )}`}
              </div>
            );
          } else {
            return <div key={key}>{`${key}${value}`}</div>;
          }
        })}
      </StyledResourceCardContent>
    </Card>
  );
};

type PlayerCardProps = {
  player: Player;
  resourceTiles: ResourceTile[];
};

const StyledPlayerCard = styled(Card)`
  width: 250px;
  margin: 8px;
  box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
`;

const PlayerCard = ({ player, resourceTiles }: PlayerCardProps) => {
  return (
    <StyledPlayerCard>
      <CardHeader title={player.name} />
      <CardContent>
        {resourceTiles.map((tile, index) => {
          const resources = Object.entries(tile).filter(
            ([_, value]) => value !== null && value !== false
          );
          return (
            <div key={index}>
              <Card>
                <ResourceTileCard resourceTiles={resources} />
              </Card>
            </div>
          );
        })}
      </CardContent>
    </StyledPlayerCard>
  );
};

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

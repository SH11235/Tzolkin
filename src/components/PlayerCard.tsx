import styled from "@emotion/styled";
import { Card, CardContent, CardHeader } from "@mui/material";
import { FirstResourcesState, ResourceTileState } from "../types/FirstResource";
import { Player } from "../types/GamePlayer";
import { ResourceTileCard } from "./ResourceTileCard";

type PlayerCardProps = {
  player: Player;
  playerIndex: number;
  firstResource: ResourceTileState[];
  setFirstResources: React.Dispatch<React.SetStateAction<FirstResourcesState>>;
};

const StyledPlayerCard = styled(Card)`
  width: 250px;
  margin: 8px;
  box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
`;

export const PlayerCard = ({
  player,
  playerIndex,
  firstResource,
  setFirstResources,
}: PlayerCardProps) => {
  const handleClickCard = (index: number) => {
    const newResourceTilesStates = [...firstResource];
    newResourceTilesStates[index].selected =
      !newResourceTilesStates[index].selected;
    setFirstResources((prevState) => {
      const newState = [...prevState];
      newState[playerIndex] = newResourceTilesStates;
      return newState;
    });
  };

  if (firstResource && firstResource.length > 0) {
    return (
      <StyledPlayerCard>
        <CardHeader title={player.name} />
        <CardContent>
          {firstResource.map((tile, index) => {
            return (
              <div key={index}>
                <ResourceTileCard
                  resourceTile={tile}
                  selected={tile.selected}
                  handleSelect={() => handleClickCard(index)}
                />
              </div>
            );
          })}
        </CardContent>
      </StyledPlayerCard>
    );
  } else {
    return null;
  }
};

import styled from "@emotion/styled";
import { Card, CardContent, CardHeader } from "@mui/material";
import { useState } from "react";
import { ResourceTile } from "../types/FirstResource";
import { Player } from "../types/GamePlayer";
import { ResourceTileCard } from "./ResourceTileCard";

type PlayerCardProps = {
  player: Player;
  resourceTiles: ResourceTile[];
};

const StyledPlayerCard = styled(Card)`
  width: 250px;
  margin: 8px;
  box-shadow: 0px 4px 16px rgba(0, 0, 0, 0.2);
`;

export const PlayerCard = ({ player, resourceTiles }: PlayerCardProps) => {
  const [selectedCards, setSelectedCards] = useState<number[]>([]);
  const handleClickCard = (index: number) => {
    if (selectedCards.includes(index)) {
      setSelectedCards(selectedCards.filter((i) => i !== index));
    } else if (selectedCards.length < 2) {
      setSelectedCards([...selectedCards, index]);
      // TODO selectedCardsの全ての要素が2の時にボタンを活性化させる
    }
  };

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
              <ResourceTileCard
                resourceTiles={resources}
                selected={selectedCards.includes(index)}
                handleSelect={() => handleClickCard(index)}
              />
            </div>
          );
        })}
      </CardContent>
    </StyledPlayerCard>
  );
};

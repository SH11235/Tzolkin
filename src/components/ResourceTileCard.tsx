import styled from "@emotion/styled";
import { Card, CardContent } from "@mui/material";
import { ResourceTile, WorkSpace } from "../types/FirstResource";

type ResourceTileCardProps = {
  resourceTile: ResourceTile;
  handleSelect?: () => void;
  selected?: boolean;
};

const StyledResourceCard = styled(Card)<{ selected?: boolean }>`
  margin-bottom: 8px;
  border: ${({ selected }) =>
    selected ? "3px solid #f8bbd0" : "3px solid #d8c289"};
`;

const StyledWorkSpaceCard = styled(CardContent)`
  background-color: #d8c289;
  padding-bottom: 0px !important;
  font-size: 8px;
  color: gray;
  display: flex;
  justify-content: center;
  align-items: center;
`;

const StyledResourceCardContent = styled(CardContent)`
  background-color: #d8c289;
`;

export const ResourceTileCard = ({
  resourceTile,
  handleSelect,
  selected,
}: ResourceTileCardProps) => {
  const propertyOrder = [
    "corn",
    "wood",
    "stone",
    "gold",
    "skull",
    "worker",
    "chaac",
    "quetzalcoatl",
    "kukulkan",
    "save_corn",
    "agriculture_skill",
    "resource_skill",
    "construction_skill",
    "temple_skill",
    "work_space",
  ];

  const resources = Object.entries(resourceTile).filter(
    ([_, value]) => value !== null && value !== false
  );

  const sortedResourceTiles = resources.sort(([a], [b]) => {
    return propertyOrder.indexOf(a) - propertyOrder.indexOf(b);
  });

  const workSpaceIndex = sortedResourceTiles.findIndex(
    ([key]) => key === "work_space"
  );

  return (
    <StyledResourceCard onClick={handleSelect} selected={selected}>
      <StyledResourceCardContent>
        {sortedResourceTiles.map(([key, value]) => {
          if (key === "work_space") {
            return null;
          } else {
            return <div key={key}>{`${key}${value}`}</div>;
          }
        })}
      </StyledResourceCardContent>
      {workSpaceIndex >= 0 && (
        <StyledWorkSpaceCard>
          {Object.entries(
            sortedResourceTiles[workSpaceIndex][1] as WorkSpace
          ).map(([key, value]) => `${key} ${value}`)}
        </StyledWorkSpaceCard>
      )}
    </StyledResourceCard>
  );
};

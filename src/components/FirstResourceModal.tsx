import styled from "@emotion/styled";
import { Button, Card, CardContent, CardHeader } from "@mui/material";
import { invoke } from "@tauri-apps/api";
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
  const handleConfirm = () => {
    players.forEach((player, index) => {
      const selectedResourceTiles = firstResources[index].filter(
        (resourceTile) => {
          return resourceTile.selected;
        }
      );
      selectedResourceTiles.forEach(async (resourceTile) => {
        // corn: number | null;
        // wood: number | null;
        // stone: number | null;
        // gold: number | null;
        // skull: number | null;
        // worker: number | null;
        // chaac: number | null;
        // quetzalcoatl: number | null;
        // kukulkan: number | null;
        // save_corn: boolean;
        // agriculture_skill: number | null;
        // resource_skill: number | null;
        // construction_skill: number | null;
        // temple_skill: number | null;
        if (resourceTile.agriculture_skill) {
          await invoke("raise_technology_level", {
            playerId: player.id,
            technologyType: "agriculture",
            rewardOption: null,
          });
        }
        if (resourceTile.resource_skill) {
          await invoke("raise_technology_level", {
            playerId: player.id,
            technologyType: "resource",
            rewardOption: null,
          });
        }
        if (resourceTile.construction_skill) {
          await invoke("raise_technology_level", {
            playerId: player.id,
            technologyType: "construction",
            rewardOption: null,
          });
        }
        if (resourceTile.temple_skill) {
          await invoke("raise_technology_level", {
            playerId: player.id,
            technologyType: "temple",
            rewardOption: null,
          });
        }
      });
    });
  };

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

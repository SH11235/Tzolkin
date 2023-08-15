import styled from "@emotion/styled";
import { Button, Card, CardContent, CardHeader } from "@mui/material";
import { invoke } from "@tauri-apps/api";
import { useState } from "react";
import { CHAAC, KUKULKAN, QUETZALCOATL } from "../constant";
import { FirstResourcesState } from "../types/FirstResource";
import { GamePlayers } from "../types/GamePlayer";
import { PlayerCard } from "./PlayerCard";

type FirstResourceProps = {
  firstResources: FirstResourcesState;
  setFirstResources: React.Dispatch<React.SetStateAction<FirstResourcesState>>;
  players: GamePlayers;
  setPlayers: React.Dispatch<React.SetStateAction<GamePlayers>>;
  setFieldSkulls: React.Dispatch<React.SetStateAction<number>>;
};

const StyledCard = styled(Card)`
  max-width: 90%;
  width: 100%;
  position: absolute;
  text-align: center;
  padding-bottom: 26px;
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
  setFieldSkulls,
}: FirstResourceProps) => {
  const [done, setDone] = useState(false);
  const handleConfirm = async () => {
    await Promise.all(
      players.map(async (player, index) => {
        const selectedResourceTiles = firstResources[index].filter(
          (resourceTile) => {
            return resourceTile.selected;
          }
        );
        await Promise.all(
          selectedResourceTiles.map(async (resourceTile) => {
            if (resourceTile.corn) {
              await invoke("add_resource", {
                playerId: player.id,
                resourceType: "corn",
                amount: resourceTile.corn,
              });
            }
            if (resourceTile.wood) {
              await invoke("add_resource", {
                playerId: player.id,
                resourceType: "wood",
                amount: resourceTile.wood,
              });
            }
            if (resourceTile.stone) {
              await invoke("add_resource", {
                playerId: player.id,
                resourceType: "stone",
                amount: resourceTile.stone,
              });
            }
            if (resourceTile.gold) {
              await invoke("add_resource", {
                playerId: player.id,
                resourceType: "gold",
                amount: resourceTile.gold,
              });
            }
            if (resourceTile.skull) {
              await invoke("add_resource", {
                playerId: player.id,
                resourceType: "skull",
                amount: resourceTile.skull,
              });
            }
            if (resourceTile.worker) {
              await invoke("add_worker", {
                playerId: player.id,
              });
            }
            if (resourceTile.chaac) {
              await invoke("raise_temple_faith", {
                playerId: player.id,
                templeType: CHAAC,
                amount: resourceTile.chaac,
              });
            }
            if (resourceTile.quetzalcoatl) {
              await invoke("raise_temple_faith", {
                playerId: player.id,
                templeType: QUETZALCOATL,
                amount: resourceTile.quetzalcoatl,
              });
            }
            if (resourceTile.kukulkan) {
              await invoke("raise_temple_faith", {
                playerId: player.id,
                templeType: KUKULKAN,
                amount: resourceTile.kukulkan,
              });
            }
            if (resourceTile.save_corn) {
              await invoke("save_corn", {
                playerId: player.id,
                saveType: "single",
              });
            }
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
          })
        );
      })
    );
    const players_state: GamePlayers = await invoke("get_players");
    setPlayers(players_state);
    const fieldSkulls: number = await invoke("get_field_skulls");
    setFieldSkulls(fieldSkulls);
    setDone(true);
  };

  if (firstResources.length > 0 && !done) {
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
          sx={{
            backgroundColor: firstResources.every((firstResource) => {
              return (
                firstResource.filter((resourceTile) => {
                  return resourceTile.selected;
                }).length === 2
              );
            })
              ? "#1976d2"
              : "#bdbdbd",
            "&:hover": {
              backgroundColor: firstResources.every((firstResource) => {
                return (
                  firstResource.filter((resourceTile) => {
                    return resourceTile.selected;
                  }).length === 2
                );
              })
                ? "#0d47a1"
                : "#bdbdbd",
            },
            color: "white",
          }}
          size="large"
        >
          決定
        </Button>
      </StyledCard>
    );
  } else {
    return null;
  }
};

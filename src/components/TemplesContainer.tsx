import React from "react";
import { Temple } from "./Temple";
import {
  ChaacBonus,
  QuetzalcoatlBonus,
  KukulkanBonus,
  CHAAC,
  ChaacTopBonusPoints,
  QuetzalcoatlTopBonusPoints,
  KukulkanTopBonusPoints,
  QUETZALCOATL,
  KUKULKAN,
} from "../constant";
import { GamePlayers } from "../types/GamePlayer";

type TemplesContainerProps = {
  players: GamePlayers;
};

export const TemplesContainer: React.FC<TemplesContainerProps> = ({
  players,
}) => {
  return (
    <span className="temples">
      <Temple
        name={`${CHAAC}`}
        playerScores={
          players.map((player) => {
            return {
              color: player.color,
              index: player.temple_faith.chaac,
            };
          }) || []
        }
        templeBonus={ChaacBonus}
        templePoints={ChaacTopBonusPoints}
        templeColor="brown"
      />
      <Temple
        name={QUETZALCOATL}
        playerScores={
          players.map((player) => {
            return {
              color: player.color,
              index: player.temple_faith.quetzalcoatl,
            };
          }) || []
        }
        templeBonus={QuetzalcoatlBonus}
        templePoints={QuetzalcoatlTopBonusPoints}
        templeColor="yellow"
      />
      <Temple
        name={KUKULKAN}
        playerScores={
          players.map((player) => {
            return {
              color: player.color,
              index: player.temple_faith.kukulkan,
            };
          }) || []
        }
        templeBonus={KukulkanBonus}
        templePoints={KukulkanTopBonusPoints}
        templeColor="green"
      />
    </span>
  );
};

import { PlayerColor } from "../types/GamePlayer";
import "./temple.css";

export type TempleBonus = {
  resource: "stone" | "gold" | "wood" | "skull" | null;
  point: number;
}[];

type TempleColor = "brown" | "yellow" | "green";

type Props = {
  name: string;
  playerScores: {
    color: PlayerColor;
    index: number;
  }[];
  templeBonus: TempleBonus;
  templeColor: TempleColor;
};

export const Temple: React.FC<Props> = ({
  name,
  playerScores,
  templeBonus,
  templeColor,
}) => {
  const offset = 9 - templeBonus.length;
  const array = Array(offset).fill(0);
  const offsetRows = array.map((_, index) => {
    return (
      <tr key={`offsetRows-${index}`}>
        <td className="temple-cell"></td>
        <td className="temple-cell"></td>
        <td className="temple-cell"></td>
      </tr>
    );
  });
  const rows = templeBonus.map((row, rowIndex) => {
    const playerScoreElements = playerScores.map((playerScore) => {
      const offset = templeBonus.length - 2;
      // chaacのとき: playerScore.index + rowIndex === 5
      // playerScore.index === -1 → rowIndex === 6
      // playerScore.index === 0 → rowIndex === 5
      // playerScore.index === 1 → rowIndex === 4
      // playerScore.index === 2 → rowIndex === 3
      // playerScore.index === 3 → rowIndex === 2
      // playerScore.index === 4 → rowIndex === 1
      // quetzalcoatlのとき playerScore.index + rowIndex === 7
      // playerScore.index === 0 → rowIndex === 7
      // playerScore.index === 1 → rowIndex === 6
      // playerScore.index === 2 → rowIndex === 5
      // playerScore.index === 3 → rowIndex === 4
      // playerScore.index === 4 → rowIndex === 3
      // playerScore.index === 5 → rowIndex === 2
      // playerScore.index === 6 → rowIndex === 1
      // playerScore.index === 7 → rowIndex === 0
      if (playerScore.index + rowIndex === offset) {
        let classNamePlayer = playerScore.color;
        return <span key={`${classNamePlayer}-${rowIndex}`} className={`circle-${classNamePlayer}`}></span>;
      }
    });
    return (
      <tr key={rowIndex}>
        <td className="temple-cell temple-cell-resource">{row.resource}</td>
        <td className="temple-cell temple-cell-player">{playerScoreElements}</td>
        <td className="temple-cell temple-cell-score">{row.point}</td>
      </tr>
    );
  });
  return (
    <div className="temple-container">
      <span>{name}</span>
      <table className={`temple-table temple-${templeColor}`} border={1}>
        <tbody>
          {offsetRows}
          {rows}
        </tbody>
      </table>
    </div>
  );
};

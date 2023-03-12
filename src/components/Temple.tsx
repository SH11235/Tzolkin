import "./Temple.css";

type PlayerColor = "red" | "blue" | "green" | "yellow" | "orange";
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
  const additonalRows = array.map((_, index) => {
    return (
      <tr key={`additonalRows-${index}`}>
        <td className="temple-cell"></td>
        <td className="temple-cell"></td>
        <td className="temple-cell"></td>
      </tr>
    );
  });
  console.log(additonalRows);
  const rows = templeBonus.map((row, rowIndex) => {
    if (rowIndex === 0) {
      return (
        <tr key={rowIndex}>
          <td className="temple-cell"></td>
          <td className="temple-cell">
            {/* <span className="circle"></span> */}
          </td>
          <td className="temple-cell">{row.point}</td>
        </tr>
      );
    } else {
      return (
        <tr key={rowIndex}>
          <td className="temple-cell">{row.resource}</td>
          <td className="temple-cell"></td>
          <td className="temple-cell">{row.point}</td>
        </tr>
      );
    }
  });
  return (
    <div className="temple-container">
      <span>{name}</span>
      <table className={`temple-table temple-${templeColor}`} border={1}>
        <tbody>
          {additonalRows}
          {rows}
        </tbody>
      </table>
    </div>
  );
};

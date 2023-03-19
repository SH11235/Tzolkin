interface PalenqueSpaceProps {
  palenqueChips: {
    wood: number;
    corn: number;
  }[];
  palenqueWorkers: (null | string)[];
}

export const PalenqueSpace = ({
  palenqueChips,
  palenqueWorkers,
}: PalenqueSpaceProps) => {
  return (
    <div className="palenque-space">
      <span className="space-name">Palanque</span>
      <span className="palenque-space-container">
        <p></p>
        <p></p>
        <p className="space">
          0
          <div
            className={`worker-space ${
              palenqueWorkers[0] ? "player-" + palenqueWorkers[0] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p></p>
        <p></p>
        <p className="space">
          1
          <div
            className={`worker-space ${
              palenqueWorkers[1] ? "player-" + palenqueWorkers[1] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p></p>
        <p className="corn-chip-space">{palenqueChips[0].corn}</p>
        <p className="space">
          2
          <div
            className={`worker-space ${
              palenqueWorkers[2] ? "player-" + palenqueWorkers[2] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p className="wood-chip-space">{palenqueChips[1].wood}</p>
        <p className="corn-chip-space">{palenqueChips[1].corn}</p>
        <p className="space">
          3
          <div
            className={`worker-space ${
              palenqueWorkers[3] ? "player-" + palenqueWorkers[3] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p className="wood-chip-space">{palenqueChips[2].wood}</p>
        <p className="corn-chip-space">{palenqueChips[2].corn}</p>
        <p className="space">
          4
          <div
            className={`worker-space ${
              palenqueWorkers[4] ? "player-" + palenqueWorkers[4] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p className="wood-chip-space">{palenqueChips[3].wood}</p>
        <p className="corn-chip-space">{palenqueChips[3].corn}</p>
        <p className="space">
          5
          <div
            className={`worker-space ${
              palenqueWorkers[5] ? "player-" + palenqueWorkers[5] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p></p>
        <p></p>
        <p className="space">
          6
          <div
            className={`worker-space ${
              palenqueWorkers[6] ? "player-" + palenqueWorkers[6] : ""
            }`}
          ></div>
        </p>
      </span>
      <span className="palenque-space-container">
        <p></p>
        <p></p>
        <p className="space">
          7
          <div
            className={`worker-space ${
              palenqueWorkers[7] ? "player-" + palenqueWorkers[7] : ""
            }`}
          ></div>
        </p>
      </span>
    </div>
  );
};

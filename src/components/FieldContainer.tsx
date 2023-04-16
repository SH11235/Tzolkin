import React from "react";
import "./FieldContainer.css";
import { PalenqueSpace } from "./PalenqueSpace";

interface FieldContainerProps {
  palenqueChips: {
    wood: number;
    corn: number;
  }[];
  palenqueWorkers: (null | string)[];
}

export const FieldContainer: React.FC<FieldContainerProps> = ({
  palenqueChips,
  palenqueWorkers,
}) => {
  return (
    <div className="field-container">
      <PalenqueSpace
        palenqueChips={palenqueChips}
        palenqueWorkers={palenqueWorkers}
      />
      <div className="yaxchilan-space">
        <span className="space-name">Yaxchilan</span>
        <span className="space-container">
          <span className="space">
            0<div className="worker-space"></div>
          </span>
          <span className="space">
            1<div className="worker-space"></div>
          </span>
          <span className="space">
            2<div className="worker-space"></div>
          </span>
          <span className="space">
            3<div className="worker-space"></div>
          </span>
          <span className="space">
            4<div className="worker-space"></div>
          </span>
          <span className="space">
            5<div className="worker-space"></div>
          </span>
          <span className="space">
            6<div className="worker-space"></div>
          </span>
          <span className="space">
            7<div className="worker-space"></div>
          </span>
        </span>
      </div>
      <div className="tikal-space">
        <span className="space-name">Tikal</span>
        <span className="space-container">
          <span className="space">
            0<div className="worker-space"></div>
          </span>
          <span className="space">
            1<div className="worker-space"></div>
          </span>
          <span className="space">
            2<div className="worker-space"></div>
          </span>
          <span className="space">
            3<div className="worker-space"></div>
          </span>
          <span className="space">
            4<div className="worker-space"></div>
          </span>
          <span className="space">
            5<div className="worker-space"></div>
          </span>
          <span className="space">
            6<div className="worker-space"></div>
          </span>
          <span className="space">
            7<div className="worker-space"></div>
          </span>
        </span>
      </div>
      <div className="uxmal-space">
        <span className="space-name">Uxmal</span>
        <span className="space-container">
          <span className="space">
            0<div className="worker-space"></div>
          </span>
          <span className="space">
            1<div className="worker-space"></div>
          </span>
          <span className="space">
            2<div className="worker-space"></div>
          </span>
          <span className="space">
            3<div className="worker-space"></div>
          </span>
          <span className="space">
            4<div className="worker-space"></div>
          </span>
          <span className="space">
            5<div className="worker-space"></div>
          </span>
          <span className="space">
            6<div className="worker-space"></div>
          </span>
          <span className="space">
            7<div className="worker-space"></div>
          </span>
        </span>
      </div>
      <div className="chichen-itza-space">
        <span className="space-name">Chichen Itza</span>
        <span className="space">
          0<div className="worker-space"></div>
        </span>
        <span className="space">
          1<div className="worker-space"></div>
        </span>
        <span className="space">
          2<div className="worker-space"></div>
        </span>
        <span className="space">
          3<div className="worker-space"></div>
        </span>
        <span className="space">
          4<div className="worker-space"></div>
        </span>
        <span className="space">
          5<div className="worker-space"></div>
        </span>
        <span className="space">
          6<div className="worker-space"></div>
        </span>
        <span className="space">
          7<div className="worker-space"></div>
        </span>
        <span className="space">
          8<div className="worker-space"></div>
        </span>
        <span className="space">
          9<div className="worker-space"></div>
        </span>
        <span className="space">
          10<div className="worker-space"></div>
        </span>
        <span className="space">
          11<div className="worker-space"></div>
        </span>
      </div>
    </div>
  );
};

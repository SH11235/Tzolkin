import { useState } from "react";
import "./App.css";
import { invoke } from "@tauri-apps/api/tauri";
import { Temple } from "./components/temple";
import { ChaacBonus, QuetzalcoatlBonus, KukulkanBonus } from "./constant";
import { Players } from "./components/players";
import { PalenqueSpace } from "./components/PalenqueSpace";

function App() {
  const [players, setPlayers] = useState<{
    name: string;
    index: number;
    color: string;
  }[]>([]);
  const [palenqueChips, setPalenqueChips] = useState<
    {
      wood: number;
      corn: number;
    }[]
  >([
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
    {
      wood: 0,
      corn: 0,
    },
  ]);
  const [palenqueWorkers, setPalenqueWorkers] = useState<(null | string)[]>([
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
  ]);

  async function player() {
    try {
      const returnNumber = await invoke("set_number_of_players", {
        number: players.length,
      });
      if (returnNumber !== players.length) {
        throw new Error("プレイ人数の設定に失敗しました");
      }
      setPalenqueChips([
        {
          wood: 0,
          corn: returnNumber,
        },
        {
          wood: returnNumber,
          corn: returnNumber,
        },
        {
          wood: returnNumber,
          corn: returnNumber,
        },
        {
          wood: returnNumber,
          corn: returnNumber,
        },
      ]);
      const input = document.getElementById(
        "number-of-players-input"
      ) as HTMLInputElement;
      if (input) {
        input.disabled = true;
      }
      const button = document.getElementById(
        "submit-button"
      ) as HTMLButtonElement;
      if (button) {
        button.disabled = true;
      }
    } catch {
      // ユーザーに警告を出す
      alert("プレイ人数の設定に失敗しました");
    }
  }

  return (
    <div className="container">
      <div className="row">
        <a
          href="https://ja.boardgamearena.com/gamepanel?game=tzolkin"
          target="_blank"
        >
          <img src="/gear_logo.png" className="logo gear" alt="Gear logo" />
        </a>
      </div>

      <div className="row">
        <Players
          players={players}
          setPlayers={setPlayers}
          onSubmit={player}
        />
      </div>

      <div className="row">
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
        <span className="temples">
          <Temple
            name="Chaac"
            playerScores={[{ color: "red", index: 0 }]}
            templeBonus={ChaacBonus}
            templeColor="brown"
          />
          <Temple
            name="Quetzalcoatl"
            playerScores={[{ color: "red", index: 0 }]}
            templeBonus={QuetzalcoatlBonus}
            templeColor="yellow"
          />
          <Temple
            name="Kukulkan"
            playerScores={[{ color: "red", index: 0 }]}
            templeBonus={KukulkanBonus}
            templeColor="green"
          />
        </span>
      </div>

      <div className="row">
        <span>ターンプレイヤー: {players.length > 0 ? players[0].name : ""}</span>
      </div>
    </div>
  );
}

export default App;

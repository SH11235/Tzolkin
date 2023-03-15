import { useState } from "react";
import "./App.css";
import { invoke } from "@tauri-apps/api/tauri";
import { Temple } from "./components/temple";
import { ChaacBonus, QuetzalcoatlBonus, KukulkanBonus } from "./constant";

function App() {
  const [numberOfPlayers, setNumberOfPlayers] = useState(0);
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

  const numberOnChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = event.target.value;
    if (/^[1-4]$/.test(newValue)) {
      setNumberOfPlayers(Number(newValue));
    }
  };

  async function player() {
    try {
      const returnNumber = await invoke("set_number_of_players", {
        number: numberOfPlayers,
      });
      if (returnNumber !== numberOfPlayers) {
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
        <form
          onSubmit={(e) => {
            e.preventDefault();
            player();
          }}
        >
          プレイ人数:{" "}
          <input
            id="number-of-players-input"
            type="number"
            value={numberOfPlayers}
            onChange={numberOnChange}
            placeholder="プレイ人数"
          />
          <button id="submit-button" type="submit">
            決定
          </button>
        </form>
      </div>

      <div className="row">
        <div className="field-container">
          <div className="palenque-space">
            <span className="space-name">Palanque</span>
            <span>
              <p></p>
              <p></p>
              <p className="space">P0</p>
            </span>
            <span>
              <p></p>
              <p className="corn-chip-space">{palenqueChips[0].corn}</p>
              <p className="space">P1</p>
            </span>
            <span>
              <p></p>
              <p className="corn-chip-space">{palenqueChips[0].corn}</p>
              <p className="space">P2</p>
            </span>
            <span>
              <p className="wood-chip-space">{palenqueChips[1].wood}</p>
              <p className="corn-chip-space">{palenqueChips[1].corn}</p>
              <p className="space">P3</p>
            </span>
            <span>
              <p className="wood-chip-space">{palenqueChips[2].wood}</p>
              <p className="corn-chip-space">{palenqueChips[2].corn}</p>
              <p className="space">P4</p>
            </span>
            <span>
              <p className="wood-chip-space">{palenqueChips[3].wood}</p>
              <p className="corn-chip-space">{palenqueChips[3].corn}</p>
              <p className="space">P5</p>
            </span>
            <span>
              <p></p>
              <p></p>
              <p className="space">P6</p>
            </span>
            <span>
              <p></p>
              <p></p>
              <p className="space">P7</p>
            </span>
          </div>
          <div className="yaxchilan-space">
            <span className="space-name">Yaxchilan</span><span className="space-container">
            <span className="space">P0</span>
            <span className="space">P1</span>
            <span className="space">P2</span>
            <span className="space">P3</span>
            <span className="space">P4</span>
            <span className="space">P5</span>
            <span className="space">P6</span>
            <span className="space">P7</span>
            </span>
          </div>
          <div className="tikal-space">
            <span className="space-name">Tikal</span>
            <span className="space-container">
              <span className="space">P0</span>
              <span className="space">P1</span>
              <span className="space">P2</span>
              <span className="space">P3</span>
              <span className="space">P4</span>
              <span className="space">P5</span>
              <span className="space">P6</span>
              <span className="space">P7</span>
            </span>
          </div>
          <div className="uxmal-space">
            <span className="space-name">Uxmal</span>
            <span className="space-container">
              <span className="space">P0</span>
              <span className="space">P1</span>
              <span className="space">P2</span>
              <span className="space">P3</span>
              <span className="space">P4</span>
              <span className="space">P5</span>
              <span className="space">P6</span>
              <span className="space">P7</span>
            </span>
          </div>
          <div className="chichen-itza-space">
            <span className="space-name">Chichen Itza</span>
            <span className="space">P0</span>
            <span className="space">P1</span>
            <span className="space">P2</span>
            <span className="space">P3</span>
            <span className="space">P4</span>
            <span className="space">P5</span>
            <span className="space">P6</span>
            <span className="space">P7</span>
            <span className="space">P8</span>
            <span className="space">P9</span>
            <span className="space">P10</span>
            <span className="space">P11</span>
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
    </div>
  );
}

export default App;

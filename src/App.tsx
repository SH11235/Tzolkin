import { useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";
import { Temple } from "./components/Temple";
import { ChaacBonus, QuetzalcoatlBonus, KukulkanBonus } from "./constant";

function App() {
  const [greetMsg, setGreetMsg] = useState("");
  const [name, setName] = useState("");

  // TODO
  // const [players, setPlayers] = useState("");

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    setGreetMsg(await invoke("greet", { name }));
  }

  return (
    <div className="container">
      <div className="row">
        <a href="https://ja.boardgamearena.com/gamepanel?game=tzolkin" target="_blank">
          <img src="/gear_logo.png" className="logo gear" alt="Gear logo" />
        </a>
      </div>

      <p>Click on the Tauri, Vite, and React logos to learn more.</p>

      <div className="row">
        <div className="field-container">

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
        <form
          onSubmit={(e) => {
            e.preventDefault();
            greet();
          }}
        >
          <input
            id="greet-input"
            onChange={(e) => setName(e.currentTarget.value)}
            placeholder="Enter a name..."
          />
          <button type="submit">Greet</button>
        </form>
      </div>
      <p>{greetMsg}</p>
    </div>
  );
}

export default App;

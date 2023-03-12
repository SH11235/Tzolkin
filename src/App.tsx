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
      <h1>Welcome to Tauri!</h1>

      <div className="row">
        <a href="https://vitejs.dev" target="_blank">
          <img src="/vite.svg" className="logo vite" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank">
          <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
        </a>
        <a href="https://reactjs.org" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>

      <p>Click on the Tauri, Vite, and React logos to learn more.</p>

      <div className="temples">
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

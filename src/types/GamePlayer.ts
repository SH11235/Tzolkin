type WorkerPosition = "Hand" | "Locked";

type Worker = {
  position: WorkerPosition;
}

type Resource = {
  golds: number;
  skulls: number;
  stones: number;
  woods: number;
}

type Technology = {
  agriculture: number;
  construction: number;
  resource: number;
  temple: number;
}

type TempleFaith = {
  chaac: number;
  kukulkan: number;
  quetzalcoatl: number;
}

export type PlayerColor = "red" | "blue" | "green" | "yellow" | "orange";

export type Player = {
  id: number;
  color: PlayerColor;
  corn_tiles: number;
  corns: number;
  name: string;
  order: number;
  points: number;
  resource: Resource;
  technology: Technology;
  temple_faith: TempleFaith;
  wood_tiles: number;
  workers: Worker[];
}

export type GamePlayers = Player[];

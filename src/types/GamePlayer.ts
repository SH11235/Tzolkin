type WorkerPosition = "Hand" | "Locked";

type Worker = {
  position: WorkerPosition;
};

type Resource = {
  golds: number;
  skulls: number;
  stones: number;
  woods: number;
};

type Technology = {
  agriculture: number;
  construction: number;
  resource: number;
  temple: number;
};

type TempleFaith = {
  chaac: number;
  kukulkan: number;
  quetzalcoatl: number;
};

export type PlayerColor = "red" | "blue" | "green" | "yellow" | "orange";

type CornSave = {
  single: number;
  triple: number;
  all: number;
};

export type Player = {
  id: number;
  name: string;
  color: PlayerColor;
  order: number;
  accelerating_ability: boolean;
  corn_save: CornSave;
  workers: Worker[];
  technology: Technology;
  temple_faith: TempleFaith;
  corns: number;
  resource: Resource;
  corn_tiles: number;
  wood_tiles: number;
  points: number;
};

export type GamePlayers = Player[];

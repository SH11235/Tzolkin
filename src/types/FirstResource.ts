export type WorkSpace = {
  [key: string]: number;
};

export type ResourceTile = {
  work_space: WorkSpace;
  corn: number | null;
  wood: number | null;
  stone: number | null;
  gold: number | null;
  skull: number | null;
  worker: number | null;
  chaac: number | null;
  quetzalcoatl: number | null;
  kukulkan: number | null;
  save_corn: boolean;
  agriculture_skill: number | null;
  resource_skill: number | null;
  construction_skill: number | null;
  temple_skill: number | null;
};

export type FirstResources = ResourceTile[][];

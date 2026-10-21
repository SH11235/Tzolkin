import { catalogHash } from './publicReplay';

export function sessionSave(session: object): string {
  return JSON.stringify({
    ...session,
    schema: 'tzolkin-session-v2',
    rulesVersion: 2,
    catalogHash: catalogHash(),
  });
}

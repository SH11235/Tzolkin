/// <reference types="vite/client" />
import catalogSource from '../../crates/tzolkin-core/data/catalog.json?raw';

export const RULES_VERSION = 2;
export const SESSION_SCHEMA = 'tzolkin-session-v2';
let catalogFingerprint = 0xcbf29ce484222325n;
for (const byte of new TextEncoder().encode(catalogSource))
  catalogFingerprint = BigInt.asUintN(64, (catalogFingerprint ^ BigInt(byte)) * 0x100000001b3n);
export const CATALOG_HASH = catalogFingerprint.toString(16).padStart(16, '0');

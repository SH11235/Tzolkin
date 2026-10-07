import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manifest = readFileSync(join(root, 'crates/tzolkin-wasm/Cargo.toml'), 'utf8');
const bindgenVersion = manifest.match(/^wasm-bindgen = "=([^"]+)"$/m)?.[1];
if (!bindgenVersion) throw new Error('The wasm-bindgen version must be pinned in Cargo.toml.');

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed with exit code ${result.status}.`);
  return result;
}

function capture(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8' });
  if (result.error || result.status !== 0) return undefined;
  return result.stdout.trim();
}

const metadata = JSON.parse(capture('cargo', ['metadata', '--format-version', '1', '--no-deps']));
const targetDirectory = metadata.target_directory;
const toolRoot = join(targetDirectory, 'wasm-tools');
const cachedBindgen = join(
  toolRoot,
  'bin',
  process.platform === 'win32' ? 'wasm-bindgen.exe' : 'wasm-bindgen',
);
let bindgen = process.env.WASM_BINDGEN_CLI || 'wasm-bindgen';
if (capture(bindgen, ['--version']) !== `wasm-bindgen ${bindgenVersion}`) {
  bindgen = cachedBindgen;
  if (capture(bindgen, ['--version']) !== `wasm-bindgen ${bindgenVersion}`) {
    console.info(`Installing wasm-bindgen ${bindgenVersion} into the project build directory.`);
    run('cargo', [
      'install',
      'wasm-bindgen-cli',
      '--version',
      bindgenVersion,
      '--locked',
      '--root',
      toolRoot,
    ]);
  }
}

const targets = capture('rustup', ['target', 'list', '--installed'])?.split(/\s+/) || [];
if (!targets.includes('wasm32-unknown-unknown'))
  run('rustup', ['target', 'add', 'wasm32-unknown-unknown']);

function build(target, outputDirectory, testApi) {
  const arguments_ = [
    'build',
    '--locked',
    '-p',
    'tzolkin-wasm',
    '--target',
    'wasm32-unknown-unknown',
    '--release',
  ];
  if (testApi) arguments_.push('--features', 'test-api');
  run('cargo', arguments_);
  mkdirSync(outputDirectory, { recursive: true });
  run(bindgen, [
    join(targetDirectory, 'wasm32-unknown-unknown/release/tzolkin_wasm.wasm'),
    '--target',
    target,
    '--out-dir',
    outputDirectory,
    '--out-name',
    'tzolkin_wasm',
  ]);
  if (target === 'nodejs') {
    writeFileSync(join(outputDirectory, 'package.json'), '{"private":true,"type":"commonjs"}\n');
  }
}

build('web', join(root, 'generated/wasm'), false);
if (process.argv.includes('--test')) {
  build('nodejs', join(root, 'generated/wasm-node'), true);
  run('cargo', ['build', '--locked', '-p', 'tzolkin-core', '--example', 'dispatch']);
}
if (!existsSync(join(root, 'generated/wasm/tzolkin_wasm_bg.wasm'))) {
  throw new Error('The WebAssembly build did not produce its browser asset.');
}

# ツォルキン — マヤの暦

Tzolk’in: The Mayan Calendar を、同じ画面を囲む 2〜4 人で遊ぶ非公式アプリです。ゲーム処理は共通の Rust コアで行い、ブラウザ版は WebAssembly、Tauri デスクトップ版はネイティブ Rust を使用します。React は表示と入力を担当します。

## 起動

開発・ビルドには Node.js 24 以上と Rust 1.90 以上を使用します。ブラウザ版も Rust コアをビルドします。

```sh
npm ci
npm run dev
```

最初の起動時に `wasm32-unknown-unknown` ターゲットを準備し、固定バージョンの `wasm-bindgen` CLI を必要に応じて `target/wasm-tools` にインストールします。生成した Wasm と JavaScript は `generated/wasm` に置きます。生成物は Git 管理対象外です。ビルド後のブラウザ版を実行する端末に Rust は必要ありません。

表示されたローカル URL を開き、プレイヤー数と名前を設定します。各プレイヤーが初期資源を選んだら対局を開始できます。

歯車の「配置する」でワーカーを置き、自分の色のワーカーを押すと回収します。アクション欄で技術、建物、交易、神殿などを選択し、「手番を終了」で次のプレイヤーへ進みます。配置と回収の排他、配置費用、歯車の回転、食料日、最終得点は自動処理です。

進行はブラウザ・アプリ内に自動保存されます。再起動後は「保存した対局を続ける」を選びます。「1つ戻す」は直近 60 操作まで取り消せます。保存ファイルの書き出し・読み込みで、対局を端末間で持ち運べます。

## デスクトップ版

[Tauri の各 OS の開発環境](https://v2.tauri.app/start/prerequisites/)が必要です。Linux では GTK 3、WebKitGTK 4.1、AppIndicator などの開発パッケージを用意します。

```sh
npm run tauri dev
npm run tauri build
```

## ゲーム内容と資料

基本ゲームの初期資源 21 枚、建物 32 枚、記念碑 13 枚、5 つの歯車、4 つの技術、3 つの神殿を扱います。2〜3 人時のダミーワーカー、森の焼き払い、物乞い、スタートプレイヤー、2 日進行、4 回の食料日、同点時の順位もルールエンジンで処理します。

[ルール参照資料](docs/rules/README.md)に出版社の日本語・英語 PDF、検索用テキスト、タイルの写真、補助 Wiki を保存しています。採用した出典と SHA-256 は [sources.json](docs/rules/sources.json) に記録しています。カード名はこのアプリで識別しやすい日本語名で、公式の固有名ではありません。

開始画面の「追加建物8枚を混ぜる」を有効にすると、既存コードにあった追加建物のモジュールも使えます。建て替え、毎食料日の報酬、記念碑の連鎖建設、技術交換、髑髏を捧げる建物を扱います。既定では無効です。

部族・予言・5 人対局は含みません。対局はローカルで行い、オンライン対戦や AI プレイヤーはありません。

## 検証

```sh
npm run format:check
npm run lint
npm run test:core
npm test
npm run build
npx playwright install chromium
npm run test:e2e
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets -- -D warnings
npm run tauri build -- --no-bundle
```

ルールの回帰テストは Rust コアを Node 向け Wasm で呼び出します。さらに、保存した 6 対局・1,740 操作の参照データに対し、ネイティブ Rust と本番用 Wasm の状態・選択肢・配置費用を各操作で比較します。保存 JSON のバージョンは `1` のままで、既存の対局を読み込めます。検証を省略するテスト用の Wasm API は本番ビルドに含めません。

TypeScript は ESLint の解析器が公式に対応する最新安定版を使用します。依存関係は npm と Cargo の lockfile で固定しています。

## 構成

- `crates/tzolkin-core/`: Tauri・ブラウザに依存しないゲーム状態、ルール、合法手、得点計算、保存データ検証。
- `crates/tzolkin-core/data/catalog.json`: 出版社の資料と照合した共通カタログ。Rust のルール処理と UI が同じデータを参照。
- `crates/tzolkin-wasm/`: ブラウザ用の Wasm 接続部分。
- `src/game/engine.ts`: Wasm／Tauri を選ぶ非同期接続部分。ゲームのルール処理は Rust に委譲。
- `src/ui/`: 歯車、プレイヤー情報、神殿、建物、アクション選択の表示。
- `src/App.tsx`: 対局セッション、保存、取り消し、画面の切り替え。
- `src-tauri/`: 同じ Rust コアを呼び出す Tauri 2 のデスクトップアプリ。

原作: Daniele Tascini & Simone Luciani / Czech Games Edition。原作の画像・ルール資料の権利は各権利者に帰属します。本プロジェクトは出版社の公式アプリではありません。

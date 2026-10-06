# ツォルキン — マヤの暦

Tzolk’in: The Mayan Calendar を、同じ画面を囲む 2〜4 人で遊ぶ非公式アプリです。ブラウザ版と Tauri デスクトップ版は、同じ TypeScript ルールエンジンで進行します。

## 起動

Node.js 24 以上を使用します。

```sh
npm ci
npm run dev
```

表示されたローカル URL を開き、プレイヤー数と名前を設定します。各プレイヤーが初期資源を選んだら対局を開始できます。

歯車の「配置する」でワーカーを置き、自分の色のワーカーを押すと回収します。アクション欄で技術、建物、交易、神殿などを選択し、「手番を終了」で次のプレイヤーへ進みます。配置と回収の排他、配置費用、歯車の回転、食料日、最終得点は自動処理です。

進行はブラウザ・アプリ内に自動保存されます。再起動後は「保存した対局を続ける」を選びます。「1つ戻す」は直近 60 操作まで取り消せます。保存ファイルの書き出し・読み込みで、対局を端末間で持ち運べます。

## デスクトップ版

Rust 1.90 以上と、[Tauri の各 OS の開発環境](https://v2.tauri.app/start/prerequisites/)が必要です。Linux では GTK 3、WebKitGTK 4.1、AppIndicator などの開発パッケージを用意します。

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
npm test
npm run build
npx playwright install chromium
npm run test:e2e
cargo fmt --manifest-path src-tauri/Cargo.toml --check
cargo clippy --locked --manifest-path src-tauri/Cargo.toml -- -D warnings
npm run tauri build -- --no-bundle
```

TypeScript は ESLint の解析器が公式に対応する最新安定版を使用します。依存関係は npm と Cargo の lockfile で固定しています。

## 構成

- `src/game/catalog.ts`: 出版社の資料と照合したタイル、技術、神殿のデータ。
- `src/game/engine.ts`: 副作用のないゲーム状態の更新、合法手と選択肢、保存データの検証。
- `src/ui/`: 歯車、プレイヤー情報、神殿、建物、アクション選択の表示。
- `src/App.tsx`: 対局セッション、保存、取り消し、画面の切り替え。
- `src-tauri/`: Tauri 2 のデスクトップアプリ。

原作: Daniele Tascini & Simone Luciani / Czech Games Edition。原作の画像・ルール資料の権利は各権利者に帰属します。本プロジェクトは出版社の公式アプリではありません。

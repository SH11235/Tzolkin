import { getNnLatencyMeasurement } from '../game/cpuTiming';

export function NnLatency({ modelChecksum }: { modelChecksum: string }) {
  const measurement = getNnLatencyMeasurement(modelChecksum);
  const summary = measurement.summary;
  return (
    <output
      role="group"
      className="nn-latency-panel"
      aria-label="NNの判断時間"
      data-nn-latency-schema={measurement.schema}
      data-nn-latency={JSON.stringify(measurement)}
    >
      <b>
        合法判断のp95：
        {summary.legalP95Ms === null ? '未測定' : `${summary.legalP95Ms.toFixed(1)} ms`}
      </b>
      <span>
        合法 {summary.legalDecisions} / 完了試行 {summary.completedAttempts} · 期限超過{' '}
        {summary.deadlines} · 中止 {summary.cancellations} · 判断エラー {summary.errors}
      </span>
      <span>
        操作反映 {summary.applied} · 反映エラー {summary.applyFailures} · 結果破棄{' '}
        {summary.applyDiscards} · 反映待ち {summary.applyPending} · 反映前 {summary.applyNotStarted}
      </span>
      <small>
        このページの直近{summary.sampleLimit}試行から同じモデルを集計。入力作成から合法性確認まで。
        モデル準備と操作反映の時間は含みません。
      </small>
    </output>
  );
}

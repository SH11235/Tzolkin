/** One monotonic budget from observation construction through current legality.
 * Timers cannot interrupt synchronous work; callers check again when it returns. */
export const NN_DECISION_DEADLINE_MS = 1000;
const SAMPLE_LIMIT = 128;
type Outcome = 'legal' | 'deadline' | 'cancelled' | 'error';
type Application = 'notStarted' | 'pending' | 'applied' | 'failed' | 'discarded';
interface Sample {
  ordinal: number;
  gameId: string;
  modelChecksum: string;
  outcome: Outcome;
  application: Application | null;
  elapsedMs: number;
  inputMs: number | null;
  notifyMs: number | null;
  chooseMs: number | null;
  legalityMs: number | null;
}
const samples: Sample[] = [];
let nextOrdinal = 0;

export class NnDecisionDeadlineError extends Error {
  constructor() {
    super('NNの判断全体が期限を超えました。この局面から再開してください。');
    this.name = 'NnDecisionDeadlineError';
  }
}

export class NnDecisionTiming {
  private readonly startedAt = performance.now();
  readonly deadlineAt = this.startedAt + NN_DECISION_DEADLINE_MS;
  private inputAt?: number;
  private notifiedAt?: number;
  private chosenAt?: number;
  private outcome?: Outcome;
  private sample?: Sample;
  private readonly timeout: ReturnType<typeof setTimeout>;
  private readonly gameId: string;
  private readonly modelChecksum: string;

  constructor(
    game: { gameId: string; modelChecksum: string },
    private readonly onDeadline?: (error: NnDecisionDeadlineError) => void,
    private readonly onChange?: () => void,
  ) {
    this.gameId = game.gameId;
    this.modelChecksum = game.modelChecksum;
    this.timeout = setTimeout(() => {
      if (!this.outcome) this.expire();
    }, NN_DECISION_DEADLINE_MS);
  }

  remainingMs(): number {
    if (this.outcome === 'deadline') throw new NnDecisionDeadlineError();
    if (this.outcome) throw new Error('NN decision timing already completed');
    const remaining = this.deadlineAt - performance.now();
    if (remaining <= 0) throw this.expire();
    return remaining;
  }

  inputReady(): void {
    this.remainingMs();
    this.inputAt = performance.now();
  }

  notified(): void {
    this.remainingMs();
    this.notifiedAt = performance.now();
  }

  chosen(): void {
    this.remainingMs();
    this.chosenAt = performance.now();
  }

  /** Only call after the App checks the actual current legal authority. */
  legal(): void {
    this.remainingMs();
    if (this.inputAt === undefined || this.notifiedAt === undefined || this.chosenAt === undefined)
      throw new Error('NN decision timing is incomplete');
    const now = performance.now();
    if (now >= this.deadlineAt) throw this.expire();
    this.finish('legal', now);
  }

  expire(): NnDecisionDeadlineError {
    const error = new NnDecisionDeadlineError();
    if (this.finish('deadline', performance.now())) this.onDeadline?.(error);
    return error;
  }

  failed(failure?: unknown): void {
    this.finish(
      failure instanceof NnDecisionDeadlineError
        ? 'deadline'
        : failure instanceof Error && failure.name === 'AbortError'
          ? 'cancelled'
          : 'error',
      performance.now(),
    );
  }

  /** Application follows the completed legality measurement and has no deadline claim. */
  applicationStarted(): void {
    this.changeApplication('notStarted', 'pending');
  }

  applicationApplied(): void {
    this.changeApplication('pending', 'applied');
  }

  applicationFailed(): void {
    if (this.sample?.application === 'discarded') this.changeApplication('discarded', 'failed');
    else this.changeApplication('pending', 'failed');
  }

  applicationDiscarded(): void {
    this.changeApplication('pending', 'discarded');
  }

  private changeApplication(previous: Application, next: Application): void {
    if (!this.sample || this.sample.application !== previous) return;
    this.sample.application = next;
    this.onChange?.();
  }

  private finish(outcome: Outcome, now: number): boolean {
    if (this.outcome) return false;
    this.outcome = outcome;
    clearTimeout(this.timeout);
    this.sample = {
      ordinal: ++nextOrdinal,
      gameId: this.gameId,
      modelChecksum: this.modelChecksum,
      outcome,
      application: outcome === 'legal' ? 'notStarted' : null,
      elapsedMs: now - this.startedAt,
      inputMs: this.inputAt === undefined ? null : this.inputAt - this.startedAt,
      notifyMs:
        this.inputAt === undefined || this.notifiedAt === undefined
          ? null
          : this.notifiedAt - this.inputAt,
      chooseMs:
        this.notifiedAt === undefined || this.chosenAt === undefined
          ? null
          : this.chosenAt - this.notifiedAt,
      legalityMs: outcome !== 'legal' || this.chosenAt === undefined ? null : now - this.chosenAt,
    };
    samples.push(this.sample);
    if (samples.length > SAMPLE_LIMIT) samples.shift();
    this.onChange?.();
    return true;
  }
}

/** The last 128 completed attempts in this page, filtered by numeric model.
 * p95 uses nearest rank over legal decisions only; failures have separate counts.
 * No raw observations, model bytes or saved game history enter this window. */
export function getNnDecisionLatency(modelChecksum: string) {
  const window = samples.filter((sample) => sample.modelChecksum === modelChecksum);
  const legal = window
    .filter((sample) => sample.outcome === 'legal')
    .map((sample) => sample.elapsedMs)
    .sort((a, b) => a - b);
  return {
    sampleLimit: SAMPLE_LIMIT,
    completedAttempts: window.length,
    legalDecisions: legal.length,
    deadlines: window.filter((sample) => sample.outcome === 'deadline').length,
    cancellations: window.filter((sample) => sample.outcome === 'cancelled').length,
    errors: window.filter((sample) => sample.outcome === 'error').length,
    applyNotStarted: window.filter((sample) => sample.application === 'notStarted').length,
    applyPending: window.filter((sample) => sample.application === 'pending').length,
    applied: window.filter((sample) => sample.application === 'applied').length,
    applyFailures: window.filter((sample) => sample.application === 'failed').length,
    applyDiscards: window.filter((sample) => sample.application === 'discarded').length,
    legalP95Ms: legal.length === 0 ? null : legal[Math.ceil(0.95 * legal.length) - 1]!,
    samples: window.map((sample) => ({ ...sample })),
  };
}

export const NN_LATENCY_MEASUREMENT_SCHEMA = 'nn-app-latency-v1';

/** Bounded DOM measurement data. Rows contain only numbers/null, with page-local ordinals.
 * Columns: ordinal, outcome (legal/deadline/cancelled/error = 0/1/2/3), application
 * (notStarted/pending/applied/failed/discarded = 0/1/2/3/4, null for nonlegal),
 * elapsed, input, notify, choose, legality in milliseconds. No game IDs are exposed. */
export function getNnLatencyMeasurement(modelChecksum: string) {
  const { samples: window, ...summary } = getNnDecisionLatency(modelChecksum);
  const outcomes = { legal: 0, deadline: 1, cancelled: 2, error: 3 };
  const applications = { notStarted: 0, pending: 1, applied: 2, failed: 3, discarded: 4 };
  return {
    schema: NN_LATENCY_MEASUREMENT_SCHEMA,
    modelChecksum,
    summary,
    rows: window.map((sample) => [
      sample.ordinal,
      outcomes[sample.outcome],
      sample.application === null ? null : applications[sample.application],
      sample.elapsedMs,
      sample.inputMs,
      sample.notifyMs,
      sample.chooseMs,
      sample.legalityMs,
    ]),
  };
}

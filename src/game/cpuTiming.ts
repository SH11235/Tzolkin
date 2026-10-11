/** One monotonic budget from observation construction through current legality.
 * Timers cannot interrupt synchronous work; callers check again when it returns. */
export const NN_DECISION_DEADLINE_MS = 1000;
const SAMPLE_LIMIT = 128;
type Outcome = 'legal' | 'deadline' | 'cancelled' | 'error';
interface Sample {
  gameId: string;
  modelChecksum: string;
  outcome: Outcome;
  elapsedMs: number;
  inputMs: number | null;
  notifyMs: number | null;
  chooseMs: number | null;
  legalityMs: number | null;
}
const samples: Sample[] = [];

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
  private readonly timeout: ReturnType<typeof setTimeout>;
  private readonly gameId: string;
  private readonly modelChecksum: string;

  constructor(
    game: { gameId: string; modelChecksum: string },
    private readonly onDeadline?: (error: NnDecisionDeadlineError) => void,
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

  private finish(outcome: Outcome, now: number): boolean {
    if (this.outcome) return false;
    this.outcome = outcome;
    clearTimeout(this.timeout);
    samples.push({
      gameId: this.gameId,
      modelChecksum: this.modelChecksum,
      outcome,
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
    });
    if (samples.length > SAMPLE_LIMIT) samples.shift();
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
    legalP95Ms: legal.length === 0 ? null : legal[Math.ceil(0.95 * legal.length) - 1]!,
    samples: window.map((sample) => ({ ...sample })),
  };
}

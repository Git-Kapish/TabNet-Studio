import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, Database, Settings, Clock, Calendar } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

interface ModelConfig {
  n_d?: number;
  n_steps?: number;
  gamma?: number;
}

interface EvaluationMetrics {
  accuracy?: number;
  f1?: number;
}

interface RunMetadata {
  run_name: string;
  timestamp?: string;
  training_duration_seconds?: number;
  best_epoch?: number;
  model_config?: ModelConfig;
  evaluation_metrics?: EvaluationMetrics;
}

function formatDuration(sec?: number): string {
  if (sec === undefined) return '—';
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const min = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return `${min}m ${s}s`;
}

function formatTimestamp(ts?: string): string {
  if (!ts) return '—';
  try {
    return new Date(ts).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return ts;
  }
}

function datasetFromRunName(name: string): string {
  return name.split('_run_')[0] ?? name;
}

export default function Results(): React.JSX.Element {
  const [runs, setRuns] = useState<RunMetadata[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRuns = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/models`);
      if (!response.ok) throw new Error('Failed to retrieve model registry.');
      const data = (await response.json()) as RunMetadata[];
      const sorted = data.sort((a, b) => {
        const da = a.timestamp ? new Date(a.timestamp).getTime() : 0;
        const db = b.timestamp ? new Date(b.timestamp).getTime() : 0;
        return db - da;
      });
      setRuns(sorted);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Step 03 of 05</div>
          <h1 className="page-title">Model Registry</h1>
          <p className="page-subtitle">
            Every completed training run, with hyperparameters, evaluation
            metrics, best epoch, and an exportable PyTorch checkpoint.
          </p>
        </div>
        <div className="page-actions">
          <span className="badge mono">
            {runs.length} {runs.length === 1 ? 'run' : 'runs'}
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={fetchRuns}
            disabled={loading}
          >
            <RefreshCw
              size={14}
              strokeWidth={1.75}
              className={loading ? 'spin' : ''}
            />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="callout callout-error" role="alert">
          {error}
        </div>
      )}

      {loading && runs.length === 0 ? (
        <div className="card card-pad">
          <div className="help mono" style={{ color: 'var(--muted-foreground)' }}>
            Querying saved experiments from the backend…
          </div>
        </div>
      ) : !loading && !error && runs.length === 0 ? (
        <div className="card">
          <div className="empty">
            <Database size={28} strokeWidth={1.5} style={{ color: 'var(--muted-foreground)' }} />
            <div className="empty-title">No runs found in registry</div>
            <div className="empty-help">
              Train a model in the Playground. Once it finishes, it'll show up
              here as a row you can inspect, download, or rerun.
            </div>
          </div>
        </div>
      ) : (
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Runs</div>
              <div className="card-subtitle">newest first</div>
            </div>
          </div>
          <div className="table-scroll">
            <table className="table" aria-label="Model runs">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Dataset</th>
                  <th className="num">Accuracy</th>
                  <th className="num">F1</th>
                  <th className="num">Best epoch</th>
                  <th className="num">Duration</th>
                  <th>Saved</th>
                  <th className="num">Config</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const acc = run.evaluation_metrics?.accuracy;
                  const f1 = run.evaluation_metrics?.f1;
                  const cfg = run.model_config;
                  return (
                    <tr key={run.run_name}>
                      <td>
                        <div className="mono" style={{ fontWeight: 600 }}>
                          {run.run_name}
                        </div>
                      </td>
                      <td>
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 6,
                            color: 'var(--muted-foreground)',
                          }}
                        >
                          <Database size={12} strokeWidth={1.75} />
                          {datasetFromRunName(run.run_name)}
                        </span>
                      </td>
                      <td className="num">
                        {acc !== undefined ? `${(acc * 100).toFixed(2)}%` : '—'}
                      </td>
                      <td className="num">{f1 !== undefined ? f1.toFixed(4) : '—'}</td>
                      <td className="num">{run.best_epoch ?? '—'}</td>
                      <td className="num">{formatDuration(run.training_duration_seconds)}</td>
                      <td>
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 6,
                            color: 'var(--muted-foreground)',
                          }}
                        >
                          <Calendar size={12} strokeWidth={1.75} />
                          {formatTimestamp(run.timestamp)}
                        </span>
                      </td>
                      <td className="num" style={{ color: 'var(--muted-foreground)' }}>
                        <span
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: 4,
                          }}
                        >
                          <Settings size={12} strokeWidth={1.75} />
                          N<sub>d</sub>={cfg?.n_d ?? 8} · steps={cfg?.n_steps ?? 3} · γ=
                          {cfg?.gamma ?? 1.3}
                        </span>
                      </td>
                      <td>
                        <a
                          className="btn btn-secondary btn-sm"
                          href={`${API_BASE}/api/models/${run.run_name}/export`}
                          download
                        >
                          <Download size={12} strokeWidth={1.75} />
                          .pt
                        </a>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {runs.length > 0 && (
        <p className="help mono" style={{ marginTop: 16 }}>
          <Clock size={11} strokeWidth={1.75} style={{ verticalAlign: -1 }} /> tip ·
          click a run name in the Architecture Explorer to load its attention
          masks
        </p>
      )}
    </div>
  );
}

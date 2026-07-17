import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Play,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  BarChart2,
  FileSpreadsheet,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL ?? `http://${window.location.hostname}:8000`;

interface Metrics {
  accuracy: number;
  f1: number;
  val_loss: number;
}

interface HistoryItem {
  epoch: number;
  train_loss: number;
  val_loss: number;
  accuracy: number;
  f1: number;
}

interface DTypes {
  [key: string]: string;
}

type TrainStatus = 'idle' | 'training' | 'completed' | 'failed';

const STATUS_BADGE: Record<TrainStatus, string> = {
  idle: 'badge',
  training: 'badge badge-running',
  completed: 'badge badge-completed',
  failed: 'badge badge-failed',
};

export default function Train(): React.JSX.Element {
  const [datasetName, setDatasetName] = useState<string>('adult');
  const [customColumns, setCustomColumns] = useState<string[]>([]);
  const [targetCol, setTargetCol] = useState<string>('income');
  const [, setDtypes] = useState<DTypes>({});
  const [rowCount, setRowCount] = useState<number>(48842);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Hyperparameters
  const [nd, setNd] = useState<number>(8);
  const [na, setNa] = useState<number>(8);
  const [nSteps, setNSteps] = useState<number>(3);
  const [gamma, setGamma] = useState<number>(1.3);
  const [lambdaSparse, setLambdaSparse] = useState<number>(0.001);
  const [lr, setLr] = useState<number>(0.02);
  const [batchSize, setBatchSize] = useState<number>(1024);
  const [epochs, setEpochs] = useState<number>(5);
  const [patience, setPatience] = useState<number>(3);
  const [seed] = useState<number>(42);

  // Training state
  const [trainingActive, setTrainingActive] = useState<boolean>(false);
  const [runName, setRunName] = useState<string>('');
  const [trainStatus, setTrainStatus] = useState<TrainStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [latestMetrics, setLatestMetrics] = useState<Metrics>({
    accuracy: 0,
    f1: 0,
    val_loss: 0,
  });

  // Early stopping states
  const [earlyStoppingTriggered, setEarlyStoppingTriggered] = useState<boolean>(false);
  const [bestEpoch, setBestEpoch] = useState<number | null>(null);
  const [stoppedEpoch, setStoppedEpoch] = useState<number | null>(null);
  const [patienceVal, setPatienceVal] = useState<number>(3);

  const pollIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    if (datasetName === 'adult') {
      setTargetCol('income');
      setRowCount(48842);
    } else if (datasetName === 'covertype') {
      setTargetCol('Cover_Type');
      setRowCount(581012);
    } else if (datasetName === 'custom') {
      setTargetCol('—');
      setRowCount(0);
      setCustomColumns([]);
    }
  }, [datasetName]);

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current !== null) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/api/dataset/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setDatasetName(data.filename);
      setCustomColumns(data.columns);
      setDtypes(data.dtypes);
      setRowCount(data.row_count);
      if (data.columns.length > 0) {
        setTargetCol(data.columns[data.columns.length - 1]);
      }
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : 'Failed to upload CSV.');
    } finally {
      setUploading(false);
    }
  };

  const startTraining = async () => {
    setTrainingActive(true);
    setTrainStatus('training');
    setErrorMessage('');
    setCurrentEpoch(0);
    setProgress(0);
    setHistory([]);
    setLatestMetrics({ accuracy: 0, f1: 0, val_loss: 0 });
    setEarlyStoppingTriggered(false);
    setBestEpoch(null);
    setStoppedEpoch(null);

    const payload = {
      dataset_name: datasetName,
      target_col: targetCol,
      n_d: nd,
      n_a: na,
      n_steps: nSteps,
      gamma: parseFloat(gamma.toString()),
      lambda_sparse: parseFloat(lambdaSparse.toString()),
      lr: parseFloat(lr.toString()),
      batch_size: parseInt(batchSize.toString()),
      epochs: parseInt(epochs.toString()),
      patience: parseInt(patience.toString()),
      seed: parseInt(seed.toString()),
      deterministic: true,
    };

    try {
      const response = await fetch(`${API_BASE}/api/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      setRunName(data.run_name);
      pollIntervalRef.current = window.setInterval(() => {
        pollStatus(data.run_name);
      }, 1500);
    } catch (err: unknown) {
      setTrainStatus('failed');
      setErrorMessage(err instanceof Error ? err.message : 'Training failed to start.');
      setTrainingActive(false);
    }
  };

  const pollStatus = async (name: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/train/status/${name}`);
      if (!response.ok) throw new Error('Status query failed');
      const data = await response.json();
      setTrainStatus(data.status);
      const curEpoch = data.epoch || 0;
      setCurrentEpoch(curEpoch);
      setProgress(Math.min(100, Math.round((curEpoch / epochs) * 100)));
      
      if (data.early_stopping_triggered !== undefined) {
        setEarlyStoppingTriggered(data.early_stopping_triggered);
      }
      if (data.best_epoch !== undefined) {
        setBestEpoch(data.best_epoch);
      }
      if (data.stopped_epoch !== undefined) {
        setStoppedEpoch(data.stopped_epoch);
      }
      if (data.patience !== undefined) {
        setPatienceVal(data.patience);
      }
      if (curEpoch > 0) {
        setLatestMetrics({
          accuracy: data.accuracy || 0,
          f1: data.f1 || 0,
          val_loss: data.val_loss || 0,
        });
        setHistory((prev) =>
          prev.some((p) => p.epoch === curEpoch)
            ? prev
            : [
                ...prev,
                {
                  epoch: curEpoch,
                  train_loss: data.train_loss || 0,
                  val_loss: data.val_loss || 0,
                  accuracy: data.accuracy || 0,
                  f1: data.f1 || 0,
                },
              ]
        );
      }
      if (data.status === 'completed' || data.status === 'failed') {
        if (pollIntervalRef.current !== null) clearInterval(pollIntervalRef.current);
        setTrainingActive(false);
        if (data.status === 'failed') {
          setErrorMessage(data.error || 'Training thread terminated unexpectedly.');
        }
      }
    } catch {
      // swallow — next tick will retry
    }
  };

  const isCustom =
    datasetName !== 'adult' && datasetName !== 'covertype' && customColumns.length > 0;

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Step 02 of 05</div>
          <h1 className="page-title">Training Playground</h1>
          <p className="page-subtitle">
            Pick a dataset, tune the architecture, and watch loss and validation
            metrics plot as the run progresses.
          </p>
        </div>
        <div className="page-actions">
          <span className={STATUS_BADGE[trainStatus]}>
            <span
              className={`dot ${trainStatus === 'training' ? 'dot-pulse' : ''}`}
              style={{ color: 'currentColor' }}
            />
            {trainStatus}
          </span>
        </div>
      </div>

      <div className="train-layout">
        {/* Left — dataset + params */}
        <div className="stack-lg">
          {/* Dataset card */}
          <section className="card" aria-label="Dataset">
            <div className="card-header">
              <div>
                <div className="card-title">Dataset</div>
                <div className="card-subtitle">CSV · preloaded or custom</div>
              </div>
              <span className="badge mono">
                {rowCount.toLocaleString()} rows
              </span>
            </div>
            <div className="card-section stack">
              <div className="field">
                <label className="label" htmlFor="dataset-select">
                  Preloaded datasets
                </label>
                <select
                  id="dataset-select"
                  className="select"
                  value={
                    datasetName === 'adult' || datasetName === 'covertype'
                      ? datasetName
                      : 'custom'
                  }
                  onChange={(e) => {
                    setDatasetName(e.target.value);
                  }}
                  disabled={trainingActive}
                >
                  <option value="adult">Adult Census Income · 48 842 rows</option>
                  <option value="covertype">Forest Cover Type · 581 012 rows</option>
                  <option value="custom">
                    {isCustom ? `Custom · ${datasetName}` : 'Custom'}
                  </option>
                </select>
              </div>

              <div className="divider-label">or upload</div>

              <label className="dropzone" htmlFor="csv-upload">
                <input
                  id="csv-upload"
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  disabled={trainingActive}
                />
                <Upload size={18} strokeWidth={1.75} aria-hidden="true" />
                <div className="dropzone-title">Drop a CSV here, or click to browse</div>
                <div className="help">
                  The first row is treated as header. The last column becomes the
                  default target.
                </div>
              </label>

              {uploading && (
                <div className="help mono" style={{ color: 'var(--info)' }}>
                  Uploading and parsing columns…
                </div>
              )}
              {uploadError && (
                <div className="callout callout-error" role="alert">
                  <AlertTriangle size={14} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 2 }} />
                  <div>{uploadError}</div>
                </div>
              )}

              {isCustom && (
                <div className="field">
                  <label className="label" htmlFor="target-select">
                    Target column
                  </label>
                  <select
                    id="target-select"
                    className="select"
                    value={targetCol}
                    onChange={(e) => setTargetCol(e.target.value)}
                    disabled={trainingActive}
                  >
                    {customColumns.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div
                className="row-between mono"
                style={{
                  fontSize: 12,
                  color: 'var(--muted-foreground)',
                  paddingTop: 8,
                  borderTop: '1px solid var(--border)',
                }}
              >
                <span>
                  target · <strong style={{ color: 'var(--foreground)' }}>{targetCol}</strong>
                </span>
                <span>
                  features ·{' '}
                  <strong style={{ color: 'var(--foreground)' }}>
                    {isCustom ? customColumns.length - 1 : 'auto'}
                  </strong>
                </span>
              </div>
            </div>
          </section>

          {/* Hyperparameters */}
          <section className="card" aria-label="Hyperparameters">
            <div className="card-header">
              <div>
                <div className="card-title">Hyperparameters</div>
                <div className="card-subtitle">architecture + optimization</div>
              </div>
              <span className="badge mono">7 params</span>
            </div>
            <div className="card-section stack">
              <div className="grid grid-cols-2">
                <SliderField
                  label="Width N_d"
                  value={nd}
                  min={8}
                  max={64}
                  step={8}
                  disabled={trainingActive}
                  onChange={setNd}
                />
                <SliderField
                  label="Width N_a"
                  value={na}
                  min={8}
                  max={64}
                  step={8}
                  disabled={trainingActive}
                  onChange={setNa}
                />
              </div>
              <div className="grid grid-cols-2">
                <SliderField
                  label="Decision steps"
                  value={nSteps}
                  min={2}
                  max={8}
                  step={1}
                  disabled={trainingActive}
                  onChange={setNSteps}
                />
                <SliderField
                  label="Gamma (relaxation)"
                  value={gamma}
                  min={1.0}
                  max={2.0}
                  step={0.1}
                  decimals={1}
                  disabled={trainingActive}
                  onChange={setGamma}
                />
              </div>
              <div className="grid grid-cols-2">
                <div className="field">
                  <label className="label" htmlFor="lambda-sparse">
                    Sparsity loss λ
                  </label>
                  <select
                    id="lambda-sparse"
                    className="select mono"
                    value={lambdaSparse}
                    onChange={(e) => setLambdaSparse(parseFloat(e.target.value))}
                    disabled={trainingActive}
                  >
                    <option value={0.0001}>0.0001</option>
                    <option value={0.001}>0.001</option>
                    <option value={0.01}>0.01</option>
                    <option value={0.1}>0.1</option>
                  </select>
                </div>
                <SliderField
                  label="Learning rate"
                  value={lr}
                  min={0.005}
                  max={0.05}
                  step={0.005}
                  decimals={3}
                  disabled={trainingActive}
                  onChange={setLr}
                />
              </div>
              <div className="grid grid-cols-3">
                <div className="field">
                  <label className="label" htmlFor="batch-size">
                    Batch size
                  </label>
                  <select
                    id="batch-size"
                    className="select mono"
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    disabled={trainingActive}
                  >
                    {[256, 512, 1024, 2048, 4096].map((n) => (
                      <option key={n} value={n}>
                        {n.toLocaleString()}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label className="label" htmlFor="epochs">
                    Epochs
                  </label>
                  <input
                    id="epochs"
                    type="number"
                    min={1}
                    max={100}
                    className="input mono"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value))}
                    disabled={trainingActive}
                  />
                </div>
                <div className="field">
                  <label className="label" htmlFor="patience">
                    Patience
                  </label>
                  <input
                    id="patience"
                    type="number"
                    min={1}
                    max={50}
                    className="input mono"
                    value={patience}
                    onChange={(e) => setPatience(parseInt(e.target.value))}
                    disabled={trainingActive}
                  />
                </div>
              </div>

              <button
                type="button"
                className="btn btn-primary btn-block"
                onClick={startTraining}
                disabled={trainingActive}
                style={{ marginTop: 8 }}
              >
                {trainingActive ? (
                  <>
                    <RefreshCw size={14} className="spin" strokeWidth={1.75} />
                    Training · epoch {currentEpoch} / {epochs}
                  </>
                ) : (
                  <>
                    <Play size={14} strokeWidth={1.75} />
                    Start training
                  </>
                )}
              </button>
            </div>
          </section>
        </div>

        {/* Right — live training console */}
        <section className="card" aria-label="Training console">
          <div className="card-header">
            <div>
              <div className="card-title">Training console</div>
              <div className="card-subtitle">
                {runName ? (
                  <>
                    run · <span className="mono">{runName}</span>
                  </>
                ) : (
                  'no active run'
                )}
              </div>
            </div>
            <span className={STATUS_BADGE[trainStatus]}>
              <span
                className={`dot ${trainStatus === 'training' ? 'dot-pulse' : ''}`}
              />
              {trainStatus}
            </span>
          </div>
          <div className="card-section">
            {trainStatus === 'idle' ? (
              <div className="empty">
                <FileSpreadsheet
                  size={28}
                  strokeWidth={1.5}
                  style={{ color: 'var(--muted-foreground)' }}
                />
                <div className="empty-title">No active training session</div>
                <div className="empty-help">
                  Configure hyperparameters on the left, then press{' '}
                  <span className="kbd">Start training</span>. Loss and
                  validation metrics will plot here as the run progresses.
                </div>
              </div>
            ) : (
              <div className="stack-lg">
                <div>
                  <div className="progress-row">
                    <span>
                      epoch {currentEpoch} / {epochs}
                    </span>
                    <span>{progress}%</span>
                  </div>
                  <div className="progress">
                    <div className="progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                  {earlyStoppingTriggered && (
                    <div style={{ marginTop: 10, fontSize: 12, color: 'var(--accent-foreground)', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <AlertTriangle size={13} style={{ color: 'var(--primary)', flexShrink: 0 }} />
                      <span>
                        Halted at epoch <strong>{stoppedEpoch}</strong> (patience limit of {patienceVal} epochs hit). Best epoch was <strong>{bestEpoch}</strong>.
                      </span>
                    </div>
                  )}
                </div>

                {currentEpoch > 0 && (
                  <div className="grid grid-cols-3">
                    <Metric label="Validation accuracy" value={`${(latestMetrics.accuracy * 100).toFixed(2)}%`} />
                    <Metric label="F1 score" value={latestMetrics.f1.toFixed(4)} />
                    <Metric label="Validation loss" value={latestMetrics.val_loss.toFixed(4)} />
                  </div>
                )}

                {trainStatus === 'completed' && (
                  <div className="callout callout-success" role="status">
                    <CheckCircle size={16} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
                    <div>
                      <div className="callout-title">Training complete</div>
                      {earlyStoppingTriggered ? (
                        <p style={{ marginTop: 2, fontSize: 13, color: 'var(--accent-foreground)' }}>
                          <strong>Early Stopping Triggered:</strong> Halted training early at epoch{' '}
                          <strong>{stoppedEpoch}</strong> because the validation loss did not improve for {patienceVal} consecutive epochs (best epoch was <strong>{bestEpoch}</strong>). Saved the optimal weights to the model registry.
                        </p>
                      ) : (
                        <p style={{ marginTop: 2 }}>
                          Checkpoint saved to the model registry. Open the
                          Architecture Explorer to read what the model learned.
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {trainStatus === 'failed' && (
                  <div className="callout callout-error" role="alert">
                    <AlertTriangle size={16} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 1 }} />
                    <div>
                      <div className="callout-title">Training process terminated</div>
                      {errorMessage}
                    </div>
                  </div>
                )}

                {history.length > 0 ? (
                  <div>
                    <div className="section-title section-title-mono" style={{ marginBottom: 8 }}>
                      <BarChart2 size={12} strokeWidth={1.75} />
                      Loss curves
                    </div>
                    <div style={{ height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="epoch" tickLine={false} axisLine={false} />
                          <YAxis tickLine={false} axisLine={false} width={48} />
                          <Tooltip />
                          <Legend
                            iconType="plainline"
                            wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="train_loss"
                            name="train loss"
                            stroke="var(--chart-1)"
                            strokeWidth={1.75}
                            dot={false}
                            activeDot={{ r: 4 }}
                          />
                          <Line
                            type="monotone"
                            dataKey="val_loss"
                            name="val loss"
                            stroke="var(--chart-3)"
                            strokeWidth={1.75}
                            dot={false}
                            activeDot={{ r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : (
                  <div className="empty" style={{ padding: 24 }}>
                    <div className="empty-help">
                      Charts will appear after the first epoch completes.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  decimals = 0,
  disabled,
  onChange,
}: SliderProps) {
  return (
    <div className="field">
      <div className="label">
        <span>{label}</span>
        <span className="label-meta mono">{value.toFixed(decimals)}</span>
      </div>
      <input
        type="range"
        className="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label}
      />
      <div className="help mono">
        {min} – {max}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="card card-pad-sm">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

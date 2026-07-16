import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { ShieldAlert, RefreshCw, Award, Zap, HardDrive } from 'lucide-react';

const API_BASE = `http://${window.location.hostname}:8000`;

interface BaselineMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface BaselineItem {
  model_name: string;
  metrics: BaselineMetrics;
  training_time_seconds: number;
  inference_time_seconds: number;
  model_size_bytes: number;
}

interface BenchmarkData {
  [key: string]: BaselineItem;
}

interface MetricItem {
  key: string;
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  trainTime: number;
  infTime: number;
  sizeMB: number;
  sizeKB: number;
}

type MetricView = 'performance' | 'speed' | 'size';

// ponytail: amber ramp for ordered series, but X-axis labels are unique model
// names so a legend is unnecessary — see DESIGN.md §13.
const MODEL_COLOR: Record<string, string> = {
  'TabNet': 'var(--chart-1)',
  'XGBoost': 'var(--chart-2)',
  'Random Forest': 'var(--chart-3)',
  'Logistic Regression': 'var(--chart-4)',
};

export default function Compare(): React.JSX.Element {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [activeMetric, setActiveMetric] = useState<MetricView>('performance');

  const fetchBenchmarks = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/benchmarks`);
      if (!response.ok) {
        throw new Error(
          'Benchmark data not found. Train the baselines first to populate it.'
        );
      }
      const json = (await response.json()) as BenchmarkData;
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBenchmarks();
  }, []);

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <div>
            <div className="page-eyebrow">Step 04 of 05</div>
            <h1 className="page-title">Baselines</h1>
          </div>
        </div>
        <div className="card card-pad">
          <div className="help mono" style={{ color: 'var(--muted-foreground)' }}>
            Loading baseline benchmarks from server…
          </div>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div>
        <div className="page-header">
          <div>
            <div className="page-eyebrow">Step 04 of 05</div>
            <h1 className="page-title">Baselines</h1>
          </div>
        </div>
        <div className="callout callout-error" role="alert">
          <ShieldAlert size={16} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <div className="callout-title">Benchmark results missing</div>
            <p style={{ marginTop: 4 }}>
              We could not load baseline comparison results because{' '}
              <code className="mono">benchmarks/results.json</code> is missing or
              corrupt. Run the baseline training pipeline in your shell:
            </p>
            <code
              className="mono"
              style={{
                display: 'block',
                marginTop: 8,
                padding: '8px 12px',
                backgroundColor: 'var(--muted)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius)',
                fontSize: 12,
              }}
            >
              python benchmarks/train_baselines.py
            </code>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={fetchBenchmarks}
              style={{ marginTop: 12 }}
            >
              <RefreshCw size={13} strokeWidth={1.75} />
              Retry fetch
            </button>
          </div>
        </div>
      </div>
    );
  }

  const models: MetricItem[] = Object.keys(data).map((key) => {
    const item = data[key];
    return {
      key,
      name: item.model_name,
      accuracy: parseFloat((item.metrics.accuracy * 100).toFixed(2)),
      precision: parseFloat(item.metrics.precision.toFixed(4)),
      recall: parseFloat(item.metrics.recall.toFixed(4)),
      f1: parseFloat(item.metrics.f1.toFixed(4)),
      trainTime: parseFloat(item.training_time_seconds.toFixed(2)),
      infTime: parseFloat((item.inference_time_seconds * 1000).toFixed(1)),
      sizeMB: parseFloat((item.model_size_bytes / (1024 * 1024)).toFixed(3)),
      sizeKB: parseFloat((item.model_size_bytes / 1024).toFixed(1)),
    };
  });

  const chartTitle: Record<MetricView, string> = {
    performance: 'Classification metrics',
    speed: 'Execution speed',
    size: 'Disk size footprint',
  };

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Step 04 of 05</div>
          <h1 className="page-title">Baselines</h1>
          <p className="page-subtitle">
            TabNet against classical scikit-learn models and tree ensembles on
            the same train/val split. Accuracy, F1, training time, inference
            latency, and on-disk size.
          </p>
        </div>
        <div className="page-actions">
          <button type="button" className="btn btn-secondary" onClick={fetchBenchmarks}>
            <RefreshCw size={14} strokeWidth={1.75} />
            Refresh
          </button>
        </div>
      </div>

      {/* Metric toggles */}
      <div className="row" style={{ marginBottom: 16, flexWrap: 'wrap' }}>
        <button
          type="button"
          className={`btn ${activeMetric === 'performance' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveMetric('performance')}
        >
          <Award size={13} strokeWidth={1.75} />
          Classification
        </button>
        <button
          type="button"
          className={`btn ${activeMetric === 'speed' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveMetric('speed')}
        >
          <Zap size={13} strokeWidth={1.75} />
          Speed
        </button>
        <button
          type="button"
          className={`btn ${activeMetric === 'size' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveMetric('size')}
        >
          <HardDrive size={13} strokeWidth={1.75} />
          Disk size
        </button>
      </div>

      <div className="stack-lg">
        <section className="card" aria-label="Baseline chart">
          <div className="card-header">
            <div>
              <div className="card-title">{chartTitle[activeMetric]}</div>
              <div className="card-subtitle">
                {activeMetric === 'performance' && 'accuracy (%) and F1 across baselines'}
                {activeMetric === 'speed' && 'training (s) and inference (ms) per model'}
                {activeMetric === 'size' && 'serialized checkpoint size (KB)'}
              </div>
            </div>
          </div>
          <div className="card-section" style={{ height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={models} margin={{ top: 8, right: 16, bottom: 0, left: -8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} fontSize={12} />
                <YAxis tickLine={false} axisLine={false} fontSize={11} width={56} />
                <Tooltip cursor={{ fill: 'var(--muted)' }} />
                <Legend
                  iconType="square"
                  wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
                />

                {activeMetric === 'performance' && (
                  <>
                    <Bar
                      dataKey="accuracy"
                      name="accuracy (%)"
                      fill="var(--chart-1)"
                      radius={[2, 2, 0, 0]}
                    >
                      {models.map((m, i) => (
                        <Cell key={i} fill={MODEL_COLOR[m.name] ?? 'var(--chart-1)'} />
                      ))}
                    </Bar>
                    <Bar
                      dataKey="f1"
                      name="F1"
                      fill="var(--chart-2)"
                      radius={[2, 2, 0, 0]}
                    />
                  </>
                )}

                {activeMetric === 'speed' && (
                  <>
                    <Bar
                      dataKey="trainTime"
                      name="train (s)"
                      fill="var(--chart-1)"
                      radius={[2, 2, 0, 0]}
                    />
                    <Bar
                      dataKey="infTime"
                      name="inference (ms)"
                      fill="var(--chart-3)"
                      radius={[2, 2, 0, 0]}
                    />
                  </>
                )}

                {activeMetric === 'size' && (
                  <Bar
                    dataKey="sizeKB"
                    name="size (KB)"
                    radius={[2, 2, 0, 0]}
                  >
                    {models.map((m, i) => (
                      <Cell key={i} fill={MODEL_COLOR[m.name] ?? 'var(--chart-1)'} />
                    ))}
                  </Bar>
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card" aria-label="Baseline table">
          <div className="card-header">
            <div>
              <div className="card-title">Comparison details</div>
              <div className="card-subtitle">all metrics, sortable view as a table</div>
            </div>
          </div>
          <div className="table-scroll" style={{ maxHeight: 480 }}>
            <table className="table" aria-label="Baseline metrics">
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">Accuracy</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                  <th className="num">F1</th>
                  <th className="num">Train</th>
                  <th className="num">Infer</th>
                  <th className="num">Size</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.key}>
                    <td>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 8,
                          fontWeight: 600,
                        }}
                      >
                        <span
                          aria-hidden="true"
                          style={{
                            display: 'inline-block',
                            width: 8,
                            height: 8,
                            borderRadius: 2,
                            backgroundColor:
                              MODEL_COLOR[m.name] ?? 'var(--chart-1)',
                          }}
                        />
                        {m.name}
                      </span>
                    </td>
                    <td className="num">{m.accuracy.toFixed(2)}%</td>
                    <td className="num">{m.precision.toFixed(4)}</td>
                    <td className="num">{m.recall.toFixed(4)}</td>
                    <td className="num" style={{ fontWeight: 600 }}>
                      {m.f1.toFixed(4)}
                    </td>
                    <td className="num">{m.trainTime.toFixed(2)} s</td>
                    <td className="num">{m.infTime.toFixed(1)} ms</td>
                    <td className="num">
                      {m.sizeKB >= 1024
                        ? `${m.sizeMB.toFixed(2)} MB`
                        : `${m.sizeKB.toFixed(1)} KB`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}

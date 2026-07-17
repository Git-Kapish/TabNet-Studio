import React, { useState, useEffect, useMemo } from 'react';
import {
  RefreshCw,
  Layers,
  GitBranch,
  Workflow,
  Cpu,
  Hash,
  Sparkles,
  CheckCircle2,
  CircleSlash,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL ?? `http://${window.location.hostname}:8000`;

interface SampleItem {
  index: number;
  raw_features: { [key: string]: string | number };
  step_masks: number[][];
  step_contributions: number[];
  local_importance: number[];
  prediction: string;
  actual: string;
}

interface ModelRun {
  run_name: string;
}

type NodeId = 'input' | 'embedding' | 'bn' | 'attn' | 'ft' | 'output';

interface NodeDef {
  id: NodeId;
  code: string;
  label: string;
  sub: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
}

const PIPELINE: NodeDef[] = [
  { id: 'input', code: 'CSV', label: 'Raw data', sub: 'tabular features', icon: Layers },
  { id: 'embedding', code: 'EMB', label: 'Embeddings', sub: 'categorical + numeric', icon: Hash },
  { id: 'bn', code: 'BN', label: 'Ghost BN', sub: 'stabilize activations', icon: CircleSlash },
  { id: 'attn', code: 'ATTN', label: 'Attentive transformer', sub: 'sparsemax mask', icon: Sparkles },
  { id: 'ft', code: 'FT', label: 'Feature transformer', sub: 'GLU block', icon: GitBranch },
  { id: 'output', code: 'CLS', label: 'Aggregator + head', sub: 'softmax output', icon: CheckCircle2 },
];

// ponytail: deterministic seed for reproducible inspector shape; small ceiling —
// replace with live model introspection when the API exposes it.
const INSPECTOR_DETAIL: Record<
  NodeId,
  { rows: { k: string; v: string }[]; note: string }
> = {
  input: {
    rows: [
      { k: 'shape', v: '(B, n_features)' },
      { k: 'dtypes', v: 'int · float · string' },
      { k: 'preprocessing', v: 'median fill + standard scale'},
    ],
    note: 'CSV is split into numeric and categorical columns. Categorical values are mapped to integer indices.',
  },
  embedding: {
    rows: [
      { k: 'numeric', v: 'passthrough + linear projection' },
      { k: 'categorical', v: 'nn.Embedding(vocab, dim)' },
      { k: 'concat', v: 'stack along feature axis' },
    ],
    note: 'Each categorical column gets its own trainable embedding table — no pre-binning required.',
  },
  bn: {
    rows: [
      { k: 'type', v: 'Ghost BatchNorm' },
      { k: 'virtual batches', v: 'B / 256' },
      { k: 'momentum', v: '0.02' },
    ],
    note: 'Ghost BN splits each batch into virtual sub-batches to regularize the running statistics.',
  },
  attn: {
    rows: [
      { k: 'normalization', v: 'sparsemax' },
      { k: 'priors', v: 'cumulative across steps' },
      { k: 'scale', v: '√d_a' },
    ],
    note: 'A sparsemax mask picks which features this decision step will use. Selected features are zeroed out for the next step via the prior.',
  },
  ft: {
    rows: [
      { k: 'shared layers', v: '⌊n_steps / 2⌋' },
      { k: 'step-dependent layers', v: 'remaining' },
      { k: 'activation', v: 'GLU' },
    ],
    note: 'The Feature Transformer extracts a hidden representation for the selected features. The first half of layers is shared across decision steps, the second half is per-step.',
  },
  output: {
    rows: [
      { k: 'aggregation', v: 'Σ eta_i · features_i' },
      { k: 'head', v: 'linear → softmax' },
      { k: 'output', v: '(B, n_classes)' },
    ],
    note: 'Per-step feature outputs are weighted by their decision-step contribution coefficient η and summed before the classification head.',
  },
};

export default function Explainability(): React.JSX.Element {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [loadingData, setLoadingData] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [features, setFeatures] = useState<string[]>([]);
  const [globalImportance, setGlobalImportance] = useState<number[]>([]);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [activeSampleIdx, setActiveSampleIdx] = useState<number>(0);
  const [activeNode, setActiveNode] = useState<NodeId>('attn');

  const fetchRuns = async () => {
    setLoadingRuns(true);
    try {
      const response = await fetch(`${API_BASE}/api/models`);
      if (!response.ok) throw new Error('Failed to retrieve model runs.');
      const data = (await response.json()) as ModelRun[];
      setRuns(data);
      if (data.length > 0) setSelectedRun(data[0].run_name);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoadingRuns(false);
    }
  };

  const fetchInterpretability = async (runName: string) => {
    if (!runName) return;
    setLoadingData(true);
    setError(null);
    try {
      const response = await fetch(
        `${API_BASE}/api/feature-importance/${runName}?num_samples=5`
      );
      if (!response.ok) {
        throw new Error('Failed to query feature importance. Verify dataset exists.');
      }
      const data = await response.json();
      setFeatures(data.feature_names);
      setGlobalImportance(data.global_importance);
      setSamples(data.samples);
      setActiveSampleIdx(0);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoadingData(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) fetchInterpretability(selectedRun);
  }, [selectedRun]);

  const handleRunChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedRun(e.target.value);
  };

  const formattedGlobalImportance = useMemo(
    () =>
      features
        .map((name, i) => ({
          name,
          importance: parseFloat((globalImportance[i] || 0).toFixed(4)),
        }))
        .sort((a, b) => b.importance - a.importance),
    [features, globalImportance]
  );

  const activeSample = samples[activeSampleIdx];
  const numSteps = activeSample ? activeSample.step_masks.length : 0;
  const inspector = INSPECTOR_DETAIL[activeNode];

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Step 05 of 05</div>
          <h1 className="page-title">Architecture Explorer</h1>
          <p className="page-subtitle">
            Click a node to inspect its role. Then jump to a specific instance
            to read the sparse masks the model assigned at each decision step.
          </p>
        </div>
        <div className="page-actions">
          <span className="badge mono">
            {features.length} features · {numSteps} steps
          </span>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={fetchRuns}
            disabled={loadingRuns || loadingData}
          >
            <RefreshCw
              size={14}
              strokeWidth={1.75}
              className={loadingRuns ? 'spin' : ''}
            />
            Reload
          </button>
        </div>
      </div>

      {/* Run selector */}
      <section className="card" style={{ marginBottom: 16 }}>
        <div className="card-section">
          <div className="row" style={{ flexWrap: 'wrap' }}>
            <div className="field" style={{ flex: 1, minWidth: 240, marginBottom: 0 }}>
              <label className="label" htmlFor="run-select">
                Trained experiment
              </label>
              <select
                id="run-select"
                className="select mono"
                value={selectedRun}
                onChange={handleRunChange}
                disabled={loadingRuns || loadingData}
              >
                {loadingRuns && <option>Loading experiment list…</option>}
                {!loadingRuns && runs.length === 0 && (
                  <option>No trained models found</option>
                )}
                {runs.map((run) => (
                  <option key={run.run_name} value={run.run_name}>
                    {run.run_name}
                  </option>
                ))}
              </select>
            </div>
            <div
              className="row"
              style={{ marginLeft: 'auto', color: 'var(--muted-foreground)' }}
            >
              <Cpu size={13} strokeWidth={1.75} />
              <span className="mono" style={{ fontSize: 12 }}>
                device · cpu
              </span>
            </div>
          </div>
        </div>
      </section>

      {error && <div className="callout callout-error" role="alert">{error}</div>}
      {loadingData && (
        <div className="help mono" style={{ color: 'var(--muted-foreground)' }}>
          Evaluating validation samples inside PyTorch TabNet…
        </div>
      )}

      {!loadingData && !error && (
        <div className="explorer-layout">
          {/* Main column */}
          <div className="stack-lg" style={{ minWidth: 0 }}>
            {/* Forward pass pipeline — clickable, the signature element */}
            <section className="card" aria-label="TabNet forward pass">
              <div className="card-header">
                <div>
                  <div className="card-title">Forward pass</div>
                  <div className="card-subtitle">
                    click a node to read its definition
                  </div>
                </div>
                <span className="badge mono">
                  <Workflow size={11} strokeWidth={1.75} />
                  6 stages
                </span>
              </div>
              <div className="card-section">
                <div className="pipeline" role="list">
                  {PIPELINE.map((node) => {
                    const Icon = node.icon;
                    const isActive = activeNode === node.id;
                    return (
                      <button
                        key={node.id}
                        type="button"
                        className="pipeline-node"
                        aria-current={isActive ? 'true' : undefined}
                        role="listitem"
                        onClick={() => setActiveNode(node.id)}
                        title={node.label}
                      >
                        <span className="pipeline-circle">
                          {isActive ? (
                            <Icon size={14} strokeWidth={1.75} />
                          ) : (
                            <span className="mono">{node.code}</span>
                          )}
                        </span>
                        <span className="pipeline-label">
                          {node.label}
                          <br />
                          <span
                            className="mono"
                            style={{
                              color: 'var(--muted-foreground)',
                              fontSize: 10,
                            }}
                          >
                            {node.sub}
                          </span>
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </section>

            {/* Global feature importance */}
            {features.length > 0 && (
              <section className="card" aria-label="Global feature importance">
                <div className="card-header">
                  <div>
                    <div className="card-title">Global feature importance</div>
                    <div className="card-subtitle">
                      top 15 features · averaged across validation samples
                    </div>
                  </div>
                </div>
                <div className="card-section" style={{ height: 320 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={formattedGlobalImportance.slice(0, 15)}
                      layout="vertical"
                      margin={{ top: 4, right: 16, bottom: 0, left: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" tickLine={false} axisLine={false} fontSize={11} />
                      <YAxis
                        type="category"
                        dataKey="name"
                        tickLine={false}
                        axisLine={false}
                        fontSize={11}
                        width={140}
                      />
                      <Tooltip cursor={{ fill: 'var(--muted)' }} />
                      <Bar
                        dataKey="importance"
                        name="importance"
                        fill="var(--chart-1)"
                        radius={[0, 2, 2, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>
            )}

            {/* Instance: samples + heatmap */}
            {activeSample && (
              <section className="card" aria-label="Instance attention">
                <div className="card-header">
                  <div>
                    <div className="card-title">Instance attention</div>
                    <div className="card-subtitle">
                      decision-step sparse masks for the selected sample
                    </div>
                  </div>
                  <div className="row">
                    {samples.map((_s, idx) => (
                      <button
                        key={idx}
                        type="button"
                        className={`btn ${
                          activeSampleIdx === idx ? 'btn-primary' : 'btn-secondary'
                        } btn-sm`}
                        onClick={() => setActiveSampleIdx(idx)}
                      >
                        sample {idx}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="card-section stack-lg">
                  {/* Prediction + ground truth */}
                  <div className="grid grid-cols-3">
                    <div className="card card-pad-sm">
                      <div className="metric-label">Ground truth</div>
                      <div className="metric-value">{activeSample.actual}</div>
                    </div>
                    <div className="card card-pad-sm">
                      <div className="metric-label">Prediction</div>
                      <div
                        className="metric-value"
                        style={{
                          color:
                            activeSample.prediction === activeSample.actual
                              ? 'var(--success)'
                              : 'var(--destructive)',
                        }}
                      >
                        {activeSample.prediction}
                      </div>
                    </div>
                    <div className="card card-pad-sm">
                      <div className="metric-label">Match</div>
                      <div className="metric-value">
                        {activeSample.prediction === activeSample.actual ? 'yes' : 'no'}
                      </div>
                    </div>
                  </div>

                  {/* Heatmap */}
                  <div>
                    <div className="section-title section-title-mono" style={{ marginBottom: 8 }}>
                      sparse mask · rows are features · columns are steps
                    </div>
                    <div className="heatmap-wrap" aria-label="Attention heatmap">
                      <div
                        className="heatmap"
                        style={{
                          gridTemplateColumns: `minmax(180px, max-content) repeat(${numSteps}, 1fr)`,
                        }}
                      >
                        <div className="heatmap-cell col-label corner mono">
                          feature ↓ · step →
                        </div>
                        {Array.from({ length: numSteps }, (_, i) => (
                          <div key={i} className="heatmap-cell col-label mono">
                            step {i + 1}
                          </div>
                        ))}
                        {features.map((featureName, fi) => (
                          <React.Fragment key={featureName}>
                            <div
                              className="heatmap-cell row-label mono"
                              title={featureName}
                            >
                              {featureName}
                            </div>
                            {activeSample.step_masks.map((stepMask, si) => {
                              const v = stepMask[fi] || 0;
                              return (
                                <div
                                  key={si}
                                  className="heatmap-cell value mono"
                                  title={`${featureName} · step ${si + 1} · ${v.toFixed(4)}`}
                                  style={{
                                    backgroundColor: `color-mix(in srgb, var(--primary) ${Math.round(
                                      v * 100
                                    )}%, var(--background))`,
                                    color:
                                      v > 0.55
                                        ? 'var(--primary-foreground)'
                                        : 'var(--foreground)',
                                  }}
                                >
                                  {v.toFixed(2)}
                                </div>
                              );
                            })}
                          </React.Fragment>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Step contributions */}
                  <div>
                    <div className="section-title section-title-mono" style={{ marginBottom: 8 }}>
                      decision-step contribution · η<sub>i</sub>
                    </div>
                    <div className="grid" style={{ gridTemplateColumns: `repeat(${numSteps}, 1fr)` }}>
                      {activeSample.step_contributions.map((eta, idx) => (
                        <div
                          key={idx}
                          className="card card-pad-sm"
                          style={{ textAlign: 'center' }}
                        >
                          <div className="metric-label">step {idx + 1}</div>
                          <div className="metric-value">{eta.toFixed(4)}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Raw features */}
                  <div>
                    <div className="section-title section-title-mono" style={{ marginBottom: 8 }}>
                      raw features · this sample
                    </div>
                    <div
                      className="table-wrap"
                      style={{ maxHeight: 220, overflow: 'auto' }}
                    >
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Feature</th>
                            <th className="num">Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(activeSample.raw_features).map(
                            ([k, v]) => (
                              <tr key={k}>
                                <td className="mono">{k}</td>
                                <td className="num">{String(v)}</td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </section>
            )}
          </div>

          {/* Inspector — right rail, contextual */}
          <aside className="inspector" aria-label="Node inspector">
            <div className="inspector-title">Inspector</div>
            <div className="inspector-name">
              {PIPELINE.find((n) => n.id === activeNode)?.label}
            </div>

            <div className="kv">
              <span className="kv-key">stage</span>
              <span className="kv-val">
                {PIPELINE.findIndex((n) => n.id === activeNode) + 1} of{' '}
                {PIPELINE.length}
              </span>
            </div>
            <div className="kv">
              <span className="kv-key">code</span>
              <span className="kv-val">
                {PIPELINE.find((n) => n.id === activeNode)?.code}
              </span>
            </div>
            {inspector.rows.map((row) => (
              <div key={row.k} className="kv">
                <span className="kv-key">{row.k}</span>
                <span className="kv-val">{row.v}</span>
              </div>
            ))}

            <p
              className="help"
              style={{
                marginTop: 16,
                paddingTop: 12,
                borderTop: '1px solid var(--border)',
                lineHeight: 1.5,
              }}
            >
              {inspector.note}
            </p>

            <button
              type="button"
              className="btn btn-secondary btn-block"
              style={{ marginTop: 16 }}
              onClick={() => {
                const next =
                  PIPELINE[
                    (PIPELINE.findIndex((n) => n.id === activeNode) + 1) %
                      PIPELINE.length
                  ];
                setActiveNode(next.id);
              }}
            >
              Next stage →
            </button>
          </aside>
        </div>
      )}
    </div>
  );
}

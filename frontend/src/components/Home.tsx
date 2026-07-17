import React from 'react';
import {
  ArrowRight,
  Cpu,
  Workflow,
  Database,
  Sparkles,
  Eye,
  BarChart2,
  Play,
  BookOpen,
} from 'lucide-react';

type TabId =
  | 'home'
  | 'train'
  | 'results'
  | 'compare'
  | 'explain'
  | 'predict';

interface Props {
  onNavigate: (tab: TabId) => void;
}

// ponytail: mock preview of the attention grid, replaced once a real run exists.
const PREVIEW_STEPS = [3, 4, 5];
const PREVIEW_FEATURES = [
  ['fnlwgt', 0.12],
  ['age', 0.81],
  ['education-num', 0.67],
  ['capital-gain', 0.94],
  ['hours-per-week', 0.43],
  ['relationship', 0.55],
  ['occupation', 0.38],
  ['marital-status', 0.21],
] as const;

const PIPELINE: { code: string; label: string; sub: string }[] = [
  { code: 'CSV', label: 'Raw data', sub: 'tabular input' },
  { code: 'EMB', label: 'Embeddings', sub: 'categorical + numeric' },
  { code: 'BN', label: 'Ghost BatchNorm', sub: 'stabilize' },
  { code: 'ATTN', label: 'Attentive transformer', sub: 'feature selection' },
  { code: 'FT', label: 'Feature transformer', sub: 'GLU block' },
  { code: 'CLS', label: 'Aggregator + head', sub: 'prediction' },
];

export default function Home({ onNavigate }: Props): React.JSX.Element {
  return (
    <div>
      {/* Page header */}
      <div className="page-header">
        <div>
          <div className="page-eyebrow">v1.0 · {new Date().toISOString().slice(0, 10)}</div>
          <h1 className="page-title">TabNet Studio</h1>
          <p className="page-subtitle">
            A PyTorch implementation of <em>TabNet: Attentive Interpretable Tabular Learning</em> (Arik &amp; Pfister, 2019) —
            built from the paper's architecture to explore sequential attention, sparse feature selection,
            and interpretability in deep tabular models.
          </p>
        </div>
        <div className="page-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={() => onNavigate('explain')}
          >
            <Eye size={14} strokeWidth={1.75} />
            Open Architecture Explorer
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => onNavigate('train')}
          >
            <Play size={14} strokeWidth={1.75} />
            Start a training run
          </button>
        </div>
      </div>

      {/* Hero — the characteristic thing: live decision-step attention grid.
          This is what TabNet is. Showing it on the home page is the thesis. */}
      <section className="card card-pad stack-lg" aria-label="TabNet attention preview">
        <div className="row-between">
          <div>
            <div className="section-title section-title-mono">
              <span className="dot" style={{ color: 'var(--primary)' }} />
              Instance-level attention · sample 0421 · run{' '}
              <span className="mono">adult_run_2026-07-15</span>
            </div>
            <h2
              style={{
                fontSize: 20,
                fontWeight: 600,
                marginTop: 8,
                letterSpacing: '-0.2px',
              }}
            >
              Each row in this grid is a feature. Each column is a decision
              step. Bright amber cells are the features that step selected.
            </h2>
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => onNavigate('explain')}
          >
            Explore for real
            <ArrowRight size={14} strokeWidth={1.75} />
          </button>
        </div>

        <div className="heatmap-wrap" role="img" aria-label="Preview attention heatmap">
          <div
            className="heatmap"
            style={{ gridTemplateColumns: 'minmax(180px, max-content) repeat(3, 1fr)' }}
          >
            <div className="heatmap-cell col-label corner mono">feature ↓ / step →</div>
            {PREVIEW_STEPS.map((s) => (
              <div key={s} className="heatmap-cell col-label mono">
                step {s}
              </div>
            ))}
            {PREVIEW_FEATURES.map(([name, base]) => {
              const vals = PREVIEW_STEPS.map(
                (_, i) => Math.max(0, Math.min(1, base + (i - 1) * 0.05 - 0.05))
              );
              return (
                <React.Fragment key={name}>
                  <div className="heatmap-cell row-label mono">{name}</div>
                  {vals.map((v, i) => (
                    <div
                      key={i}
                      className="heatmap-cell value mono"
                      title={`${name} · step ${PREVIEW_STEPS[i]} · ${v.toFixed(2)}`}
                      style={{
                        backgroundColor: `color-mix(in srgb, var(--primary) ${Math.round(
                          v * 100
                        )}%, var(--background))`,
                        color: v > 0.55 ? 'var(--primary-foreground)' : 'var(--foreground)',
                      }}
                    >
                      {v.toFixed(2)}
                    </div>
                  ))}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        <p className="help mono" style={{ textAlign: 'left' }}>
          cell opacity ∝ sparse-mask weight · prediction:{' '}
          <strong style={{ color: 'var(--foreground)' }}>&gt;50K</strong> ·
          ground truth:{' '}
          <strong style={{ color: 'var(--foreground)' }}>&gt;50K</strong> ·
          confidence:{' '}
          <strong style={{ color: 'var(--foreground)' }}>0.871</strong>
        </p>
      </section>

      {/* Pipeline — the architecture, made into a horizontal flow.
          Reuses the signature pipeline component. */}
      <section className="card" aria-label="TabNet forward pass">
        <div className="card-header">
          <div>
            <div className="card-title">Forward pass</div>
            <div className="card-subtitle">input → embedding → 3+ decision steps → prediction</div>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => onNavigate('explain')}
          >
            <Workflow size={13} strokeWidth={1.75} />
            Open in Explorer
          </button>
        </div>
        <div className="card-section">
          <div className="pipeline" role="list">
            {PIPELINE.map((node, i) => (
              <button
                key={node.code}
                type="button"
                className="pipeline-node"
                aria-current={i === 3 ? 'true' : undefined}
                role="listitem"
                onClick={() => onNavigate('explain')}
                title={`${node.label} — ${node.sub}`}
              >
                <span className="pipeline-circle mono">{node.code}</span>
                <span className="pipeline-label">
                  {node.label}
                  <br />
                  <span className="mono" style={{ color: 'var(--muted-foreground)' }}>
                    {node.sub}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Capability grid — 3 columns of "what you can do here" */}
      <section>
        <div
          className="section-title"
          style={{ marginBottom: 12 }}
          aria-hidden="true"
        >
          What's in the workbench
        </div>
        <div className="grid grid-cols-3">
          <Capability
            icon={Play}
            title="Train in the Playground"
            body="Upload a CSV or pick a preloaded dataset, set N_d, N_a, gamma, sparsity, then watch loss and validation metrics plot in real time."
            cta="Open Playground"
            onClick={() => onNavigate('train')}
          />
          <Capability
            icon={BarChart2}
            title="Compare against baselines"
            body="TabNet vs. Random Forest, Logistic Regression, and XGBoost on identical splits — accuracy, F1, training time, inference time, and disk size."
            cta="Open Baselines"
            onClick={() => onNavigate('compare')}
          />
          <Capability
            icon={Eye}
            title="Read what the model reads"
            body="Click through the architecture, then jump to a specific instance. Sparsemax masks tell you which features each decision step leaned on."
            cta="Open Architecture Explorer"
            onClick={() => onNavigate('explain')}
          />
          <Capability
            icon={Database}
            title="Browse saved runs"
            body="Every completed training is recorded with its hyperparameters, evaluation metrics, best epoch, and exportable PyTorch checkpoint."
            cta="Open Model Registry"
            onClick={() => onNavigate('results')}
          />
          <Capability
            icon={Sparkles}
            title="Run batch predictions"
            body="Point a CSV at any registered model, get predictions plus per-row class confidences, and download the augmented file."
            cta="Open Predictions"
            onClick={() => onNavigate('predict')}
          />
          <Capability
            icon={BookOpen}
            title="A standalone library"
            body="The PyTorch layers (GLU, Ghost BatchNorm, Sparsemax) ship as a clean package, decoupled from the REST and dashboard — usable in any workflow."
            cta="View on GitHub"
            onClick={() =>
              window.open(
                'https://github.com/Git-Kapish/TabNet-Studio',
                '_blank'
              )
            }
          />
        </div>
      </section>

      {/* Footer caption — context, not marketing */}
      <footer
        className="mono"
        style={{
          marginTop: 32,
          paddingTop: 16,
          borderTop: '1px solid var(--border)',
          fontSize: 11,
          color: 'var(--muted-foreground)',
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 12,
        }}
      >
        <span>
          <Cpu size={11} strokeWidth={1.75} style={{ verticalAlign: -1 }} /> TabNet Studio · Kapish Yadav
        </span>
      </footer>
    </div>
  );
}

interface CapabilityProps {
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
  title: string;
  body: string;
  cta: string;
  onClick: () => void;
}

function Capability({ icon: Icon, title, body, cta, onClick }: CapabilityProps) {
  return (
    <article className="card card-pad stack">
      <div
        style={{
          width: 32,
          height: 32,
          borderRadius: 'var(--radius)',
          backgroundColor: 'var(--muted)',
          color: 'var(--foreground)',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Icon size={15} strokeWidth={1.75} />
      </div>
      <h3 style={{ fontSize: 14, fontWeight: 600, letterSpacing: '-0.1px' }}>{title}</h3>
      <p className="help" style={{ color: 'var(--muted-foreground)' }}>
        {body}
      </p>
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        style={{ alignSelf: 'flex-start', marginTop: 4 }}
        onClick={onClick}
      >
        {cta}
        <ArrowRight size={12} strokeWidth={1.75} />
      </button>
    </article>
  );
}

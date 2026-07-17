import React, { useState, useEffect } from 'react';
import {
  Upload,
  Play,
  Download,
  RefreshCw,
  FileText,
  CheckCircle2,
  ShieldAlert,
} from 'lucide-react';

interface ModelRun {
  run_name: string;
}

const API_BASE = import.meta.env.VITE_API_URL ?? `http://${window.location.hostname}:8000`;

interface PredictResult {
  predictions: string[];
  probabilities: number[][];
  has_ground_truth: boolean;
}

interface PreviewRow {
  [key: string]: string;
}

export default function Predict(): React.JSX.Element {
  const [runs, setRuns] = useState<ModelRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>('');
  const [loadingRuns, setLoadingRuns] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const [fileSelected, setFileSelected] = useState<File | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predResult, setPredResult] = useState<PredictResult | null>(null);
  const [previewRows, setPreviewRows] = useState<PreviewRow[]>([]);
  const [previewHeaders, setPreviewHeaders] = useState<string[]>([]);

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

  useEffect(() => {
    fetchRuns();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileSelected(file);
    setPredResult(null);
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event?.target?.result as string | null;
      if (!text) return;
      const lines = text.split('\n').filter((l) => l.trim() !== '');
      const headers = lines[0].split(',').map((h) => h.trim());
      const rows = lines.slice(1, 6).map((line) => {
        const values = line.split(',');
        const rowObj: PreviewRow = {};
        headers.forEach((h, index) => {
          rowObj[h] = values[index] ? values[index].trim() : '';
        });
        return rowObj;
      });
      setPreviewHeaders(headers);
      setPreviewRows(rows);
    };
    reader.readAsText(file);
  };

  const executeInference = async () => {
    if (!fileSelected || !selectedRun) return;
    setPredicting(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', fileSelected);
    try {
      const response = await fetch(
        `${API_BASE}/api/predict?run_name=${selectedRun}`,
        { method: 'POST', body: formData }
      );
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as PredictResult;
      setPredResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Batch prediction failed.');
    } finally {
      setPredicting(false);
    }
  };

  const downloadCSV = () => {
    if (!predResult || !fileSelected) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e?.target?.result as string | null;
      if (!text) return;
      const lines = text.split('\n').filter((l) => l.trim() !== '');
      const header = lines[0].trim() + ',Predicted_Class,Confidence_Probabilities\n';
      const newLines = lines.slice(1).map((line, index) => {
        const pred = predResult.predictions[index] || '';
        const probs = predResult.probabilities[index] || [];
        const maxProb = probs.length > 0 ? Math.max(...probs).toFixed(4) : '';
        return `${line.trim()},${pred},${maxProb}\n`;
      });
      const blob = new Blob([header, ...newLines], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `predictions_${selectedRun}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };
    reader.readAsText(fileSelected);
  };

  const previewFeatureCols = previewHeaders.slice(0, 4);

  return (
    <div>
      <div className="page-header">
        <div>
          <div className="page-eyebrow">Deploy</div>
          <h1 className="page-title">Batch Predictions</h1>
          <p className="page-subtitle">
            Point a CSV at any registered model, get per-row predictions and
            class confidence, and download the augmented file.
          </p>
        </div>
        <div className="page-actions">
          {predResult && (
            <span className="badge badge-completed">
              <CheckCircle2 size={11} strokeWidth={1.75} />
              {predResult.predictions.length} rows
            </span>
          )}
          {predResult && (
            <button type="button" className="btn btn-primary" onClick={downloadCSV}>
              <Download size={14} strokeWidth={1.75} />
              Download CSV
            </button>
          )}
        </div>
      </div>

      <div className="predict-layout">
        {/* Left — pick model + file + run */}
        <section className="card" aria-label="Inference configuration">
          <div className="card-header">
            <div>
              <div className="card-title">Inference</div>
              <div className="card-subtitle">model · file · run</div>
            </div>
          </div>
          <div className="card-section stack">
            <div className="field">
              <label className="label" htmlFor="predict-run">
                Model
              </label>
              <select
                id="predict-run"
                className="select mono"
                value={selectedRun}
                onChange={(e) => setSelectedRun(e.target.value)}
                disabled={loadingRuns || predicting}
              >
                {loadingRuns && <option>Loading experiment list…</option>}
                {runs.map((run) => (
                  <option key={run.run_name} value={run.run_name}>
                    {run.run_name}
                  </option>
                ))}
              </select>
            </div>

            <div className="divider" />

            <div className="field">
              <label className="label">Input CSV</label>
              <label className="dropzone" htmlFor="predict-csv">
                <input
                  id="predict-csv"
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  disabled={predicting}
                />
                <Upload size={18} strokeWidth={1.75} aria-hidden="true" />
                <div className="dropzone-title">
                  {fileSelected ? fileSelected.name : 'Select a CSV file'}
                </div>
                <div className="help">
                  {fileSelected
                    ? `${(fileSelected.size / 1024).toFixed(1)} KB`
                    : 'Columns must match the training schema'}
                </div>
              </label>
            </div>

            <button
              type="button"
              className="btn btn-primary btn-block"
              onClick={executeInference}
              disabled={!fileSelected || !selectedRun || predicting}
            >
              {predicting ? (
                <>
                  <RefreshCw size={14} className="spin" strokeWidth={1.75} />
                  Generating predictions…
                </>
              ) : (
                <>
                  <Play size={14} strokeWidth={1.75} />
                  Run predictions
                </>
              )}
            </button>
          </div>
        </section>

        {/* Right — preview + result */}
        <div className="stack-lg predict-results">
          {error && (
            <div className="callout callout-error" role="alert">
              <ShieldAlert size={16} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 2 }} />
              <div>
                <div className="callout-title">Prediction failed</div>
                {error}
              </div>
            </div>
          )}

          {!error && !fileSelected && (
            <div className="card">
              <div className="empty">
                <FileText
                  size={28}
                  strokeWidth={1.5}
                  style={{ color: 'var(--muted-foreground)' }}
                />
                <div className="empty-title">No file selected</div>
                <div className="empty-help">
                  Pick a CSV on the left. The first 5 rows will preview here
                  before you run inference, so you can confirm the schema.
                </div>
              </div>
            </div>
          )}

          {previewRows.length > 0 && (
            <section className="card" aria-label="Preview">
              <div className="card-header">
                <div>
                  <div className="card-title">
                    {predResult ? 'Predictions preview' : 'Raw data preview'}
                  </div>
                  <div className="card-subtitle">
                    first {previewRows.length} of {fileSelected ? 'many' : '—'} rows
                  </div>
                </div>
                {predResult && (
                  <span className="badge mono">
                    {previewFeatureCols.length} feature cols shown
                  </span>
                )}
              </div>
              <div className="table-scroll" style={{ maxHeight: 420 }}>
                <table className="table" aria-label="Preview">
                  <thead>
                    <tr>
                      <th className="num">#</th>
                      {previewFeatureCols.map((key) => (
                        <th key={key}>{key}</th>
                      ))}
                      {predResult && (
                        <>
                          <th>Predicted class</th>
                          <th className="num">Confidence</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {previewRows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        <td className="num">{rIdx + 1}</td>
                        {previewFeatureCols.map((key) => (
                          <td key={key} className="mono">
                            {row[key] ?? ''}
                          </td>
                        ))}
                        {predResult && (
                          <>
                            <td style={{ fontWeight: 600 }}>
                              {predResult.predictions[rIdx]}
                            </td>
                            <td className="num">
                              {(
                                Math.max(...(predResult.probabilities[rIdx] || [0])) * 100
                              ).toFixed(2)}
                              %
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect, useMemo } from 'react';
import {
  Home as HomeIcon,
  Play,
  Database,
  BarChart2,
  Eye,
  Sparkles,
  Sun,
  Moon,
  ExternalLink,
  Activity,
} from 'lucide-react';

import Home from './components/Home';
import Train from './components/Train';
import Results from './components/Results';
import Compare from './components/Compare';
import Explainability from './components/Explainability';
import Predict from './components/Predict';

type Theme = 'light' | 'dark';

type TabId =
  | 'home'
  | 'train'
  | 'results'
  | 'compare'
  | 'explain'
  | 'predict';

interface NavItem {
  id: TabId;
  label: string;
  hint: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
}

const NAV: NavItem[] = [
  { id: 'home', label: 'Studio', hint: 'Overview', icon: HomeIcon },
  { id: 'train', label: 'Playground', hint: 'Configure & train', icon: Play },
  { id: 'results', label: 'Model Registry', hint: 'Saved runs', icon: Database },
  { id: 'compare', label: 'Baselines', hint: 'TabNet vs. others', icon: BarChart2 },
  { id: 'explain', label: 'Architecture Explorer', hint: 'Feature attribution', icon: Eye },
  { id: 'predict', label: 'Predictions', hint: 'Batch inference', icon: Sparkles },
];

const PAGE_META: Record<TabId, { eyebrow: string; crumb: string }> = {
  home: { eyebrow: 'TabNet Studio', crumb: 'Studio' },
  train: { eyebrow: 'Train · step 02', crumb: 'Playground' },
  results: { eyebrow: 'Train · step 03', crumb: 'Model Registry' },
  compare: { eyebrow: 'Evaluate', crumb: 'Baselines' },
  explain: { eyebrow: 'Interpret', crumb: 'Architecture Explorer' },
  predict: { eyebrow: 'Deploy', crumb: 'Predictions' },
};

export default function App(): React.JSX.Element {
  const [activeTab, setActiveTab] = useState<TabId>('home');
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('tabnet-studio-theme');
    if (stored === 'light' || stored === 'dark') return stored;
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('tabnet-studio-theme', theme);
  }, [theme]);

  const meta = useMemo(() => PAGE_META[activeTab], [activeTab]);

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            TN
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span className="brand-name">TabNet Studio</span>
            <span className="brand-tag">v1.0</span>
          </div>
        </div>

        <div>
          <div className="nav-group-label">Workflow</div>
          <ul className="nav-list">
            {NAV.map((item) => {
              const Icon = item.icon;
              const current = activeTab === item.id;
              return (
                <li key={item.id}>
                  <button
                    type="button"
                    className="nav-item"
                    aria-current={current ? 'page' : undefined}
                    onClick={() => setActiveTab(item.id)}
                    style={{ width: '100%', textAlign: 'left' }}
                  >
                    <span className="nav-glyph">
                      <Icon size={15} strokeWidth={1.75} />
                    </span>
                    <span style={{ flex: 1 }}>{item.label}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="sidebar-footer mono">
          <div>backend · {window.location.hostname}:8000</div>
          <div style={{ marginTop: 2, opacity: 0.7 }}>
            pytorch · 2.x · cpu
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar" role="banner">
          <div className="topbar-left">
            <span className="crumb">
              <span>TabNet Studio</span>
              <span aria-hidden="true">/</span>
              <span className="crumb-current">{meta.crumb}</span>
            </span>
            <span className="kbd" aria-hidden="true">
              {meta.eyebrow}
            </span>
          </div>
          <div className="topbar-right">
            <div className="topbar-meta">
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                <Activity size={12} />
                <span>idle</span>
              </span>
            </div>
            <a
              className="icon-btn"
              href="https://github.com/google-research/google-research/tree/master/tabnet"
              target="_blank"
              rel="noreferrer"
              aria-label="Open upstream TabNet repository"
            >
              <ExternalLink size={15} strokeWidth={1.75} />
            </a>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? (
                <Moon size={16} strokeWidth={1.75} />
              ) : (
                <Sun size={16} strokeWidth={1.75} />
              )}
            </button>
          </div>
        </header>

        <div className="page">
          {activeTab === 'home' && <Home onNavigate={setActiveTab} />}
          {activeTab === 'train' && <Train />}
          {activeTab === 'results' && <Results />}
          {activeTab === 'compare' && <Compare />}
          {activeTab === 'explain' && <Explainability />}
          {activeTab === 'predict' && <Predict />}
        </div>
      </main>
    </div>
  );
}

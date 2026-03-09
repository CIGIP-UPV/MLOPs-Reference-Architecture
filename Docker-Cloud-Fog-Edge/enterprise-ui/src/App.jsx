import React, { useEffect, useState } from 'react';

const apiKey = 'enterprise-demo-key';

function useJson(url) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        if (active) {
          setState({ loading: false, data, error: null });
        }
      } catch (error) {
        if (active) {
          setState({ loading: false, data: null, error: error.message });
        }
      }
    }
    load();
    const timer = setInterval(load, 10000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [url]);
  return state;
}

async function postAction(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey
    },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json();
}

function MetricCard({ label, value, helper }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{helper}</small>
    </article>
  );
}

export default function App() {
  const overview = useJson('/api/overview');
  const models = useJson('/api/models');
  const deployments = useJson('/api/deployments');
  const [actionState, setActionState] = useState('');

  const overviewData = overview.data || {};
  const recentHealth = overviewData.recent_health || {};
  const production = overviewData.production || {};
  const latestDrift = overviewData.latest_drift || {};
  const deploymentItems = deployments.data?.items || [];
  const modelItems = models.data?.items || [];

  const runClosedLoop = async () => {
    setActionState('Launching closed-loop cycle...');
    try {
      const response = await postAction('/api/closed-loop/run', { trigger: 'enterprise-dashboard' });
      setActionState(`Closed-loop completed: ${JSON.stringify(response.actions || [])}`);
    } catch (error) {
      setActionState(`Error: ${error.message}`);
    }
  };

  const rollback = async () => {
    setActionState('Requesting rollback...');
    try {
      const response = await postAction('/api/deployments/rollback', { reason: 'Manual rollback from React dashboard' });
      setActionState(`Rollback response: ${JSON.stringify(response)}`);
    } catch (error) {
      setActionState(`Error: ${error.message}`);
    }
  };

  if (overview.loading) {
    return <main className="shell"><p>Loading enterprise telemetry...</p></main>;
  }

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">Enterprise Tier</p>
          <h1>Industrial MLOps Control Room</h1>
          <p className="lede">
            This dashboard exposes the governance loop requested by the paper: drift surveillance,
            versioned deployment commands, OTA synchronization and rollback supervision.
          </p>
        </div>
        <div className="action-panel">
          <button onClick={runClosedLoop}>Run Closed Loop</button>
          <button className="secondary" onClick={rollback}>Rollback</button>
          <p>{actionState}</p>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard label="Production Version" value={production.version || 'n/a'} helper={production.current_stage || 'No stage'} />
        <MetricCard label="Accuracy Proxy" value={recentHealth.accuracy ?? 'n/a'} helper={recentHealth.status || 'unknown'} />
        <MetricCard label="Average Drift" value={recentHealth.average_drift ?? 'n/a'} helper={latestDrift?.drift_severity || 'no drift report'} />
        <MetricCard label="False Alarms" value={recentHealth.false_alarm_rate ?? 'n/a'} helper={`Events: ${recentHealth.events || 0}`} />
      </section>

      <section className="panel-grid">
        <article className="panel">
          <h2>Deployment Timeline</h2>
          <ul>
            {deploymentItems.map((item) => (
              <li key={item.id}>
                <strong>v{item.model_version}</strong>
                <span>{item.action}</span>
                <small>{item.reason}</small>
              </li>
            ))}
          </ul>
        </article>

        <article className="panel">
          <h2>Model Registry Snapshot</h2>
          <ul>
            {modelItems.map((item) => (
              <li key={item.version}>
                <strong>v{item.version}</strong>
                <span>{item.current_stage || 'None'}</span>
                <small>f1={item.tags?.metric_f1 || 'n/a'} checksum={item.tags?.sha256?.slice(0, 10) || 'n/a'}</small>
              </li>
            ))}
          </ul>
        </article>
      </section>

      <section className="panel">
        <h2>Recent Prediction Stream</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                <th>Version</th>
                <th>Risk</th>
                <th>Prediction</th>
                <th>Actual</th>
                <th>Drift</th>
              </tr>
            </thead>
            <tbody>
              {(overviewData.recent_predictions || []).slice(-12).reverse().map((row) => (
                <tr key={`${row.id}-${row.cycle_id}`}>
                  <td>{row.cycle_id}</td>
                  <td>{row.model_version}</td>
                  <td>{row.risk_score}</td>
                  <td>{row.prediction}</td>
                  <td>{row.actual_breakage}</td>
                  <td>{row.drift_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

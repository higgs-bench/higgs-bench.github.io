import { useMemo, useState } from "react";
import { useJson } from "../lib/useJson";

const VERSIONS = [
  { id: "A", label: "1:1" },
  { id: "B", label: "1:10" },
  { id: "C", label: "1:50" },
];
const MODES = [
  { id: "seed", label: "Repeat the seed" },
  { id: "boot", label: "Resample the test set" },
];
const MODEL_NAMES = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  random_forest: "Random forest",
  catboost: "CatBoost",
  adaboost: "AdaBoost",
  voting_3: "Voting (3)",
  voting_4: "Voting (4)",
};

export default function UncertaintyToggle() {
  const { data, error, loading } = useJson("ranking.json");
  const [version, setVersion] = useState("C");
  const [mode, setMode] = useState("seed");

  const rows = useMemo(() => {
    if (!data) return [];
    return data
      .filter((d) => d.version === version && d.technique === "baseline")
      .sort((a, b) => b.auprc - a.auprc);
  }, [data, version]);

  const bounds = (r) =>
    mode === "seed"
      ? [r.seed_ci_lo, r.seed_ci_hi]
      : [r.auprc_ci_lo, r.auprc_ci_hi];

  const domain = useMemo(() => {
    if (!rows.length) return [0, 1];
    const all = rows.flatMap(bounds);
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const pad = (hi - lo) * 0.12 || 0.005;
    return [lo - pad, hi + pad];
  }, [rows, mode]);

  if (loading) return <div className="missing">Loading rankings…</div>;
  if (error) return <div className="missing state-error">{error}</div>;

  const [lo, hi] = domain;
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;

  const best = rows[0];
  const [bestLo] = best ? bounds(best) : [0];
  const overlapping = rows.filter((r) => bounds(r)[1] >= bestLo).length;

  return (
    <div>
      <div className="controls">
        <div>
          <span className="control-label">Signal-to-background ratio</span>
          <div className="seg">
            {VERSIONS.map((v) => (
              <button
                key={v.id}
                aria-pressed={version === v.id}
                onClick={() => setVersion(v.id)}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <span className="control-label">What the error bar measures</span>
          <div className="seg">
            {MODES.map((m) => (
              <button
                key={m.id}
                aria-pressed={mode === m.id}
                onClick={() => setMode(m.id)}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="rank-list">
        {rows.map((r) => {
          const [a, b] = bounds(r);
          return (
            <div className="rank-row" key={r.model}>
              <div className="rank-name">{MODEL_NAMES[r.model]}</div>
              <div className="rank-track">
                <div
                  className="rank-ci"
                  style={{
                    left: `${pct(a)}%`,
                    width: `${Math.max(pct(b) - pct(a), 0.5)}%`,
                  }}
                />
                <div className="rank-dot" style={{ left: `${pct(r.auprc)}%` }} />
              </div>
              <div className="rank-value">{r.auprc.toFixed(4)}</div>
            </div>
          );
        })}
      </div>

      <div className="overlap-note">
        {mode === "seed" ? (
          <>
            Rerunning with a different random seed barely moves anything —
            several of these classifiers are deterministic, so the bar collapses
            to a point. Reported this way, the ranking looks settled:{" "}
            <strong>{overlapping} of {rows.length}</strong> classifiers overlap
            the leader. This is how most benchmark tables are reported.
          </>
        ) : (
          <>
            Resample the test set instead and the same numbers stop separating.{" "}
            <strong>{overlapping} of {rows.length}</strong> classifiers now
            overlap the leader. The ranking was never resolvable at this test-set
            size; the seed just wasn't the thing that varied.
          </>
        )}
      </div>
    </div>
  );
}
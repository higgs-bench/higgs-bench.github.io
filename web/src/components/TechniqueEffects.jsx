import { useMemo, useState } from "react";
import { useJson } from "../lib/useJson";

const VERSIONS = [
  { id: "A", label: "1:1" },
  { id: "B", label: "1:10" },
  { id: "C", label: "1:50" },
];

const TECH_NAMES = {
  class_weight: "Class weights",
  smote: "SMOTE",
  undersample: "Undersampling",
  focal: "Focal loss",
};

const MODEL_NAMES = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  random_forest: "Random forest",
  catboost: "CatBoost",
  adaboost: "AdaBoost",
  voting_3: "Voting (3)",
  voting_4: "Voting (4)",
};

const COLOR = {
  hurts: "var(--pink)",
  helps: "var(--mint)",
  null: "#4a4a56",
};

function groupNote(sign) {
  if (!sign) return null;
  if (sign.n_effective === 0) {
    return "Mathematically inert at this ratio — the technique reduces to no treatment.";
  }
  const worse = sign.n_models - sign.n_helped;
  const p = sign.sign_test_p;
  const pTxt = p < 0.05 ? `p = ${p.toFixed(3)}` : `p = ${p.toFixed(2)}`;
  return `Worse than baseline in ${worse} of ${sign.n_models} classifiers · ${pTxt}`;
}

export default function TechniqueEffects() {
  const deltas = useJson("deltas.json");
  const signs = useJson("sign_tests.json");
  const [version, setVersion] = useState("C");

  const rows = useMemo(
    () => (deltas.data || []).filter((d) => d.version === version),
    [deltas.data, version]
  );

  const domain = useMemo(() => {
    if (!rows.length) return [-1, 1];
    const lo = Math.min(0, ...rows.map((r) => r.delta_ci_lo));
    const hi = Math.max(0, ...rows.map((r) => r.delta_ci_hi));
    const pad = (hi - lo) * 0.08 || 0.01;
    return [lo - pad, hi + pad];
  }, [rows]);

  if (deltas.loading || signs.loading) {
    return <div className="missing">Loading comparisons…</div>;
  }
  if (deltas.error || signs.error) {
    return <div className="missing state-error">{deltas.error || signs.error}</div>;
  }

  const [lo, hi] = domain;
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;
  const zero = pct(0);

  const techniques = ["class_weight", "smote", "undersample", "focal"];

  return (
    <div className="forest">
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
      </div>

      <div style={{ marginTop: 28 }}>
        {techniques.map((tech) => {
          const group = rows.filter((r) => r.technique === tech);
          if (!group.length) return null;
          const sign = (signs.data || []).find(
            (s) => s.version === version && s.technique === tech
          );

          return (
            <div className="forest-group" key={tech}>
              <div className="forest-group-head">
                <span className="forest-group-name">{TECH_NAMES[tech]}</span>
                <span className="forest-group-note">{groupNote(sign)}</span>
              </div>

              {group.map((r) => (
                <div className="forest-row" key={r.model}>
                  <div className="forest-name">{MODEL_NAMES[r.model]}</div>
                  <div className="forest-track">
                    <div className="forest-zero" style={{ left: `${zero}%` }} />
                    <div
                      className="forest-ci"
                      style={{
                        left: `${pct(r.delta_ci_lo)}%`,
                        width: `${Math.max(pct(r.delta_ci_hi) - pct(r.delta_ci_lo), 0.4)}%`,
                        background: COLOR[r.direction],
                        opacity: r.direction === "null" ? 0.75 : 1,
                      }}
                    />
                    <div
                      className="forest-dot"
                      style={{
                        left: `${pct(r.delta)}%`,
                        background: COLOR[r.direction],
                      }}
                    />
                  </div>
                  <div
                    className="forest-delta"
                    style={{ color: COLOR[r.direction] }}
                  >
                    {r.delta >= 0 ? "+" : ""}
                    {r.delta.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>

      <div className="forest-axis">
        <span>{lo.toFixed(3)}</span>
        <span>worse ← 0 → better</span>
        <span>+{hi.toFixed(3)}</span>
      </div>

      <div className="key">
        <span><i style={{ background: COLOR.hurts }} />significantly worse</span>
        <span><i style={{ background: COLOR.helps }} />significantly better</span>
        <span><i style={{ background: COLOR.null }} />no detectable difference</span>
      </div>
    </div>
  );
}
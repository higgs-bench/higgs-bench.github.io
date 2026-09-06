import { useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useJson } from "../lib/useJson";

const VERSIONS = [
  { id: "A", label: "1:1" },
  { id: "B", label: "1:10" },
  { id: "C", label: "1:50" },
];
const MODELS = [
  ["xgboost", "XGBoost"], ["lightgbm", "LightGBM"],
  ["random_forest", "Random forest"], ["catboost", "CatBoost"],
  ["adaboost", "AdaBoost"], ["voting_3", "Voting (3 models)"],
  ["voting_4", "Voting (4 models)"],
];
const TECHNIQUES = [
  ["baseline", "No treatment"], ["class_weight", "Class weights"],
  ["smote", "SMOTE"], ["undersample", "Undersampling"],
  ["focal", "Focal loss"],
];

function Segmented({ label, options, value, onChange }) {
  return (
    <div>
      <span className="control-label">{label}</span>
      <div className="seg">
        {options.map((o) => (
          <button
            key={o.id}
            aria-pressed={value === o.id}
            onClick={() => onChange(o.id)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Readout({ value, label, color }) {
  return (
    <div>
      <div className="readout-value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="readout-label">{label}</div>
    </div>
  );
}

export default function ThresholdExplorer() {
  const { data, error, loading } = useJson("threshold_curves.json");
  const [version, setVersion] = useState("C");
  const [model, setModel] = useState("xgboost");
  const [technique, setTechnique] = useState("baseline");
  const [idx, setIdx] = useState(49); // ~0.50

  const cell = data?.cells?.[`${version}_${model}_${technique}`];

  const series = useMemo(() => {
    if (!data || !cell) return [];
    return data.thresholds.map((t, i) => ({
      t,
      precision: cell.precision[i],
      recall: cell.recall[i],
      f1: cell.f1[i],
    }));
  }, [data, cell]);

  if (loading) return <div className="missing">Loading predictions…</div>;
  if (error) return <div className="missing state-error">{error}</div>;

  const threshold = data.thresholds[idx];
  let tunedIdx = 0;
  if (cell) {
    let bestGap = Infinity;
    data.thresholds.forEach((t, i) => {
      const gap = Math.abs(t - cell.tuned_threshold);
      if (gap < bestGap) {
        bestGap = gap;
        tunedIdx = i;
      }
    });
  }

  return (
    <>
      <div className="controls">
        <Segmented
          label="Signal-to-background ratio"
          options={VERSIONS}
          value={version}
          onChange={setVersion}
        />
        <div>
          <span className="control-label">Classifier</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map(([id, name]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </div>
        <div>
          <span className="control-label">Imbalance technique</span>
          <select value={technique} onChange={(e) => setTechnique(e.target.value)}>
            {TECHNIQUES.map(([id, name]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      {!cell ? (
        <div className="missing">
          Focal loss only applies to XGBoost and LightGBM — the other
          classifiers have no custom-objective interface. Pick a different
          combination.
        </div>
      ) : (
        <div className="explorer">
          <div className="readout">
            <Readout value={cell.f1[idx].toFixed(3)} label="F1 score" color="var(--pink)" />
            <Readout value={cell.precision[idx].toFixed(3)} label="Precision" />
            <Readout value={cell.recall[idx].toFixed(3)} label="Recall" />
            <Readout value={cell.tp[idx].toLocaleString()} label="Signal events found" />
            <Readout value={cell.fp[idx].toLocaleString()} label="Background misidentified" />
          </div>

          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={series} margin={{ top: 4, right: 8, left: -18, bottom: 4 }}>
                <CartesianGrid stroke="#24242e" vertical={false} />
                <XAxis
                  dataKey="t" type="number" domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  stroke="#8b8894" tick={{ fontSize: 12 }}
                />
                <YAxis domain={[0, 1]} stroke="#8b8894" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    background: "#101014", border: "1px solid #24242e",
                    borderRadius: 6, fontSize: 13,
                  }}
                  labelFormatter={(v) => `Threshold ${Number(v).toFixed(2)}`}
                  formatter={(v, n) => [Number(v).toFixed(3), n]}
                />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <ReferenceLine x={threshold} stroke="#ff2d78" strokeWidth={2} />
                <ReferenceLine
                  x={cell.tuned_threshold} stroke="#ffc24b"
                  strokeDasharray="4 4"
                />
                <Line type="monotone" dataKey="f1" name="F1" stroke="#ff2d78" dot={false} strokeWidth={2.4} />
                <Line type="monotone" dataKey="precision" name="Precision" stroke="#4fffc1" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="recall" name="Recall" stroke="#ff7ab0" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="slider-row">
            <div className="slider-head">
              <label htmlFor="thr">Decision threshold</label>
              <span className="num" style={{ color: "var(--pink)" }}>
                {threshold.toFixed(2)}
              </span>
            </div>
            <input
              id="thr" type="range" min={0} max={data.thresholds.length - 1}
              value={idx} onChange={(e) => setIdx(Number(e.target.value))}
            />
            <div className="tuned-note">
              Tuned on validation data, this classifier picks{" "}
              <span className="num" style={{ color: "var(--amber)" }}>
                {cell.tuned_threshold.toFixed(2)}
              </span>
              , where F1 reaches{" "}
              <span className="num">{cell.f1[tunedIdx].toFixed(3)}</span>.
              <button onClick={() => setIdx(tunedIdx)}>Jump to it</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
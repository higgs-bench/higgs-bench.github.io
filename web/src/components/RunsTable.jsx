import { useMemo, useState } from "react";
import { useJson } from "../lib/useJson";

const COLUMNS = [
  { key: "version", label: "Ratio", num: false },
  { key: "model", label: "Classifier", num: false },
  { key: "technique", label: "Technique", num: false },
  { key: "seed", label: "Seed", num: true, dp: 0 },
  { key: "test_auprc", label: "AUPRC", num: true, dp: 4 },
  { key: "test_auc_roc", label: "AUC", num: true, dp: 4 },
  { key: "test_f1", label: "F1 (tuned)", num: true, dp: 4 },
  { key: "fixed_f1", label: "F1 (cut at 0.5)", num: true, dp: 4 },
  { key: "test_precision", label: "Precision", num: true, dp: 4 },
  { key: "test_recall", label: "Recall", num: true, dp: 4 },
  { key: "test_brier", label: "Brier", num: true, dp: 4 },
  { key: "tuned_threshold", label: "Threshold", num: true, dp: 2 },
  { key: "fit_seconds", label: "Fit (s)", num: true, dp: 1 },
];

const RATIO = { A: "1:1", B: "1:10", C: "1:50" };
const MODEL_NAMES = {
  xgboost: "XGBoost", lightgbm: "LightGBM", random_forest: "Random forest",
  catboost: "CatBoost", adaboost: "AdaBoost",
  voting_3: "Voting (3)", voting_4: "Voting (4)",
};
const TECH_NAMES = {
  baseline: "No treatment", class_weight: "Class weights", smote: "SMOTE",
  undersample: "Undersampling", focal: "Focal loss",
};

export default function RunsTable() {
  const { data, error, loading } = useJson("runs.json");
  const [version, setVersion] = useState("all");
  const [technique, setTechnique] = useState("all");
  const [sort, setSort] = useState({ key: "test_auprc", dir: "desc" });

  const rows = useMemo(() => {
    if (!data) return [];
    let r = data;
    if (version !== "all") r = r.filter((d) => d.version === version);
    if (technique !== "all") r = r.filter((d) => d.technique === technique);

    const { key, dir } = sort;
    return [...r].sort((a, b) => {
      const x = a[key];
      const y = b[key];
      const cmp = typeof x === "number" ? x - y : String(x).localeCompare(String(y));
      return dir === "asc" ? cmp : -cmp;
    });
  }, [data, version, technique, sort]);

  const toggleSort = (key) =>
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === "asc" ? "desc" : "asc" }
        : { key, dir: "desc" }
    );

  const downloadCsv = () => {
    const head = COLUMNS.map((c) => c.key).join(",");
    const body = rows
      .map((r) => COLUMNS.map((c) => r[c.key]).join(","))
      .join("\n");
    const blob = new Blob([`${head}\n${body}\n`], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "higgs_runs.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="missing">Loading runs…</div>;
  if (error) return <div className="missing state-error">{error}</div>;

  return (
    <div>
      <div className="controls">
        <div>
          <span className="control-label">Signal-to-background ratio</span>
          <select value={version} onChange={(e) => setVersion(e.target.value)}>
            <option value="all">All ratios</option>
            <option value="A">1:1</option>
            <option value="B">1:10</option>
            <option value="C">1:50</option>
          </select>
        </div>
        <div>
          <span className="control-label">Technique</span>
          <select value={technique} onChange={(e) => setTechnique(e.target.value)}>
            <option value="all">All techniques</option>
            {Object.entries(TECH_NAMES).map(([id, name]) => (
              <option key={id} value={id}>{name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  onClick={() => toggleSort(c.key)}
                  aria-sort={
                    sort.key === c.key
                      ? sort.dir === "asc" ? "ascending" : "descending"
                      : "none"
                  }
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key}>
                {COLUMNS.map((c) => {
                  let v = r[c.key];
                  if (c.key === "version") v = RATIO[v];
                  else if (c.key === "model") v = MODEL_NAMES[v];
                  else if (c.key === "technique") v = TECH_NAMES[v];
                  else if (c.num) v = Number(v).toFixed(c.dp);
                  return (
                    <td key={c.key} className={c.num ? "n" : undefined}>
                      {v}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-bar">
        <span className="table-count">
          Showing {rows.length} of {data.length} runs
        </span>
        <button className="dl" onClick={downloadCsv}>
          Download these rows as CSV
        </button>
      </div>
    </div>
  );
}
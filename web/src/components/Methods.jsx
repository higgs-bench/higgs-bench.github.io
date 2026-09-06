const CELLS = [
  {
    title: "Building the imbalance",
    body: `A fixed pool of 200,000 background and 200,000 signal events is drawn
      from the 11 million in the dataset, split 60/20/20 into training,
      validation and test. Rarer ratios are made by truncating the signal
      class, so background rows are identical across all three and the 1:50
      splits are subsets of the 1:10 splits.`,
  },
  {
    title: "Keeping the test set honest",
    body: `Test sets stay imbalanced — that is the situation a detector actually
      faces. Resampling touches training data only. The decision threshold is
      chosen on validation data and never on test, which is what separates a
      technique's effect from a threshold's effect.`,
  },
  {
    title: "Scoring",
    body: `AUPRC is the headline metric and is always shown against the
      prevalence floor, since a random classifier scores exactly the
      prevalence. AUC-ROC, F1, precision, recall, Brier score and MCC are
      recorded alongside it for every run.`,
  },
  {
    title: "Confidence intervals",
    body: `2,000 stratified bootstrap resamples of the test set. For technique
      comparisons the same resample is applied to the technique and its
      baseline, so the interval sits on the difference rather than on two
      independent numbers.`,
  },
  {
    title: "Significance",
    body: `A sign test across the seven classifiers, which bottoms out at
      p = 0.016 when all seven agree. Testing across five seeds instead
      could never reach p < 0.05 and would be measuring the wrong thing.`,
  },
  {
    title: "What was not tuned",
    body: `Every classifier uses 500 estimators at depth 6, learning rate 0.1.
      That makes the comparison consistent, not optimal — a tuned model of
      any of these families would score higher. AdaBoost uses 200 estimators
      because it trains sequentially and the budget was fixed.`,
  },
];

export default function Methods() {
  return (
    <>
      <div className="method-grid">
        {CELLS.map((c) => (
          <div className="method-cell" key={c.title}>
            <h3>{c.title}</h3>
            <p>{c.body}</p>
          </div>
        ))}
      </div>

      <div className="caveat">
        <h3>On signal significance</h3>
        <p>
          The quantity Z = TP/√(TP+FP) appears in the run data for continuity
          with earlier work, but it grows with the square root of test-set size
          while classifier quality stays put. It is not a discovery
          significance and is not compared against Z ≥ 5 anywhere here.
        </p>
      </div>
    </>
  );
}
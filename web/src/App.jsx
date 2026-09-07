import Reveal from "./components/Reveal";
import ScrollProgress from "./components/ScrollProgress";
import ThresholdExplorer from "./components/ThresholdExplorer";
import { useJson } from "./lib/useJson";
import ParticleField from "./components/ParticleField";
import TechniqueEffects from "./components/TechniqueEffects";
import UncertaintyToggle from "./components/UncertaintyToggle";
import RunsTable from "./components/RunsTable";
import Methods from "./components/Methods";

function Stat({ value, label }) {
  return (
    <div>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export default function App() {
  const { data: meta, error, loading } = useJson("meta.json");

  if (loading) return <div className="shell state">Loading results…</div>;
  if (error) {
    return (
      <div className="shell state state-error">
        Could not load results data: {error}
      </div>
    );
  }

  return (
    <>
      <ScrollProgress />
      <ParticleField />
      <main>
        <section className="shell section">
          <h1>
            <span className="hero-line">Five ways to fix class imbalance.</span>
            <span className="hero-line">None of them beat doing nothing.</span>
          </h1>
          <p className="lede hero-sub" style={{ marginTop: 28 }}>
            We ran {meta.n_runs} experiments on the HIGGS boson dataset —
            seven classifiers, five imbalance-handling techniques, three
            signal-to-background ratios, five seeds each. Once the decision
            threshold is tuned on held-out data, rebalancing the training set
            stops helping. Across {meta.n_cells_imbalanced} comparisons on
            imbalanced data it produced one improvement and{" "}
            {meta.n_sig_hurt} measurable losses.
          </p>
          <div className="stats hero-stats">
            <Stat value={meta.n_runs} label="experiments run" />
            <Stat
              value={`${meta.n_sig_hurt} / ${meta.n_sig_help}`}
              label="comparisons made significantly worse, versus significantly better"
            />
            <Stat
              value={`${meta.uncertainty_ratio}×`}
              label="how much a single-seed error bar understates real uncertainty"
            />
          </div>
          <p className="hero-caption">
            <span className="dot" />
            One in fifty drifting particles is pink. That is the signal rate
            a classifier faces at the hardest ratio tested here.
          </p>
        </section>

        <section className="shell section">
          <Reveal>
            <h2>The threshold does the work</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              A classifier outputs a probability; you choose where to cut it.
              At a 1:50 ratio an untreated model scores 0.006 on F1 at the
              default cut of 0.5, and 0.169 once that cut is tuned on held-out
              data. Rebalancing the training set moves the probabilities
              around, which looks like an improvement only while the cut stays
              fixed. Drag the threshold and watch.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <ThresholdExplorer />
          </Reveal>
        </section>


        <section className="shell section">
          <Reveal>
            <h2>Every comparison, with honest error bars</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              Each row compares one technique against no treatment, on the same
              classifier and the same test set. Intervals come from 2,000 paired
              bootstrap resamples: the same resample is applied to both sides,
              so the interval sits on the difference itself. A bar that clears
              zero is a real effect. At 1:1 the techniques have nothing to
              correct, and the picture changes sharply as signal gets rarer.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <TechniqueEffects />
          </Reveal>
        </section>

        <section className="shell section">
          <Reveal>
            <h2>The error bar you choose decides the answer</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              Same {meta.n_runs} runs, same classifiers, same scores. The only
              thing that changes below is what the interval is measuring. Repeat
              the run with a new seed and you learn how stable the training
              algorithm is. Resample the test set and you learn how much of the
              gap between two classifiers is an accident of which events landed
              in it. Only the second question is the one a ranking table claims
              to answer.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <UncertaintyToggle />
          </Reveal>
        </section>

        <section className="shell section">
          <Reveal>
            <h2>Check it yourself</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              Every run, unfiltered. Sort by any column, narrow to a ratio or a
              technique, take the rows away as CSV. The two F1 columns are the
              whole argument in miniature: one uses a threshold tuned on
              validation data, the other leaves the cut at 0.5.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <RunsTable />
          </Reveal>
        </section>
        
        <section className="shell section">
          <Reveal>
            <h2>How this was measured</h2>
            <p className="lede" style={{ marginTop: 20 }}>
              The protocol is the argument. Most of what separates this from a
              benchmark table that reports the opposite result comes down to
              where the threshold was chosen and what the error bars measure.
            </p>
          </Reveal>
          <Reveal delay={120}>
            <Methods />
          </Reveal>
        </section>
      </main>

      
      <footer className="shell">
        <p>
          Data from the HIGGS dataset in the UCI Machine Learning Repository:
          11 million simulated collision events, 28 features, no missing values.
          All code, cached predictions and result files are in the repository.
        </p>
        <div className="footer-links">
          <a href="https://github.com/higgs-bench/higgs-bench.github.io">
            Source and data
          </a>
          <a href="https://archive.ics.uci.edu/dataset/280/higgs">
            HIGGS dataset
          </a>
          <a href="/summary.html">Text-only summary</a>
        </div>
      </footer>
    </>
  );
}
import Link from "next/link";
import { ArrowRight, ArrowUpRight, GitBranch, ShieldCheck } from "lucide-react";
import SampleInvestigation from "@/components/SampleInvestigation";
import "./home.css";

export default function Home() {
  return (
    <div className="lumiq-home">
      <a className="home-skip" href="#main">
        Skip to content
      </a>
      <header className="home-nav home-container">
        <Link className="home-brand" href="/" aria-label="Lumiq home">
          <span className="brand-mark">L</span>Lumiq
        </Link>
        <nav aria-label="Main navigation">
          <a href="#approach">The approach</a>
          <a href="#inside">Under the hood</a>
        </nav>
        <Link className="home-signin" href="/login">
          Sign in <ArrowUpRight size={16} />
        </Link>
      </header>
      <main id="main">
        <section
          className="home-hero home-container"
          aria-labelledby="hero-title"
        >
          <div className="hero-copy">
            <p className="home-kicker">
              An investigation workspace for your data
            </p>
            <h1 id="hero-title">
              Less guesswork.
              <br />
              <span>More evidence.</span>
            </h1>
            <p className="hero-description">
              Ask your CSV a question. Follow the calculation. See what the
              numbers actually support.
            </p>
            <Link className="home-cta" href="/chat">
              Open your workspace <ArrowUpRight size={20} />
            </Link>
          </div>
          <SampleInvestigation />
        </section>
        <div className="home-principle home-container">
          <span>A question is the beginning.</span>
          <span>
            The evidence is the point.{" "}
            <ArrowRight size={18} aria-hidden="true" />
          </span>
        </div>
        <section
          id="approach"
          className="home-approach home-container"
          aria-labelledby="approach-title"
        >
          <div className="approach-heading">
            <h2 id="approach-title">
              An answer you can
              <br />
              <span>take apart.</span>
            </h2>
            <p>
              Every investigation keeps the question, the calculation, and the
              interpretation connected. So you can check the work.
            </p>
          </div>
          <div className="approach-steps">
            <article>
              <span className="step-symbol" aria-hidden="true">
                ↗
              </span>
              <div>
                <h3>Start with what’s there.</h3>
                <p>
                  Upload a CSV. See its columns, date coverage, missing values,
                  and duplicates before asking more of it.
                </p>
              </div>
            </article>
            <article>
              <span className="step-symbol" aria-hidden="true">
                ＝
              </span>
              <div>
                <h3>Make the calculation visible.</h3>
                <p>
                  Inspect the metrics, filters, and time windows. Change the
                  plan and save a new revision alongside the original.
                </p>
              </div>
            </article>
            <article>
              <span className="step-symbol" aria-hidden="true">
                ↳
              </span>
              <div>
                <h3>Follow the finding to its source.</h3>
                <p>
                  Open the measured evidence behind an interpretation. Download
                  the table. Keep the assumptions and limits in view.
                </p>
              </div>
            </article>
          </div>
        </section>
        <section
          id="inside"
          className="home-engine home-container"
          aria-labelledby="engine-title"
        >
          <div className="engine-top">
            <GitBranch size={25} strokeWidth={1.5} />
            <h2 id="engine-title">
              Two specialists.
              <br />
              One inspectable investigation.
            </h2>
            <p>
              A planner shapes the question. Trusted tools calculate. An analyst
              interprets the evidence.
            </p>
          </div>
          <ol className="engine-flow" aria-label="Investigation workflow">
            <li>
              <span>Planner</span>
              <small>Question → calculation plan</small>
            </li>
            <li className="flow-tool">
              <span>Analytical tools</span>
              <small>Plan → measured evidence</small>
            </li>
            <li>
              <span>Analyst</span>
              <small>Evidence → interpretation</small>
            </li>
          </ol>
          <div className="engine-resilience">
            <ShieldCheck size={20} />
            <p>
              <strong>Useful work survives a model failure.</strong> Gemini and
              Groq share a bounded fallback path. Saved calculations stay
              available when interpretation needs to resume.
            </p>
          </div>
        </section>
        <section
          className="home-finale home-container"
          aria-labelledby="finale-title"
        >
          <div>
            <p>Bring your data. Keep your curiosity.</p>
            <h2 id="finale-title">
              There’s a story
              <br />
              in your spreadsheet.
            </h2>
          </div>
          <Link className="home-cta" href="/chat">
            Open your workspace <ArrowUpRight size={20} />
          </Link>
        </section>
      </main>
      <footer className="home-footer home-container">
        <Link className="home-brand" href="/">
          Lumiq
        </Link>
        <p>From question to evidence.</p>
        <a
          href="https://github.com/Shreyas-Sarkar/Lumiq"
          target="_blank"
          rel="noopener noreferrer"
        >
          Explore the source <ArrowUpRight size={14} />
        </a>
      </footer>
    </div>
  );
}

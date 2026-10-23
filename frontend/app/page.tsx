import BrandMark from "@/components/BrandMark";
import Link from "next/link";
import { ArrowRight, ArrowUpRight, GitBranch, ShieldCheck } from "lucide-react";
import SampleInvestigation from "@/components/SampleInvestigation";
import HomeMotion from "@/components/HomeMotion";
import InvestigationTrace from "@/components/InvestigationTrace";
import "./home.css";

export default function Home() {
  return (
    <div className="lumiq-home">
      <HomeMotion />
      <a className="home-skip" href="#main">
        Skip to content
      </a>
      <header className="home-nav home-container">
        <Link className="home-brand" href="/" aria-label="Lumiq home">
          <BrandMark />
          Lumiq
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
          <div className="home-sample-stage">
            <div
              className="sample-sheet sample-sheet-back"
              aria-hidden="true"
            />
            <div
              className="sample-sheet sample-sheet-front"
              aria-hidden="true"
            />
            <SampleInvestigation />
          </div>
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
          <div className="approach-heading" data-reveal>
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
            <article data-reveal>
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
            <article data-reveal>
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
            <article data-reveal>
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
          <div className="engine-top" data-reveal>
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
          <InvestigationTrace />
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
          data-reveal
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
          <BrandMark size={24} />
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

"use client";

import { useEffect, useState } from "react";

import {
  fetchPollsterCoverage,
  type PollCoverageFetchResult,
  type PollCoverageGap,
  type PollCoveragePollResult,
} from "./pollstersApi";

const COVERAGE_ENDPOINT = "/api/v1/pollsters/coverage";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatRatio(value: number): string {
  return `${value.toFixed(2)}x`;
}

function formatDate(value: string): string {
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function statusStyles(status: string) {
  if (status === "undersampled") {
    return {
      color: "#8b5b00",
      background: "rgba(214, 158, 46, 0.12)",
      label: "Undersampled",
    };
  }
  if (status === "oversampled") {
    return {
      color: "#4b6d90",
      background: "rgba(75, 109, 144, 0.12)",
      label: "Oversampled",
    };
  }
  return {
    color: "var(--color-text-muted)",
    background: "var(--color-surface)",
    label: "Representative",
  };
}

function topAffectedTypes(gap: PollCoverageGap): string {
  if (!gap.affected_types.length) return "No type breakout available";
  return gap.affected_types
    .slice(0, 2)
    .map((item) => `${item.display_name} (${formatPercent(item.group_share)})`)
    .join(" • ");
}

function selectFeaturedPolls(polls: PollCoveragePollResult[]): PollCoveragePollResult[] {
  const withUndersampling = polls
    .filter((poll) => poll.n_undersampled > 0)
    .sort((a, b) => b.n_undersampled - a.n_undersampled || b.gaps.length - a.gaps.length);

  if (withUndersampling.length > 0) {
    return withUndersampling.slice(0, 2);
  }

  return polls.slice(0, 1);
}

export function PollCoverageSection() {
  const [result, setResult] = useState<PollCoverageFetchResult | { status: "loading" }>({
    status: "loading",
  });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const next = await fetchPollsterCoverage(COVERAGE_ENDPOINT);
      if (!cancelled) {
        setResult(next);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (result.status === "loading") {
    return (
      <section
        aria-labelledby="poll-coverage-heading"
        className="mt-10 rounded-xl border px-5 py-6"
        style={{
          borderColor: "var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        <h2
          id="poll-coverage-heading"
          className="text-2xl font-semibold"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Poll Coverage Diagnostics
        </h2>
        <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
          Loading poll coverage diagnostics...
        </p>
      </section>
    );
  }

  if (result.status === "unavailable") {
    return (
      <section
        aria-labelledby="poll-coverage-heading"
        className="mt-10 rounded-xl border px-5 py-6"
        style={{
          borderColor: "var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        <h2
          id="poll-coverage-heading"
          className="text-2xl font-semibold"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Poll Coverage Diagnostics
        </h2>
        <div
          data-testid="coverage-unavailable"
          className="mt-4 rounded-lg border px-4 py-4"
          style={{
            borderColor: "rgba(214, 158, 46, 0.35)",
            background: "rgba(214, 158, 46, 0.08)",
          }}
        >
          <p className="text-sm font-semibold" style={{ color: "var(--color-text)" }}>
            Coverage diagnostics unavailable
          </p>
          <p className="mt-2 text-sm" style={{ color: "var(--color-text-muted)" }}>
            The current report has not been generated yet.
          </p>
          <p className="mt-2 text-xs" style={{ color: "var(--color-text-subtle)" }}>
            {result.message}
          </p>
        </div>
      </section>
    );
  }

  if (result.status === "error") {
    return (
      <section
        aria-labelledby="poll-coverage-heading"
        className="mt-10 rounded-xl border px-5 py-6"
        style={{
          borderColor: "var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        <h2
          id="poll-coverage-heading"
          className="text-2xl font-semibold"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Poll Coverage Diagnostics
        </h2>
        <p className="mt-3 text-sm" style={{ color: "var(--color-text-muted)" }}>
          {result.message}
        </p>
      </section>
    );
  }

  const { metadata, summary, per_poll_results } = result.data;
  const topGroups = summary.undersampled_ranking.slice(0, 5);
  const featuredPolls = selectFeaturedPolls(per_poll_results);

  return (
    <section aria-labelledby="poll-coverage-heading" className="mt-10" data-testid="coverage-section">
      <div className="mb-5">
        <h2
          id="poll-coverage-heading"
          className="text-2xl font-semibold"
          style={{ fontFamily: "var(--font-serif)", color: "var(--color-text)" }}
        >
          Poll Coverage Diagnostics
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          This report compares the weighted demographic composition published in each poll
          against the electorate baseline used by WetherVane. Ratios below{" "}
          {metadata.undersample_threshold.toFixed(2)}x are flagged as undersampled; ratios above{" "}
          {metadata.oversample_threshold.toFixed(2)}x are flagged as oversampled.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-xl border px-4 py-4" style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}>
          <p className="text-xs uppercase tracking-[0.18em]" style={{ color: "var(--color-text-subtle)" }}>
            Total Polls
          </p>
          <p className="mt-2 text-2xl font-semibold" style={{ color: "var(--color-text)" }}>
            {metadata.total_polls}
          </p>
        </div>
        <div className="rounded-xl border px-4 py-4" style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}>
          <p className="text-xs uppercase tracking-[0.18em]" style={{ color: "var(--color-text-subtle)" }}>
            With XT Data
          </p>
          <p className="mt-2 text-2xl font-semibold" style={{ color: "var(--color-text)" }}>
            {metadata.polls_with_xt_data}
          </p>
        </div>
        <div className="rounded-xl border px-4 py-4" style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}>
          <p className="text-xs uppercase tracking-[0.18em]" style={{ color: "var(--color-text-subtle)" }}>
            Polls Analyzed
          </p>
          <p className="mt-2 text-2xl font-semibold" style={{ color: "var(--color-text)" }}>
            {metadata.polls_analyzed}
          </p>
        </div>
        <div className="rounded-xl border px-4 py-4" style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}>
          <p className="text-xs uppercase tracking-[0.18em]" style={{ color: "var(--color-text-subtle)" }}>
            Mappable Groups
          </p>
          <p className="mt-2 text-2xl font-semibold" style={{ color: "var(--color-text)" }}>
            {metadata.mappable_xt_columns.length}
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border px-4 py-4" style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}>
        <p className="text-xs uppercase tracking-[0.18em]" style={{ color: "var(--color-text-subtle)" }}>
          Active XT Columns
        </p>
        <p className="mt-2 text-sm leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          {metadata.active_xt_columns.join(", ")}
        </p>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr,1.4fr]">
        <div
          className="rounded-xl border px-5 py-5"
          style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}
        >
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold" style={{ color: "var(--color-text)" }}>
              Top undersampled groups
            </h3>
            <span className="text-xs" style={{ color: "var(--color-text-subtle)" }}>
              Ranked by flagged polls
            </span>
          </div>
          <div className="mt-4 space-y-3" data-testid="coverage-top-groups">
            {topGroups.map((group) => {
              const summaryItem = summary.by_group[group.group];
              return (
                <div
                  key={group.group}
                  className="rounded-lg border px-4 py-3"
                  style={{ borderColor: "var(--color-border)" }}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <p className="font-medium" style={{ color: "var(--color-text)" }}>
                      {group.label}
                    </p>
                    <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
                      {group.n_polls_undersampled} polls
                    </p>
                  </div>
                  <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-subtle)" }}>
                    Representative in {summaryItem?.n_representative ?? 0} polls; oversampled in{" "}
                    {summaryItem?.n_oversampled ?? 0}.
                  </p>
                  <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-subtle)" }}>
                    Most affected types:{" "}
                    {summaryItem?.top_affected_types.length
                      ? summaryItem.top_affected_types
                          .slice(0, 2)
                          .map((item) => `${item.type_label} (${item.n_races_affected})`)
                          .join(" • ")
                      : "None listed"}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        <div
          className="rounded-xl border px-5 py-5"
          style={{ borderColor: "var(--color-border)", background: "var(--color-surface)" }}
        >
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold" style={{ color: "var(--color-text)" }}>
              Per-poll gap breakdown
            </h3>
            <span className="text-xs" style={{ color: "var(--color-text-subtle)" }}>
              Largest undersampling examples
            </span>
          </div>
          <div className="mt-4 space-y-5" data-testid="coverage-featured-polls">
            {featuredPolls.map((poll) => (
              <article key={`${poll.pollster}-${poll.race}-${poll.date}`} className="rounded-lg border px-4 py-4" style={{ borderColor: "var(--color-border)" }}>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-medium" style={{ color: "var(--color-text)" }}>
                      {poll.pollster} • {poll.race}
                    </p>
                    <p className="mt-1 text-xs" style={{ color: "var(--color-text-subtle)" }}>
                      {poll.state} • {formatDate(poll.date)} • Sample {poll.n_sample ?? "n/a"}
                    </p>
                  </div>
                  <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
                    {poll.n_undersampled} undersampled / {poll.n_oversampled} oversampled
                  </p>
                </div>

                <div className="mt-4 space-y-3">
                  {poll.gaps.map((gap) => {
                    const style = statusStyles(gap.status);
                    return (
                      <div
                        key={`${poll.pollster}-${poll.race}-${gap.demographic_group}`}
                        className="rounded-lg border px-3 py-3"
                        style={{ borderColor: "var(--color-border)" }}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-3">
                          <div>
                            <p className="font-medium" style={{ color: "var(--color-text)" }}>
                              {gap.label}
                            </p>
                            <p className="mt-1 text-xs" style={{ color: "var(--color-text-subtle)" }}>
                              Poll share {formatPercent(gap.poll_share)} vs electorate {formatPercent(gap.population_share)}
                            </p>
                          </div>
                          <span
                            className="rounded-full px-2 py-1 text-xs font-medium"
                            style={{ color: style.color, background: style.background }}
                          >
                            {style.label} • {formatRatio(gap.ratio)}
                          </span>
                        </div>
                        <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--color-text-subtle)" }}>
                          Most exposed community types: {topAffectedTypes(gap)}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

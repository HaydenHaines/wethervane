import { ELECTION_YEAR, GOVERNOR_RACES_COUNT } from "@/lib/config/election";

/**
 * Governor overview page.
 *
 * Governor race data is not yet available from the API. Rather than showing
 * placeholder cards with fabricated EVEN/Tossup ratings, this page displays
 * an honest "coming soon" message explaining when forecasts will be available.
 */
export default function GovernorPage() {
  return (
    <div>
      <h1 className="font-serif text-2xl font-bold mb-4">{ELECTION_YEAR} Governor Races</h1>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-muted)" }}>
        {GOVERNOR_RACES_COUNT} governors are on the ballot in {ELECTION_YEAR}, including 11 open seats.
      </p>

      {/* Coming soon notice */}
      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: 8,
          padding: "32px 24px",
          background: "var(--color-surface)",
          textAlign: "center",
          maxWidth: 560,
          margin: "0 auto",
        }}
      >
        <div
          style={{
            fontSize: 32,
            marginBottom: 12,
            color: "var(--color-text-muted)",
          }}
        >
          &#9788;
        </div>
        <h2
          style={{
            fontFamily: "var(--font-serif)",
            fontSize: 20,
            fontWeight: 700,
            marginBottom: 12,
          }}
        >
          Governor forecasts coming soon
        </h2>
        <p
          style={{
            fontSize: 15,
            lineHeight: 1.65,
            color: "var(--color-text-muted)",
            margin: "0 0 8px",
          }}
        >
          Governor race predictions are based on structural model priors derived
          from the electoral type system. Detailed state-level forecasts will
          appear here as polling data arrives for each race.
        </p>
        <p
          style={{
            fontSize: 14,
            lineHeight: 1.65,
            color: "var(--color-text-muted)",
            margin: 0,
          }}
        >
          In the meantime, explore{" "}
          <a
            href="/forecast/senate"
            style={{ color: "var(--color-dem)", textDecoration: "none" }}
          >
            Senate forecasts
          </a>{" "}
          or the{" "}
          <a
            href="/forecast"
            style={{ color: "var(--color-dem)", textDecoration: "none" }}
          >
            national map
          </a>
          .
        </p>
      </div>
    </div>
  );
}

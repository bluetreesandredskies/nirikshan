import { S } from "../strings.js";

const lvl = (l) => (["low", "moderate", "high"].includes(l) ? l : "unknown");
const list = (v) => (Array.isArray(v) ? v : typeof v === "string" && v ? [v] : []);
const pretty = (s) => String(s).replace(/_/g, " ");

/** Two SEPARATE sections (image / exposure), never merged. result = /risk-fusion response. */
export default function RiskBadge({ result, lang }) {
  const t = S[lang];
  const ir = result.image_risk || {};
  const er = result.exposure_risk || {};
  const il = lvl(ir.image_risk_level);
  const el = lvl(er.exposure_risk_level);
  // The backend's exact exposure-explanation field name isn't confirmed; accept common shapes.
  const reasons = [...list(er.reasons), ...list(er.explanations), ...list(er.explanation)];
  const why = reasons.length ? reasons : list(er.routes_triggered).map(pretty);

  return (
    <div>
      {result.model_status === "stage1_placeholder" && <div className="small-note">{t.prelim}</div>}

      {result.safety_escalated && (
        <div className="notice" role="alert">
          <strong>{t.safety}</strong>
          {ir.safety_explanation}
        </div>
      )}

      <section className={`badge lvl-${il}`} style={{ "--i": 0 }}>
        <div className="kind">{t.imageRisk}</div>
        <div className="level">{t.levels[il]}</div>
        {ir.explanation && <p>{ir.explanation}</p>}
      </section>

      <section className={`badge lvl-${el}`} style={{ "--i": 1 }}>
        <div className="kind">{t.exposureRisk}</div>
        <div className="level">{t.levels[el]}</div>
        {why.length > 0 && (
          <>
            <p>{t.because}</p>
            <ul>{why.map((r, i) => <li key={i}>{r}</li>)}</ul>
          </>
        )}
      </section>

      {result.recommendation?.text && (
        <section className="card" style={{ animation: "rise var(--dur) var(--ease) 500ms both" }}>
          <h2>{t.next_}</h2>
          <p>{result.recommendation.text}</p>
        </section>
      )}
    </div>
  );
}

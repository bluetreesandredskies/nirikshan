import { useEffect, useState } from "react";
import { getFairness } from "../api.js";
import LoadingScreen from "../components/LoadingScreen.jsx";
import { S } from "../strings.js";

export default function FairnessDashboard({ lang, onBack }) {
  const t = S[lang];
  const [data, setData] = useState(null);
  const [err, setErr] = useState(false);
  const [go, setGo] = useState(false);

  useEffect(() => {
    let dead = false;
    getFairness().then((d) => !dead && setData(d)).catch(() => !dead && setErr(true));
    return () => { dead = true; };
  }, []);

  useEffect(() => {
    if (!data) return;
    const id = requestAnimationFrame(() => requestAnimationFrame(() => setGo(true)));
    return () => cancelAnimationFrame(id);
  }, [data]);

  return (
    <section className="card enter">
      <h1>{t.fairTitle}</h1>
      {err && <div className="err" role="alert">{t.errNet}</div>}
      {!data && !err && <LoadingScreen variant="analyzing" lang={lang} compact />}
      {data && (
        <>
          {!data.generated_at && data.notes && <p className="small-note">{data.notes}</p>}
          {data.bins.map((b) => {
            const has = b.n > 0 && typeof b.accuracy === "number";
            const pct = has ? Math.round(b.accuracy * 1000) / 10 : 0;
            return (
              <div className="bar-row" key={b.ita_bin}>
                <div className="bar-head">
                  <span>{t.bins[b.ita_bin] || b.ita_bin}{b.low_confidence && ` (${t.small})`}</span>
                  <span>{has ? `${pct}%` : t.noData} · n={b.n}</span>
                </div>
                <div className="track" role="img" aria-label={`${t.bins[b.ita_bin] || b.ita_bin}: ${has ? pct + "%" : t.noData}, n=${b.n}`}>
                  {has ? <div className={`fill${b.low_confidence ? " hatch" : ""}`} style={{ width: go ? `${pct}%` : 0 }} /> : <span className="nodata">{t.noData}</span>}
                </div>
                {has && typeof b.macro_f1 === "number" && <div className="help">macro-F1 {b.macro_f1.toFixed(2)}</div>}
              </div>
            );
          })}
        </>
      )}
      <div className="btn-row"><button className="btn" onClick={onBack}>{t.backBtn}</button></div>
    </section>
  );
}

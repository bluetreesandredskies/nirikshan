import { useEffect, useRef, useState } from "react";
import PhotoCapture from "./components/PhotoCapture.jsx";
import ExposureIntakeForm from "./components/ExposureIntakeForm.jsx";
import LoadingScreen from "./components/LoadingScreen.jsx";
import RiskBadge from "./components/RiskBadge.jsx";
import GradCamOverlay from "./components/GradCamOverlay.jsx";
import FairnessDashboard from "./pages/FairnessDashboard.jsx";
import { postFusion } from "./api.js";
import { enqueue, listQueued, countQueued, removeQueued } from "./offlineQueue.js";
import { S } from "./strings.js";

// phase: capture | form | uploading | analyzing | queued | result | dashboard
export default function App() {
  const [lang, setLang] = useState("en");
  const [phase, setPhase] = useState("capture");
  const [photo, setPhoto] = useState(null);
  const [answers, setAnswers] = useState(null); // raw form state, kept so a failed submit doesn't lose answers
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(0);
  const [sentBack, setSentBack] = useState(false);
  const t = S[lang];
  const flushing = useRef(false);
  const latest = useRef({});

  useEffect(() => () => photo && URL.revokeObjectURL(photo.url), [photo]);

  function msg(e) {
    if (e.kind === "detail") return e.detail;
    if (e.kind === "validation") return t.err422;
    if (e.kind === "network") return t.errNet;
    if (e.kind === "toobig") return t.errBig;
    return t.errSrv;
  }

  async function submit(payload, consent, raw) {
    setAnswers(raw); setError(null); setSentBack(false); setPhase("uploading");
    const exposureJson = JSON.stringify(payload);
    try {
      const res = await postFusion({
        blob: photo.blob, exposureJson, consent,
        onUploaded: () => setPhase((p) => (p === "uploading" ? "analyzing" : p)),
      });
      setResult(res); setPhase("result"); // loader unmounts the instant the response lands
    } catch (e) {
      if (e.kind === "network") {
        try { await enqueue(photo.blob, exposureJson, consent); setPending(await countQueued()); setPhase("queued"); return; }
        catch { /* IndexedDB unavailable: fall through to the normal error */ }
      }
      setError(msg(e)); setPhase("form");
    }
  }

  /** Replays queued screenings (same multipart request). Only runs while the user is idle. */
  async function flush() {
    const p = latest.current;
    const idle = p.phase === "queued" || (p.phase === "capture" && !p.photo);
    if (flushing.current || !idle || !navigator.onLine) return;
    flushing.current = true;
    try {
      for (const item of await listQueued()) {
        try {
          const res = await postFusion({ blob: item.blob, exposureJson: item.exposureJson, consent: item.consent });
          await removeQueued(item.id);
          setPhoto({ blob: item.blob, url: URL.createObjectURL(item.blob) });
          setResult(res); setSentBack(true); setPhase("result");
        } catch (e) {
          if (e.kind === "network" || e.kind === "server") break; // try again later
          await removeQueued(item.id); // 4xx will never succeed; drop it and tell the user
          setError(msg(e)); setPhase("capture");
        }
      }
    } finally {
      flushing.current = false;
      setPending(await countQueued().catch(() => 0));
    }
  }
  latest.current = { phase, photo, flush };

  useEffect(() => {
    const go = () => latest.current.flush();
    window.addEventListener("online", go);
    return () => window.removeEventListener("online", go);
  }, []);
  useEffect(() => { latest.current.flush(); }, [phase]); // app start, and whenever the user returns to an idle screen

  function reset() { setPhoto(null); setAnswers(null); setResult(null); setError(null); setSentBack(false); setPhase("capture"); }

  return (
    <div className="app">
      <div className="shell">
        <header className="topbar">
          <span className="brand">Nirikshan</span>
          <div className="lang" role="group" aria-label="Language">
            <button aria-pressed={lang === "en"} onClick={() => setLang("en")}>EN</button>
            <button aria-pressed={lang === "hi"} onClick={() => setLang("hi")}>हिं</button>
          </div>
        </header>
        <main key={phase === "uploading" || phase === "analyzing" ? "wait" : phase} className="page">
          {phase === "capture" && (
            <>
              {error && <div className="err" role="alert">{error}</div>}
              <PhotoCapture lang={lang} photo={photo} onReady={setPhoto} />
              {photo && <button className="btn btn-primary" onClick={() => setPhase("form")}>{t.cont}</button>}
            </>
          )}
          {phase === "form" && (
            <ExposureIntakeForm lang={lang} initial={answers} error={error} onSubmit={submit} onBack={() => setPhase("capture")} />
          )}
          {(phase === "uploading" || phase === "analyzing") && <LoadingScreen variant={phase} lang={lang} />}
          {phase === "queued" && (
            <section className="card enter">
              <div className="blobs" aria-hidden="true"><i /><i /><i /></div>
              <h1>{t.queuedTitle}</h1>
              <p>{t.queuedBody}</p>
              <p className="small-note">{pending} {t.pending}</p>
              <div className="btn-row">
                <button className="btn btn-primary" onClick={() => latest.current.flush()}>{t.sendNow}</button>
                <button className="btn" onClick={reset}>{t.again}</button>
              </div>
            </section>
          )}
          {phase === "result" && result && (
            <>
              {sentBack && <div className="small-note">{t.sentBack}</div>}
              <section className="card enter">
                <GradCamOverlay photoUrl={photo.url} heatmapBase64={result.gradcam_png_base64} lang={lang} />
              </section>
              <RiskBadge result={result} lang={lang} />
              <div className="btn-row">
                <button className="btn btn-warm" onClick={() => setPhase("dashboard")}>{t.fairness}</button>
                <button className="btn" onClick={reset}>{t.again}</button>
              </div>
            </>
          )}
          {phase === "dashboard" && <FairnessDashboard lang={lang} onBack={() => setPhase(result ? "result" : "capture")} />}
        </main>
      </div>
    </div>
  );
}

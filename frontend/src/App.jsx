import { useEffect, useState } from "react";
import PhotoCapture from "./components/PhotoCapture.jsx";
import ExposureIntakeForm from "./components/ExposureIntakeForm.jsx";
import LoadingScreen from "./components/LoadingScreen.jsx";
import RiskBadge from "./components/RiskBadge.jsx";
import GradCamOverlay from "./components/GradCamOverlay.jsx";
import FairnessDashboard from "./pages/FairnessDashboard.jsx";
import { postFusion } from "./api.js";
import { S } from "./strings.js";

// phase: capture | form | uploading | analyzing | result | dashboard
export default function App() {
  const [lang, setLang] = useState("en");
  const [phase, setPhase] = useState("capture");
  const [photo, setPhoto] = useState(null);
  const [answers, setAnswers] = useState(null); // raw form state, kept so a failed submit doesn't lose answers
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const t = S[lang];

  useEffect(() => () => photo && URL.revokeObjectURL(photo.url), [photo]);

  function msg(e) {
    if (e.kind === "detail") return e.detail;
    if (e.kind === "validation") return t.err422;
    if (e.kind === "network") return t.errNet;
    if (e.kind === "toobig") return t.errBig;
    return t.errSrv;
  }

  async function submit(payload, consent, raw) {
    setAnswers(raw); setError(null); setPhase("uploading");
    try {
      const res = await postFusion({
        blob: photo.blob, exposureJson: JSON.stringify(payload), consent,
        onUploaded: () => setPhase((p) => (p === "uploading" ? "analyzing" : p)),
      });
      setResult(res); setPhase("result"); // loader unmounts the instant the response lands
    } catch (e) { setError(msg(e)); setPhase("form"); }
  }

  function reset() { setPhoto(null); setAnswers(null); setResult(null); setError(null); setPhase("capture"); }

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
              <PhotoCapture lang={lang} photo={photo} onReady={setPhoto} />
              {photo && <button className="btn btn-primary" onClick={() => setPhase("form")}>{t.cont}</button>}
            </>
          )}
          {phase === "form" && (
            <ExposureIntakeForm lang={lang} initial={answers} error={error} onSubmit={submit} onBack={() => setPhase("capture")} />
          )}
          {(phase === "uploading" || phase === "analyzing") && <LoadingScreen variant={phase} lang={lang} />}
          {phase === "result" && result && (
            <>
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

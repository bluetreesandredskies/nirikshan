import { useState } from "react";
import { downscale } from "../api.js";
import { S } from "../strings.js";

/** Picks/captures a photo, downscales it (~1600px JPEG), and hands { blob, url } to onReady. */
export default function PhotoCapture({ lang, photo, onReady }) {
  const t = S[lang];
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  async function onFile(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setError(null); setBusy(true);
    try {
      const blob = await downscale(file);
      onReady({ blob, url: URL.createObjectURL(blob) });
    } catch (err) {
      setError(err.kind === "toobig" ? t.errBig : t.errImg);
    } finally { setBusy(false); }
  }

  return (
    <section className="card enter">
      <h1>{t.capTitle}</h1>
      <p>{t.capHelp}</p>
      {photo && <img className="photo" src={photo.url} alt="" />}
      {error && <div className="err" role="alert">{error}</div>}
      <div className="btn-row">
        <label className="btn btn-warm">
          {t.camera}
          <input type="file" accept="image/*" capture="environment" onChange={onFile} disabled={busy} />
        </label>
        <label className="btn">
          {photo ? t.retake : t.gallery}
          <input type="file" accept="image/*" onChange={onFile} disabled={busy} />
        </label>
      </div>
    </section>
  );
}

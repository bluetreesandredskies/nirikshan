import { useState } from "react";
import LoadingScreen from "./LoadingScreen.jsx";
import { S } from "../strings.js";

/** heatmapBase64 null -> plain photo only. String -> cross-fade plain photo to heatmap. */
export default function GradCamOverlay({ photoUrl, heatmapBase64, lang }) {
  const [ready, setReady] = useState(false);
  const [on, setOn] = useState(true);
  const t = S[lang];
  if (!heatmapBase64) return <img className="photo" src={photoUrl} alt="" />;
  return (
    <div>
      <div className={`stack${ready && on ? " on" : ""}`}>
        <img className="photo" src={photoUrl} alt="" style={{ marginBottom: 0 }} />
        <img className="heat" src={`data:image/png;base64,${heatmapBase64}`} alt="" onLoad={() => setReady(true)} />
      </div>
      {!ready && <LoadingScreen variant="preparing" lang={lang} compact />}
      {ready && <div className="btn-row"><button className="btn" onClick={() => setOn(!on)}>{on ? t.heatOff : t.heatOn}</button></div>}
    </div>
  );
}

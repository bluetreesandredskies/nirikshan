import { S } from "../strings.js";

/** Purely presentational: the parent unmounts it the moment real data arrives (no timers here).
 *  variant: "uploading" | "analyzing" | "preparing" */
export default function LoadingScreen({ variant = "analyzing", lang = "en", compact = false }) {
  return (
    <div className={`loading${compact ? " compact" : ""}`} role="status" aria-live="polite">
      <div>
        <div className="blobs" aria-hidden="true"><i /><i /><i /></div>
        <p>{S[lang].load[variant]}</p>
      </div>
    </div>
  );
}

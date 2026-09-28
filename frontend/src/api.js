const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const MAX_BYTES = 10 * 1024 * 1024;

export class ApiError extends Error {
  // kind: "network" | "detail" | "validation" | "toobig" | "badimage" | "server"
  constructor(kind, detail = "", status = 0) { super(kind); this.kind = kind; this.detail = detail; this.status = status; }
}

/** Downscale to longest side ~1600px, re-encode as JPEG. */
export async function downscale(file, maxSide = 1600) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch { throw new ApiError("badimage"); }
  const k = Math.min(1, maxSide / Math.max(bmp.width, bmp.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k);
  c.getContext("2d").drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.85));
  if (!blob) throw new ApiError("badimage");
  if (blob.size > MAX_BYTES) throw new ApiError("toobig");
  return blob;
}

/** multipart POST via XHR so we know when the upload really finished (uploading -> analyzing). */
export function postFusion({ blob, exposureJson, consent, onUploaded }) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("image", blob, "photo.jpg");
    fd.append("exposure", exposureJson);
    fd.append("consent_store_anonymized", consent ? "true" : "false");
    const x = new XMLHttpRequest();
    x.open("POST", `${API}/risk-fusion`);
    x.upload.onload = () => onUploaded?.();
    x.onerror = x.ontimeout = () => reject(new ApiError("network"));
    x.onload = () => {
      let body = null;
      try { body = JSON.parse(x.responseText); } catch { /* keep null */ }
      if (x.status >= 200 && x.status < 300 && body) return resolve(body);
      const d = body?.detail;
      if (Array.isArray(d)) return reject(new ApiError("validation", "", x.status));
      if (typeof d === "string") return reject(new ApiError("detail", d, x.status));
      reject(new ApiError("server", "", x.status));
    };
    x.send(fd);
  });
}

export async function getFairness() {
  let r;
  try { r = await fetch(`${API}/fairness-report`); } catch { throw new ApiError("network"); }
  if (!r.ok) throw new ApiError("server", "", r.status);
  return r.json();
}

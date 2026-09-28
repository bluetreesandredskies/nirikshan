// IndexedDB queue. Stores ONLY: image Blob, exposure JSON string, consent flag (plus the auto key).
const DB = "nirikshan-queue", STORE = "screenings";

function open() {
  return new Promise((res, rej) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function tx(mode, fn) {
  const db = await open();
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const out = fn(t.objectStore(STORE));
    t.oncomplete = () => { db.close(); res(out.result); };
    t.onerror = t.onabort = () => { db.close(); rej(t.error); };
  });
}

export const enqueue = (blob, exposureJson, consent) => tx("readwrite", (s) => s.add({ blob, exposureJson, consent: !!consent }));
export const listQueued = () => tx("readonly", (s) => s.getAll());
export const countQueued = () => tx("readonly", (s) => s.count());
export const removeQueued = (id) => tx("readwrite", (s) => s.delete(id));

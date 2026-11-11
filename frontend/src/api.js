// Toutes les requetes passent par /api (proxy Vite en dev, nginx en Docker).
const API = import.meta.env.VITE_API_URL ?? "/api";

async function getJson(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(`${path} : erreur ${response.status}`);
  return response.json();
}

export const fetchHealth = () => getJson("/health");
export const fetchMetadata = () => getJson("/model/metadata");

// Envoie une image (Blob ou File) et mesure le temps d'aller-retour complet.
export async function predict(imageBlob, filename = "frame.jpg") {
  const form = new FormData();
  form.append("file", imageBlob, filename);
  const start = performance.now();
  const response = await fetch(`${API}/predict`, { method: "POST", body: form });
  const roundTripMs = performance.now() - start;
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Erreur ${response.status}`);
  }
  return { ...(await response.json()), roundTripMs };
}

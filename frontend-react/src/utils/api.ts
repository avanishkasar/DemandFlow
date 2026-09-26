const BASE_URL = import.meta.env.VITE_API_URL || "/api";

interface ApiOptions {
  method?: string;
  body?: unknown;
  headers?: Record<string, string>;
}

export async function apiFetch<T>(path: string, opts: ApiOptions = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: opts.method || "GET",
    headers: {
      "Content-Type": "application/json",
      ...opts.headers,
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(err.message || `API error: ${res.status}`);
  }

  return res.json();
}

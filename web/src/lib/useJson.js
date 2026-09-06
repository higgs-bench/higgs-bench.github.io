import { useEffect, useState } from "react";

const BASE = import.meta.env.BASE_URL;

/** Load a JSON file from public/data. Returns { data, error, loading }. */
export function useJson(name) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let alive = true;
    fetch(`${BASE}data/${name}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => alive && setState({ data: d, error: null, loading: false }))
      .catch((e) => alive && setState({ data: null, error: e.message, loading: false }));
    return () => { alive = false; };
  }, [name]);

  return state;
}
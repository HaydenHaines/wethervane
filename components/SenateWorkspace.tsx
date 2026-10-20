"use client";

import { useEffect, useRef, useState } from "react";
import SenateExplorer from "./SenateExplorer";
import ComponentStudy from "./ComponentStudy";

/** A second data contract within the same page, preserving the full-chamber view. */
export default function SenateWorkspace() {
  const [study, setStudy] = useState<string | null | undefined>(undefined);
  const previous = useRef("/");
  useEffect(() => {
    const restore = () => {
      const query = new URLSearchParams(location.search);
      if (query.get("chamber") === "house") return;
      setStudy(query.get("study"));
    };
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  function openStudy() {
    previous.current = location.pathname + location.search;
    history.pushState(null, "", "/?study=ordinary-2018-fp");
    setStudy("ordinary-2018-fp");
  }
  function closeStudy() {
    history.pushState(null, "", previous.current);
    setStudy(null);
  }
  if (study === undefined) return <div className="load-page" role="status">Preparing your perspective…</div>;
  return study !== null ? <ComponentStudy initialId={study} onReturn={closeStudy} /> : <SenateExplorer onStudy={openStudy} />;
}

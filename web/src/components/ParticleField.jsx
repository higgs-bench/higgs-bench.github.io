import { useEffect, useRef } from "react";

const COUNT = 110;
const SIGNAL_RATE = 1 / 50; // matches Version C prevalence

export default function ParticleField() {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    let raf = 0;
    let w = 0;
    let h = 0;
    let particles = [];

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const spawn = () => {
      particles = Array.from({ length: COUNT }, () => {
        const signal = Math.random() < SIGNAL_RATE;
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          vx: (Math.random() - 0.5) * 0.16,
          vy: -0.06 - Math.random() * 0.18,
          r: signal ? 1.9 + Math.random() : 0.7 + Math.random() * 0.8,
          a: signal ? 0.75 : 0.1 + Math.random() * 0.18,
          signal,
          phase: Math.random() * Math.PI * 2,
        };
      });
    };

    const draw = (time) => {
      ctx.clearRect(0, 0, w, h);
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.y < -10) { p.y = h + 10; p.x = Math.random() * w; }
        if (p.x < -10) p.x = w + 10;
        if (p.x > w + 10) p.x = -10;

        const twinkle = p.signal
          ? 0.6 + 0.4 * Math.sin(time / 700 + p.phase)
          : 1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        if (p.signal) {
          ctx.shadowBlur = 12;
          ctx.shadowColor = "rgba(255,45,120,0.9)";
          ctx.fillStyle = `rgba(255,45,120,${p.a * twinkle})`;
        } else {
          ctx.shadowBlur = 0;
          ctx.fillStyle = `rgba(246,242,245,${p.a})`;
        }
        ctx.fill();
      }
      ctx.shadowBlur = 0;
      raf = requestAnimationFrame(draw);
    };

    const onVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(raf);
      } else {
        raf = requestAnimationFrame(draw);
      }
    };

    resize();
    spawn();
    raf = requestAnimationFrame(draw);

    const onResize = () => { resize(); spawn(); };
    window.addEventListener("resize", onResize);
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return <canvas ref={canvasRef} className="particles" aria-hidden="true" />;
}
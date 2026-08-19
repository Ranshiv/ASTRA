import { useEffect, useRef, useState } from "react";
import { engine } from "@/lib/engine";

export function FitsViewer({ path }: { path: string }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    void engine.fitsImage(path).then((payload) => {
      if (cancelled || !canvas.current) return;
      const [height, width] = payload.shape;
      canvas.current.width = width; canvas.current.height = height;
      const context = canvas.current.getContext("2d");
      if (!context) return;
      const image = context.createImageData(width, height);
      payload.pixels.forEach((pixel, index) => {
        image.data[index * 4] = pixel; image.data[index * 4 + 1] = pixel;
        image.data[index * 4 + 2] = pixel; image.data[index * 4 + 3] = 255;
      });
      context.putImageData(image, 0, 0);
    }).catch((err) => { if (!cancelled) setError(String(err)); });
    return () => { cancelled = true; };
  }, [path]);
  return <div>
    <canvas ref={canvas} className="max-h-80 max-w-full rounded border border-[var(--color-edge)]" />
    {error && <p className="text-xs text-[var(--color-bad)]">{error}</p>}
  </div>;
}

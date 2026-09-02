import { useEffect, useRef, useState } from "react";
import { engine } from "@/lib/engine";
import { Note } from "@/components/ui";

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
      const data = image.data;
      const pixels = payload.pixels;
      // A plain indexed loop over the RGBA buffer, not `forEach`: for a
      // full-frame image (millions of pixels) the per-call overhead of a
      // callback invoked once per source pixel is the dominant cost here.
      for (let index = 0, offset = 0; index < pixels.length; index += 1, offset += 4) {
        const value = pixels[index];
        data[offset] = value;
        data[offset + 1] = value;
        data[offset + 2] = value;
        data[offset + 3] = 255;
      }
      context.putImageData(image, 0, 0);
    }).catch((err) => { if (!cancelled) setError(String(err)); });
    return () => { cancelled = true; };
  }, [path]);
  return <div>
    <canvas ref={canvas} role="img" aria-label={`FITS image for ${path}`} className="max-h-80 max-w-full rounded border border-[var(--color-edge)]" />
    {error && <Note tone="bad">{error}</Note>}
  </div>;
}

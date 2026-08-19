import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const dist = resolve(process.argv[2] ?? "dist");
const indexPath = resolve(dist, "index.html");

if (!existsSync(indexPath)) {
  throw new Error(`Desktop entry check failed: ${indexPath} does not exist. Run npm run build first.`);
}

const html = readFileSync(indexPath, "utf8");
if (!html.includes('id="root"') || !html.includes("Starting ASTRA")) {
  throw new Error("Desktop entry check failed: index.html has no visible ASTRA startup fallback.");
}

const assetPaths = [...html.matchAll(/(?:src|href)="\/?(assets\/[^"]+)"/g)].map((match) => match[1]);
if (assetPaths.length === 0) {
  throw new Error("Desktop entry check failed: no built JavaScript or stylesheet assets were found.");
}

for (const assetPath of assetPaths) {
  if (!existsSync(resolve(dist, assetPath))) {
    throw new Error(`Desktop entry check failed: missing built asset ${assetPath}.`);
  }
}

console.log(`Desktop entry check passed (${assetPaths.length} bundled assets, startup fallback present).`);

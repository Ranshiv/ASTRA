declare module "aladin-lite" {
  const A: {
    init: Promise<void>;
    aladin: (selector: string, options: Record<string, unknown>) => {
      setFov: (fov: number) => void;
    };
  };
  export default A;
}

import { Link } from "react-router-dom";

interface ModelExportProps {
  styleId: string;
}

// Rendered when the training job reaches the `done` state. Congratulates
// the user and links them back to the inference library.
export function ModelExport({ styleId }: ModelExportProps) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-lg border border-emerald-700/50 bg-emerald-950/30 p-4 text-sm">
      <div className="font-medium text-emerald-200">
        Training complete
      </div>
      <div className="text-zinc-200">
        Saved as{" "}
        <code className="rounded bg-zinc-900 px-1.5 py-0.5 font-mono text-emerald-300">
          {styleId}
        </code>
      </div>
      <Link
        to="/"
        className="rounded-md bg-emerald-500 px-3 py-1.5 text-xs font-medium text-zinc-950 hover:bg-emerald-400"
      >
        Back to Library
      </Link>
    </div>
  );
}

export default ModelExport;

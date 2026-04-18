import type { TrainingConfig } from "../types";

interface HyperparamFormProps {
  value: TrainingConfig;
  onChange: (next: TrainingConfig) => void;
  disabled?: boolean;
}

// Production-quality defaults matching spec §3.3.
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  style_name: "my_style",
  style_weight: 1e10,
  content_weight: 1e5,
  tv_weight: 1e-6,
  learning_rate: 1e-3,
  epochs: 2,
  batch_size: 4,
  image_size: 256,
};

interface NumericFieldProps {
  label: string;
  hint?: string;
  value: number;
  onChange: (v: number) => void;
  step?: string;
  min?: number;
  disabled?: boolean;
}

function NumericField({
  label,
  hint,
  value,
  onChange,
  step = "any",
  min,
  disabled,
}: NumericFieldProps) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-zinc-300">{label}</span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        disabled={disabled}
        onChange={(e) => {
          const next = e.target.value;
          // Allow scientific notation: `parseFloat` handles "1e10" etc.
          const n = parseFloat(next);
          if (!Number.isNaN(n)) onChange(n);
        }}
        className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 font-mono text-xs text-zinc-100 focus:border-violet-500 focus:outline-none disabled:opacity-50"
      />
      {hint && <span className="text-[10px] text-zinc-500">{hint}</span>}
    </label>
  );
}

export function HyperparamForm({
  value,
  onChange,
  disabled,
}: HyperparamFormProps) {
  const set = <K extends keyof TrainingConfig>(
    key: K,
    v: TrainingConfig[K]
  ) => onChange({ ...value, [key]: v });

  return (
    <div className="grid grid-cols-1 gap-3 rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 sm:grid-cols-2 lg:grid-cols-4">
      <label className="col-span-1 flex flex-col gap-1 text-sm sm:col-span-2">
        <span className="text-zinc-300">Style name</span>
        <input
          type="text"
          value={value.style_name}
          disabled={disabled}
          onChange={(e) => set("style_name", e.target.value)}
          placeholder="my_style"
          className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 focus:border-violet-500 focus:outline-none disabled:opacity-50"
        />
        <span className="text-[10px] text-zinc-500">
          Used as the on-disk model id (lowercase, underscores).
        </span>
      </label>

      <NumericField
        label="Style weight"
        hint="Perceptual style term (gram matrices)"
        value={value.style_weight}
        onChange={(v) => set("style_weight", v)}
        disabled={disabled}
      />
      <NumericField
        label="Content weight"
        hint="Perceptual content term (relu3_3)"
        value={value.content_weight}
        onChange={(v) => set("content_weight", v)}
        disabled={disabled}
      />
      <NumericField
        label="TV weight"
        hint="Total variation smoothing"
        value={value.tv_weight}
        onChange={(v) => set("tv_weight", v)}
        disabled={disabled}
      />
      <NumericField
        label="Learning rate"
        hint="Adam LR"
        value={value.learning_rate}
        onChange={(v) => set("learning_rate", v)}
        disabled={disabled}
      />
      <NumericField
        label="Epochs"
        value={value.epochs}
        min={1}
        step="1"
        onChange={(v) => set("epochs", Math.max(1, Math.floor(v)))}
        disabled={disabled}
      />
      <NumericField
        label="Batch size"
        value={value.batch_size}
        min={1}
        step="1"
        onChange={(v) => set("batch_size", Math.max(1, Math.floor(v)))}
        disabled={disabled}
      />
      <NumericField
        label="Image size"
        hint="Training crop (px)"
        value={value.image_size}
        min={64}
        step="1"
        onChange={(v) => set("image_size", Math.max(64, Math.floor(v)))}
        disabled={disabled}
      />
    </div>
  );
}

export default HyperparamForm;

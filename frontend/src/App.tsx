import { NavLink, Route, Routes } from "react-router-dom";
import InferencePage from "./pages/InferencePage";
import TrainingPage from "./pages/TrainingPage";

function NavItem({
  to,
  label,
  end,
}: {
  to: string;
  label: string;
  end?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        [
          "rounded-md px-3 py-1.5 text-sm font-medium transition",
          isActive
            ? "bg-violet-600 text-white"
            : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100",
        ].join(" ")
      }
    >
      {label}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="flex min-h-full flex-col bg-zinc-950 text-zinc-100">
      <header className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur">
        <nav className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-violet-500 shadow-[0_0_8px_rgba(139,92,246,0.9)]" />
            <span className="text-sm font-semibold tracking-wide text-zinc-100">
              LiveArt
            </span>
            <span className="text-[10px] uppercase tracking-widest text-zinc-500">
              Neural Style Transfer
            </span>
          </div>
          <div className="flex items-center gap-1">
            <NavItem to="/" label="Inference" end />
            <NavItem to="/training" label="Training" />
          </div>
        </nav>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<InferencePage />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </main>
    </div>
  );
}

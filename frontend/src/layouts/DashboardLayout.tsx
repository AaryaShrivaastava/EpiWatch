import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Activity, 
  AlertOctagon, 
  TrendingUp, 
  AlertTriangle, 
  CheckSquare, 
  Stethoscope, 
  Map, 
  Binary, 
  Brain,
  ShieldCheck,
  Radio
} from 'lucide-react';

export default function DashboardLayout({ children }: { children: ReactNode }) {
  const navItems = [
    { label: 'Executive Overview', path: '/', icon: <LayoutDashboard size={18} /> },
    { label: 'Disease Analytics', path: '/analytics', icon: <Activity size={18} /> },
    { label: 'Outbreak Detection', path: '/outbreaks', icon: <AlertOctagon size={18} /> },
    { label: 'Trajectory Forecasts', path: '/forecasts', icon: <TrendingUp size={18} /> },
    { label: 'Intelligent Alerts', path: '/alerts', icon: <AlertTriangle size={18} /> },
    { label: 'Action Directives', path: '/recommendations', icon: <CheckSquare size={18} /> },
    { label: 'Healthcare & Stress', path: '/healthcare', icon: <Stethoscope size={18} /> },
    { label: 'Spatial Risk Map', path: '/spatial', icon: <Map size={18} /> },
    { label: 'SIR/SIRD Simulation', path: '/simulation', icon: <Binary size={18} /> },
    { label: 'AI Surveillance', path: '/agent', icon: <Radio size={18} className="text-teal-400 animate-pulse" /> },
  ];

  return (
    <div className="flex h-screen w-full bg-slate-900 text-slate-100 overflow-hidden font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-950 border-r border-slate-800 flex flex-col shrink-0">
        <div className="p-4 border-b border-slate-800 flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-teal-500/20 border border-teal-500/40 flex items-center justify-center text-teal-400 font-bold">
            EW
          </div>
          <div>
            <div className="font-bold text-lg leading-tight tracking-wide text-white">EpiWatch</div>
            <div className="text-[10px] text-teal-400 font-mono uppercase tracking-wider">Epidemiological Intelligence</div>
          </div>
        </div>

        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
            Surveillance & Analytics
          </div>
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-teal-500/15 text-teal-300 border border-teal-500/30 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                }`
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="p-3 border-t border-slate-800/80 bg-slate-950/60 text-xs text-slate-400">
          <div className="flex items-center gap-2 font-mono text-[11px] text-emerald-400">
            <Radio size={12} className="animate-pulse" /> Surveillance Core Online
          </div>
          <div className="text-[10px] text-slate-400 mt-1">MP Central India Pilot (5 Regions)</div>
        </div>
      </aside>

      {/* Main Workstation Container */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-900 overflow-hidden">
        {/* Top Header & Demo Banner */}
        <header className="bg-slate-950/90 border-b border-slate-800 px-6 py-2.5 shrink-0 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-amber-500/15 border border-amber-500/30 text-amber-300 text-xs font-semibold uppercase tracking-wider">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
              Demo Mode Active
            </span>
            <p className="text-xs text-slate-300 hidden md:block">
              <strong>DEMO MODE</strong> — Synthetic epidemiological data. Results are illustrative and are not live public-health surveillance.
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs">
            <div className="hidden sm:flex items-center gap-1.5 text-slate-400">
              <ShieldCheck size={14} className="text-teal-400" />
              <span>Census 2011 Verified Anchors</span>
            </div>
            <div className="text-[11px] text-slate-400 font-mono bg-slate-900 px-2 py-1 rounded border border-slate-800">
              UTC {new Date().toISOString().substring(0, 10)}
            </div>
          </div>
        </header>

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-slate-900 text-slate-100">
          <div className="w-full max-w-[1600px] mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}

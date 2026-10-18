import { useEffect, useState } from 'react';
import { 
  fetchDiseases, 
  fetchSimulationParameters, 
  runSirSimulation 
} from '../services/apiClient';
import { 
  Binary, 
  Play, 
  Sliders, 
  ShieldAlert, 
  Users, 
  HeartCrack, 
  CheckCircle, 
  Info,
  Sparkles
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

export default function Simulation() {
  const [diseases, setDiseases] = useState<string[]>([]);
  const [selectedDisease, setSelectedDisease] = useState<string>('COVID-19');

  // Compartmental Parameters
  const [population, setPopulation] = useState<number>(2368145);
  const [initialInfected, setInitialInfected] = useState<number>(250);
  const [beta, setBeta] = useState<number>(0.28);
  const [gamma, setGamma] = useState<number>(0.10);
  const [mu, setMu] = useState<number>(0.003);
  const [days, setDays] = useState<number>(90);
  const [intervention, setIntervention] = useState<number>(0);

  const [simulationResult, setSimulationResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);

  useEffect(() => {
    fetchDiseases().then((diss) => {
      setDiseases(diss);
      if (diss.length > 0) setSelectedDisease(diss[0]);
    }).catch(console.error);
  }, []);

  // Update parameters when disease changes
  useEffect(() => {
    fetchSimulationParameters(selectedDisease).then((p) => {
      if (p) {
        setBeta(p.beta || 0.28);
        setGamma(p.gamma || 0.10);
        setMu(p.mu || 0.003);
      }
      handleRun();
    }).catch(console.error);
  }, [selectedDisease]);

  const handleRun = async () => {
    setRunning(true);
    try {
      const res = await runSirSimulation({
        disease: selectedDisease,
        population,
        initial_infected: initialInfected,
        beta,
        gamma,
        mu,
        days,
        intervention_reduction_percent: intervention
      });
      setSimulationResult(res);
    } catch (err) {
      console.error('Simulation error:', err);
    } finally {
      setRunning(false);
      setLoading(false);
    }
  };

  const trajectory = simulationResult?.trajectory || [];

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Binary className="text-cyan-400" size={22} />
            Mathematical Epidemic Modeling (SIR / SIRD)
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Deterministic ordinary differential equations with public health intervention scenario analysis
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span>Target Pathogen:</span>
          <select
            value={selectedDisease}
            onChange={(e) => setSelectedDisease(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
          >
            {diseases.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
      </div>

      {/* Main Simulation Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Parameter Controls */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-4">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Sliders size={16} className="text-teal-400" />
            Epidemiological Parameters
          </h2>

          <div className="space-y-3.5 text-xs">
            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Transmission Rate (β):</span>
                <span className="font-mono text-cyan-400 font-bold">{beta.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.80"
                step="0.01"
                value={beta}
                onChange={(e) => setBeta(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Recovery Rate (γ):</span>
                <span className="font-mono text-emerald-400 font-bold">{gamma.toFixed(3)}</span>
              </div>
              <input
                type="range"
                min="0.02"
                max="0.35"
                step="0.01"
                value={gamma}
                onChange={(e) => setGamma(parseFloat(e.target.value))}
                className="w-full accent-emerald-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-[10px] text-slate-400 mt-0.5">Mean infectious period: {(1 / gamma).toFixed(1)} days</div>
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Mortality Rate (μ):</span>
                <span className="font-mono text-rose-400 font-bold">{mu.toFixed(4)}</span>
              </div>
              <input
                type="range"
                min="0.0005"
                max="0.02"
                step="0.0005"
                value={mu}
                onChange={(e) => setMu(parseFloat(e.target.value))}
                className="w-full accent-rose-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-slate-300 mb-1">
                <span>Intervention Transmission Reduction:</span>
                <span className="font-mono text-amber-400 font-bold">{intervention}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="80"
                step="5"
                value={intervention}
                onChange={(e) => setIntervention(parseInt(e.target.value))}
                className="w-full accent-amber-400 bg-slate-800 h-1.5 rounded-lg appearance-none cursor-pointer"
              />
              <div className="text-[10px] text-slate-400 mt-0.5">Lockdown, mask mandates, contact tracing effect</div>
            </div>

            <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800">
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-mono">Initial Infected (I₀)</label>
                <input
                  type="number"
                  value={initialInfected}
                  onChange={(e) => setInitialInfected(parseInt(e.target.value) || 10)}
                  className="w-full mt-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
              <div>
                <label className="text-[10px] text-slate-400 uppercase font-mono">Simulation Days</label>
                <input
                  type="number"
                  value={days}
                  onChange={(e) => setDays(parseInt(e.target.value) || 60)}
                  className="w-full mt-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white font-mono"
                />
              </div>
            </div>

            <button
              onClick={handleRun}
              disabled={running}
              className="w-full mt-2 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 text-cyan-300 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-all shadow-sm"
            >
              <Play size={14} className={running ? 'animate-spin' : ''} />
              {running ? 'Integrating Differential Equations...' : 'Execute SIRD Simulation'}
            </button>
          </div>
        </div>

        {/* Right 2 Columns: Mathematical Simulation Output */}
        <div className="lg:col-span-2 space-y-4">
          {/* Key Metric Indicators */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
              <span className="text-[11px] text-slate-400">Basic Reproduction (R₀)</span>
              <div className="text-xl font-bold text-cyan-400 font-mono mt-0.5">
                {simulationResult?.r0 || '--'}
              </div>
              <div className="text-[10px] text-slate-400">β / γ baseline</div>
            </div>

            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
              <span className="text-[11px] text-slate-400">Effective R with Policy</span>
              <div className={`text-xl font-bold font-mono mt-0.5 ${simulationResult?.effective_r0_with_intervention < 1.0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {simulationResult?.effective_r0_with_intervention || '--'}
              </div>
              <div className="text-[10px] text-slate-400">{intervention}% suppression applied</div>
            </div>

            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
              <span className="text-[11px] text-slate-400">Projected Peak Infected</span>
              <div className="text-xl font-bold text-amber-400 font-mono mt-0.5">
                {simulationResult?.peak_infected?.toLocaleString() || '--'}
              </div>
              <div className="text-[10px] text-slate-400">Day {simulationResult?.peak_day || '--'} peak</div>
            </div>

            <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800">
              <span className="text-[11px] text-slate-400">Projected Deaths</span>
              <div className="text-xl font-bold text-rose-400 font-mono mt-0.5">
                {simulationResult?.total_deceased?.toLocaleString() || '--'}
              </div>
              <div className="text-[10px] text-slate-400">Across {days} days</div>
            </div>
          </div>

          {/* Compartment Trajectory Curve */}
          <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                  SIRD Compartment Trajectories (S, I, R, D)
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">Daily state counts from numerical integration</p>
              </div>
              <span className="text-xs font-mono text-cyan-400 bg-cyan-950/40 px-2 py-0.5 rounded border border-cyan-800/40">
                Runge-Kutta ODE
              </span>
            </div>

            <div className="h-72 w-full">
              {loading ? (
                <div className="h-full flex items-center justify-center text-slate-400 text-xs">Simulating model dynamics...</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trajectory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="day" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'Day', position: 'insideBottomRight', offset: -4, fill: '#64748b', fontSize: 10 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                    <Line type="monotone" dataKey="infected" stroke="#f59e0b" strokeWidth={2.5} name="Infected (I)" dot={false} />
                    <Line type="monotone" dataKey="recovered" stroke="#10b981" strokeWidth={2} name="Recovered (R)" dot={false} />
                    <Line type="monotone" dataKey="deceased" stroke="#f43f5e" strokeWidth={1.8} name="Deceased (D)" dot={false} />
                    <Line type="monotone" dataKey="susceptible" stroke="#64748b" strokeWidth={1.5} strokeDasharray="3 3" name="Susceptible (S)" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="mt-3 p-2.5 rounded bg-slate-900 border border-slate-800 text-[11px] text-slate-400 flex items-start gap-2">
              <Info size={14} className="shrink-0 text-cyan-400 mt-0.5" />
              <span>
                <strong>Model Note:</strong> This compartmental model simulates theoretical population dynamics for policy comparison and is explicitly distinct from empirical statistical machine learning forecasts.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

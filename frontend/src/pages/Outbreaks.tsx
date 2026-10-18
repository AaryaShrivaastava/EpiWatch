import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchDiseases, 
  fetchOutbreaksHistory, 
  fetchOutbreaksSummary, 
  triggerOutbreakDetection 
} from '../services/apiClient';
import { 
  AlertOctagon, 
  AlertTriangle, 
  Play, 
  Filter, 
  ShieldAlert, 
  Activity, 
  CheckCircle, 
  Info,
  Clock,
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

export default function Outbreaks() {
  const [regions, setRegions] = useState<string[]>([]);
  const [diseases, setDiseases] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('Bhopal');
  const [selectedDisease, setSelectedDisease] = useState<string>('Dengue');
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');

  const [outbreaks, setOutbreaks] = useState<any[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningDetection, setRunningDetection] = useState<boolean>(false);
  const [detectionFeedback, setDetectionFeedback] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([fetchRegions(), fetchDiseases()]).then(([regs, diss]) => {
      setRegions(regs);
      setDiseases(diss);
      if (regs.length > 0) setSelectedRegion(regs[0]);
      if (diss.length > 0) setSelectedDisease('Dengue');
    });
  }, []);

  const loadOutbreaks = async () => {
    setLoading(true);
    try {
      const [histData, sumData] = await Promise.all([
        fetchOutbreaksHistory(selectedRegion, selectedDisease, 100),
        fetchOutbreaksSummary(selectedRegion)
      ]);
      setOutbreaks(histData || []);
      setSummary(sumData || null);
    } catch (err) {
      console.error('Error fetching outbreaks:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedRegion) {
      loadOutbreaks();
    }
  }, [selectedRegion, selectedDisease]);

  const handleRunDetection = async () => {
    setRunningDetection(true);
    setDetectionFeedback(null);
    try {
      const res = await triggerOutbreakDetection(selectedRegion, selectedDisease);
      setDetectionFeedback(`Detection completed: ${res.active_anomalies_detected} signals identified in observation window.`);
      await loadOutbreaks();
    } catch (err) {
      setDetectionFeedback('Error executing detection pipeline.');
    } finally {
      setRunningDetection(false);
    }
  };

  // Filtered rows
  const filteredOutbreaks = outbreaks.filter((row) => {
    if (severityFilter === 'ALL') return true;
    return row.severity?.toUpperCase() === severityFilter;
  });

  const getSeverityBadge = (sev: string) => {
    switch (sev?.toUpperCase()) {
      case 'CRITICAL':
        return 'bg-red-500/20 text-red-400 border border-red-500/40';
      case 'HIGH':
        return 'bg-amber-500/20 text-amber-400 border border-amber-500/40';
      case 'MODERATE':
        return 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/40';
      default:
        return 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Filter and Action Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <AlertOctagon className="text-red-400" size={22} />
            Early Outbreak Detection Engine
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Statistical 28-day baseline deviations, Isolation Forest, and supervised classification
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span>District:</span>
            <select
              value={selectedRegion}
              onChange={(e) => setSelectedRegion(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              {regions.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span>Disease:</span>
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              {diseases.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span>Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MODERATE">Moderate</option>
            </select>
          </div>

          <button
            onClick={handleRunDetection}
            disabled={runningDetection}
            className="flex items-center gap-2 px-3.5 py-1.5 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 rounded-lg text-xs font-semibold transition-all shadow-sm"
          >
            <Play size={13} className={runningDetection ? 'animate-spin' : ''} />
            {runningDetection ? 'Running...' : 'Run Outbreak Detection'}
          </button>
        </div>
      </div>

      {detectionFeedback && (
        <div className="p-3 rounded-lg bg-teal-500/15 border border-teal-500/30 text-teal-300 text-xs flex items-center gap-2">
          <Sparkles size={16} />
          {detectionFeedback}
        </div>
      )}

      {/* KPI Detection Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Total Anomaly Episodes</div>
          <div className="text-2xl font-bold text-white mt-1">
            {summary ? summary.total_outbreak_events : outbreaks.length}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Crossing threshold</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Critical Surges (&gt;3σ)</div>
          <div className="text-2xl font-bold text-red-400 mt-1">
            {summary ? summary.critical_events : outbreaks.filter(o => o.severity === 'CRITICAL').length}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Rapid doubling phase</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">High Severity Events (&gt;2σ)</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">
            {summary ? summary.high_events : outbreaks.filter(o => o.severity === 'HIGH').length}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Elevated surveillance</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Algorithms Ensembled</div>
          <div className="text-sm font-bold text-teal-400 mt-1 font-mono">
            Z-Score + IF + GB
          </div>
          <div className="text-[11px] text-slate-400 mt-1">3-Tier Consensus</div>
        </div>
      </div>

      {/* Anomaly Score Timeline Chart */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Activity size={16} className="text-red-400" />
              Observed Cases vs 2-Sigma Baseline Upper Threshold
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Spikes above the amber line represent statistically verified outbreak signals</p>
          </div>
          <span className="text-xs text-slate-400 font-mono">28-Day Rolling Causal Window</span>
        </div>

        <div className="h-64 w-full">
          {outbreaks.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-400 text-xs">No outbreak telemetry available</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={outbreaks.slice(-45)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '6px' }} />
                <Line type="monotone" dataKey="new_cases" stroke="#38bdf8" strokeWidth={2} name="Daily Cases" />
                <Line type="monotone" dataKey="upper_threshold" stroke="#f59e0b" strokeWidth={2} strokeDasharray="4 4" dot={false} name="Upper Threshold (μ + 2σ)" />
                <Line type="monotone" dataKey="baseline_mean" stroke="#64748b" strokeWidth={1.5} dot={false} name="Moving Baseline (μ)" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Outbreak Events Table */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Detected Outbreak Episodes ({selectedRegion})
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Showing {filteredOutbreaks.length} verified surveillance detections</p>
          </div>
          <div className="text-xs text-slate-400 font-mono">
            {filteredOutbreaks.length} Records
          </div>
        </div>

        {loading ? (
          <div className="text-center py-12 text-xs text-slate-400">Loading outbreak detections...</div>
        ) : filteredOutbreaks.length === 0 ? (
          <div className="p-8 text-center bg-slate-900/60 rounded-lg border border-slate-800 text-slate-400 text-xs">
            No outbreak events matching the current severity filter ({severityFilter}) for {selectedRegion} ({selectedDisease}).
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-xs whitespace-nowrap">
              <thead className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">Date</th>
                  <th className="py-2.5 px-3">Pathogen</th>
                  <th className="py-2.5 px-3">Cases</th>
                  <th className="py-2.5 px-3">Upper Bound</th>
                  <th className="py-2.5 px-3">Severity</th>
                  <th className="py-2.5 px-3">Score (0-100)</th>
                  <th className="py-2.5 px-3">Contributing Evidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-medium text-slate-300">
                {filteredOutbreaks.slice(-25).reverse().map((row, idx) => {
                  let signals = [];
                  try {
                    signals = typeof row.contributing_signals === 'string' ? JSON.parse(row.contributing_signals) : row.contributing_signals;
                  } catch (e) {
                    signals = [row.contributing_signals || 'Elevated case count'];
                  }

                  return (
                    <tr key={idx} className="hover:bg-slate-900/80 transition-colors">
                      <td className="py-2.5 px-3 font-mono text-white flex items-center gap-1.5">
                        <Clock size={12} className="text-slate-400" />
                        {row.date}
                      </td>
                      <td className="py-2.5 px-3 font-semibold text-slate-200">{row.disease}</td>
                      <td className="py-2.5 px-3 font-mono font-bold text-white">{row.new_cases}</td>
                      <td className="py-2.5 px-3 font-mono text-amber-400">{row.upper_threshold || '--'}</td>
                      <td className="py-2.5 px-3">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${getSeverityBadge(row.severity)}`}>
                          {row.severity}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-cyan-400 font-bold">
                        {row.overall_outbreak_score ? Number(row.overall_outbreak_score).toFixed(1) : '--'}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 max-w-xs truncate text-[11px]">
                        {Array.isArray(signals) ? signals.join('; ') : signals}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

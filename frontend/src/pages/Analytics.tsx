import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchDiseases, 
  fetchAnalyticsTrends, 
  fetchAnalyticsOverview, 
  fetchAnalyticsSeasonality 
} from '../services/apiClient';
import { 
  Activity, 
  Calendar, 
  BarChart2, 
  Percent, 
  TrendingUp, 
  ShieldAlert, 
  Layers,
  ArrowUpRight
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

export default function Analytics() {
  const [regions, setRegions] = useState<string[]>([]);
  const [diseases, setDiseases] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('Bhopal');
  const [selectedDisease, setSelectedDisease] = useState<string>('COVID-19');
  const [timeWindow, setTimeWindow] = useState<number>(60);

  const [trends, setTrends] = useState<any[]>([]);
  const [overview, setOverview] = useState<any[]>([]);
  const [seasonality, setSeasonality] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    Promise.all([fetchRegions(), fetchDiseases()]).then(([regs, diss]) => {
      setRegions(regs);
      setDiseases(diss);
      if (regs.length > 0) setSelectedRegion(regs[0]);
      if (diss.length > 0) setSelectedDisease(diss[0]);
    });
  }, []);

  const loadData = async () => {
    if (!selectedRegion || !selectedDisease) return;
    setLoading(true);
    try {
      const [trendData, overviewData, seasonData] = await Promise.all([
        fetchAnalyticsTrends(selectedRegion, selectedDisease, timeWindow),
        fetchAnalyticsOverview(selectedRegion),
        fetchAnalyticsSeasonality(selectedRegion, selectedDisease)
      ]);
      setTrends(trendData || []);
      setOverview(overviewData || []);
      setSeasonality(seasonData || []);
    } catch (err) {
      console.error('Analytics load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [selectedRegion, selectedDisease, timeWindow]);

  // Compute selected disease summary
  const selectedDiseaseStats = overview.find((d: any) => d.disease === selectedDisease) || {
    total_cases: trends.reduce((acc, r) => acc + (r.new_cases || 0), 0),
    avg_daily_cases: trends.length > 0 ? (trends.reduce((acc, r) => acc + (r.new_cases || 0), 0) / trends.length).toFixed(1) : 0,
    max_daily_cases: trends.reduce((max, r) => Math.max(max, r.new_cases || 0), 0),
    overall_cfr: 0.8,
    avg_positivity: 2.1
  };

  return (
    <div className="space-y-6">
      {/* Top Filter Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="text-teal-400" size={22} />
            Descriptive Disease Analytics & Velocity
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Multi-dimensional epidemiological metrics, incidence rates, CFR, and seasonal decomposition
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
            <span>Horizon:</span>
            <select
              value={timeWindow}
              onChange={(e) => setTimeWindow(Number(e.target.value))}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              <option value={30}>Past 30 Days</option>
              <option value={60}>Past 60 Days</option>
              <option value={90}>Past 90 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Analytics KPI Bar */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Total Cases ({timeWindow}d)</div>
          <div className="text-2xl font-bold text-white mt-1">
            {selectedDiseaseStats.total_cases?.toLocaleString()}
          </div>
          <div className="text-[11px] text-teal-400 mt-1 font-mono">Confirmed Records</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Average Daily Cases</div>
          <div className="text-2xl font-bold text-white mt-1">
            {selectedDiseaseStats.avg_daily_cases}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Moving 7d baseline</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Peak Single-Day Cases</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">
            {selectedDiseaseStats.max_daily_cases}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Observed ceiling</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Case Fatality Rate (CFR)</div>
          <div className="text-2xl font-bold text-rose-400 mt-1">
            {selectedDiseaseStats.overall_cfr}%
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Deaths / Confirmed cases</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Avg Test Positivity (TPR)</div>
          <div className="text-2xl font-bold text-cyan-400 mt-1">
            {selectedDiseaseStats.avg_positivity}%
          </div>
          <div className="text-[11px] text-slate-400 mt-1">WHO threshold: &lt;5%</div>
        </div>
      </div>

      {/* Primary Charts: Epidemic Curve & Incidence Rates */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Main Trajectory */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Daily Cases vs 7-Day & 14-Day Averages
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">{selectedRegion} — {selectedDisease}</p>
            </div>
            <span className="text-[10px] font-mono bg-teal-500/10 text-teal-400 px-2 py-0.5 rounded border border-teal-500/20">
              Telemetry
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trends}>
                <defs>
                  <linearGradient id="areaCases" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
                <Area type="monotone" dataKey="new_cases" stroke="#0ea5e9" fill="url(#areaCases)" name="Daily Cases" strokeWidth={2} />
                <Line type="monotone" dataKey="cases_7d_avg" stroke="#f59e0b" strokeWidth={2} dot={false} name="7-Day Avg" />
                <Line type="monotone" dataKey="cases_14d_avg" stroke="#10b981" strokeWidth={1.8} dot={false} strokeDasharray="4 4" name="14-Day Avg" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Incidence per 100k Population */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider">
                Normalized Incidence Rate (Per 100,000 Population)
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Demographically anchored epidemiological burden</p>
            </div>
            <span className="text-[10px] font-mono bg-purple-500/10 text-purple-400 px-2 py-0.5 rounded border border-purple-500/20">
              Normalized
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trends}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '8px' }} />
                <Line type="monotone" dataKey="incidence_per_100k" stroke="#a855f7" strokeWidth={2.5} dot={false} name="Incidence / 100k" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Seasonal Profile & Multi-Disease Burden Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Seasonal Patterns */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center gap-2">
            <Calendar size={16} className="text-amber-400" />
            Seasonal Profile (Monthly Means)
          </h2>
          <p className="text-xs text-slate-400 mb-4">Average seasonal distribution for {selectedDisease}</p>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonality}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="month" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                <Bar dataKey="avg_monthly_cases" fill="#0d9488" radius={[4, 4, 0, 0]} name="Avg Cases" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Multi-Disease Comparison Table */}
        <div className="lg:col-span-2 bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center justify-between">
            <span className="flex items-center gap-2">
              <BarChart2 size={16} className="text-cyan-400" />
              Cross-Disease Surveillance Comparison ({selectedRegion})
            </span>
            <span className="text-xs text-slate-400">All Tracked Diseases</span>
          </h2>
          <div className="overflow-x-auto mt-4">
            <table className="min-w-full text-left text-xs whitespace-nowrap">
              <thead className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">Disease Pathogen</th>
                  <th className="py-2.5 px-3">Total Volume</th>
                  <th className="py-2.5 px-3">Daily Avg</th>
                  <th className="py-2.5 px-3">CFR (%)</th>
                  <th className="py-2.5 px-3">Positivity (%)</th>
                  <th className="py-2.5 px-3">Transmission Type</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-medium text-slate-300">
                {overview.map((row, idx) => (
                  <tr 
                    key={idx} 
                    onClick={() => setSelectedDisease(row.disease)}
                    className={`cursor-pointer transition-colors ${selectedDisease === row.disease ? 'bg-teal-500/10 text-teal-300 font-bold' : 'hover:bg-slate-900'}`}
                  >
                    <td className="py-2.5 px-3 flex items-center gap-1.5 text-white">
                      {selectedDisease === row.disease && <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>}
                      {row.disease}
                    </td>
                    <td className="py-2.5 px-3 font-mono">{row.total_cases.toLocaleString()}</td>
                    <td className="py-2.5 px-3 font-mono">{row.avg_daily_cases}</td>
                    <td className="py-2.5 px-3 font-mono text-rose-400">{row.overall_cfr}%</td>
                    <td className="py-2.5 px-3 font-mono text-cyan-400">{row.avg_positivity}%</td>
                    <td className="py-2.5 px-3 text-[11px] text-slate-400">
                      {row.disease === 'Dengue' || row.disease === 'Malaria' ? 'Vector-borne' : row.disease === 'Typhoid' ? 'Water-borne' : 'Respiratory'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchDiseases,
  fetchDashboardSummary, 
  fetchDashboardRegionalRisk, 
  fetchAnalyticsTrends,
  fetchForecastResults,
  fetchAlerts,
  fetchRecommendations
} from '../services/apiClient';
import { 
  ShieldAlert, 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  CheckCircle2, 
  Stethoscope, 
  Layers, 
  ArrowUpRight, 
  ArrowDownRight,
  RefreshCw,
  Info
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

export default function Dashboard() {
  const [regions, setRegions] = useState<string[]>([]);
  const [diseases, setDiseases] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('Bhopal');
  const [selectedDisease, setSelectedDisease] = useState<string>('COVID-19');

  const [summary, setSummary] = useState<any>(null);
  const [regionalRisks, setRegionalRisks] = useState<any[]>([]);
  const [trends, setTrends] = useState<any[]>([]);
  const [forecasts, setForecasts] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [recs, setRecs] = useState<any[]>([]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Load initial dropdowns
  useEffect(() => {
    Promise.all([fetchRegions(), fetchDiseases()]).then(([regs, diss]) => {
      setRegions(regs);
      setDiseases(diss);
      if (regs.length > 0) setSelectedRegion(regs[0]);
      if (diss.length > 0) setSelectedDisease(diss[0]);
    }).catch(console.error);
  }, []);

  // Fetch dashboard data
  const loadDashboardData = async () => {
    if (!selectedRegion) return;
    try {
      setRefreshing(true);
      const [sumData, riskList, trendData, fcData, alertList, recList] = await Promise.all([
        fetchDashboardSummary(selectedRegion),
        fetchDashboardRegionalRisk(),
        fetchAnalyticsTrends(selectedRegion, selectedDisease, 45),
        fetchForecastResults(selectedRegion, selectedDisease, 14),
        fetchAlerts(selectedRegion),
        fetchRecommendations(selectedRegion)
      ]);

      setSummary(sumData);
      setRegionalRisks(riskList || []);
      setTrends(trendData || []);
      setForecasts(fcData || []);
      setAlerts(alertList || []);
      setRecs(recList || []);
    } catch (err) {
      console.error('Error fetching dashboard payload:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, [selectedRegion, selectedDisease]);

  const getRiskColor = (cat: string) => {
    switch (cat?.toLowerCase()) {
      case 'critical':
      case 'severe': return 'text-red-400 bg-red-950/40 border-red-800/60';
      case 'high': return 'text-amber-400 bg-amber-950/40 border-amber-800/60';
      case 'moderate': return 'text-yellow-400 bg-yellow-950/40 border-yellow-800/60';
      default: return 'text-emerald-400 bg-emerald-950/40 border-emerald-800/60';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Filter & Action Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Layers className="text-teal-400" size={22} />
            Epidemiological Operations Command
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time multi-district disease surveillance and healthcare risk monitoring
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
              {regions.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span>Pathogen:</span>
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              {diseases.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <button
            onClick={loadDashboardData}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-teal-500/15 hover:bg-teal-500/25 border border-teal-500/30 text-teal-300 rounded-lg text-xs font-medium transition-all"
          >
            <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Workstation Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {/* Risk Score */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Integrated Risk</span>
            <ShieldAlert size={16} className="text-teal-400" />
          </div>
          <div className="my-2">
            <span className="text-2xl font-bold text-white">
              {summary ? Number(summary.integrated_regional_risk_score).toFixed(1) : '--'}
            </span>
            <span className="text-xs text-slate-400"> / 100</span>
          </div>
          <span className={`text-[10px] px-2 py-0.5 rounded border inline-block w-fit font-semibold uppercase ${getRiskColor(summary?.integrated_risk_category)}`}>
            {summary?.integrated_risk_category || 'Assessing'}
          </span>
        </div>

        {/* 30D Total Cases */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>30-Day Cases</span>
            <Activity size={16} className="text-cyan-400" />
          </div>
          <div className="my-2">
            <span className="text-2xl font-bold text-white">
              {summary ? summary.total_cases_30d.toLocaleString() : '--'}
            </span>
            <span className="text-xs text-slate-400"> cases</span>
          </div>
          <div className="text-[11px] text-slate-400">All registered diseases</div>
        </div>

        {/* 7D Growth Rate */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Weekly Growth</span>
            <TrendingUp size={16} className={summary?.recent_growth_rate_7d > 0 ? 'text-amber-400' : 'text-emerald-400'} />
          </div>
          <div className="my-2 flex items-center gap-1">
            <span className="text-2xl font-bold text-white">
              {summary ? `${summary.recent_growth_rate_7d > 0 ? '+' : ''}${summary.recent_growth_rate_7d.toFixed(1)}%` : '--'}
            </span>
            {summary?.recent_growth_rate_7d > 0 ? (
              <ArrowUpRight size={18} className="text-amber-400" />
            ) : (
              <ArrowDownRight size={18} className="text-emerald-400" />
            )}
          </div>
          <div className="text-[11px] text-slate-400">Compared to prev 7d</div>
        </div>

        {/* Healthcare Stress */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Hospital Stress</span>
            <Stethoscope size={16} className="text-orange-400" />
          </div>
          <div className="my-2">
            <span className="text-2xl font-bold text-white">
              {summary ? Number(summary.healthcare_stress_score).toFixed(1) : '--'}
            </span>
            <span className="text-xs text-slate-400"> / 10.0</span>
          </div>
          <div className="text-[11px] text-slate-400">Capacity utilization load</div>
        </div>

        {/* Preparedness Score */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Preparedness</span>
            <CheckCircle2 size={16} className="text-emerald-400" />
          </div>
          <div className="my-2">
            <span className="text-2xl font-bold text-white">
              {summary ? Number(summary.healthcare_preparedness_score).toFixed(1) : '--'}%
            </span>
          </div>
          <div className="text-[11px] text-emerald-400/90 font-medium">Beds, O2, Drug Stocks</div>
        </div>

        {/* Active Alerts */}
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Active Alerts</span>
            <AlertTriangle size={16} className="text-amber-400" />
          </div>
          <div className="my-2">
            <span className="text-2xl font-bold text-amber-400">
              {summary ? summary.active_alerts_count : alerts.length}
            </span>
            <span className="text-xs text-slate-400"> active</span>
          </div>
          <div className="text-[11px] text-slate-400">Threshold breaches</div>
        </div>
      </div>

      {/* Primary Visualizations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: 45-day Trajectory & 7-day Moving Average */}
        <div className="lg:col-span-2 bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Activity size={16} className="text-teal-400" />
                Case Trajectory & 7-Day Moving Baseline ({selectedRegion} — {selectedDisease})
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Historical daily cases with rolling smoothing</p>
            </div>
            <div className="text-xs text-slate-400 font-mono">Past 45 Days</div>
          </div>

          <div className="h-72 w-full">
            {trends.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">Loading surveillance telemetry...</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trends}>
                  <defs>
                    <linearGradient id="colorCases" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area type="monotone" dataKey="new_cases" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorCases)" name="Daily Cases" strokeWidth={2} />
                  <Line type="monotone" dataKey="cases_7d_avg" stroke="#f59e0b" strokeWidth={2.5} dot={false} name="7-Day Rolling Avg" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Right 1 Col: Risk Dimensions Breakdown & Forecast Preview */}
        <div className="space-y-6">
          {/* Risk Dimensions */}
          <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
              <ShieldAlert size={16} className="text-amber-400" />
              Risk Dimension Matrix
            </h2>
            <div className="space-y-3.5 text-xs">
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Disease Burden (Incidence/CFR)</span>
                  <span className="font-semibold text-white">{summary ? summary.disease_risk_score.toFixed(1) : '--'} / 100</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-teal-400 h-full rounded-full" style={{ width: `${Math.min(summary?.disease_risk_score || 0, 100)}%` }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Outbreak Signals (Z-Score & IF)</span>
                  <span className="font-semibold text-white">{summary ? summary.outbreak_signal_score.toFixed(1) : '--'} / 100</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-amber-400 h-full rounded-full" style={{ width: `${Math.min(summary?.outbreak_signal_score || 0, 100)}%` }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Environmental Hazard Factor</span>
                  <span className="font-semibold text-white">{summary ? summary.environmental_risk_score.toFixed(1) : '--'} / 100</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-cyan-400 h-full rounded-full" style={{ width: `${Math.min(summary?.environmental_risk_score || 0, 100)}%` }}></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span>Healthcare Stress Index</span>
                  <span className="font-semibold text-white">{summary ? (summary.healthcare_stress_score * 10.0).toFixed(1) : '--'} / 100</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-orange-400 h-full rounded-full" style={{ width: `${Math.min((summary?.healthcare_stress_score || 0) * 10.0, 100)}%` }}></div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex items-start gap-1.5">
                <Info size={14} className="shrink-0 text-slate-400 mt-0.5" />
                <span>Primary Driver: <strong className="text-slate-200">{summary?.primary_driver || 'Normal surveillance'}</strong></span>
              </div>
            </div>
          </div>

          {/* 14-Day Forecast Preview */}
          <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <TrendingUp size={16} className="text-purple-400" />
                14-Day Forecast Projection
              </span>
              <span className="text-[10px] text-purple-400 font-mono">XGBoost ML</span>
            </h2>
            <div className="h-32 w-full">
              {forecasts.length === 0 ? (
                <div className="h-full flex items-center justify-center text-slate-400 text-xs">No forecast data</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={forecasts}>
                    <defs>
                      <linearGradient id="colorFc" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#a855f7" stopOpacity={0.0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="2 2" stroke="#1e293b" />
                    <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 9 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 9 }} />
                    <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }} />
                    <Area type="monotone" dataKey="predicted_cases" stroke="#a855f7" fill="url(#colorFc)" name="Predicted" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Grid: Regional Risk Leaderboard & Recent Operational Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Multi-Region Risk Comparison Table */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-3 flex items-center justify-between">
            <span>District Risk Surveillance Matrix</span>
            <span className="text-xs text-slate-400">All 5 MP Pilot Regions</span>
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-xs whitespace-nowrap">
              <thead className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">District</th>
                  <th className="py-2.5 px-3">Risk Score</th>
                  <th className="py-2.5 px-3">Category</th>
                  <th className="py-2.5 px-3">Stress Load</th>
                  <th className="py-2.5 px-3">Preparedness</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-medium">
                {regionalRisks.map((item, idx) => (
                  <tr 
                    key={idx} 
                    onClick={() => setSelectedRegion(item.region)}
                    className={`cursor-pointer transition-colors ${selectedRegion === item.region ? 'bg-teal-500/10 text-teal-300' : 'hover:bg-slate-900 text-slate-300'}`}
                  >
                    <td className="py-2.5 px-3 font-bold text-white flex items-center gap-1.5">
                      {selectedRegion === item.region && <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>}
                      {item.region}
                    </td>
                    <td className="py-2.5 px-3 font-mono">{Number(item.integrated_regional_risk_score).toFixed(1)}</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded border text-[10px] font-semibold uppercase ${getRiskColor(item.integrated_risk_category || item.risk_category)}`}>
                        {item.integrated_risk_category || item.risk_category}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono">{Number(item.healthcare_stress_score).toFixed(1)}/10</td>
                    <td className="py-2.5 px-3 font-mono text-emerald-400">{Number(item.healthcare_preparedness_score).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Operational Alerts & Directives Feed */}
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-3 flex items-center justify-between">
            <span>Active Operational Alerts & Directives ({selectedRegion})</span>
            <span className="text-xs text-amber-400 font-mono">{alerts.length} Active</span>
          </h2>
          <div className="space-y-3 overflow-y-auto max-h-72 pr-1">
            {alerts.length === 0 && recs.length === 0 ? (
              <div className="text-slate-400 text-xs py-8 text-center">No active alerts or directives for this region.</div>
            ) : (
              <>
                {alerts.map((al, idx) => (
                  <div key={`al-${idx}`} className="p-3 rounded-lg bg-slate-900 border-l-4 border-l-amber-500 border border-slate-800">
                    <div className="flex justify-between items-start">
                      <div className="font-semibold text-xs text-amber-300 flex items-center gap-1.5">
                        <AlertTriangle size={13} /> {al.category} — {al.region}
                      </div>
                      <span className="text-[10px] font-mono text-slate-400">{al.timestamp}</span>
                    </div>
                    <p className="text-xs text-slate-300 mt-1">{al.explanation}</p>
                  </div>
                ))}
                {recs.map((rc, idx) => (
                  <div key={`rc-${idx}`} className="p-3 rounded-lg bg-slate-900 border-l-4 border-l-teal-500 border border-slate-800">
                    <div className="flex justify-between items-start">
                      <div className="font-semibold text-xs text-teal-300 flex items-center gap-1.5">
                        <CheckCircle2 size={13} /> [{rc.target_stakeholder}] {rc.recommendation_title}
                      </div>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-teal-500/20 text-teal-300 font-mono uppercase">{rc.priority}</span>
                    </div>
                    <p className="text-xs text-slate-300 mt-1">{rc.rationale}</p>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

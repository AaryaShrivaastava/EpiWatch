import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchAlerts, 
  fetchAlertsStatistics, 
  updateAlertStatus 
} from '../services/apiClient';
import { 
  AlertTriangle, 
  ShieldAlert, 
  CheckCircle, 
  Clock, 
  Search, 
  Filter, 
  Check, 
  XCircle,
  Sparkles,
  Info
} from 'lucide-react';

export default function Alerts() {
  const [regions, setRegions] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('ALL');
  const [selectedStatus, setSelectedStatus] = useState<string>('ALL');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [alerts, setAlerts] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({ active: 0, acknowledged: 0, resolved: 0, total: 0 });
  const [loading, setLoading] = useState<boolean>(true);
  const [feedback, setFeedback] = useState<string | null>(null);

  useEffect(() => {
    fetchRegions().then(setRegions).catch(console.error);
  }, []);

  const loadAlerts = async () => {
    setLoading(true);
    try {
      const reg = selectedRegion === 'ALL' ? undefined : selectedRegion;
      const stat = selectedStatus === 'ALL' ? undefined : selectedStatus;
      const [alertList, statsData] = await Promise.all([
        fetchAlerts(reg, stat),
        fetchAlertsStatistics()
      ]);
      setAlerts(alertList || []);
      setStats(statsData || { active: 0, acknowledged: 0, resolved: 0, total: 0 });
    } catch (err) {
      console.error('Error loading alerts:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAlerts();
  }, [selectedRegion, selectedStatus]);

  const handleUpdateStatus = async (alertId: string, newStatus: string) => {
    try {
      await updateAlertStatus(alertId, newStatus);
      setFeedback(`Alert ${alertId} updated to ${newStatus}.`);
      await loadAlerts();
    } catch (err) {
      setFeedback(`Failed to update status for ${alertId}`);
    }
  };

  const filteredAlerts = alerts.filter((al) => {
    if (selectedSeverity !== 'ALL' && al.severity?.toUpperCase() !== selectedSeverity) {
      return false;
    }
    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      const match = (
        al.category?.toLowerCase().includes(q) ||
        al.region?.toLowerCase().includes(q) ||
        al.explanation?.toLowerCase().includes(q) ||
        al.triggering_metric?.toLowerCase().includes(q)
      );
      if (!match) return false;
    }
    return true;
  });

  const getSeverityStyle = (sev: string) => {
    switch (sev?.toLowerCase()) {
      case 'critical':
      case 'high':
        return 'border-l-red-500 bg-red-950/20 border-slate-800 text-red-400';
      case 'warning':
      case 'moderate':
        return 'border-l-amber-500 bg-amber-950/20 border-slate-800 text-amber-400';
      default:
        return 'border-l-teal-500 bg-teal-950/20 border-slate-800 text-teal-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Search/Filter Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <AlertTriangle className="text-amber-400" size={22} />
            Intelligent Operational Alerts
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Automated multi-condition rule evaluations, threshold breaches, and escalation management
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Search Box */}
          <div className="relative">
            <Search size={14} className="absolute left-3 top-2.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search alerts..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-teal-400 w-44"
            />
          </div>

          {/* Region Filter */}
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
          >
            <option value="ALL">All Districts</option>
            {regions.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
          >
            <option value="ALL">All Statuses</option>
            <option value="active">Active Only</option>
            <option value="acknowledged">Acknowledged</option>
            <option value="resolved">Resolved</option>
          </select>

          {/* Severity Filter */}
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
          >
            <option value="ALL">All Severities</option>
            <option value="HIGH">High / Critical</option>
            <option value="MODERATE">Moderate</option>
            <option value="INFO">Info</option>
          </select>
        </div>
      </div>

      {feedback && (
        <div className="p-3 rounded-lg bg-teal-500/15 border border-teal-500/30 text-teal-300 text-xs flex items-center gap-2">
          <Sparkles size={16} />
          {feedback}
        </div>
      )}

      {/* Statistics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Total Recorded Alerts</div>
          <div className="text-2xl font-bold text-white mt-1">{stats.total}</div>
          <div className="text-[11px] text-slate-400 mt-1">Surveillance events</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Active Unresolved</div>
          <div className="text-2xl font-bold text-amber-400 mt-1">{stats.active}</div>
          <div className="text-[11px] text-slate-400 mt-1">Requires attention</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Acknowledged / In Review</div>
          <div className="text-2xl font-bold text-cyan-400 mt-1">{stats.acknowledged}</div>
          <div className="text-[11px] text-slate-400 mt-1">Under investigation</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="text-xs text-slate-400">Resolved Alerts</div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">{stats.resolved}</div>
          <div className="text-[11px] text-slate-400 mt-1">Stabilized events</div>
        </div>
      </div>

      {/* Alert Feed */}
      <div className="space-y-4">
        {loading ? (
          <div className="text-center py-12 text-xs text-slate-400">Loading alerts feed...</div>
        ) : filteredAlerts.length === 0 ? (
          <div className="p-8 text-center bg-slate-950/60 rounded-xl border border-slate-800 text-slate-400 text-xs">
            <CheckCircle className="mx-auto text-emerald-400 mb-2" size={28} />
            No alerts match the selected criteria. The surveillance systems report stable operating parameters.
          </div>
        ) : (
          filteredAlerts.map((al, idx) => (
            <div 
              key={idx} 
              className={`p-5 rounded-xl border-l-4 border shadow-sm transition-all bg-slate-950/80 ${getSeverityStyle(al.severity)}`}
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-white flex items-center gap-1.5">
                    <AlertTriangle size={16} />
                    {al.category} — {al.region}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 uppercase font-semibold">
                    {al.severity}
                  </span>
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded uppercase font-semibold ${
                    al.status === 'active' ? 'bg-amber-500/20 text-amber-300' : al.status === 'acknowledged' ? 'bg-cyan-500/20 text-cyan-300' : 'bg-emerald-500/20 text-emerald-300'
                  }`}>
                    {al.status}
                  </span>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-400 font-mono flex items-center gap-1">
                    <Clock size={12} /> {al.timestamp}
                  </span>
                  
                  {/* Status Action Buttons */}
                  {al.status !== 'acknowledged' && al.status !== 'resolved' && (
                    <button
                      onClick={() => handleUpdateStatus(al.alert_id, 'acknowledged')}
                      className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px] font-semibold border border-slate-700 transition-colors"
                    >
                      Acknowledge
                    </button>
                  )}
                  {al.status !== 'resolved' && (
                    <button
                      onClick={() => handleUpdateStatus(al.alert_id, 'resolved')}
                      className="px-2.5 py-1 rounded bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 text-[11px] font-semibold border border-emerald-500/40 transition-colors"
                    >
                      Resolve
                    </button>
                  )}
                </div>
              </div>

              <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                {al.explanation}
              </p>

              <div className="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap gap-4 text-[11px] text-slate-400 font-mono">
                <span>Triggering Rule: <strong className="text-slate-200 font-sans">{al.triggering_metric}</strong></span>
                <span>Observed Reading: <strong className="text-teal-400 font-sans">{al.observed_value}</strong></span>
                <span>Alert ID: <strong className="text-slate-400 font-sans">{al.alert_id}</strong></span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

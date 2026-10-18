import { useState, useEffect, useRef } from 'react';
import { 
  fetchSurveillanceStatus,
  startSurveillance,
  stopSurveillance,
  triggerSurveillanceScan,
  fetchSurveillanceFindings,
  fetchSurveillanceStatistics,
  fetchSurveillanceTimeline,
  fetchSurveillanceConfig,
  updateSurveillanceConfig,
  sendAssistantChat
} from '../services/apiClient';
import { 
  Radio, 
  Play, 
  Square, 
  RotateCw, 
  Zap, 
  AlertTriangle, 
  CheckCircle2, 
  ShieldAlert, 
  Clock, 
  Sliders, 
  Filter, 
  Search, 
  Eye, 
  FileText, 
  Brain, 
  Bot, 
  User, 
  Send, 
  Sparkles, 
  ChevronRight, 
  X,
  Activity,
  Layers,
  TrendingUp,
  Cpu
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend 
} from 'recharts';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  tools?: string[];
  timestamp: string;
}

export default function Agent() {
  const [activeTab, setActiveTab] = useState<'control' | 'analytics' | 'assistant'>('control');

  // Status & Control State
  const [status, setStatus] = useState<any>(null);
  const [findings, setFindings] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [timeline, setTimeline] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [actionMessage, setActionMessage] = useState<string>('');

  // Filters
  const [filterRegion, setFilterRegion] = useState<string>('All');
  const [filterDisease, setFilterDisease] = useState<string>('All');
  const [filterSeverity, setFilterSeverity] = useState<string>('All');

  // Finding Detail Modal
  const [selectedFinding, setSelectedFinding] = useState<any | null>(null);

  // Configuration Modal
  const [showConfigModal, setShowConfigModal] = useState<boolean>(false);
  const [configData, setConfigData] = useState<any>(null);
  const [editInterval, setEditInterval] = useState<number>(900);
  const [configMsg, setConfigMsg] = useState<string>('');

  // Assistant Chat State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: "Hello! I am the **EpiWatch AI Surveillance Agent**. I continuously monitor regional disease baselines, ICU bed loads, and environmental indicators across Madhya Pradesh. Ask me to summarize active surveillance signals, explain regional risk factors, or query 14-day forecasts.",
      tools: [],
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [chatLoading, setChatLoading] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load All Surveillance Data
  const loadSurveillanceData = async () => {
    try {
      const [st, fnd, stt, tml] = await Promise.all([
        fetchSurveillanceStatus(),
        fetchSurveillanceFindings({
          region: filterRegion !== 'All' ? filterRegion : undefined,
          disease: filterDisease !== 'All' ? filterDisease : undefined,
          severity: filterSeverity !== 'All' ? filterSeverity : undefined
        }),
        fetchSurveillanceStatistics(),
        fetchSurveillanceTimeline(40)
      ]);
      setStatus(st);
      setFindings(fnd || []);
      setStats(stt);
      setTimeline(tml || []);
    } catch (e: any) {
      console.error('Error loading surveillance data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSurveillanceData();
    const interval = setInterval(loadSurveillanceData, 8000); // Polling every 8s
    return () => clearInterval(interval);
  }, [filterRegion, filterDisease, filterSeverity]);

  // Handle Start
  const handleStart = async () => {
    setActionMessage('Starting background surveillance scheduler...');
    try {
      const res = await startSurveillance();
      setActionMessage(res.message || 'Surveillance started successfully.');
      loadSurveillanceData();
    } catch (e: any) {
      setActionMessage(`Error starting surveillance: ${e.message}`);
    }
  };

  // Handle Stop
  const handleStop = async () => {
    setActionMessage('Stopping background surveillance scheduler...');
    try {
      const res = await stopSurveillance();
      setActionMessage(res.message || 'Surveillance stopped.');
      loadSurveillanceData();
    } catch (e: any) {
      setActionMessage(`Error stopping surveillance: ${e.message}`);
    }
  };

  // Handle Manual Scan
  const handleTriggerScan = async () => {
    setIsScanning(true);
    setActionMessage('Executing live epidemiological surveillance scan across all datasets...');
    try {
      const res = await triggerSurveillanceScan();
      setActionMessage(`Scan complete (${res.duration_seconds}s). Evaluated ${res.records_evaluated} records, generated ${res.findings_generated} findings.`);
      loadSurveillanceData();
    } catch (e: any) {
      setActionMessage(`Scan failed: ${e.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  // Handle Open Config Modal
  const handleOpenConfig = async () => {
    setShowConfigModal(true);
    setConfigMsg('');
    try {
      const cfg = await fetchSurveillanceConfig();
      setConfigData(cfg);
      setEditInterval(cfg?.scan_interval_seconds || 900);
    } catch (e: any) {
      setConfigMsg(`Could not load config: ${e.message}`);
    }
  };

  // Handle Save Config
  const handleSaveConfig = async () => {
    try {
      await updateSurveillanceConfig({ scan_interval_seconds: editInterval });
      setConfigMsg('Configuration updated successfully.');
      setTimeout(() => setShowConfigModal(false), 800);
      loadSurveillanceData();
    } catch (e: any) {
      setConfigMsg(`Error saving config: ${e.message}`);
    }
  };

  // Handle Send Chat
  const handleSendChat = async (prompt?: string) => {
    const text = prompt || chatInput;
    if (!text.trim() || chatLoading) return;

    const userMsg: ChatMessage = {
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setChatMessages((prev) => [...prev, userMsg]);
    if (!prompt) setChatInput('');
    setChatLoading(true);

    try {
      const res = await sendAssistantChat(text);
      const botMsg: ChatMessage = {
        role: 'assistant',
        content: res.reply || 'No response generated.',
        tools: res.tools_executed || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setChatMessages((prev) => [...prev, botMsg]);
    } catch (e: any) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an issue querying the surveillance core.',
          tools: [],
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* 1. AGENT STATUS HEADER */}
      <div className="bg-slate-950/90 p-5 rounded-xl border border-slate-800 shadow-xl space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
                <Radio className="text-teal-400 animate-pulse" size={24} />
                Active AI Disease Surveillance Center
              </h1>
              {status && (
                <span
                  className={`px-3 py-0.5 rounded-full text-xs font-mono font-bold uppercase tracking-wider flex items-center gap-1.5 ${
                    status.agent_status === 'RUNNING'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : status.agent_status === 'DEGRADED'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'bg-slate-800 text-slate-400 border border-slate-700'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      status.agent_status === 'RUNNING' ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'
                    }`}
                  ></span>
                  {status.agent_status}
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Automated in-process surveillance worker continuously evaluating multi-pathogen baselines, 7-day growth rates, and ICU bed thresholds.
            </p>
          </div>

          {/* Quick Tabs & LLM Status */}
          <div className="flex items-center gap-3">
            {status?.llm_status === 'GEMINI_ACTIVE' ? (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-cyan-950/40 border border-cyan-800/40 text-cyan-300 font-mono text-xs">
                <Sparkles size={14} className="text-cyan-400" />
                Gemini 1.5 Flash Connected
              </span>
            ) : (
              <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 font-mono text-xs">
                <Cpu size={14} className="text-teal-400" />
                Deterministic Surveillance Core
              </span>
            )}

            <div className="flex bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
              <button
                onClick={() => setActiveTab('control')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  activeTab === 'control' ? 'bg-teal-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                Surveillance Control
              </button>
              <button
                onClick={() => setActiveTab('analytics')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  activeTab === 'analytics' ? 'bg-teal-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                Surveillance Analytics
              </button>
              <button
                onClick={() => setActiveTab('assistant')}
                className={`px-3 py-1.5 rounded-md font-medium transition-all ${
                  activeTab === 'assistant' ? 'bg-teal-600 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                AI Assistant
              </button>
            </div>
          </div>
        </div>

        {/* Status Metrics Ribbon */}
        {status && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 pt-3 border-t border-slate-800/80 text-xs">
            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Last Successful Scan</div>
              <div className="font-mono text-white text-xs font-bold mt-1">
                {status.last_successful_scan ? new Date(status.last_successful_scan).toLocaleTimeString() : 'Pending'}
              </div>
            </div>

            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Next Scheduled Scan</div>
              <div className="font-mono text-cyan-400 text-xs font-bold mt-1">
                {status.next_scheduled_scan ? new Date(status.next_scheduled_scan).toLocaleTimeString() : 'Paused'}
              </div>
            </div>

            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Scan Interval</div>
              <div className="font-mono text-white text-xs font-bold mt-1">
                {Math.round(status.scan_interval_seconds / 60)} min ({status.scan_interval_seconds}s)
              </div>
            </div>

            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Total Scans Run</div>
              <div className="font-mono text-emerald-400 text-xs font-bold mt-1">
                {status.total_scans_completed}
              </div>
            </div>

            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Anomalies Detected</div>
              <div className="font-mono text-amber-400 text-xs font-bold mt-1">
                {status.findings_detected}
              </div>
            </div>

            <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400">Active Alert Signals</div>
              <div className="font-mono text-red-400 text-xs font-bold mt-1">
                {status.active_alerts}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 2. CONTROLS TOOLBAR */}
      <div className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          {status?.agent_status === 'RUNNING' ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-600/20 hover:bg-red-600/30 text-red-300 border border-red-500/40 font-medium transition-colors"
            >
              <Square size={13} />
              <span>Stop Surveillance</span>
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/40 font-medium transition-colors"
            >
              <Play size={13} />
              <span>Start Surveillance</span>
            </button>
          )}

          <button
            onClick={handleTriggerScan}
            disabled={isScanning}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white font-medium disabled:opacity-50 transition-colors shadow-sm"
          >
            <Zap size={13} className={isScanning ? 'animate-spin' : ''} />
            <span>{isScanning ? 'Scanning...' : 'Run Scan Now'}</span>
          </button>

          <button
            onClick={loadSurveillanceData}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors"
          >
            <RotateCw size={13} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleOpenConfig}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700 transition-colors"
          >
            <Sliders size={13} />
            <span>Configure Thresholds</span>
          </button>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="text-slate-400">Region:</span>
            <select
              value={filterRegion}
              onChange={(e) => setFilterRegion(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs"
            >
              <option value="All">All Regions</option>
              <option value="Bhopal">Bhopal</option>
              <option value="Indore">Indore</option>
              <option value="Jabalpur">Jabalpur</option>
              <option value="Gwalior">Gwalior</option>
              <option value="Ujjain">Ujjain</option>
            </select>
          </div>

          <div className="flex items-center gap-1">
            <span className="text-slate-400">Disease:</span>
            <select
              value={filterDisease}
              onChange={(e) => setFilterDisease(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs"
            >
              <option value="All">All Diseases</option>
              <option value="COVID-19">COVID-19</option>
              <option value="Dengue">Dengue</option>
              <option value="Malaria">Malaria</option>
              <option value="Influenza">Influenza</option>
              <option value="Typhoid">Typhoid</option>
            </select>
          </div>

          <div className="flex items-center gap-1">
            <span className="text-slate-400">Severity:</span>
            <select
              value={filterSeverity}
              onChange={(e) => setFilterSeverity(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2 py-1 text-xs"
            >
              <option value="All">All Severities</option>
              <option value="Critical">Critical</option>
              <option value="High">High</option>
              <option value="Moderate">Moderate</option>
              <option value="Low">Low</option>
            </select>
          </div>
        </div>
      </div>

      {actionMessage && (
        <div className="text-xs text-teal-300 font-mono bg-teal-950/50 p-2.5 rounded-lg border border-teal-800/40 flex items-center justify-between">
          <span>{actionMessage}</span>
          <button onClick={() => setActionMessage('')} className="text-slate-400 hover:text-white">
            <X size={14} />
          </button>
        </div>
      )}

      {/* TAB 1: CONTROL CENTER (TIMELINE + FINDINGS TABLE) */}
      {activeTab === 'control' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Findings Table (8 Cols) */}
          <div className="lg:col-span-8 bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <ShieldAlert size={16} className="text-teal-400" />
                  <span>Surveillance Findings & Active Signals</span>
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Anomalies detected through statistical rolling Z-scores and moving baselines
                </p>
              </div>
              <span className="text-xs font-mono text-slate-400 bg-slate-900 px-2 py-1 rounded border border-slate-800">
                {findings.length} findings
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full text-left text-xs whitespace-nowrap">
                <thead className="border-b border-slate-800 text-slate-400 uppercase font-mono text-[10px]">
                  <tr>
                    <th className="py-2.5 px-3">Finding ID / Time</th>
                    <th className="py-2.5 px-3">District</th>
                    <th className="py-2.5 px-3">Pathogen</th>
                    <th className="py-2.5 px-3">Severity</th>
                    <th className="py-2.5 px-3">Detection Metric</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-medium text-slate-300">
                  {findings.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="py-8 text-center text-slate-500">
                        No active anomalies detected matching current filters. All regional baselines within moving bounds.
                      </td>
                    </tr>
                  ) : (
                    findings.map((f) => (
                      <tr key={f.finding_id} className="hover:bg-slate-900/80 transition-colors">
                        <td className="py-2.5 px-3 font-mono">
                          <div className="text-white font-bold">{f.finding_id}</div>
                          <div className="text-[10px] text-slate-500">{new Date(f.timestamp).toLocaleTimeString()}</div>
                        </td>
                        <td className="py-2.5 px-3 font-bold text-slate-200">{f.region}</td>
                        <td className="py-2.5 px-3 font-mono text-teal-300">{f.disease}</td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                              f.severity === 'Critical'
                                ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                                : f.severity === 'High'
                                ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40'
                                : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40'
                            }`}
                          >
                            {f.severity}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 font-mono text-slate-300">
                          <div>{f.triggering_metric}</div>
                          <div className="text-[10px] text-slate-500">Value: {f.observed_value} (Th: {f.threshold})</div>
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[10px] text-slate-400">
                          {f.status}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          <button
                            onClick={() => setSelectedFinding(f)}
                            className="px-2.5 py-1 bg-slate-900 hover:bg-slate-800 text-teal-400 border border-slate-700 rounded text-[11px] font-medium transition-colors"
                          >
                            Inspect Evidence
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Live Activity Timeline (4 Cols) */}
          <div className="lg:col-span-4 bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <Clock size={16} className="text-cyan-400" />
                <span>Live Activity Timeline</span>
              </h2>
              <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span> Real-time
              </span>
            </div>

            <div className="space-y-3 max-h-[500px] overflow-y-auto pr-1">
              {timeline.length === 0 ? (
                <div className="text-xs text-slate-500 text-center py-8">
                  No activity logs recorded yet. Run a scan to initiate timeline.
                </div>
              ) : (
                timeline.map((evt) => (
                  <div key={evt.id} className="p-2.5 bg-slate-900/60 rounded-lg border border-slate-800/80 text-xs space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-200">{evt.title}</span>
                      <span className="text-[10px] font-mono text-slate-500">
                        {new Date(evt.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">
                      {evt.details}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: SURVEILLANCE ANALYTICS CHARTS */}
      {activeTab === 'analytics' && (
        <div className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Findings by Severity */}
            <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Findings by Severity Distribution
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats?.by_severity || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                    <Bar dataKey="count" fill="#f59e0b" name="Findings" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Regional Distribution */}
            <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Anomalies Detected by District
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats?.by_region || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                    <Bar dataKey="count" fill="#0ea5e9" name="District Anomalies" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Findings by Disease */}
            <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Findings by Tracked Pathogen
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats?.by_disease || []}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                    <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
                    <Bar dataKey="count" fill="#10b981" name="Disease Signals" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Scan History Success/Failure */}
            <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
              <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Scheduled Worker Execution Reliability
              </h3>
              <div className="grid grid-cols-2 gap-4 pt-4">
                <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 text-center">
                  <div className="text-2xl font-bold font-mono text-emerald-400">{stats?.scan_success_count || 0}</div>
                  <div className="text-xs text-slate-400 mt-1">Successful Scans</div>
                </div>
                <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 text-center">
                  <div className="text-2xl font-bold font-mono text-red-400">{stats?.scan_failure_count || 0}</div>
                  <div className="text-xs text-slate-400 mt-1">Failed Scans</div>
                </div>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed mt-2">
                Worker scheduled in-process with threading locks preventing overlapping execution.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: AGENT REASONING & CONVERSATIONAL ASSISTANT */}
      {activeTab === 'assistant' && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
          {/* Reasoning & Grounding Cards (5 Cols) */}
          <div className="lg:col-span-5 space-y-4">
            <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
              <h3 className="font-bold text-sm text-white flex items-center gap-2">
                <Brain className="text-teal-400" size={18} />
                Agent Analytical Reasoning Framework
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                The surveillance agent follows a deterministic public-health pipeline. The LLM is restricted to approved tools and does not invent ungrounded assertions.
              </p>

              <div className="space-y-2.5 text-xs text-slate-300 pt-2">
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="font-bold text-teal-300">1. Trigger Condition</div>
                  <div className="text-slate-400 mt-0.5">Rolling Z-score exceeding 2.0σ or 7-day case growth exceeding 15% with &gt;15 cases.</div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="font-bold text-cyan-300">2. Approved Local Tools</div>
                  <div className="text-slate-400 mt-0.5"><code>detect_outbreaks</code>, <code>forecast_disease</code>, <code>get_regional_risk_assessment</code>, <code>get_surveillance_findings</code>.</div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="font-bold text-purple-300">3. Deduplication Guardrail</div>
                  <div className="text-slate-400 mt-0.5">24-hour suppression window per district and pathogen preventing notification spam.</div>
                </div>

                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="font-bold text-amber-300">4. Uncertainty Acknowledgment</div>
                  <div className="text-slate-400 mt-0.5">Statistical anomalies are marked as unconfirmed signals until verified by epidemiologists.</div>
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Chat Box (7 Cols) */}
          <div className="lg:col-span-7 bg-slate-950/90 rounded-xl border border-slate-800 flex flex-col h-[580px] shadow-xl">
            <div className="p-3.5 border-b border-slate-800 flex items-center justify-between text-xs font-bold text-slate-300">
              <span className="flex items-center gap-2">
                <Bot size={16} className="text-teal-400" />
                Conversational Surveillance Assistant
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Grounded in verified artifacts</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {chatMessages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex gap-3 text-xs ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {m.role === 'assistant' && (
                    <div className="w-7 h-7 rounded-lg bg-teal-500/20 border border-teal-500/30 flex items-center justify-center text-teal-400 shrink-0">
                      <Bot size={15} />
                    </div>
                  )}

                  <div
                    className={`max-w-xl p-3 rounded-xl leading-relaxed ${
                      m.role === 'user'
                        ? 'bg-teal-600 text-white font-medium rounded-tr-none'
                        : 'bg-slate-900 border border-slate-800 text-slate-200 rounded-tl-none'
                    }`}
                  >
                    <div className="whitespace-pre-line">{m.content}</div>
                    {m.tools && m.tools.length > 0 && (
                      <div className="mt-2 pt-1.5 border-t border-slate-800/80 flex flex-wrap gap-1 text-[10px] text-teal-400 font-mono">
                        <span className="text-slate-500">Tools:</span>
                        {m.tools.map((t) => (
                          <span key={t} className="bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                            {t}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {m.role === 'user' && (
                    <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 shrink-0">
                      <User size={15} />
                    </div>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Quick Prompts */}
            <div className="p-2 border-t border-slate-800/80 flex gap-2 overflow-x-auto text-[11px] text-slate-400 bg-slate-950/60">
              <button
                onClick={() => handleSendChat("Summarize active surveillance anomalies across all districts")}
                className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 whitespace-nowrap text-teal-300"
              >
                Summarize active anomalies
              </button>
              <button
                onClick={() => handleSendChat("What is the current risk and hospital capacity for Indore?")}
                className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 whitespace-nowrap"
              >
                Indore hospital stress
              </button>
              <button
                onClick={() => handleSendChat("Explain public health directives for Bhopal")}
                className="px-2 py-1 rounded bg-slate-900 hover:bg-slate-800 border border-slate-800 whitespace-nowrap"
              >
                Bhopal directives
              </button>
            </div>

            {/* Chat Input */}
            <div className="p-3 border-t border-slate-800 flex items-center gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendChat()}
                placeholder="Ask about district anomalies, forecasts, or bed stress..."
                className="flex-1 bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-2 focus:outline-none focus:border-teal-500"
              />
              <button
                onClick={() => handleSendChat()}
                disabled={chatLoading}
                className="p-2 rounded-lg bg-teal-600 hover:bg-teal-500 text-white transition-colors disabled:opacity-50"
              >
                <Send size={15} />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 5. FINDING DETAIL MODAL */}
      {selectedFinding && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-2xl w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-bold text-white">{selectedFinding.finding_id}</h3>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${
                      selectedFinding.severity === 'Critical'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                        : selectedFinding.severity === 'High'
                        ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40'
                        : 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40'
                    }`}
                  >
                    {selectedFinding.severity}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  {selectedFinding.region} • {selectedFinding.disease} • Detected {new Date(selectedFinding.timestamp).toLocaleString()}
                </div>
              </div>

              <button
                onClick={() => setSelectedFinding(null)}
                className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-900"
              >
                <X size={18} />
              </button>
            </div>

            {/* Finding Summary */}
            <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs text-slate-300 leading-relaxed">
              <strong>Summary:</strong> {selectedFinding.summary}
            </div>

            {/* Baseline vs Observed Metrics */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1">
                <div className="text-slate-400 font-medium">Historical Baseline (28-day)</div>
                <div className="font-mono text-slate-200 text-xs">
                  Mean: <strong>{selectedFinding.historical_baseline?.baseline_mean}</strong>
                </div>
                <div className="font-mono text-slate-200 text-xs">
                  Std Dev: <strong>{selectedFinding.historical_baseline?.baseline_std}</strong>
                </div>
                <div className="font-mono text-slate-400 text-[11px]">
                  2-Sigma Upper: {selectedFinding.historical_baseline?.upper_threshold_2std}
                </div>
              </div>

              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-1">
                <div className="text-slate-400 font-medium">Triggering Indicator</div>
                <div className="font-mono text-teal-300 font-bold text-xs">
                  {selectedFinding.triggering_metric}: {selectedFinding.observed_value}
                </div>
                <div className="font-mono text-slate-400 text-[11px]">
                  Configured Threshold: {selectedFinding.threshold}
                </div>
                <div className="text-[10px] text-slate-500">
                  Method: {selectedFinding.detection_method}
                </div>
              </div>
            </div>

            {/* Supporting Evidence Payload */}
            {selectedFinding.supporting_evidence && (
              <div className="bg-slate-900/40 p-3 rounded-lg border border-slate-800/60 space-y-1.5 text-xs">
                <div className="text-slate-300 font-semibold">Supporting Clinical & Environmental Signals</div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400">
                  <div>Recent Daily Cases: <strong className="text-white">{selectedFinding.supporting_evidence.recent_daily_cases}</strong></div>
                  <div>7d Growth: <strong className="text-white">{selectedFinding.supporting_evidence['7d_growth_rate_percent']}%</strong></div>
                  <div>Incidence/100k: <strong className="text-white">{selectedFinding.supporting_evidence.incidence_per_100k}</strong></div>
                  <div>Hospital Stress: <strong className="text-white">{selectedFinding.supporting_evidence.hospital_stress_index}/10</strong></div>
                </div>
              </div>
            )}

            {/* Recommended Action */}
            <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 text-xs space-y-1">
              <div className="text-teal-300 font-bold">Recommended Next Analytical Step</div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                {selectedFinding.recommended_next_step}
              </p>
            </div>

            {/* Footer Provenance */}
            <div className="flex items-center justify-between text-[10px] text-slate-500 pt-2 border-t border-slate-800">
              <div>Provenance: {selectedFinding.data_provenance}</div>
              <div>Related Alert ID: {selectedFinding.related_alert_id || 'None'}</div>
            </div>
          </div>
        </div>
      )}

      {/* 6. CONFIGURATION MODAL */}
      {showConfigModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h3 className="font-bold text-sm text-white flex items-center gap-2">
                <Sliders size={16} className="text-teal-400" />
                Configure Surveillance Worker
              </h3>
              <button onClick={() => setShowConfigModal(false)} className="text-slate-400 hover:text-white">
                <X size={16} />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-300 font-medium block mb-1">
                  Scheduled Scan Interval (seconds):
                </label>
                <input
                  type="number"
                  min={30}
                  max={86400}
                  value={editInterval}
                  onChange={(e) => setEditInterval(parseInt(e.target.value, 10))}
                  className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-white font-mono text-xs focus:outline-none focus:border-teal-500"
                />
                <span className="text-[10px] text-slate-500 mt-1 block">
                  Recommended: 900s (15 min) for production, 60s for testing.
                </span>
              </div>

              {configMsg && (
                <div className="text-xs text-teal-300 font-mono bg-teal-950/40 p-2 rounded border border-teal-800/40">
                  {configMsg}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
              <button
                onClick={() => setShowConfigModal(false)}
                className="px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-300 text-xs border border-slate-700"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveConfig}
                className="px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

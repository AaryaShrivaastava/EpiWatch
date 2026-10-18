import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchHealthcareOverview, 
  fetchHealthcareCapacity 
} from '../services/apiClient';
import { 
  Stethoscope, 
  Activity, 
  Bed, 
  HeartPulse, 
  ShieldCheck, 
  AlertTriangle, 
  CheckCircle2, 
  Info,
  Layers
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend 
} from 'recharts';

export default function Healthcare() {
  const [regions, setRegions] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('Bhopal');
  const [overview, setOverview] = useState<any>(null);
  const [allCapacity, setAllCapacity] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchRegions().then((regs) => {
      setRegions(regs);
      if (regs.length > 0) setSelectedRegion(regs[0]);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedRegion) return;
    setLoading(true);
    Promise.all([
      fetchHealthcareOverview(selectedRegion),
      fetchHealthcareCapacity()
    ]).then(([ov, caps]) => {
      setOverview(ov || null);
      setAllCapacity(caps || []);
      setLoading(false);
    }).catch((err) => {
      console.error('Error fetching healthcare data:', err);
      setLoading(false);
    });
  }, [selectedRegion]);

  const metrics = overview?.metrics || {};
  const gaps = overview?.projected_gaps || {};

  return (
    <div className="space-y-6">
      {/* Top Header & Filter */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <Stethoscope className="text-teal-400" size={22} />
            Healthcare Capacity & Preparedness Workstation
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Hospital beds, ICU capacity, physician density, dynamic stress scoring, and 7-day resource gap estimation
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-300">
          <span>District:</span>
          <select
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
          >
            {regions.map((r) => <option key={r} value={r}>{r}</option>)}
          </select>
        </div>
      </div>

      {/* Capacity & Stress KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Staffed Hospital Beds</span>
            <Bed size={16} className="text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {metrics.hospital_beds?.toLocaleString() || '--'}
          </div>
          <div className="text-[11px] text-teal-400 font-mono mt-1">
            {metrics.beds_per_100k || '--'} beds / 100k
          </div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>ICU Critical Beds</span>
            <HeartPulse size={16} className="text-rose-400" />
          </div>
          <div className="text-2xl font-bold text-white mt-1">
            {metrics.icu_beds?.toLocaleString() || '--'}
          </div>
          <div className="text-[11px] text-rose-400 font-mono mt-1">
            {metrics.icu_beds_per_100k || '--'} ICU / 100k
          </div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Healthcare Stress Load</span>
            <Activity size={16} className="text-orange-400" />
          </div>
          <div className="text-2xl font-bold text-orange-400 mt-1">
            {metrics.healthcare_stress_score ? Number(metrics.healthcare_stress_score).toFixed(1) : '--'} / 10
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Capacity utilization</div>
        </div>

        <div className="bg-slate-950/80 p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Preparedness Index</span>
            <ShieldCheck size={16} className="text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {metrics.healthcare_preparedness_score ? Number(metrics.healthcare_preparedness_score).toFixed(1) : '--'}%
          </div>
          <div className="text-[11px] text-slate-400 mt-1">Oxygen & Medical Readiness</div>
        </div>
      </div>

      {/* Hospital Resource Gap & Projection Table */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center justify-between">
          <span className="flex items-center gap-2">
            <AlertTriangle size={16} className="text-amber-400" />
            Projected 7-Day Patient Surge & Resource Gap Assessment ({selectedRegion})
          </span>
          <span className="text-xs text-slate-400 font-mono">Dynamic Model</span>
        </h2>
        <p className="text-xs text-slate-400 mb-4">
          Estimates future inpatient and ICU admissions based on projected 7-day disease trajectory vs available staffed reserves
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
            <span className="text-[11px] text-slate-400">Projected 7D Patient Surge</span>
            <div className="text-lg font-bold text-white mt-0.5">{gaps.projected_patient_surge_7d || 150} cases</div>
          </div>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
            <span className="text-[11px] text-slate-400">Projected Bed Deficit</span>
            <div className={`text-lg font-bold mt-0.5 ${gaps.projected_bed_deficit > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
              {gaps.projected_bed_deficit > 0 ? `-${gaps.projected_bed_deficit} beds` : 'None (Surplus)'}
            </div>
          </div>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
            <span className="text-[11px] text-slate-400">Oxygen Stock Status</span>
            <div className="text-lg font-bold text-teal-400 mt-0.5">{gaps.oxygen_security_status || 'SECURE'}</div>
          </div>
          <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
            <span className="text-[11px] text-slate-400">Medicine Availability</span>
            <div className="text-lg font-bold text-cyan-400 mt-0.5">{gaps.medicine_stock_status || 'ADEQUATE'}</div>
          </div>
        </div>
      </div>

      {/* Regional Capacity Comparison Chart */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider mb-2 flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Layers size={16} className="text-teal-400" />
            Cross-District Hospital Beds & Doctor Density (Per 100k Population)
          </span>
          <span className="text-xs text-slate-400 font-mono">All 5 Pilot Districts</span>
        </h2>
        <div className="h-64 w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={allCapacity}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="region" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              <Bar dataKey="beds_per_100k" fill="#0ea5e9" name="Beds / 100k" radius={[4, 4, 0, 0]} />
              <Bar dataKey="doctors_per_100k" fill="#10b981" name="Doctors / 100k" radius={[4, 4, 0, 0]} />
              <Bar dataKey="icu_beds_per_100k" fill="#f43f5e" name="ICU / 100k" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

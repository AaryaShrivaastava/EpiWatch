import { useEffect, useState } from 'react';
import { 
  fetchRegions, 
  fetchRecommendations, 
  generateRecommendations 
} from '../services/apiClient';
import { 
  CheckSquare, 
  Users, 
  Stethoscope, 
  Landmark, 
  Sparkles, 
  Play, 
  ShieldCheck, 
  Info,
  Calendar
} from 'lucide-react';

export default function Recommendations() {
  const [regions, setRegions] = useState<string[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string>('Bhopal');
  const [selectedStakeholder, setSelectedStakeholder] = useState<string>('ALL');

  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [generating, setGenerating] = useState<boolean>(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  useEffect(() => {
    fetchRegions().then((regs) => {
      setRegions(regs);
      if (regs.length > 0) setSelectedRegion(regs[0]);
    }).catch(console.error);
  }, []);

  const loadRecs = async () => {
    if (!selectedRegion) return;
    setLoading(true);
    try {
      const stakeholder = selectedStakeholder === 'ALL' ? undefined : selectedStakeholder;
      const data = await fetchRecommendations(selectedRegion, stakeholder);
      setRecommendations(data || []);
    } catch (err) {
      console.error('Error loading recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRecs();
  }, [selectedRegion, selectedStakeholder]);

  const handleGenerate = async () => {
    setGenerating(true);
    setFeedback(null);
    try {
      const res = await generateRecommendations(selectedRegion);
      setFeedback(`Generated ${res.recommendations_count} fresh evidence-grounded directives for ${selectedRegion}`);
      await loadRecs();
    } catch (err) {
      setFeedback('Failed to execute recommendation rules matrix.');
    } finally {
      setGenerating(false);
    }
  };

  const getPriorityStyle = (priority: string) => {
    switch (priority?.toUpperCase()) {
      case 'HIGH':
      case 'CRITICAL':
        return 'bg-red-500/15 text-red-400 border border-red-500/30';
      case 'MODERATE':
        return 'bg-amber-500/15 text-amber-400 border border-amber-500/30';
      default:
        return 'bg-teal-500/15 text-teal-300 border border-teal-500/30';
    }
  };

  const getStakeholderIcon = (sh: string) => {
    switch (sh?.toLowerCase()) {
      case 'health authorities':
        return <Landmark size={16} className="text-purple-400" />;
      case 'healthcare facilities':
        return <Stethoscope size={16} className="text-teal-400" />;
      default:
        return <Users size={16} className="text-cyan-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Action Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            <CheckSquare className="text-teal-400" size={22} />
            Evidence-Based Public Health Action Directives
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Operational recommendations mapped to risk scoring, hospital surge capacity, and transmission velocity
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
            <span>Audience:</span>
            <select
              value={selectedStakeholder}
              onChange={(e) => setSelectedStakeholder(e.target.value)}
              className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-semibold text-white focus:outline-none focus:border-teal-400"
            >
              <option value="ALL">All Stakeholders</option>
              <option value="Health Authorities">Health Authorities</option>
              <option value="Healthcare Facilities">Healthcare Facilities</option>
              <option value="Citizens">Citizens / Public</option>
            </select>
          </div>

          <button
            onClick={handleGenerate}
            disabled={generating}
            className="flex items-center gap-2 px-3.5 py-1.5 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/40 text-teal-300 rounded-lg text-xs font-semibold transition-all shadow-sm"
          >
            <Play size={13} className={generating ? 'animate-spin' : ''} />
            {generating ? 'Evaluating Rules...' : 'Re-Evaluate Decision Matrix'}
          </button>
        </div>
      </div>

      {feedback && (
        <div className="p-3 rounded-lg bg-teal-500/15 border border-teal-500/30 text-teal-300 text-xs flex items-center gap-2">
          <Sparkles size={16} />
          {feedback}
        </div>
      )}

      {/* Stakeholder Category Tabs */}
      <div className="flex gap-2 border-b border-slate-800 pb-2">
        {['ALL', 'Health Authorities', 'Healthcare Facilities', 'Citizens'].map((sh) => (
          <button
            key={sh}
            onClick={() => setSelectedStakeholder(sh)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              selectedStakeholder === sh
                ? 'bg-slate-800 text-white font-semibold border border-slate-700'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {sh === 'ALL' ? 'All Target Audiences' : sh}
          </button>
        ))}
      </div>

      {/* Recommendations Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {loading ? (
          <div className="col-span-2 text-center py-12 text-xs text-slate-400">Loading public health directives...</div>
        ) : recommendations.length === 0 ? (
          <div className="col-span-2 p-8 text-center bg-slate-950/60 rounded-xl border border-slate-800 text-slate-400 text-xs">
            No directives generated for {selectedRegion} under the selected filters.
          </div>
        ) : (
          recommendations.map((rec, idx) => (
            <div 
              key={idx} 
              className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 flex flex-col justify-between shadow-sm hover:border-slate-700 transition-all"
            >
              <div>
                <div className="flex items-start justify-between gap-3 mb-2">
                  <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
                    {getStakeholderIcon(rec.target_stakeholder)}
                    <span>{rec.target_stakeholder}</span>
                  </div>
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-bold uppercase ${getPriorityStyle(rec.priority)}`}>
                    {rec.priority} Priority
                  </span>
                </div>

                <h3 className="font-bold text-sm text-white mt-1 leading-snug">
                  {rec.recommendation_title}
                </h3>

                <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                  {rec.rationale}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-800/80 space-y-2 text-[11px]">
                {rec.evidence_summary && (
                  <div className="p-2.5 rounded bg-slate-900 border border-slate-800/80 text-slate-300">
                    <strong className="text-teal-400 font-mono">Triggering Evidence: </strong>
                    {rec.evidence_summary}
                  </div>
                )}
                <div className="flex justify-between items-center text-slate-500 font-mono text-[10px]">
                  <span>District: <strong className="text-slate-400">{rec.region}</strong></span>
                  <span className="flex items-center gap-1">
                    <Calendar size={11} /> {rec.generation_date}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

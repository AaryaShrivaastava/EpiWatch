import { useEffect, useState, useMemo } from 'react';
import { 
  MapContainer, 
  TileLayer, 
  GeoJSON, 
  CircleMarker, 
  Tooltip as LeafletTooltip, 
  useMap 
} from 'react-leaflet';
import type { Layer } from 'leaflet';
import { 
  fetchSpatialRiskMap, 
  fetchSpatialBoundaries, 
  fetchSpatialClusters, 
  runSpatialCluster,
  type SpatialFilterParams 
} from '../services/apiClient';
import { 
  Map as MapIcon, 
  Layers, 
  AlertOctagon, 
  ShieldCheck, 
  Activity, 
  Stethoscope, 
  Trees, 
  TrendingUp, 
  Info, 
  RotateCcw, 
  SlidersHorizontal,
  ChevronRight,
  Maximize2,
  Calendar,
  Filter
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend 
} from 'recharts';

// Center of Madhya Pradesh
const MP_CENTER: [number, number] = [23.55, 77.80];
const MP_DEFAULT_ZOOM = 7;

function MapController({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom, { animate: true });
  }, [center, zoom, map]);
  return null;
}

export default function Spatial() {
  const [regions, setRegions] = useState<any[]>([]);
  const [boundaries, setBoundaries] = useState<any>(null);
  const [clusterData, setClusterData] = useState<any>(null);
  const [selectedRegion, setSelectedRegion] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  // Filters
  const [selectedDisease, setSelectedDisease] = useState<string>('All');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('All');
  const [selectedCluster, setSelectedCluster] = useState<string>('All');
  const [showBoundaries, setShowBoundaries] = useState<boolean>(true);
  const [showCircles, setShowCircles] = useState<boolean>(true);
  const [circleMetric, setCircleMetric] = useState<'case_count' | 'population'>('case_count');

  // Clustering modal state
  const [isClustering, setIsClustering] = useState<boolean>(false);
  const [clusterK, setClusterK] = useState<number>(3);
  const [clusterMsg, setClusterMsg] = useState<string>('');

  const diseasesList = ['All', 'COVID-19', 'Dengue', 'Malaria', 'Influenza', 'Typhoid'];
  const riskLevels = ['All', 'Low', 'Moderate', 'High', 'Critical'];

  // Load Data
  const loadData = () => {
    setLoading(true);
    const params: SpatialFilterParams = {};
    if (selectedDisease !== 'All') params.disease = selectedDisease;
    if (selectedRiskLevel !== 'All') params.risk_level = selectedRiskLevel;
    if (selectedCluster !== 'All') params.cluster_id = parseInt(selectedCluster, 10);

    Promise.all([
      fetchSpatialRiskMap(params),
      fetchSpatialBoundaries(),
      fetchSpatialClusters()
    ])
      .then(([regs, bnds, cls]) => {
        const regList = regs || [];
        setRegions(regList);
        setBoundaries(bnds);
        setClusterData(cls);

        // Keep or update selected region
        if (selectedRegion) {
          const updated = regList.find((r: any) => r.region_name === selectedRegion.region_name);
          if (updated) setSelectedRegion(updated);
        } else if (regList.length > 0) {
          setSelectedRegion(regList[0]);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error loading spatial data:', err);
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
  }, [selectedDisease, selectedRiskLevel, selectedCluster]);

  const handleResetMap = () => {
    setSelectedDisease('All');
    setSelectedRiskLevel('All');
    setSelectedCluster('All');
  };

  const handleRecluster = async () => {
    setIsClustering(true);
    setClusterMsg('Running K-Means algorithm on standardized indicators...');
    try {
      const res = await runSpatialCluster(clusterK, 'kmeans');
      if (res.status === 'success') {
        setClusterMsg(`K-Means successfully clustered regions (k=${clusterK}, inertia=${res.inertia}).`);
        loadData();
      } else {
        setClusterMsg(res.message || 'Clustering failed.');
      }
    } catch (e: any) {
      setClusterMsg(`Error running clustering: ${e.message}`);
    } finally {
      setIsClustering(false);
    }
  };

  // GeoJSON styling function
  const getGeoJsonStyle = (feature: any) => {
    const props = feature?.properties || {};
    const rName = props.region_name;
    const match = regions.find((r) => r.region_name.toLowerCase() === rName?.toLowerCase());
    const isSelected = selectedRegion?.region_name.toLowerCase() === rName?.toLowerCase();
    const color = match ? match.risk_color : props.risk_color || '#10b981';

    return {
      fillColor: color,
      weight: isSelected ? 3 : 1.5,
      opacity: 1,
      color: isSelected ? '#ffffff' : '#334155',
      dashArray: isSelected ? '' : '3',
      fillOpacity: isSelected ? 0.65 : 0.4
    };
  };

  // Radius calculation
  const getCircleRadius = (r: any) => {
    if (circleMetric === 'case_count') {
      const cases = r.case_count || 10;
      return Math.max(14, Math.min(42, Math.sqrt(cases) * 4.5));
    } else {
      const pop = r.population || 2000000;
      return Math.max(14, Math.min(40, Math.sqrt(pop / 100000) * 5));
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Header & Provenance */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-950/80 p-4 rounded-xl border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <MapIcon className="text-teal-400" size={22} />
              Interactive Spatial Risk Map & Clustering
            </h1>
            <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold bg-teal-500/20 text-teal-300 border border-teal-500/30">
              Madhya Pradesh Administrative Division
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Color-coded regional risk assessment, Census 2011 anchors, and spatial vulnerability clustering across 5 central India districts.
          </p>
        </div>

        {/* Data Provenance Badge */}
        <div className="flex items-center gap-3">
          <div className="text-right hidden sm:block">
            <div className="text-[11px] text-slate-300 font-mono font-semibold">Survey of India / Census 2011</div>
            <div className="text-[10px] text-slate-500">Synthetic biological model calibration</div>
          </div>
          <button
            onClick={handleResetMap}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-xs text-slate-300 border border-slate-700 transition-colors"
          >
            <RotateCcw size={13} />
            <span>Reset View</span>
          </button>
        </div>
      </div>

      {/* Map Filter Controls Bar */}
      <div className="bg-slate-950/90 p-3.5 rounded-xl border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          {/* Disease Selector */}
          <div className="flex items-center gap-1.5">
            <Filter size={14} className="text-teal-400" />
            <span className="text-slate-400 font-medium">Disease:</span>
            <select
              value={selectedDisease}
              onChange={(e) => setSelectedDisease(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-teal-500"
            >
              {diseasesList.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Risk Level Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">Risk Level:</span>
            <select
              value={selectedRiskLevel}
              onChange={(e) => setSelectedRiskLevel(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-teal-500"
            >
              {riskLevels.map((lvl) => (
                <option key={lvl} value={lvl}>{lvl}</option>
              ))}
            </select>
          </div>

          {/* Cluster Selector */}
          <div className="flex items-center gap-1.5">
            <span className="text-slate-400 font-medium">Cluster:</span>
            <select
              value={selectedCluster}
              onChange={(e) => setSelectedCluster(e.target.value)}
              className="bg-slate-900 border border-slate-700 text-slate-200 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-teal-500"
            >
              <option value="All">All Clusters</option>
              <option value="0">Cluster 0 (High Urban Density)</option>
              <option value="1">Cluster 1 (Urban Core)</option>
              <option value="2">Cluster 2 (Semi-Urban Buffer)</option>
            </select>
          </div>
        </div>

        {/* Layer Toggles & Metric Scaling */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showBoundaries}
              onChange={(e) => setShowBoundaries(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-teal-500 focus:ring-0"
            />
            <span>District Polygons</span>
          </label>

          <label className="flex items-center gap-1.5 text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showCircles}
              onChange={(e) => setShowCircles(e.target.checked)}
              className="rounded bg-slate-900 border-slate-700 text-teal-500 focus:ring-0"
            />
            <span>Risk Circles</span>
          </label>

          <div className="flex items-center gap-1 text-[11px] bg-slate-900 px-2 py-1 rounded border border-slate-800 text-slate-400">
            <span>Radius:</span>
            <button
              onClick={() => setCircleMetric('case_count')}
              className={`px-1.5 py-0.5 rounded ${circleMetric === 'case_count' ? 'bg-teal-500/20 text-teal-300 font-bold' : 'hover:text-white'}`}
            >
              Cases
            </button>
            <span>/</span>
            <button
              onClick={() => setCircleMetric('population')}
              className={`px-1.5 py-0.5 rounded ${circleMetric === 'population' ? 'bg-teal-500/20 text-teal-300 font-bold' : 'hover:text-white'}`}
            >
              Pop
            </button>
          </div>
        </div>
      </div>

      {/* Main Map & Detail Panel Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Large Interactive Leaflet Map (Occupies 8 Cols) */}
        <div className="lg:col-span-8 bg-slate-950/90 rounded-xl border border-slate-800 overflow-hidden relative flex flex-col shadow-xl">
          <div className="h-[540px] w-full z-0 relative">
            <MapContainer
              center={MP_CENTER}
              zoom={MP_DEFAULT_ZOOM}
              scrollWheelZoom={true}
              className="h-full w-full"
              style={{ backgroundColor: '#020617' }}
            >
              <MapController center={MP_CENTER} zoom={MP_DEFAULT_ZOOM} />
              
              {/* Dark Carto Tile Layer */}
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                maxZoom={18}
              />

              {/* GeoJSON District Boundaries */}
              {showBoundaries && boundaries && (
                <GeoJSON
                  key={`geojson-${selectedRegion?.region_name}-${selectedDisease}-${selectedRiskLevel}`}
                  data={boundaries}
                  style={getGeoJsonStyle}
                  onEachFeature={(feature: any, layer: Layer) => {
                    const rName = feature?.properties?.region_name;
                    layer.on({
                      click: () => {
                        const match = regions.find((r) => r.region_name.toLowerCase() === rName?.toLowerCase());
                        if (match) setSelectedRegion(match);
                      }
                    });
                  }}
                />
              )}

              {/* Scaled Risk Circles with Coordinates */}
              {showCircles &&
                regions.map((r) => (
                  <CircleMarker
                    key={r.region_id}
                    center={[r.latitude, r.longitude]}
                    radius={getCircleRadius(r)}
                    pathOptions={{
                      fillColor: r.risk_color,
                      fillOpacity: 0.85,
                      color: selectedRegion?.region_id === r.region_id ? '#ffffff' : '#020617',
                      weight: selectedRegion?.region_id === r.region_id ? 3 : 1.5
                    }}
                    eventHandlers={{
                      click: () => setSelectedRegion(r)
                    }}
                  >
                    <LeafletTooltip direction="top" offset={[0, -10]} opacity={0.95}>
                      <div className="p-1 text-xs">
                        <div className="font-bold text-slate-900">{r.region_name}</div>
                        <div className="text-slate-600">Risk: <strong style={{ color: r.risk_color }}>{r.risk_category} ({r.risk_score})</strong></div>
                        <div className="text-slate-600">7d Cases: <strong>{r.case_count}</strong></div>
                        <div className="text-[10px] text-slate-500">Cluster: {r.cluster_label}</div>
                      </div>
                    </LeafletTooltip>
                  </CircleMarker>
                ))}
            </MapContainer>

            {/* In-Map Floating Legend */}
            <div className="absolute bottom-4 left-4 z-10 bg-slate-950/90 backdrop-blur-md p-3 rounded-lg border border-slate-800 text-xs shadow-lg space-y-2 pointer-events-auto">
              <div className="font-bold text-slate-200 text-[11px] uppercase tracking-wider">
                Risk Classification Palette
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px]">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50"></span>
                  <span className="text-slate-300">Low Risk (&lt;25)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-yellow-500 shadow-sm shadow-yellow-500/50"></span>
                  <span className="text-slate-300">Moderate (25-54)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-orange-500 shadow-sm shadow-orange-500/50"></span>
                  <span className="text-slate-300">High Risk (55-74)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-red-500 shadow-sm shadow-red-500/50"></span>
                  <span className="text-slate-300">Critical (&ge;75)</span>
                </div>
              </div>
              <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-800/80">
                Circle Radius: Scaled by {circleMetric === 'case_count' ? '7-Day Active Cases' : 'Census Population'}
              </div>
            </div>
          </div>
        </div>

        {/* Regional Summary & Deep-Dive Panel (4 Cols) */}
        <div className="lg:col-span-4 space-y-4">
          {selectedRegion ? (
            <div className="bg-slate-950/90 rounded-xl border border-slate-800 p-5 space-y-4 shadow-xl">
              {/* Region Header */}
              <div className="flex items-start justify-between pb-3 border-b border-slate-800/80">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-lg font-bold text-white">{selectedRegion.region_name}</h2>
                    <span className="text-[10px] font-mono bg-slate-900 px-2 py-0.5 rounded text-slate-400 border border-slate-800">
                      {selectedRegion.region_id}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {selectedRegion.state}, India • {selectedRegion.selected_disease}
                  </div>
                </div>

                <div className="text-right">
                  <span
                    className="inline-block px-2.5 py-1 rounded text-xs font-bold uppercase tracking-wider"
                    style={{
                      backgroundColor: `${selectedRegion.risk_color}25`,
                      color: selectedRegion.risk_color,
                      border: `1px solid ${selectedRegion.risk_color}50`
                    }}
                  >
                    {selectedRegion.risk_category} Risk
                  </span>
                  <div className="text-[10px] text-slate-400 font-mono mt-1">
                    Score: <strong>{selectedRegion.risk_score}</strong>/100
                  </div>
                </div>
              </div>

              {/* Primary Surveillance Metrics */}
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium">Recent 7d Cases</div>
                  <div className="text-base font-bold text-white font-mono mt-0.5">{selectedRegion.case_count}</div>
                  <div className={`text-[10px] font-medium flex items-center gap-1 mt-0.5 ${selectedRegion.case_growth >= 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                    <TrendingUp size={11} /> {selectedRegion.case_growth_percent}% 7d change
                  </div>
                </div>

                <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium">Outbreak Status</div>
                  <div className="text-sm font-bold text-teal-300 font-mono mt-1 flex items-center gap-1">
                    <AlertOctagon size={14} />
                    {selectedRegion.outbreak_status}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    Severity: {selectedRegion.outbreak_severity}
                  </div>
                </div>

                <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium">14d Case Forecast</div>
                  <div className="text-base font-bold text-cyan-300 font-mono mt-0.5">
                    {selectedRegion.forecast_cases_14d !== null ? selectedRegion.forecast_cases_14d : 'Unavailable'}
                  </div>
                  <div className="text-[10px] text-slate-500">XGBoost Horizon 14</div>
                </div>

                <div className="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-medium">Preparedness Score</div>
                  <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
                    {selectedRegion.preparedness_score}%
                  </div>
                  <div className="text-[10px] text-slate-500">Stress: {selectedRegion.healthcare_stress}/10</div>
                </div>
              </div>

              {/* Environmental & Demographics Context */}
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 space-y-2 text-xs">
                <div className="text-slate-300 font-semibold flex items-center gap-1.5">
                  <Trees size={14} className="text-emerald-400" />
                  <span>Demographics & Environmental Vulnerability</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400">
                  <div>Population: <strong className="text-slate-200">{selectedRegion.population?.toLocaleString()}</strong></div>
                  <div>Density: <strong className="text-slate-200">{selectedRegion.population_density}/km²</strong></div>
                  <div>Urbanization: <strong className="text-slate-200">{selectedRegion.urbanization_percent}%</strong></div>
                  <div>Env Risk Score: <strong className="text-slate-200">{selectedRegion.environmental_risk_score}</strong></div>
                </div>
                <div className="text-[11px] text-slate-400 pt-1 border-t border-slate-800/60">
                  Primary Transmission Driver: <span className="text-teal-300">{selectedRegion.primary_driver}</span>
                </div>
              </div>

              {/* Cluster Profile Summary */}
              <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800/80 text-xs space-y-1.5">
                <div className="flex items-center justify-between text-slate-300 font-semibold">
                  <span>Cluster Profile</span>
                  <span className="font-mono text-purple-400 text-[11px]">Cluster {selectedRegion.cluster_id}</span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {selectedRegion.cluster_label}
                </p>
              </div>

              {/* Disease Breakdown Table */}
              {selectedRegion.disease_breakdown && Object.keys(selectedRegion.disease_breakdown).length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-slate-300 text-xs font-semibold">Multi-Pathogen Surveillance Breakdown</div>
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-left text-[11px]">
                      <thead className="text-slate-500 font-mono border-b border-slate-800">
                        <tr>
                          <th className="py-1">Disease</th>
                          <th className="py-1">Cases</th>
                          <th className="py-1">Positivity</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-slate-300 font-mono">
                        {Object.entries(selectedRegion.disease_breakdown).map(([dis, data]: any) => (
                          <tr key={dis}>
                            <td className="py-1 font-sans">{dis}</td>
                            <td className="py-1 text-teal-300">{data.recent_cases}</td>
                            <td className="py-1 text-slate-400">{data.positivity_rate}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Metadata Provenance Footnote */}
              <div className="text-[10px] text-slate-500 border-t border-slate-800/80 pt-2 space-y-0.5">
                <div>Provenance: {selectedRegion.data_source}</div>
                <div>Completeness: {selectedRegion.data_completeness}</div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-8 text-center text-slate-400 text-xs">
              Click any district polygon or circle marker on the map to inspect regional risk indicators.
            </div>
          )}
        </div>
      </div>

      {/* Cluster Profiles Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-purple-400 uppercase font-bold">Cluster 0</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-red-500/20 text-red-300 font-semibold uppercase">High Vigilance</span>
          </div>
          <h3 className="font-bold text-sm text-white">High Urban Density — Elevated Transmission</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Districts characterized by population densities &gt;800/km² and high urbanization (&gt;70%). Requires aggressive vector abatement and early clinic surveillance.
          </p>
          <div className="text-xs text-slate-300 font-mono pt-1">
            Districts: <strong className="text-white">Bhopal, Indore</strong>
          </div>
        </div>

        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-cyan-400 uppercase font-bold">Cluster 1</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 font-semibold uppercase">Moderate Load</span>
          </div>
          <h3 className="font-bold text-sm text-white">Urban Core — Balanced Infrastructure</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Significant urban centers with density ~450/km² and balanced hospital bed ratios. Stable containment capabilities with localized seasonal monitoring.
          </p>
          <div className="text-xs text-slate-300 font-mono pt-1">
            Districts: <strong className="text-white">Jabalpur, Gwalior</strong>
          </div>
        </div>

        <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono text-teal-400 uppercase font-bold">Cluster 2</span>
            <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-semibold uppercase">Resilient</span>
          </div>
          <h3 className="font-bold text-sm text-white">Semi-Urban / Agricultural Buffer</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Lower density (~320/km²) with lower baseline incidence, requiring vigilant rural primary health center supply chains during monsoon vector seasons.
          </p>
          <div className="text-xs text-slate-300 font-mono pt-1">
            Districts: <strong className="text-white">Ujjain</strong>
          </div>
        </div>
      </div>

      {/* Cross-District Density vs Incidence Comparison Chart */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Activity size={16} className="text-teal-400" />
            <span>District Spatial Vulnerability & Incidence Correlation</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">Epidemiological Spatial Features</span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={regions}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="region_name" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <RechartsTooltip contentStyle={{ backgroundColor: '#020617', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }} />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              <Bar dataKey="population_density" fill="#0ea5e9" name="Density (/km²)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="risk_score" fill="#f59e0b" name="Risk Score (0-100)" radius={[4, 4, 0, 0]} />
              <Bar dataKey="case_count" fill="#10b981" name="7d Cases" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Hotspots & Re-Clustering Control Section */}
      <div className="bg-slate-950/80 p-5 rounded-xl border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h3 className="font-bold text-sm text-white">Spatial Clustering vs Hotspot Statistical Disclaimers</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
            Ordinary K-Means and DBSCAN clustering partition regional vulnerability profiles using standardized demographic and clinical vectors.
            This does not constitute statistically significant spatial autocorrelation hotspot detection (e.g. Moran’s I or Getis-Ord Gi*), which requires higher observation counts (N &ge; 10).
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-1.5 text-xs text-slate-300">
            <span>Clusters (k):</span>
            <select
              value={clusterK}
              onChange={(e) => setClusterK(parseInt(e.target.value, 10))}
              className="bg-slate-900 border border-slate-700 text-white rounded px-2 py-1 text-xs"
            >
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4</option>
            </select>
          </div>
          <button
            onClick={handleRecluster}
            disabled={isClustering}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-teal-600 hover:bg-teal-500 disabled:opacity-50 text-white font-medium text-xs rounded-lg transition-colors"
          >
            <SlidersHorizontal size={13} />
            <span>{isClustering ? 'Clustering...' : 'Re-Run K-Means'}</span>
          </button>
        </div>
      </div>
      {clusterMsg && (
        <div className="text-xs text-teal-300 font-mono bg-teal-950/40 p-2.5 rounded-lg border border-teal-800/40">
          {clusterMsg}
        </div>
      )}
    </div>
  );
}

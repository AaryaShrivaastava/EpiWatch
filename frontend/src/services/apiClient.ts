import axios from 'axios';

// Ensure apiClient points to /api/v1 (or fallback to /api)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor to handle errors gracefully
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.warn('API Client Error:', error.config?.url, error.message);
    return Promise.reject(error);
  }
);

// ---------------------------------------------------------------------------
// System & Metadata
// ---------------------------------------------------------------------------
export const fetchHealth = async () => {
  const res = await apiClient.get('/health');
  return res.data.data;
};

export const fetchSystemStatus = async () => {
  const res = await apiClient.get('/system/status');
  return res.data.data;
};

export const fetchRegions = async (): Promise<string[]> => {
  const res = await apiClient.get('/data/regions');
  return res.data.data || ['Bhopal', 'Indore', 'Jabalpur', 'Gwalior', 'Ujjain'];
};

export const fetchDiseases = async (): Promise<string[]> => {
  const res = await apiClient.get('/data/diseases');
  return res.data.data || ['COVID-19', 'Dengue', 'Malaria', 'Influenza', 'Typhoid'];
};

// ---------------------------------------------------------------------------
// Dashboard & Analytics
// ---------------------------------------------------------------------------
export const fetchDashboardSummary = async (region: string = 'Bhopal') => {
  const res = await apiClient.get(`/dashboard/summary?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

export const fetchRegionRisk = async (regionId: string) => {
  const res = await apiClient.get(`/risk/regions/${encodeURIComponent(regionId)}`);
  return res.data.data;
};

export const fetchDashboardRegionalRisk = async () => {
  const res = await apiClient.get('/dashboard/regional-risk');
  return res.data.data;
};

export const fetchAnalyticsTrends = async (region: string, disease: string, days: number = 60) => {
  const res = await apiClient.get(`/analytics/trends?region=${encodeURIComponent(region)}&disease=${encodeURIComponent(disease)}&days=${days}`);
  return res.data.data;
};

export const fetchAnalyticsOverview = async (region: string) => {
  const res = await apiClient.get(`/analytics/overview?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

export const fetchAnalyticsSeasonality = async (region: string, disease: string) => {
  const res = await apiClient.get(`/analytics/seasonality?region=${encodeURIComponent(region)}&disease=${encodeURIComponent(disease)}`);
  return res.data.data;
};

// ---------------------------------------------------------------------------
// Outbreak Detection
// ---------------------------------------------------------------------------
export const fetchOutbreaksHistory = async (region?: string, disease?: string, limit: number = 100) => {
  let url = `/outbreaks/history?limit=${limit}`;
  if (region) url += `&region=${encodeURIComponent(region)}`;
  if (disease) url += `&disease=${encodeURIComponent(disease)}`;
  const res = await apiClient.get(url);
  return res.data.data;
};

export const fetchOutbreaksSummary = async (region: string = 'Bhopal') => {
  const res = await apiClient.get(`/outbreaks/summary?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

export const triggerOutbreakDetection = async (region: string, disease: string) => {
  const res = await apiClient.post('/outbreaks/detect', { region, disease });
  return res.data;
};

// ---------------------------------------------------------------------------
// Forecasting
// ---------------------------------------------------------------------------
export const fetchForecastResults = async (region: string, disease: string, horizon: number = 14) => {
  const res = await apiClient.get(`/forecast/results?region=${encodeURIComponent(region)}&disease=${encodeURIComponent(disease)}&horizon=${horizon}`);
  return res.data.data;
};

export const fetchForecastModels = async () => {
  const res = await apiClient.get('/forecast/models');
  return res.data.data;
};

export const fetchForecastEvaluation = async (region: string = 'Bhopal') => {
  const res = await apiClient.get(`/forecast/evaluation?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

export const runForecastSimulation = async (region: string, disease: string, horizon: number, model: string) => {
  const res = await apiClient.post('/forecast/run', { region, disease, horizon, model });
  return res.data;
};

// ---------------------------------------------------------------------------
// Healthcare Preparedness & Capacity
// ---------------------------------------------------------------------------
export const fetchHealthcareOverview = async (region: string = 'Bhopal') => {
  const res = await apiClient.get(`/healthcare/overview?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

export const fetchHealthcareCapacity = async () => {
  const res = await apiClient.get('/healthcare/capacity');
  return res.data.data;
};

export const fetchHealthcarePreparedness = async (region: string = 'Bhopal') => {
  const res = await apiClient.get(`/healthcare/preparedness?region=${encodeURIComponent(region)}`);
  return res.data.data;
};

// ---------------------------------------------------------------------------
// Spatial Analytics & Regional Risk
// ---------------------------------------------------------------------------
export interface SpatialFilterParams {
  disease?: string;
  risk_level?: string;
  cluster_id?: number;
  start_date?: string;
  end_date?: string;
}

export const fetchSpatialRegions = async () => {
  const res = await apiClient.get('/spatial/regions');
  return res.data.data;
};

export const fetchSpatialRiskMap = async (params?: SpatialFilterParams) => {
  let url = '/spatial/risk-map';
  if (params) {
    const q: string[] = [];
    if (params.disease) q.push(`disease=${encodeURIComponent(params.disease)}`);
    if (params.risk_level) q.push(`risk_level=${encodeURIComponent(params.risk_level)}`);
    if (params.cluster_id !== undefined) q.push(`cluster_id=${params.cluster_id}`);
    if (params.start_date) q.push(`start_date=${encodeURIComponent(params.start_date)}`);
    if (params.end_date) q.push(`end_date=${encodeURIComponent(params.end_date)}`);
    if (q.length > 0) url += `?${q.join('&')}`;
  }
  const res = await apiClient.get(url);
  return res.data.data;
};

export const fetchSpatialBoundaries = async () => {
  const res = await apiClient.get('/spatial/boundaries');
  return res.data;
};

export const fetchSpatialClusters = async () => {
  const res = await apiClient.get('/spatial/clusters');
  return res.data;
};

export const runSpatialCluster = async (n_clusters: number = 3, method: string = 'kmeans') => {
  const res = await apiClient.post('/spatial/cluster', { n_clusters, method });
  return res.data;
};

export const fetchSpatialRegionDetail = async (regionId: string, disease?: string) => {
  let url = `/spatial/regions/${encodeURIComponent(regionId)}`;
  if (disease) url += `?disease=${encodeURIComponent(disease)}`;
  const res = await apiClient.get(url);
  return res.data.data;
};

export const fetchSpatialHotspots = async () => {
  const res = await apiClient.get('/spatial/hotspots');
  return res.data.data;
};

// ---------------------------------------------------------------------------
// Active Epidemiological Surveillance Engine
// ---------------------------------------------------------------------------
export const fetchSurveillanceStatus = async () => {
  const res = await apiClient.get('/surveillance/status');
  return res.data.data;
};

export const startSurveillance = async () => {
  const res = await apiClient.post('/surveillance/start');
  return res.data;
};

export const stopSurveillance = async () => {
  const res = await apiClient.post('/surveillance/stop');
  return res.data;
};

export const triggerSurveillanceScan = async () => {
  const res = await apiClient.post('/surveillance/scan');
  return res.data.data;
};

export const fetchSurveillanceScans = async () => {
  const res = await apiClient.get('/surveillance/scans');
  return res.data.data;
};

export const fetchSurveillanceFindings = async (filters?: {
  region?: string;
  disease?: string;
  severity?: string;
}) => {
  let url = '/surveillance/findings';
  if (filters) {
    const q: string[] = [];
    if (filters.region) q.push(`region=${encodeURIComponent(filters.region)}`);
    if (filters.disease) q.push(`disease=${encodeURIComponent(filters.disease)}`);
    if (filters.severity) q.push(`severity=${encodeURIComponent(filters.severity)}`);
    if (q.length > 0) url += `?${q.join('&')}`;
  }
  const res = await apiClient.get(url);
  return res.data.data;
};

export const fetchSurveillanceFindingDetail = async (findingId: string) => {
  const res = await apiClient.get(`/surveillance/findings/${encodeURIComponent(findingId)}`);
  return res.data.data;
};

export const fetchSurveillanceStatistics = async () => {
  const res = await apiClient.get('/surveillance/statistics');
  return res.data.data;
};

export const fetchSurveillanceConfig = async () => {
  const res = await apiClient.get('/surveillance/config');
  return res.data.data;
};

export const updateSurveillanceConfig = async (payload: {
  scan_interval_seconds?: number;
  monitored_regions?: string[];
  monitored_diseases?: string[];
  thresholds?: Record<string, number>;
}) => {
  const res = await apiClient.patch('/surveillance/config', payload);
  return res.data;
};

export const fetchSurveillanceTimeline = async (limit: number = 50) => {
  const res = await apiClient.get(`/surveillance/timeline?limit=${limit}`);
  return res.data.data;
};

// ---------------------------------------------------------------------------
// Recommendations & Alerts
// ---------------------------------------------------------------------------
export const fetchRecommendations = async (regionId?: string, stakeholder?: string) => {
  let url = '/recommendations';
  const params: string[] = [];
  if (regionId) params.push(`region_id=${encodeURIComponent(regionId)}`);
  if (stakeholder) params.push(`stakeholder=${encodeURIComponent(stakeholder)}`);
  if (params.length > 0) url += `?${params.join('&')}`;
  const res = await apiClient.get(url);
  return res.data.data;
};

export const generateRecommendations = async (region: string) => {
  const res = await apiClient.post('/recommendations/generate', { region });
  return res.data;
};

export const fetchAlerts = async (regionId?: string, status?: string) => {
  let url = '/alerts';
  const params: string[] = [];
  if (regionId) params.push(`region_id=${encodeURIComponent(regionId)}`);
  if (status) params.push(`status=${encodeURIComponent(status)}`);
  if (params.length > 0) url += `?${params.join('&')}`;
  const res = await apiClient.get(url);
  return res.data.data;
};

export const fetchAlertsStatistics = async () => {
  const res = await apiClient.get('/alerts/statistics');
  return res.data.data;
};

export const updateAlertStatus = async (alertId: string, status: string) => {
  const res = await apiClient.patch(`/alerts/${encodeURIComponent(alertId)}/status`, { status });
  return res.data;
};

// ---------------------------------------------------------------------------
// SIR / SIRD Simulation
// ---------------------------------------------------------------------------
export interface SirSimParams {
  disease: string;
  population: number;
  initial_infected: number;
  beta: number;
  gamma: number;
  mu: number;
  days: number;
  intervention_reduction_percent: number;
}

export const runSirSimulation = async (params: SirSimParams) => {
  const res = await apiClient.post('/simulation/run', params);
  return res.data.data;
};

export const fetchSimulationParameters = async (disease: string = 'COVID-19') => {
  const res = await apiClient.get(`/simulation/parameters?disease=${encodeURIComponent(disease)}`);
  return res.data.data;
};

// ---------------------------------------------------------------------------
// AI Assistant & Models
// ---------------------------------------------------------------------------
export const sendAssistantChat = async (message: string) => {
  const res = await apiClient.post('/assistant/chat', { message });
  return res.data.data;
};

export const fetchModels = async () => {
  const res = await apiClient.get('/models');
  return res.data.data;
};

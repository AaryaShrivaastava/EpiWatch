import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardLayout from './layouts/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Alerts from './pages/Alerts';
import Recommendations from './pages/Recommendations';
import Analytics from './pages/Analytics';
import Outbreaks from './pages/Outbreaks';
import Forecasts from './pages/Forecasts';
import Healthcare from './pages/Healthcare';
import Spatial from './pages/Spatial';
import Simulation from './pages/Simulation';
import Agent from './pages/Agent';

function App() {
  return (
    <BrowserRouter>
      <DashboardLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/outbreaks" element={<Outbreaks />} />
          <Route path="/forecasts" element={<Forecasts />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/recommendations" element={<Recommendations />} />
          <Route path="/healthcare" element={<Healthcare />} />
          <Route path="/spatial" element={<Spatial />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/agent" element={<Agent />} />
        </Routes>
      </DashboardLayout>
    </BrowserRouter>
  );
}

export default App;

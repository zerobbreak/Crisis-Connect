import { useState, useEffect, useCallback } from "react";


// AlertForm.tsx
//import { useState } from 'react';

type AlertSentPayload = {
  id: string;
  message: string;
  severity: AlertSeverity;
  riskScore: number;
  languages: string[];
  channels: string[];
  recipients: string;
  location: {
    lat: number;
    lng: number;
    radius: number;
    address: string;
  };
  createdAt: Date;
};

interface AlertFormProps {
  onSent?: (payload: AlertSentPayload) => void; // optional callback with alert payload
  embedded?: boolean; // render compact form-only variant for embedding
}

 

import { 
  Send, 
  AlertTriangle, 
  Globe, 
  MessageSquare, 
  Users, 
  BarChart3, 
  Settings, 
  Bell,
  Shield,
  CheckCircle,
  XCircle,
  Clock,
  Target,
  Map,
  MapPin,
  Zap,
  Radio,
  Eye
} from "lucide-react";

// Types
type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';
type AlertStatus = 'draft' | 'sending' | 'sent' | 'failed';

/*type AlertFormProps = {
  onSent?: () => void;
};*/






interface Alert {
  id: string;
  message: string;
  severity: AlertSeverity;
  riskScore: number;
  languages: string[];
  channels: string[];
  recipients: string;
  status: AlertStatus;
  createdAt: Date;
  sentAt?: Date;
  location?: {
    lat: number;
    lng: number;
    radius: number;
    address: string;
  };
}

interface Language {
  code: string;
  name: string;
  flag: string;
}

interface Channel {
  id: string;
  name: string;
  icon: string;
  enabled: boolean;
}

// Constants
const LANGUAGES: Language[] = [
  { code: 'en', name: 'English', flag: '🇬🇧' },
  { code: 'af', name: 'Afrikaans', flag: '🇿🇦' },
  { code: 'zu', name: 'Zulu', flag: '🇿🇦' },
  { code: 'xh', name: 'Xhosa', flag: '🇿🇦' },
  { code: 'tn', name: 'Tswana', flag: '🇧🇼' },
  { code: 'st', name: 'Sotho', flag: '🇱🇸' },
];

const CHANNELS: Channel[] = [
  { id: 'whatsapp', name: 'WhatsApp', icon: '💬', enabled: true },
  { id: 'sms', name: 'SMS', icon: '📱', enabled: true },
  { id: 'email', name: 'Email', icon: '📧', enabled: true },
  { id: 'push', name: 'Push Notification', icon: '🔔', enabled: true },
  { id: 'radio', name: 'Radio Broadcast', icon: '📻', enabled: false },
  { id: 'siren', name: 'Emergency Siren', icon: '🚨', enabled: false },
];

const ALERT_TEMPLATES = 
[
  "⚠️ Heavy rain expected. Move to higher ground immediately.",
  "🔥 Fire alert in your area. Evacuate now via designated routes.",
  "🌪️ Severe weather warning. Seek shelter indoors.",
  "🚨 Emergency evacuation required. Follow local authorities' instructions.",
  "💧 Flood warning issued. Avoid low-lying areas.",
];


export default function AlertForm(props: AlertFormProps) 
{
  const { onSent, embedded = true } = props;
  // Navigation state
  const [activeTab, setActiveTab] = useState<'dashboard' | 'create' | 'map' | 'history' | 'settings'>('dashboard');
  
  // Form state
  const [formData, setFormData] = useState(
 {
    message: ALERT_TEMPLATES[0],
    severity: 'medium' as AlertSeverity,
    languages: ['en', 'af'],
    channels: ['whatsapp', 'sms'],
    recipients: '',
    targetRadius: 5, // km
    location: 
    {
      lat: -25.7479, // Pretoria coordinates
      lng: 28.2293,
      address: 'Pretoria, Gauteng, South Africa'
    }
 }
);
  // Load defaults from localStorage (preferred languages and location) on mount
  useEffect(() => {
    try {
      const storedLangs = localStorage.getItem('preferredLanguages');
      const storedLoc = localStorage.getItem('preferredLocation');
      const next: any = {};
      if (storedLangs) {
        const parsed = JSON.parse(storedLangs);
        if (Array.isArray(parsed) && parsed.length > 0) {
          next.languages = parsed;
        }
      }
      if (storedLoc) {
        const parsedLoc = JSON.parse(storedLoc);
        if (parsedLoc && typeof parsedLoc.lat === 'number' && typeof parsedLoc.lng === 'number') {
          next.location = {
            lat: parsedLoc.lat,
            lng: parsedLoc.lng,
            address: parsedLoc.address || `Saved: ${parsedLoc.lat.toFixed(4)}, ${parsedLoc.lng.toFixed(4)}`
          };
        }
      }
      if (Object.keys(next).length > 0) {
        setFormData((prev) => ({ ...prev, ...next }));
      }
    } catch {}
  }, []);

  



  // System state
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [riskScore, setRiskScore] = useState<number>(42);
  const [mapCenter, setMapCenter] = useState({ lat: -25.7479, lng: 28.2293 });
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  // Calculate risk score based on form data
  useEffect(() => {
    const baseScore = 30;
    const severityMultiplier = {
      low: 1,
      medium: 1.5,
      high: 2,
      critical: 2.5
    };
    const channelBonus = formData.channels.length * 5;
    const languageBonus = formData.languages.length * 3;
    
    const calculated = Math.min(100, 
      Math.round(baseScore * severityMultiplier[formData.severity] + channelBonus + languageBonus)
    );
    setRiskScore(calculated);
  }, [formData.severity, formData.channels.length, formData.languages.length]);

  // Utility functions
  const toggleArrayItem = useCallback(<T,>(array: T[], item: T): T[] => {
    return array.includes(item) 
      ? array.filter(i => i !== item) 
      : [...array, item];
  }, []);

  const getSeverityColor = (severity: AlertSeverity) => {
    const colors = {
      low: 'bg-green-100 text-green-800 border-green-200',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      high: 'bg-orange-100 text-orange-800 border-orange-200',
      critical: 'bg-red-100 text-red-800 border-red-200'
    };
    return colors[severity];
  };

  const getRiskScoreColor = (score: number) => {
    if (score <= 30) return 'text-green-600';
    if (score <= 60) return 'text-yellow-600';
    if (score <= 80) return 'text-orange-600';
    return 'text-red-600';
  };

  // Form handlers
  const handleSubmit = async () => {
    setIsLoading(true);

    const newAlert: Alert = {
      id: Date.now().toString(),
      message: formData.message,
      severity: formData.severity,
      riskScore,
      languages: formData.languages,
      channels: formData.channels,
      recipients: formData.recipients,
      status: 'sending',
      createdAt: new Date(),
      location: {
        lat: formData.location.lat,
        lng: formData.location.lng,
        radius: formData.targetRadius,
        address: formData.location.address
      }
    };

    setAlerts(prev => [newAlert, ...prev]);

    // Simulate API call
    setTimeout(() => {
      setAlerts(prev => prev.map(alert => 
        alert.id === newAlert.id 
          ? { ...alert, status: 'sent' as AlertStatus, sentAt: new Date() }
          : alert
      ));
      setIsLoading(false);
      setActiveTab('history');
      if (onSent) {
        onSent({
          id: newAlert.id,
          message: newAlert.message,
          severity: newAlert.severity,
          riskScore: newAlert.riskScore,
          languages: newAlert.languages,
          channels: newAlert.channels,
          recipients: newAlert.recipients,
          location: newAlert.location!,
          createdAt: newAlert.createdAt,
        });
      }
      // persist alert meta for statistics
      try {
        const existing = localStorage.getItem('sentAlerts');
        const list = existing ? JSON.parse(existing) : [];
        list.unshift({
          id: newAlert.id,
          createdAt: newAlert.createdAt,
          location: newAlert.location,
          severity: newAlert.severity,
          riskScore: newAlert.riskScore,
        });
        localStorage.setItem('sentAlerts', JSON.stringify(list.slice(0, 200)));
      } catch {}
    }, 2000);
  };

  // Navigation component
  const Navigation = () => (
    <nav className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Shield className="h-8 w-8 text-blue-600" />
          <h1 className="text-xl font-bold text-gray-900">Emergency Alert System</h1>
        </div>
        
        <div className="flex space-x-1">
          {[
            { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
            { id: 'create', label: 'Create Alert', icon: Send },
            { id: 'map', label: 'Live Map', icon: Map },
            { id: 'history', label: 'History', icon: Clock },
            { id: 'settings', label: 'Settings', icon: Settings },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id as any)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === id
                  ? 'bg-blue-100 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <Icon className="h-4 w-4" />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>
    </nav>
  );

  // Dashboard component
  const Dashboard = () => (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-900">System Overview</h2>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Alerts</p>
              <p className="text-3xl font-bold text-gray-900">{alerts.length}</p>
            </div>
            <Bell className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Sent Today</p>
              <p className="text-3xl font-bold text-green-600">
                {alerts.filter(a => a.status === 'sent').length}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-500" />
          </div>
        </div>
        
        <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Channels</p>
              <p className="text-3xl font-bold text-blue-600">
                {CHANNELS.filter(c => c.enabled).length}
              </p>
            </div>
            <MessageSquare className="h-8 w-8 text-blue-500" />
          </div>
        </div>
        
        <div className="bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Current Risk</p>
              <p className={`text-3xl font-bold ${getRiskScoreColor(riskScore)}`}>
                {riskScore}
              </p>
            </div>
            <Target className="h-8 w-8 text-orange-500" />
          </div>
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Alerts</h3>
        </div>
        <div className="p-6">
          {alerts.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No alerts sent yet</p>
          ) : (
            <div className="space-y-4">
              {alerts.slice(0, 5).map((alert) => (
                <div key={alert.id} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 truncate">{alert.message}</p>
                    <p className="text-sm text-gray-500">
                      {alert.createdAt.toLocaleString()} • Risk: {alert.riskScore}
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getSeverityColor(alert.severity)}`}>
                      {alert.severity}
                    </span>
                    {alert.status === 'sent' && <CheckCircle className="h-5 w-5 text-green-500" />}
                    {alert.status === 'sending' && <Clock className="h-5 w-5 text-yellow-500" />}
                    {alert.status === 'failed' && <XCircle className="h-5 w-5 text-red-500" />}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Map component with interactive features
  const LiveMap = () => {
    const [mapView, setMapView] = useState<'satellite' | 'terrain' | 'street'>('street');
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [activeIncidents, setActiveIncidents] = useState([
      { id: '1', type: 'flood', lat: -25.7679, lng: 28.2093, severity: 'high', time: '10 min ago' },
      { id: '2', type: 'fire', lat: -25.7279, lng: 28.2493, severity: 'critical', time: '5 min ago' },
      { id: '3', type: 'weather', lat: -25.7879, lng: 28.1893, severity: 'medium', time: '15 min ago' }
    ]);

    const handleMapClick = (lat: number, lng: number) => {
      setFormData(prev => ({
        ...prev,
        location: {
          lat,
          lng,
          address: `Location: ${lat.toFixed(4)}, ${lng.toFixed(4)}`
        }
      }));
      setActiveTab('create');
    };

    return (
      <div className="p-6 h-full">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm h-full flex flex-col">
          {/* Map Header */}
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                  <Map className="h-5 w-5 mr-2" />
                  Live Emergency Map
                </h2>
                <p className="text-sm text-gray-600 mt-1">Monitor incidents and target alert zones</p>
              </div>
              
              <div className="flex items-center space-x-4">
                {/* Map View Toggle */}
                <div className="flex items-center space-x-2">
                  <label htmlFor="map-view" className="block text-sm font-medium text-gray-700 mb-1">
  Map View
</label>
<select
  id="map-view"
  value={mapView}
  onChange={(e) => setMapView(e.target.value as any)}
  className="text-sm border border-gray-300 rounded px-2 py-1"
>
  <option value="street">Street</option>
  <option value="satellite">Satellite</option>
  <option value="terrain">Terrain</option>
</select>

                </div>
                
                {/* Heatmap Toggle */}
                <button
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`flex items-center space-x-2 px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                    showHeatmap ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'
                  }`}
                >
                  <Zap className="h-4 w-4" />
                  <span>Risk Heatmap</span>
                </button>
              </div>
            </div>
          </div>

          {/* Map Container */}
          <div className="flex-1 relative">
            {/* Interactive Map Placeholder */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50 to-green-50 flex items-center justify-center">
              <div className="relative w-full h-full max-w-4xl max-h-96 mx-auto">
                {/* Map Background */}
                <div className="absolute inset-0 bg-gradient-to-br from-green-100 via-blue-100 to-gray-100 rounded-lg border-2 border-dashed border-gray-300">
                  {/* Grid Lines */}
                  <div className="absolute inset-0 opacity-20">
                    {[...Array(10)].map((_, i) => (
                      <div key={i}>
                        <div 
                          className="absolute border-t border-gray-400" 
                          style={{ top: `${i * 10}%`, left: 0, right: 0 }} 
                        />
                        <div 
                          className="absolute border-l border-gray-400" 
                          style={{ left: `${i * 10}%`, top: 0, bottom: 0 }} 
                        />
                      </div>
                    ))}
                  </div>

                  {/* Mock Incidents */}
                  {activeIncidents.map((incident, index) => (
                    <div
                      key={incident.id}
                      className={`absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer`}
                      style={{
                        left: `${30 + index * 20}%`,
                        top: `${40 + index * 15}%`
                      }}
                      onClick={() => setSelectedAlert(alerts.find(a => a.id === incident.id) || null)}
                    >
                      <div className={`relative ${
                        incident.severity === 'critical' ? 'animate-pulse' : ''
                      }`}>
                        <div className={`w-6 h-6 rounded-full border-2 border-white shadow-lg ${
                          incident.severity === 'critical' ? 'bg-red-500' :
                          incident.severity === 'high' ? 'bg-orange-500' :
                          'bg-yellow-500'
                        }`}>
                          {incident.type === 'fire' && <span className="block text-center text-white text-xs">🔥</span>}
                          {incident.type === 'flood' && <span className="block text-center text-white text-xs">🌊</span>}
                          {incident.type === 'weather' && <span className="block text-center text-white text-xs">⛈️</span>}
                        </div>
                        
                        {/* Ripple Effect */}
                        <div className={`absolute inset-0 rounded-full border-2 animate-ping ${
                          incident.severity === 'critical' ? 'border-red-400' :
                          incident.severity === 'high' ? 'border-orange-400' :
                          'border-yellow-400'
                        }`} />
                      </div>
                    </div>
                  ))}

                  {/* Alert Coverage Areas */}
                  {alerts.filter(a => a.location).map((alert, index) => (
                    <div
                      key={alert.id}
                      className="absolute transform -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                      style={{
                        left: `${50 + index * 10}%`,
                        top: `${30 + index * 10}%`
                      }}
                      onClick={() => setSelectedAlert(alert)}
                    >
                      {/* Coverage Circle */}
                      <div className={`w-16 h-16 rounded-full border-2 border-dashed opacity-60 ${
                        alert.severity === 'critical' ? 'border-red-500 bg-red-100' :
                        alert.severity === 'high' ? 'border-orange-500 bg-orange-100' :
                        alert.severity === 'medium' ? 'border-yellow-500 bg-yellow-100' :
                        'border-green-500 bg-green-100'
                      }`} />
                      
                      {/* Center Pin */}
                      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                        <MapPin className={`h-4 w-4 ${
                          alert.severity === 'critical' ? 'text-red-600' :
                          alert.severity === 'high' ? 'text-orange-600' :
                          alert.severity === 'medium' ? 'text-yellow-600' :
                          'text-green-600'
                        }`} />
                      </div>
                    </div>
                  ))}

                  {/* Click to Add Location */}
                  <div className="absolute inset-0 cursor-crosshair" onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * 100;
                    const y = ((e.clientY - rect.top) / rect.height) * 100;
                    const lat = -25.8 + (y / 100) * 0.2; // Mock coordinates
                    const lng = 28.1 + (x / 100) * 0.4;
                    handleMapClick(lat, lng);
                  }} />
                </div>

                {/* Map Legend */}
                <div className="absolute top-4 left-4 bg-white bg-opacity-90 backdrop-blur-sm rounded-lg p-4 shadow-lg">
                  <h4 className="font-semibold text-sm mb-2">Legend</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <span>Critical Alert</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                      <span>High Priority</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <span>Medium Priority</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <MapPin className="h-3 w-3 text-gray-600" />
                      <span>Alert Zone</span>
                    </div>
                  </div>
                </div>

                {/* Coordinates Display */}
                <div className="absolute top-4 right-4 bg-white bg-opacity-90 backdrop-blur-sm rounded-lg p-3 shadow-lg">
                  <div className="text-xs font-mono">
                    <div>Lat: {mapCenter.lat.toFixed(4)}</div>
                    <div>Lng: {mapCenter.lng.toFixed(4)}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Map Controls */}
          <div className="p-4 border-t border-gray-200 bg-gray-50">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={() => setActiveTab('create')}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Target className="h-4 w-4" />
                  <span>Create Alert Here</span>
                </button>
                
                <div className="text-sm text-gray-600">
                  Click anywhere on the map to set alert location
                </div>
              </div>
              
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">Active Incidents:</span>
                <span className="px-2 py-1 bg-red-100 text-red-700 rounded-full text-xs font-medium">
                  {activeIncidents.length}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Selected Alert Details */}
        {selectedAlert && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white rounded-xl max-w-md w-full p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Alert Details</h3>
                <button
                  onClick={() => setSelectedAlert(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium text-gray-500">Message</label>
                  <p className="text-sm">{selectedAlert.message}</p>
                </div>
                
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-sm font-medium text-gray-500">Severity</label>
                    <p className={`text-sm font-medium capitalize ${
                      selectedAlert.severity === 'critical' ? 'text-red-600' :
                      selectedAlert.severity === 'high' ? 'text-orange-600' :
                      selectedAlert.severity === 'medium' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {selectedAlert.severity}
                    </p>
                  </div>
                  
                  <div>
                    <label className="text-sm font-medium text-gray-500">Risk Score</label>
                    <p className="text-sm font-medium">{selectedAlert.riskScore}</p>
                  </div>
                </div>
                
                {selectedAlert.location && (
                  <div>
                    <label className="text-sm font-medium text-gray-500">Location</label>
                    <p className="text-sm">{selectedAlert.location.address}</p>
                    <p className="text-xs text-gray-500">
                      Radius: {selectedAlert.location.radius}km
                    </p>
                  </div>
                )}
                
                <div>
                  <label className="text-sm font-medium text-gray-500">Status</label>
                  <div className="flex items-center space-x-2">
                    {selectedAlert.status === 'sent' && <CheckCircle className="h-4 w-4 text-green-500" />}
                    {selectedAlert.status === 'sending' && <Clock className="h-4 w-4 text-yellow-500" />}
                    {selectedAlert.status === 'failed' && <XCircle className="h-4 w-4 text-red-500" />}
                    <span className="text-sm capitalize">{selectedAlert.status}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };
  const CreateAlertForm = () => (
    <div className={`${embedded ? '' : 'p-6'}`}>
      <div className={`${embedded ? '' : 'max-w-2xl mx-auto'}`}>
        <div className={`${embedded ? '' : 'bg-white rounded-xl border border-gray-200 shadow-sm'}`}>
          {!embedded && (
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900">Create New Alert</h2>
              <p className="text-sm text-gray-600 mt-1">Send emergency notifications to targeted recipients</p>
            </div>
          )}
          
          <div className={`${embedded ? 'space-y-6' : 'p-6 space-y-6'}`}>
            {/* Alert Message */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Alert Message
              </label>
              <div className="space-y-2">
                <textarea
                  className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={4}
                  placeholder="Enter your emergency message..."
                  value={formData.message}
                  onChange={(e) => setFormData(prev => ({ ...prev, message: e.target.value }))}
                  required
                />
                <div className="flex flex-wrap gap-2">
                  {ALERT_TEMPLATES.map((template, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => setFormData(prev => ({ ...prev, message: template }))}
                      className="text-xs px-3 py-1 bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
                    >
                      Template {index + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>
        </div>

            {/* Severity & Risk Score */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
  <label 
    htmlFor="alert-severity" 
    className="block text-sm font-medium text-gray-700 mb-2"
  >
    Severity Level
  </label>
  <select
    id="alert-severity"
    value={formData.severity}
    onChange={(e) =>
      setFormData(prev => ({ ...prev, severity: e.target.value as AlertSeverity }))
    }
    className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
    aria-label="Select alert severity level"
  >
    <option value="low">Low - Information</option>
    <option value="medium">Medium - Advisory</option>
    <option value="high">High - Warning</option>
    <option value="critical">Critical - Emergency</option>
  </select>
</div>

              
              <div className="flex items-end">
                <div className="w-full">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Risk Score
                  </label>
                  <div className={`w-full p-3 bg-gray-50 rounded-lg border text-center font-bold text-lg ${getRiskScoreColor(riskScore)}`}>
                    {riskScore}/100
                  </div>
                </div>
              </div>
            </div>

            {/* Languages */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                <Globe className="inline h-4 w-4 mr-1" />
                Languages ({formData.languages.length} selected)
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {LANGUAGES.map((lang) => (
                  <label
                    key={lang.code}
                    className={`flex items-center justify-between p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      formData.languages.includes(lang.code)
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">{lang.flag}</span>
                      <div>
                        <p className="font-medium text-sm">{lang.name}</p>
                        <p className="text-xs text-gray-500">{lang.code.toUpperCase()}</p>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={formData.languages.includes(lang.code)}
                      onChange={() => setFormData(prev => ({
                        ...prev,
                        languages: toggleArrayItem(prev.languages, lang.code)
                      }))}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                      formData.languages.includes(lang.code)
                        ? 'bg-blue-500 border-blue-500'
                        : 'border-gray-300'
                    }`}>
                      {formData.languages.includes(lang.code) && (
                        <CheckCircle className="h-3 w-3 text-white" />
                      )}
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Channels */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                <MessageSquare className="inline h-4 w-4 mr-1" />
                Delivery Channels ({formData.channels.length} selected)
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {CHANNELS.map((channel) => (
                  <label
                    key={channel.id}
                    className={`flex items-center justify-between p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      !channel.enabled
                        ? 'opacity-50 cursor-not-allowed'
                        : formData.channels.includes(channel.id)
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-xl">{channel.icon}</span>
                      <div>
                        <p className="font-medium text-sm">{channel.name}</p>
                        <p className="text-xs text-gray-500">
                          {channel.enabled ? 'Available' : 'Unavailable'}
                        </p>
                      </div>
                    </div>
                    <input
                      type="checkbox"
                      checked={formData.channels.includes(channel.id)}
                      onChange={() => setFormData(prev => ({
                        ...prev,
                        channels: toggleArrayItem(prev.channels, channel.id)
                      }))}
                      disabled={!channel.enabled}
                      className="sr-only"
                    />
                    <div className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                      formData.channels.includes(channel.id) && channel.enabled
                        ? 'bg-green-500 border-green-500'
                        : 'border-gray-300'
                    }`}>
                      {formData.channels.includes(channel.id) && channel.enabled && (
                        <CheckCircle className="h-3 w-3 text-white" />
                      )}
                    </div>
                  </label>
                ))}
              </div>
            </div>

            {/* Location & Targeting */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                <MapPin className="inline h-4 w-4 mr-1" />
                Target Location & Radius
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <input
                    type="text"
                    placeholder="Enter address or coordinates"
                    value={formData.location.address}
                    onChange={(e) => setFormData(prev => ({
                      ...prev,
                      location: { ...prev.location, address: e.target.value }
                    }))}
                    className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  {!embedded && (
                    <button
                      type="button"
                      onClick={() => setActiveTab('map')}
                      className="mt-2 text-sm text-blue-600 hover:text-blue-700 flex items-center"
                    >
                      <Eye className="h-4 w-4 mr-1" />
                      Select on Map
                    </button>
                  )}
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="Latitude"
                      value={formData.location.lat}
                      onChange={(e) => {
                        const next = parseFloat(e.target.value);
                        setFormData(prev => ({
                          ...prev,
                          location: { ...prev.location, lat: isNaN(next) ? prev.location.lat : next }
                        }));
                      }}
                      className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <input
                      type="number"
                      step="0.0001"
                      placeholder="Longitude"
                      value={formData.location.lng}
                      onChange={(e) => {
                        const next = parseFloat(e.target.value);
                        setFormData(prev => ({
                          ...prev,
                          location: { ...prev.location, lng: isNaN(next) ? prev.location.lng : next }
                        }));
                      }}
                      className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      if (!navigator.geolocation) return;
                      navigator.geolocation.getCurrentPosition((pos) => {
                        const { latitude, longitude } = pos.coords;
                        setFormData(prev => ({
                          ...prev,
                          location: {
                            ...prev.location,
                            lat: latitude,
                            lng: longitude,
                            address: `My Location: ${latitude.toFixed(4)}, ${longitude.toFixed(4)}`
                          }
                        }));
                      });
                    }}
                    className="mt-2 text-xs text-blue-600 hover:text-blue-700"
                  >
                    Use My Location
                  </button>
                </div>
                
             

<div>
  <label 
    htmlFor="alert-radius" 
    className="block text-xs text-gray-500 mb-1"
  >
    Alert Radius (km)
  </label>
  <input
    id="alert-radius"
    type="range"
    min={1}
    max={50}
    value={formData.targetRadius}
    onChange={(e) =>
      setFormData(prev => ({ ...prev, targetRadius: parseInt(e.target.value) }))
    }
    className="w-full"
  />
  <div className="flex justify-between text-xs text-gray-500 mt-1">
    <span>1km</span>
    <span className="font-medium">{formData.targetRadius}km</span>
    <span>50km</span>
  </div>

  <div className="mt-2 p-3 bg-blue-50 rounded-lg">
    <p className="text-sm text-blue-700">
      <MapPin className="inline h-4 w-4 mr-1" />
      Target: {formData.location.address} • Radius: {formData.targetRadius}km
    </p>
  </div>
</div>






              <label className="block text-sm font-medium text-gray-700 mb-2">
                <Users className="inline h-4 w-4 mr-1" />
                Recipients
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
                placeholder="Enter phone numbers (+27123456789), email addresses, or group IDs, separated by commas..."
                value={formData.recipients}
                onChange={(e) => setFormData(prev => ({ ...prev, recipients: e.target.value }))}
                required
              />
              <p className="text-xs text-gray-500 mt-1">
                Supports phone numbers, email addresses, and predefined groups
              </p>
            </div>

            {/* Submit Button */}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={isLoading || !formData.message || !formData.recipients}
              className={`w-full flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                isLoading || !formData.message || !formData.recipients
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white shadow-lg hover:shadow-xl'
              }`}
            >
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Sending Alert...</span>
                </>
              ) : (
                <>
                  <Send className="h-4 w-4" />
                  <span>Send Alert Now</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // Alert History component
  const AlertHistory = () => (
    <div className="p-6">
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Alert History</h2>
          <p className="text-sm text-gray-600 mt-1">Track all sent and pending alerts</p>
        </div>
        
        <div className="overflow-x-auto">
          {alerts.length === 0 ? (
            <div className="p-12 text-center">
              <Bell className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No alerts in history</p>
              <button
                onClick={() => setActiveTab('create')}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create First Alert
              </button>
            </div>
          ) : (
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Message
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Severity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Risk
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {alerts.map((alert) => (
                  <tr key={alert.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4">
                      <div className="max-w-xs">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {alert.message}
                        </p>
                        <p className="text-xs text-gray-500">
                          {alert.languages.length} languages • {alert.channels.length} channels
                        </p>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getSeverityColor(alert.severity)}`}>
                        {alert.severity}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        {alert.status === 'sent' && <CheckCircle className="h-4 w-4 text-green-500" />}
                        {alert.status === 'sending' && <Clock className="h-4 w-4 text-yellow-500" />}
                        {alert.status === 'failed' && <XCircle className="h-4 w-4 text-red-500" />}
                        <span className="text-sm capitalize">{alert.status}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`text-sm font-medium ${getRiskScoreColor(alert.riskScore)}`}>
                        {alert.riskScore}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">
                      {alert.createdAt.toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );

  // Settings component
  const Settings = () => (
    <div className="p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">System Settings</h2>
            <p className="text-sm text-gray-600 mt-1">Configure alert system preferences</p>
          </div>
          
          <div className="p-6 space-y-6">
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Channel Configuration</h3>
              <div className="space-y-3">
                {CHANNELS.map((channel) => (
                  <div key={channel.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      <span className="text-xl">{channel.icon}</span>
                      <div>
                        <p className="font-medium">{channel.name}</p>
                        <p className="text-sm text-gray-500">
                          {channel.enabled ? 'Currently active' : 'Currently disabled'}
                        </p>
                      </div>
                    </div>
                    <div className={`w-12 h-6 rounded-full p-1 cursor-pointer transition-colors ${
                      channel.enabled ? 'bg-green-500' : 'bg-gray-300'
                    }`}>
                      <div className={`w-4 h-4 rounded-full bg-white transition-transform ${
                        channel.enabled ? 'translate-x-6' : 'translate-x-0'
                      }`} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Main render
  if (embedded) {
    return <CreateAlertForm />;
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navigation />
      
      {activeTab === 'dashboard' && <Dashboard />}
      {activeTab === 'create' && <CreateAlertForm />}
      {activeTab === 'map' && <LiveMap />}
      {activeTab === 'history' && <AlertHistory />}
      {activeTab === 'settings' && <Settings />}
    </div>
  );
}
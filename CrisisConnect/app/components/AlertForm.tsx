import { useState, useEffect, useCallback } from "react";
import { Send, Globe, MessageSquare, Users, Bell, MapPin, Zap, Clock, CheckCircle, XCircle } from "lucide-react";

// --- Types ---
type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';
type AlertStatus = 'draft' | 'sending' | 'sent' | 'failed';

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

// --- Props ---
interface AlertFormProps {
  onSent?: (payload: Omit<Alert, 'status' | 'createdAt'>) => void;
  embedded?: boolean;
}

// --- Static Data ---
const LANGUAGES: Language[] = [
  { code: 'en', name: 'English', flag: '🇬🇧' },
  { code: 'zu', name: 'isiZulu', flag: '🇿🇦' },
  { code: 'xh', name: 'isiXhosa', flag: '🇿🇦' },
  { code: 'af', name: 'Afrikaans', flag: '🇿🇦' },
  { code: 'nso', name: 'Sesotho sa Leboa', flag: '🇿🇦' },
];

const CHANNELS: Channel[] = [
  { id: 'sms', name: 'SMS', icon: '📱', enabled: true },
  { id: 'email', name: 'Email', icon: '✉️', enabled: true },
  { id: 'whatsapp', name: 'WhatsApp', icon: '💬', enabled: true },
  { id: 'push', name: 'Push Notification', icon: '🔔', enabled: true },
  { id: 'radio', name: 'Radio Broadcast', icon: '📻', enabled: false },
  { id: 'siren', name: 'Emergency Siren', icon: '🚨', enabled: false },
];

const ALERT_TEMPLATES = [
  "⚠️ Heavy rain expected. Move to higher ground immediately.",
  "🔥 Fire alert in your area. Evacuate now via designated routes.",
  "🌪️ Severe weather warning. Seek shelter indoors.",
  "🚨 Emergency evacuation required. Follow local authorities' instructions.",
  "💧 Flood warning issued. Avoid low-lying areas.",
];

export default function AlertForm({ onSent, embedded = false }: AlertFormProps) {
  // --- Form State ---
  const [formData, setFormData] = useState({
    message: ALERT_TEMPLATES[0],
    severity: 'medium' as AlertSeverity,
    languages: ['en', 'zu'] as string[],
    channels: ['sms', 'whatsapp'] as string[],
    recipients: '',
    targetRadius: 10,
    location: {
      lat: -25.7479,
      lng: 28.2293,
      address: 'Pretoria, Gauteng, South Africa'
    }
  });

  // --- System State ---
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [riskScore, setRiskScore] = useState<number>(42);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  // --- Load from localStorage ---
  useEffect(() => {
    try {
      const storedLangs = localStorage.getItem('preferredLanguages');
      const storedLoc = localStorage.getItem('preferredLocation');

      if (storedLangs) {
        const langs = JSON.parse(storedLangs);
        setFormData(prev => ({ ...prev, languages: langs }));
      }

      if (storedLoc) {
        const loc = JSON.parse(storedLoc);
        setFormData(prev => ({ ...prev, location: loc }));
      }
    } catch (err) {
      console.error("Failed to load preferences", err);
    }
  }, []);

  // --- Save to localStorage ---
  useEffect(() => {
    localStorage.setItem('preferredLanguages', JSON.stringify(formData.languages));
    localStorage.setItem('preferredLocation', JSON.stringify(formData.location));
  }, [formData.languages, formData.location]);

  // --- Fetch Risk Score from Backend ---
  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const res = await fetch("http://localhost:8000/resources", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lat: formData.location.lat,
            lon: formData.location.lng,
            household_size: 4
          })
        });

        if (!res.ok) return;

        const data = await res.json();
        const score = Math.round(data.composite_risk_score || 0);
        setRiskScore(score);
      } catch (err) {
        console.error("Failed to fetch risk score", err);
      }
    };

    if (formData.location.lat && formData.location.lng) {
      fetchRisk();
    }
  }, [formData.location]);

  // --- Handle Form Changes ---
  const handleChange = useCallback((next: Partial<typeof formData>) => {
    if (Object.keys(next).length > 0) {
      setFormData(prev => ({ ...prev, ...next }));
    }
  }, []);

  // --- Handle Submit ---
  const handleSubmit = async () => {
    if (!formData.message || !formData.recipients) return;
    setIsLoading(true);

    try {
      // Step 1: Simulate disaster
      const scenario = formData.severity === 'critical' ? 'flood' :
                      formData.severity === 'high' ? 'storm' :
                      formData.severity === 'medium' ? 'drought' : 'heatwave';

      const simRes = await fetch("http://localhost:8000/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          location: formData.location.address,
          lat: formData.location.lat,
          lon: formData.location.lng,
          scenario,
          household_size: 4
        })
      });

      if (!simRes.ok) throw new Error("Simulation failed");

      const simData = await simRes.json();

      // Step 2: Run prediction
      await fetch("http://localhost:8000/predict", { method: "POST" });

      // Step 3: Generate alerts
      const alertRes = await fetch("http://localhost:8000/alerts/generate", { method: "POST" });
      const alertData = await alertRes.json();

      // Step 4: Create local alert
      const newAlert: Alert = {
        id: Date.now().toString(),
        message: simData.message_en || formData.message,
        severity: formData.severity,
        riskScore: simData.risk_score || riskScore,
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

      // Simulate delay
      setTimeout(() => {
        setAlerts(prev => prev.map(alert =>
          alert.id === newAlert.id
            ? { ...alert, status: 'sent' as AlertStatus, sentAt: new Date() }
            : alert
        ));
        setIsLoading(false);
        if (onSent) onSent({
          id: newAlert.id,
          message: newAlert.message,
          severity: newAlert.severity,
          riskScore: newAlert.riskScore,
          languages: newAlert.languages,
          channels: newAlert.channels,
          recipients: newAlert.recipients,
          location: newAlert.location
        });
      }, 2000);

    } catch (err) {
      console.error("Alert creation failed", err);
      setAlerts(prev => prev.map(alert =>
        alert.id === Date.now().toString()
          ? { ...alert, status: 'failed' }
          : alert
      ));
      setIsLoading(false);
      alert("❌ Failed to send alert. Check backend.");
    }
  };

  // --- UI Helpers ---
  const getSeverityColor = (severity: AlertSeverity) => {
    const colors = {
      low: 'border-green-200 text-green-800 bg-green-100',
      medium: 'border-yellow-200 text-yellow-800 bg-yellow-100',
      high: 'border-orange-200 text-orange-800 bg-orange-100',
      critical: 'border-red-200 text-red-800 bg-red-100'
    };
    return colors[severity];
  };

  const getRiskScoreColor = (score: number) => {
    if (score <= 30) return 'text-green-600';
    if (score <= 60) return 'text-yellow-600';
    if (score <= 80) return 'text-orange-600';
    return 'text-red-600';
  };

  // --- Render ---
  return (
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
              <label className="block text-sm font-medium text-gray-700 mb-2">Alert Message</label>
              <div className="space-y-2">
                <textarea
                  className="w-full border border-gray-300 rounded-lg p-3 text-sm text-gray-900 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={4}
                  placeholder="Enter your emergency message..."
                  value={formData.message}
                  onChange={(e) => handleChange({ message: e.target.value })}
                  required
                />
                <div className="flex flex-wrap gap-2">
                  {ALERT_TEMPLATES.map((template, index) => (
                    <button
                      key={index}
                      type="button"
                      onClick={() => handleChange({ message: template })}
                      className="text-xs px-3 py-1 bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
                    >
                      Template {index + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Severity & Risk Score */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label htmlFor="alert-severity" className="block text-sm font-medium text-gray-700 mb-2">Severity Level</label>
                <select
                  id="alert-severity"
                  value={formData.severity}
                  onChange={(e) => handleChange({ severity: e.target.value as AlertSeverity })}
                  className="w-full border border-gray-300 rounded-lg p-2 text-sm"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Risk Score</label>
                <div className={`w-full p-3 bg-gray-50 rounded-lg border text-center font-bold text-lg ${getRiskScoreColor(riskScore)}`}>
                  {riskScore}/100
                </div>
              </div>
            </div>

            {/* Languages */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                <Globe className="inline h-4 w-4 mr-1" /> Languages ({formData.languages.length} selected)
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {LANGUAGES.map((lang) => (
                  <label
                    key={lang.code}
                    className={`flex items-center justify-between p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      formData.languages.includes(lang.code) ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <span>{lang.flag}</span>
                      {lang.name}
                    </span>
                    {formData.languages.includes(lang.code) && (
                      <CheckCircle className="h-4 w-4 text-blue-500" />
                    )}
                  </label>
                ))}
              </div>
            </div>

            {/* Channels */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                <MessageSquare className="inline h-4 w-4 mr-1" /> Delivery Channels
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {CHANNELS.map((channel) => (
                  <label
                    key={channel.id}
                    className={`flex items-center justify-between p-3 rounded-lg border-2 cursor-pointer transition-all ${
                      formData.channels.includes(channel.id) ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <span className="flex items-center gap-2">
                      <span>{channel.icon}</span>
                      {channel.name}
                    </span>
                    {formData.channels.includes(channel.id) && (
                      <CheckCircle className="h-4 w-4 text-blue-500" />
                    )}
                  </label>
                ))}
              </div>
            </div>

            {/* Recipients */}
            <div>
              <label htmlFor="recipients" className="block text-sm font-medium text-gray-700 mb-2">Recipients</label>
              <input
                id="recipients"
                type="text"
                value={formData.recipients}
                onChange={(e) => handleChange({ recipients: e.target.value })}
                placeholder="e.g. Emergency Teams, Local Authorities"
                className="w-full border border-gray-300 rounded-lg p-3 text-sm"
                required
              />
            </div>

            {/* Location */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Target Location</label>
              <div className="space-y-2">
                <input
                  type="text"
                  value={formData.location.address}
                  onChange={(e) => handleChange({ location: { ...formData.location, address: e.target.value } })}
                  placeholder="e.g. eThekwini (Durban)"
                  className="w-full border border-gray-300 rounded-lg p-3 text-sm"
                />
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      if (!navigator.geolocation) return;
                      navigator.geolocation.getCurrentPosition((pos) => {
                        const { latitude, longitude } = pos.coords;
                        handleChange({
                          location: {
                            lat: latitude,
                            lng: longitude,
                            address: `My Location: ${latitude.toFixed(4)}, ${longitude.toFixed(4)}`
                          }
                        });
                      });
                    }}
                    className="mt-2 text-xs text-blue-600 hover:text-blue-700"
                  >
                    Use My Location
                  </button>
                </div>
              </div>
            </div>

            {/* Alert Radius */}
            <div>
              <label htmlFor="alert-radius" className="block text-xs text-gray-500 mb-1">Alert Radius (km)</label>
              <input
                id="alert-radius"
                type="range"
                min={1}
                max={50}
                value={formData.targetRadius}
                onChange={(e) => handleChange({ targetRadius: Number(e.target.value) })}
                className="w-full"
              />
              <div className="text-center text-sm text-gray-600">{formData.targetRadius} km</div>
            </div>

            {/* Submit Button */}
            <div>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isLoading}
                className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-lg flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <Clock className="animate-spin h-5 w-5" />
                    <span>Sending Alert...</span>
                  </>
                ) : (
                  <>
                    <Send className="h-5 w-5" />
                    <span>Send Alert</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Selected Alert Details Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-xl max-w-md w-full p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Alert Details</h3>
              <button onClick={() => setSelectedAlert(null)} className="text-gray-400 hover:text-gray-600">✕</button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-500">Message</label>
                <p className="text-sm">{selectedAlert.message}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-gray-500">Location</label>
                <p className="text-sm">{selectedAlert.location?.address}</p>
                <p className="text-xs text-gray-500">Radius: {selectedAlert.location?.radius}km</p>
              </div>
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
}
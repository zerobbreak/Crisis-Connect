import { openDB } from 'idb';

interface AlertRecord {
  id: string;
  message: string;
  languages: string[];
  channels: string[];
  riskScore?: number;
  createdAt: string;
}

const DB_NAME = 'crisisconnect';
const STORE = 'alerts';

export async function saveRecentAlerts(alerts: AlertRecord[]) {
  const db = await openDB(DB_NAME, 1, {
    upgrade(db) {
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    }
  });
  await db.put(STORE, alerts.slice(0, 10), 'recent');
}

export async function getRecentAlerts(): Promise<AlertRecord[] | null> {
  const db = await openDB(DB_NAME, 1);
  return (await db.get(STORE, 'recent')) || null;
}

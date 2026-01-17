"""
EM-DAT Database Fetcher

Fetches historical disaster data from the EM-DAT (Emergency Events Database)
maintained by the Centre for Research on the Epidemiology of Disasters.

EM-DAT contains 25,000+ events since 1900 and is freely available with registration.
"""

import httpx
import pandas as pd
from datetime import datetime
from typing import List, Optional, Dict, Any
import asyncio
import logging
import os

logger = logging.getLogger(__name__)


class EMDATFetcher:
    """Fetch historical disaster data from EM-DAT API"""
    
    # Southern Africa countries (ISO3 codes)
    SOUTHERN_AFRICA_COUNTRIES = [
        "ZAF",  # South Africa
        "ZWE",  # Zimbabwe
        "BWA",  # Botswana
        "NAM",  # Namibia
        "MOZ",  # Mozambique
        "ZMB",  # Zambia
        "MWI",  # Malawi
        "AGO",  # Angola
        "SWZ",  # Eswatini
        "LSO",  # Lesotho
    ]
    
    # Relevant disaster types for flood prediction
    DISASTER_TYPES = [
        "Flood",
        "Storm",
        "Drought",
        "Landslide",
        "Extreme temperature",
    ]
    
    # Country name to ISO3 mapping
    COUNTRY_TO_ISO3 = {
        "South Africa": "ZAF",
        "Zimbabwe": "ZWE",
        "Botswana": "BWA",
        "Namibia": "NAM",
        "Mozambique": "MOZ",
        "Zambia": "ZMB",
        "Malawi": "MWI",
        "Angola": "AGO",
        "Eswatini": "SWZ",
        "Lesotho": "LSO",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize EM-DAT fetcher
        
        Args:
            api_key: EM-DAT API key (from env if not provided)
        """
        self.api_key = api_key or os.getenv("EMDAT_API_KEY")
        self.base_url = os.getenv("EMDAT_API_URL", "https://public.emdat.be/api")
        self._client: Optional[httpx.AsyncClient] = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._client
    
    async def fetch_disasters(
        self,
        countries: Optional[List[str]] = None,
        start_year: int = 2010,
        end_year: int = 2024,
        disaster_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch disaster records from EM-DAT
        
        Args:
            countries: List of country names or ISO codes (defaults to Southern Africa)
            start_year: Start year for data range
            end_year: End year for data range
            disaster_types: List of disaster types to include
            
        Returns:
            DataFrame with normalized columns:
            - date, location, latitude, longitude
            - disaster_type, disaster_subtype
            - country, country_iso
            - deaths, affected, damage_usd
            - event_name, source
        """
        # Default to Southern Africa countries
        if countries is None:
            country_codes = self.SOUTHERN_AFRICA_COUNTRIES
        else:
            # Convert country names to ISO3 codes
            country_codes = [
                self.COUNTRY_TO_ISO3.get(c, c) for c in countries
            ]
        
        # Default to relevant disaster types
        if disaster_types is None:
            disaster_types = self.DISASTER_TYPES
            
        logger.info(
            f"Fetching EM-DAT disasters: countries={country_codes}, "
            f"years={start_year}-{end_year}, types={disaster_types}"
        )
        
        all_records = []
        
        try:
            client = await self._get_client()
            
            # Fetch data for each country
            for country_code in country_codes:
                try:
                    records = await self._fetch_country_disasters(
                        client, country_code, start_year, end_year, disaster_types
                    )
                    all_records.extend(records)
                    logger.info(f"  {country_code}: {len(records)} records")
                    
                    # Rate limiting - be respectful to the API
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"Error fetching {country_code}: {e}")
                    continue
            
            if not all_records:
                logger.warning("No records fetched from EM-DAT")
                return self._empty_dataframe()
            
            # Normalize to standard schema
            df = self._normalize_disasters(all_records)
            logger.info(f"Total EM-DAT records: {len(df)}")
            
            return df
            
        except Exception as e:
            logger.error(f"EM-DAT fetch failed: {e}", exc_info=True)
            return self._empty_dataframe()
    
    async def _fetch_country_disasters(
        self,
        client: httpx.AsyncClient,
        country_code: str,
        start_year: int,
        end_year: int,
        disaster_types: List[str],
    ) -> List[Dict[str, Any]]:
        """Fetch disasters for a single country with pagination"""
        
        records = []
        page = 1
        page_size = 100
        
        while True:
            params = {
                "country": country_code,
                "from": start_year,
                "to": end_year,
                "page": page,
                "pageSize": page_size,
            }
            
            # Add disaster type filter if supported
            if disaster_types:
                params["disasterType"] = ",".join(disaster_types)
            
            try:
                response = await client.get(
                    f"{self.base_url}/disasters",
                    params=params
                )
                
                if response.status_code == 401:
                    logger.error("EM-DAT authentication failed. Check API key.")
                    break
                    
                if response.status_code == 429:
                    logger.warning("EM-DAT rate limit hit. Waiting...")
                    await asyncio.sleep(5)
                    continue
                    
                response.raise_for_status()
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, dict):
                    page_records = data.get("data", data.get("disasters", []))
                    total_pages = data.get("totalPages", 1)
                elif isinstance(data, list):
                    page_records = data
                    total_pages = 1
                else:
                    break
                
                records.extend(page_records)
                
                # Check if more pages
                if page >= total_pages or len(page_records) < page_size:
                    break
                    
                page += 1
                await asyncio.sleep(0.2)  # Rate limiting between pages
                
            except httpx.HTTPStatusError as e:
                logger.warning(f"HTTP error for {country_code} page {page}: {e}")
                break
            except Exception as e:
                logger.warning(f"Error fetching page {page}: {e}")
                break
        
        return records
    
    def _normalize_disasters(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Normalize EM-DAT response to standard schema"""
        
        normalized = []
        
        for record in raw_data:
            try:
                # Parse date - EM-DAT uses various date fields
                date = self._parse_emdat_date(record)
                
                normalized.append({
                    # Core identifiers
                    "disaster_id": record.get("disNo", record.get("id")),
                    "event_name": record.get("eventName", record.get("name", "")),
                    
                    # Temporal
                    "date": date,
                    "year": record.get("year", date.year if date else None),
                    
                    # Geographic
                    "country": record.get("country", record.get("countryName", "")),
                    "country_iso": record.get("iso", record.get("countryCode", "")),
                    "location": record.get("location", record.get("region", "")),
                    "latitude": self._safe_float(record.get("latitude", record.get("lat"))),
                    "longitude": self._safe_float(record.get("longitude", record.get("lon"))),
                    
                    # Classification
                    "disaster_type": record.get("disasterType", record.get("type", "")),
                    "disaster_subtype": record.get("disasterSubtype", record.get("subtype", "")),
                    "disaster_group": record.get("disasterGroup", ""),
                    
                    # Impact
                    "deaths": self._safe_int(record.get("totalDeaths", record.get("deaths", 0))),
                    "injured": self._safe_int(record.get("noInjured", record.get("injured", 0))),
                    "affected": self._safe_int(record.get("totalAffected", record.get("affected", 0))),
                    "homeless": self._safe_int(record.get("noHomeless", record.get("homeless", 0))),
                    "damage_usd": self._safe_float(record.get("totalDamage", record.get("damage", 0))) * 1000,  # EM-DAT reports in thousands
                    
                    # Metadata
                    "source": "emdat",
                    "fetched_at": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.warning(f"Error normalizing record: {e}")
                continue
        
        return pd.DataFrame(normalized)
    
    def _parse_emdat_date(self, record: Dict[str, Any]) -> Optional[datetime]:
        """Parse date from EM-DAT record"""
        
        # Try different date field combinations
        year = record.get("year") or record.get("startYear")
        month = record.get("startMonth") or record.get("month") or 6
        day = record.get("startDay") or record.get("day") or 15
        
        if year:
            try:
                return datetime(int(year), int(month), int(day))
            except (ValueError, TypeError):
                try:
                    return datetime(int(year), 1, 1)
                except:
                    pass
        
        # Try ISO date string
        for field in ["date", "startDate", "eventDate"]:
            if field in record and record[field]:
                try:
                    return pd.to_datetime(record[field]).to_pydatetime()
                except:
                    pass
        
        return None
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_int(self, value: Any) -> int:
        """Safely convert value to int"""
        if value is None or value == "":
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0
    
    def _empty_dataframe(self) -> pd.DataFrame:
        """Return empty DataFrame with expected columns"""
        return pd.DataFrame(columns=[
            "disaster_id", "event_name", "date", "year",
            "country", "country_iso", "location", "latitude", "longitude",
            "disaster_type", "disaster_subtype", "disaster_group",
            "deaths", "injured", "affected", "homeless", "damage_usd",
            "source", "fetched_at"
        ])
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

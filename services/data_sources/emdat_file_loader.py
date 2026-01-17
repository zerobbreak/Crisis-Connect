"""
EM-DAT File Loader

Loads EM-DAT data from manually downloaded Excel or JSON files.
Use this when API access is not available.

Download from: https://www.emdat.be/ -> Public Data -> Download
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class EMDATFileLoader:
    """Load EM-DAT data from downloaded files"""
    
    # Column name mappings from EM-DAT export format
    COLUMN_MAPPINGS = {
        # Common EM-DAT column names
        "Dis No": "disaster_id",
        "DisNo": "disaster_id",
        "DisNo.": "disaster_id",  # EM-DAT 2023+ format
        "Disaster No": "disaster_id",
        "Year": "year",
        "Start Year": "year",
        "Start Month": "month",
        "Start Day": "day",
        "End Year": "end_year",
        "End Month": "end_month",
        "End Day": "end_day",
        "Country": "country",
        "ISO": "country_iso",
        "Region": "region",
        "Continent": "continent",
        "Location": "location",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Disaster Type": "disaster_type",
        "Disaster Subtype": "disaster_subtype",
        "Disaster Group": "disaster_group",
        "Disaster Subgroup": "disaster_subgroup",
        "Event Name": "event_name",
        "Origin": "origin",
        "Associated Dis": "associated_disaster",
        "Associated Dis2": "associated_disaster_2",
        "OFDA Response": "ofda_response",
        "Appeal": "appeal",
        "Declaration": "declaration",
        "Aid Contribution": "aid_contribution",
        "Dis Mag Value": "magnitude_value",
        "Dis Mag Scale": "magnitude_scale",
        "Total Deaths": "deaths",
        "No. Injured": "injured",  # EM-DAT 2023+ format
        "No Injured": "injured",
        "No. Affected": "affected",  # EM-DAT 2023+ format
        "No Affected": "affected",
        "No. Homeless": "homeless",  # EM-DAT 2023+ format
        "No Homeless": "homeless",
        "Total Affected": "total_affected",
        "Reconstruction Costs ('000 US$)": "reconstruction_costs",
        "Insured Damages ('000 US$)": "insured_damages",
        "Total Damages ('000 US$)": "damage_usd",
        "Total Damage ('000 US$)": "damage_usd",  # EM-DAT 2023+ format
        "CPI": "cpi",
    }
    
    # Southern Africa country filter
    SOUTHERN_AFRICA = [
        "South Africa", "Zimbabwe", "Botswana", "Namibia", "Mozambique",
        "Zambia", "Malawi", "Angola", "Eswatini", "Lesotho", "Madagascar",
        "Mauritius", "Comoros", "Seychelles"
    ]
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize loader
        
        Args:
            data_dir: Directory containing downloaded EM-DAT files
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def load_excel(
        self,
        filepath: str,
        filter_region: bool = True,
        start_year: int = 2000,
    ) -> pd.DataFrame:
        """
        Load EM-DAT data from Excel file
        
        Args:
            filepath: Path to Excel file
            filter_region: If True, filter to Southern Africa
            start_year: Minimum year to include
            
        Returns:
            Normalized DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        logger.info(f"Loading EM-DAT Excel: {filepath}")
        
        try:
            # EM-DAT exports often have multiple sheets or header rows
            df = pd.read_excel(filepath, sheet_name=0)
            
            # If first row looks like headers, use it
            if df.iloc[0].astype(str).str.contains('Year|Country|Disaster', case=False).any():
                df.columns = df.iloc[0]
                df = df.iloc[1:].reset_index(drop=True)
            
            logger.info(f"  Loaded {len(df)} raw records")
            
        except Exception as e:
            logger.error(f"Error reading Excel: {e}")
            return pd.DataFrame()
        
        return self._process_data(df, filter_region, start_year)
    
    def load_json(
        self,
        filepath: str,
        filter_region: bool = True,
        start_year: int = 2000,
    ) -> pd.DataFrame:
        """
        Load EM-DAT data from JSON file
        
        Args:
            filepath: Path to JSON file
            filter_region: If True, filter to Southern Africa
            start_year: Minimum year to include
            
        Returns:
            Normalized DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        logger.info(f"Loading EM-DAT JSON: {filepath}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"  Loaded {len(df)} raw records")
            
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return pd.DataFrame()
        
        return self._process_data(df, filter_region, start_year)
    
    def _process_data(
        self,
        df: pd.DataFrame,
        filter_region: bool,
        start_year: int,
    ) -> pd.DataFrame:
        """Process and normalize EM-DAT data"""
        
        if len(df) == 0:
            return df
        
        # Normalize column names
        df = df.rename(columns=self.COLUMN_MAPPINGS)
        
        # Also handle case variations
        df.columns = df.columns.str.strip()
        lower_mapping = {k.lower(): v for k, v in self.COLUMN_MAPPINGS.items()}
        df = df.rename(columns=lambda x: lower_mapping.get(x.lower(), x))
        
        # Create date column
        df = self._create_date_column(df)
        
        # Filter by region
        if filter_region and "country" in df.columns:
            before = len(df)
            df = df[df["country"].isin(self.SOUTHERN_AFRICA)]
            logger.info(f"  Filtered to Southern Africa: {before} -> {len(df)} records")
        
        # Filter by year
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            before = len(df)
            df = df[df["year"] >= start_year]
            logger.info(f"  Filtered to year >= {start_year}: {before} -> {len(df)} records")
        
        # Convert numeric columns
        numeric_cols = ["deaths", "injured", "affected", "homeless", "damage_usd",
                       "latitude", "longitude", "year"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Add metadata
        df["source"] = "emdat_file"
        df["loaded_at"] = datetime.now().isoformat()
        
        logger.info(f"  Final: {len(df)} records with {len(df.columns)} columns")
        
        return df
    
    def _create_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create date column from year/month/day"""
        
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            return df
        
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            
            if "month" in df.columns:
                df["month"] = pd.to_numeric(df["month"], errors="coerce").fillna(6)
            else:
                df["month"] = 6
            
            if "day" in df.columns:
                df["day"] = pd.to_numeric(df["day"], errors="coerce").fillna(15)
            else:
                df["day"] = 15
            
            # Create date
            df["date"] = pd.to_datetime(
                df[["year", "month", "day"]].astype(int, errors="ignore"),
                errors="coerce"
            )
        
        return df
    
    def load_and_save(
        self,
        filepath: str,
        output_name: str = "emdat",
    ) -> pd.DataFrame:
        """
        Load EM-DAT file and save to raw data directory
        
        Args:
            filepath: Path to downloaded file
            output_name: Name for output file
            
        Returns:
            Processed DataFrame
        """
        filepath = Path(filepath)
        
        # Detect file type
        if filepath.suffix.lower() in [".xlsx", ".xls"]:
            df = self.load_excel(filepath)
        elif filepath.suffix.lower() == ".json":
            df = self.load_json(filepath)
        else:
            logger.error(f"Unsupported file type: {filepath.suffix}")
            return pd.DataFrame()
        
        if len(df) > 0:
            # Save to raw directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.raw_dir / f"{output_name}_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"  Saved to: {output_path}")
        
        return df


def load_emdat_download(filepath: str) -> pd.DataFrame:
    """
    Quick function to load an EM-DAT download
    
    Usage:
        from services.data_sources.emdat_file_loader import load_emdat_download
        df = load_emdat_download("path/to/emdat_public_2024.xlsx")
    """
    loader = EMDATFileLoader()
    return loader.load_and_save(filepath)

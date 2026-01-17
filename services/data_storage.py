"""
Data Storage Management

Manages raw and processed data files with versioning, deduplication,
and consolidation capabilities.
"""

import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging
import json
import hashlib

logger = logging.getLogger(__name__)


class DataStorage:
    """Manage data files and versions"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data storage
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track file versions
        self.manifest_path = self.data_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load or create file manifest"""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "sources": {},
            "versions": [],
        }
    
    def _save_manifest(self):
        """Save file manifest"""
        try:
            with open(self.manifest_path, "w") as f:
                json.dump(self.manifest, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Error saving manifest: {e}")
    
    def save_raw(self, df: pd.DataFrame, source: str) -> Path:
        """
        Save raw data from a source with timestamp
        
        Args:
            df: DataFrame to save
            source: Source identifier (e.g., 'emdat', 'weather')
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source}_{timestamp}.csv"
        filepath = self.raw_dir / filename
        
        # Save with proper encoding
        df.to_csv(filepath, index=False, encoding="utf-8")
        
        # Update manifest
        if source not in self.manifest["sources"]:
            self.manifest["sources"][source] = []
        
        self.manifest["sources"][source].append({
            "filename": filename,
            "timestamp": timestamp,
            "records": len(df),
            "columns": list(df.columns),
        })
        self._save_manifest()
        
        logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath
    
    def load_raw(self, source: str, version: str = "latest") -> pd.DataFrame:
        """
        Load raw data from source
        
        Args:
            source: Source identifier
            version: 'latest', 'all', or specific timestamp
            
        Returns:
            DataFrame with raw data
        """
        files = list(self.raw_dir.glob(f"{source}_*.csv"))
        
        if not files:
            logger.warning(f"No raw data found for source: {source}")
            return pd.DataFrame()
        
        if version == "latest":
            # Get most recent file
            latest = sorted(files)[-1]
            return pd.read_csv(latest)
        
        elif version == "all":
            # Combine all versions
            dfs = []
            for f in sorted(files):
                try:
                    dfs.append(pd.read_csv(f))
                except Exception as e:
                    logger.warning(f"Error reading {f}: {e}")
            
            if not dfs:
                return pd.DataFrame()
            
            return pd.concat(dfs, ignore_index=True).drop_duplicates()
        
        else:
            # Look for specific version
            matching = [f for f in files if version in f.name]
            if matching:
                return pd.read_csv(matching[0])
            
            logger.warning(f"Version {version} not found for {source}")
            return pd.DataFrame()
    
    def save_processed(self, df: pd.DataFrame, name: str, version: bool = True) -> Path:
        """
        Save processed data
        
        Args:
            df: DataFrame to save
            name: Dataset name (e.g., 'disasters_master')
            version: Whether to create versioned copy
            
        Returns:
            Path to saved file
        """
        # Save main file
        main_path = self.processed_dir / f"{name}.csv"
        df.to_csv(main_path, index=False, encoding="utf-8")
        
        # Save versioned copy if requested
        if version:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_path = self.processed_dir / f"{name}_{timestamp}.csv"
            df.to_csv(version_path, index=False, encoding="utf-8")
        
        logger.info(f"Saved processed data: {main_path} ({len(df)} records)")
        return main_path
    
    def load_processed(self, name: str) -> pd.DataFrame:
        """
        Load processed data
        
        Args:
            name: Dataset name
            
        Returns:
            DataFrame with processed data
        """
        filepath = self.processed_dir / f"{name}.csv"
        
        if not filepath.exists():
            logger.warning(f"Processed data not found: {name}")
            return pd.DataFrame()
        
        return pd.read_csv(filepath)
    
    def consolidate_disasters(
        self,
        sources: Optional[List[str]] = None,
        include_local: bool = True,
    ) -> pd.DataFrame:
        """
        Combine all disaster sources into one dataset
        
        Args:
            sources: List of raw data sources to include
            include_local: Whether to include existing local data
            
        Returns:
            Consolidated DataFrame with duplicates removed
        """
        if sources is None:
            sources = ["emdat"]
        
        all_data = []
        
        # Load raw data from each source
        for source in sources:
            df = self.load_raw(source, version="latest")
            if len(df) > 0:
                df["data_source"] = source
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {source}")
        
        # Include existing local data (Excel file)
        if include_local:
            local_path = self.data_dir / "data_disaster.xlsx"
            if local_path.exists():
                try:
                    local_df = pd.read_excel(local_path)
                    local_df["data_source"] = "local_excel"
                    all_data.append(local_df)
                    logger.info(f"Loaded {len(local_df)} records from local Excel")
                except Exception as e:
                    logger.warning(f"Error loading local Excel: {e}")
        
        if not all_data:
            logger.warning("No data to consolidate")
            return pd.DataFrame()
        
        # Combine all data
        consolidated = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data: {len(consolidated)} total records")
        
        # Remove duplicates based on key fields
        before = len(consolidated)
        dedup_cols = ["date", "latitude", "longitude", "disaster_type"]
        available_cols = [c for c in dedup_cols if c in consolidated.columns]
        
        if available_cols:
            consolidated = consolidated.drop_duplicates(subset=available_cols, keep="first")
            logger.info(f"Removed {before - len(consolidated)} duplicate records")
        
        return consolidated
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all stored data"""
        summary = {
            "raw_sources": {},
            "processed_datasets": [],
            "total_raw_files": 0,
            "total_processed_files": 0,
        }
        
        # Count raw files by source
        for source, versions in self.manifest.get("sources", {}).items():
            summary["raw_sources"][source] = {
                "versions": len(versions),
                "latest_records": versions[-1]["records"] if versions else 0,
            }
            summary["total_raw_files"] += len(versions)
        
        # List processed datasets
        for f in self.processed_dir.glob("*.csv"):
            if not any(c.isdigit() for c in f.stem.split("_")[-1]):
                # Skip versioned copies
                summary["processed_datasets"].append({
                    "name": f.stem,
                    "size_mb": f.stat().st_size / (1024 * 1024),
                })
                summary["total_processed_files"] += 1
        
        return summary
    
    def cleanup_old_versions(self, keep_last: int = 5):
        """
        Remove old versions of raw data, keeping most recent
        
        Args:
            keep_last: Number of versions to keep per source
        """
        for source, versions in self.manifest.get("sources", {}).items():
            if len(versions) <= keep_last:
                continue
            
            # Remove old versions
            to_remove = versions[:-keep_last]
            for version in to_remove:
                filepath = self.raw_dir / version["filename"]
                if filepath.exists():
                    filepath.unlink()
                    logger.info(f"Removed old version: {version['filename']}")
            
            # Update manifest
            self.manifest["sources"][source] = versions[-keep_last:]
        
        self._save_manifest()

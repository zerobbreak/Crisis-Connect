"""
Pattern Detection Agent

Discovers recurring weather patterns that precede disasters by analyzing
historical sequences and finding clusters of similar conditions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("crisisconnect.agents.pattern_detection")


@dataclass
class Pattern:
    """Represents a discovered disaster pattern"""
    pattern_id: str
    pattern_type: str  # "flood", "drought", "storm", etc.
    sequence: np.ndarray  # The actual pattern (normalized)
    frequency: int  # How many times it occurred
    confidence: float  # How reliable is this pattern?
    precedes_disaster_pct: float  # In what % of cases did a disaster follow?
    days_to_disaster_avg: float  # Average days between pattern and disaster
    example_events: List[Dict] = field(default_factory=list)  # Real disaster events matching this pattern
    last_seen: datetime = field(default_factory=datetime.now)  # When was it last observed?
    feature_importance: Dict[str, float] = field(default_factory=dict)  # Which features matter most


class PatternDetectionAgent:
    """
    Agent that discovers recurring weather patterns preceding disasters
    
    Key insight: Disasters aren't random. They follow patterns:
    - Floods: 3-5 days of heavy rainfall
    - Droughts: Months of low rainfall and high temperature
    - Storms: Specific pressure and wind patterns
    
    This agent learns these patterns from historical data.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, min_pattern_occurrences: int = 3):
        """
        Args:
            similarity_threshold: How similar must sequences be to be considered the same pattern?
            min_pattern_occurrences: Minimum times a pattern must occur to be considered valid
        """
        self.similarity_threshold = similarity_threshold
        self.min_pattern_occurrences = min_pattern_occurrences
        self.discovered_patterns: Dict[str, Pattern] = {}
        self.pattern_counter = 0
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns: List[str] = []
        
        logger.info(f"PatternDetectionAgent initialized: threshold={similarity_threshold}, min_occurrences={min_pattern_occurrences}")
    
    def discover_patterns(self, 
                         sequences_data: pd.DataFrame,
                         labels: np.ndarray,
                         feature_columns: List[str],
                         disaster_type_column: str = 'disaster_type') -> Dict[str, Pattern]:
        """
        Discover patterns by analyzing all disaster sequences
        
        Algorithm:
        1. Normalize all sequences
        2. For each disaster sequence, find similar sequences (same disaster type)
        3. If N sequences are highly similar, create a pattern
        4. Calculate how reliable the pattern is
        
        Args:
            sequences_data: DataFrame containing weather sequences
            labels: Array of 0/1 indicating if disaster occurred
            feature_columns: List of feature column names
            disaster_type_column: Column containing disaster type
            
        Returns:
            Dictionary of discovered patterns
        """
        logger.info("Starting pattern discovery...")
        
        self.feature_columns = feature_columns
        
        # Extract feature values
        feature_values = sequences_data[feature_columns].values
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        
        # Normalize features
        sequences_normalized = self.scaler.fit_transform(feature_values)
        self.is_trained = True
        
        # Get disaster types
        if disaster_type_column in sequences_data.columns:
            disaster_types = sequences_data[disaster_type_column].unique()
        else:
            disaster_types = ['unknown']
        
        for disaster_type in disaster_types:
            if pd.isna(disaster_type):
                continue
                
            logger.info(f"Analyzing {disaster_type} patterns...")
            
            # Get all sequences for this disaster type where disaster occurred
            if disaster_type_column in sequences_data.columns:
                type_mask = sequences_data[disaster_type_column] == disaster_type
            else:
                type_mask = np.ones(len(sequences_data), dtype=bool)
            
            disaster_mask = labels == 1
            combined_mask = type_mask & disaster_mask
            
            disaster_indices = np.where(combined_mask)[0]
            
            if len(disaster_indices) < self.min_pattern_occurrences:
                logger.warning(f"Not enough {disaster_type} samples ({len(disaster_indices)})")
                continue
            
            disaster_sequences = sequences_normalized[disaster_indices]
            
            # Find clusters of similar sequences
            clusters = self._cluster_similar_sequences(
                disaster_sequences, 
                min_size=self.min_pattern_occurrences
            )
            
            # Create a pattern for each cluster
            for cluster_indices in clusters:
                # Map cluster indices back to original data indices
                original_indices = disaster_indices[cluster_indices]
                
                pattern = self._create_pattern_from_cluster(
                    cluster_indices,
                    disaster_sequences,
                    str(disaster_type),
                    sequences_data,
                    labels,
                    original_indices
                )
                
                if pattern.precedes_disaster_pct >= 0.6:  # At least 60% reliable
                    self.discovered_patterns[pattern.pattern_id] = pattern
                    logger.info(
                        f"  [OK] Pattern {pattern.pattern_type}#{pattern.pattern_id}: "
                        f"{pattern.frequency} occurrences, "
                        f"{pattern.precedes_disaster_pct:.1%} reliability"
                    )
        
        logger.info(f"Discovered {len(self.discovered_patterns)} patterns")
        return self.discovered_patterns
    
    def _cluster_similar_sequences(self, 
                                   sequences: np.ndarray,
                                   min_size: int = 3) -> List[List[int]]:
        """
        Find groups of similar sequences using cosine similarity
        
        Args:
            sequences: Normalized sequence array
            min_size: Minimum cluster size
            
        Returns:
            List of cluster index lists
        """
        if len(sequences) == 0:
            return []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(sequences)
        
        clusters = []
        used = set()
        
        for i in range(len(sequences)):
            if i in used:
                continue
            
            # Find all sequences similar to this one
            similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx not in used]
            
            if len(similar_indices) >= min_size:
                clusters.append(similar_indices)
                used.update(similar_indices)
        
        return clusters
    
    def _create_pattern_from_cluster(self,
                                     cluster_indices: List[int],
                                     sequences: np.ndarray,
                                     disaster_type: str,
                                     sequences_data: pd.DataFrame,
                                     labels: np.ndarray,
                                     original_indices: np.ndarray) -> Pattern:
        """Create a Pattern object from a cluster of similar sequences"""
        
        # Average sequence represents the pattern
        pattern_sequence = sequences[cluster_indices].mean(axis=0)
        
        # Calculate reliability
        disaster_count = np.sum(labels[original_indices])
        total_count = len(original_indices)
        reliability = disaster_count / total_count if total_count > 0 else 0
        
        # Calculate feature importance (which features vary least = most important)
        feature_std = sequences[cluster_indices].std(axis=0)
        feature_importance = {}
        for i, col in enumerate(self.feature_columns):
            # Lower std = more consistent = more important
            importance = 1.0 / (feature_std[i] + 0.1)
            feature_importance[col] = float(importance)
        
        # Normalize importance scores
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        # Estimate days to disaster
        days_to_disaster = self._estimate_days_to_disaster(sequences_data, original_indices)
        
        # Create example events
        example_events = []
        for idx in original_indices[:5]:  # Top 5 examples
            event = {
                "index": int(idx),
                "disaster_occurred": bool(labels[idx])
            }
            if 'date' in sequences_data.columns:
                event["date"] = str(sequences_data.iloc[idx].get('date', 'N/A'))
            if 'location' in sequences_data.columns:
                event["location"] = str(sequences_data.iloc[idx].get('location', 'N/A'))
            elif 'country' in sequences_data.columns:
                event["location"] = str(sequences_data.iloc[idx].get('country', 'N/A'))
            example_events.append(event)
        
        # Create pattern ID
        pattern_id = f"{disaster_type}_{self.pattern_counter}"
        self.pattern_counter += 1
        
        # Calculate confidence based on cluster cohesion
        if len(cluster_indices) > 1:
            cluster_similarity = cosine_similarity(sequences[cluster_indices])
            avg_similarity = (cluster_similarity.sum() - len(cluster_indices)) / (len(cluster_indices) * (len(cluster_indices) - 1))
            confidence = float(avg_similarity)
        else:
            confidence = 0.5
        
        return Pattern(
            pattern_id=pattern_id,
            pattern_type=disaster_type,
            sequence=pattern_sequence,
            frequency=len(cluster_indices),
            confidence=confidence,
            precedes_disaster_pct=reliability,
            days_to_disaster_avg=days_to_disaster,
            example_events=example_events,
            last_seen=datetime.now(),
            feature_importance=feature_importance
        )
    
    def _estimate_days_to_disaster(self, 
                                   sequences_data: pd.DataFrame,
                                   indices: np.ndarray) -> float:
        """
        Estimate average days from pattern to disaster occurrence
        
        In production, this would track actual delays from historical data.
        For now, we use a simplified estimate based on disaster type.
        """
        # Check if we have date information
        if 'date' in sequences_data.columns and 'disaster_date' in sequences_data.columns:
            try:
                dates = pd.to_datetime(sequences_data.iloc[indices]['date'])
                disaster_dates = pd.to_datetime(sequences_data.iloc[indices]['disaster_date'])
                days_diff = (disaster_dates - dates).dt.days
                return float(days_diff.mean())
            except Exception:
                pass
        
        # Default estimates by disaster type (based on typical development times)
        if len(indices) > 0 and 'disaster_type' in sequences_data.columns:
            disaster_type = sequences_data.iloc[indices[0]].get('disaster_type', 'unknown')
            type_estimates = {
                'flood': 3.0,
                'Flood': 3.0,
                'storm': 2.0,
                'Storm': 2.0,
                'drought': 30.0,
                'Drought': 30.0,
                'landslide': 1.0,
                'Landslide': 1.0,
            }
            return type_estimates.get(disaster_type, 5.0)
        
        return 5.0  # Default estimate
    
    def match_current_sequence(self, current_sequence: np.ndarray) -> List[Tuple[Pattern, float]]:
        """
        Match current weather sequence against discovered patterns
        
        Args:
            current_sequence: Current weather data (1D or 2D array)
            
        Returns:
            List of (matching_pattern, similarity_score) sorted by similarity
        """
        if not self.is_trained:
            logger.warning("Pattern agent not trained yet")
            return []
        
        if len(self.discovered_patterns) == 0:
            logger.warning("No patterns discovered yet")
            return []
        
        # Ensure 2D shape
        if current_sequence.ndim == 1:
            current_sequence = current_sequence.reshape(1, -1)
        
        # Handle if sequence is 3D (batch, timesteps, features)
        if current_sequence.ndim == 3:
            # Flatten timesteps and features or take mean
            current_sequence = current_sequence.mean(axis=1)
        
        # Normalize using the same scaler
        try:
            current_normalized = self.scaler.transform(current_sequence)[0]
        except Exception as e:
            logger.warning(f"Error normalizing sequence: {e}")
            return []
        
        matches = []
        
        for pattern_id, pattern in self.discovered_patterns.items():
            try:
                similarity = cosine_similarity(
                    current_normalized.reshape(1, -1),
                    pattern.sequence.reshape(1, -1)
                )[0][0]
                
                if similarity > self.similarity_threshold * 0.8:  # Slightly lower threshold for matching
                    matches.append((pattern, float(similarity)))
            except Exception as e:
                logger.warning(f"Error matching pattern {pattern_id}: {e}")
                continue
        
        # Sort by similarity (best matches first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of discovered patterns"""
        if not self.discovered_patterns:
            return {"total_patterns": 0, "patterns": []}
        
        summary = {
            "total_patterns": len(self.discovered_patterns),
            "patterns_by_type": {},
            "patterns": []
        }
        
        for pattern_id, pattern in self.discovered_patterns.items():
            if pattern.pattern_type not in summary["patterns_by_type"]:
                summary["patterns_by_type"][pattern.pattern_type] = 0
            summary["patterns_by_type"][pattern.pattern_type] += 1
            
            summary["patterns"].append({
                "id": pattern.pattern_id,
                "type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "reliability": pattern.precedes_disaster_pct,
                "days_to_disaster": pattern.days_to_disaster_avg
            })
        
        return summary
    
    def save_patterns(self, filepath: str):
        """Save discovered patterns to file"""
        import json
        
        patterns_data = {}
        for pattern_id, pattern in self.discovered_patterns.items():
            patterns_data[pattern_id] = {
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "sequence": pattern.sequence.tolist(),
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
                "precedes_disaster_pct": pattern.precedes_disaster_pct,
                "days_to_disaster_avg": pattern.days_to_disaster_avg,
                "example_events": pattern.example_events,
                "last_seen": pattern.last_seen.isoformat(),
                "feature_importance": pattern.feature_importance
            }
        
        with open(filepath, 'w') as f:
            json.dump(patterns_data, f, indent=2)
        
        logger.info(f"Saved {len(patterns_data)} patterns to {filepath}")
    
    def load_patterns(self, filepath: str):
        """Load patterns from file"""
        import json
        
        with open(filepath, 'r') as f:
            patterns_data = json.load(f)
        
        self.discovered_patterns = {}
        for pattern_id, data in patterns_data.items():
            self.discovered_patterns[pattern_id] = Pattern(
                pattern_id=data["pattern_id"],
                pattern_type=data["pattern_type"],
                sequence=np.array(data["sequence"]),
                frequency=data["frequency"],
                confidence=data["confidence"],
                precedes_disaster_pct=data["precedes_disaster_pct"],
                days_to_disaster_avg=data["days_to_disaster_avg"],
                example_events=data.get("example_events", []),
                last_seen=datetime.fromisoformat(data.get("last_seen", datetime.now().isoformat())),
                feature_importance=data.get("feature_importance", {})
            )
        
        logger.info(f"Loaded {len(self.discovered_patterns)} patterns from {filepath}")

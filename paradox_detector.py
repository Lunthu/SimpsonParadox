"""
Simpson's Paradox and Hidden Pattern Detector
Advanced statistical analysis for revealing counterintuitive patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ParadoxDetector:
    """Detect Simpson's Paradox and other hidden statistical patterns"""
    
    def __init__(self, data: pd.DataFrame, 
                 metrics: List[str], 
                 dimensions: List[str],
                 significance_level: float = 0.05):
        """
        Initialize paradox detector
        
        Args:
            data: DataFrame with analysis data
            metrics: List of metric column names
            dimensions: List of dimension column names
            significance_level: P-value threshold
        """
        self.data = data
        self.metrics = metrics
        self.dimensions = dimensions
        self.significance_level = significance_level
        self.paradoxes = []
        
    def detect_simpsons_paradox(self) -> List[Dict]:
        """
        Detect Simpson's Paradox: where trends in groups reverse when aggregated
        
        Example: Each hospital has higher success rate for treatment A,
        but combined data shows treatment B is better (due to case mix)
        """
        paradoxes = []
        
        for metric_x in self.metrics:
            for metric_y in self.metrics:
                if metric_x >= metric_y:
                    continue
                
                for dimension in self.dimensions:
                    paradox = self._check_simpson_for_pair(
                        metric_x, metric_y, dimension
                    )
                    if paradox:
                        paradoxes.append(paradox)
        
        self.paradoxes.extend(paradoxes)
        return paradoxes
    
    def _check_simpson_for_pair(self, 
                                metric_x: str, 
                                metric_y: str, 
                                dimension: str) -> Optional[Dict]:
        """
        Check if a Simpson's Paradox exists for a metric pair
        """
        # Clean data
        clean_data = self.data[[metric_x, metric_y, dimension]].dropna()
        
        if len(clean_data) < 10:
            return None
        
        # Overall correlation
        overall_corr, overall_p = stats.pearsonr(
            clean_data[metric_x], 
            clean_data[metric_y]
        )
        
        # Group-specific correlations
        groups = clean_data.groupby(dimension)
        group_corrs = {}
        group_directions = []
        
        for group_name, group_df in groups:
            if len(group_df) < 3:
                continue
            
            # Skip if no variance in either variable
            if group_df[metric_x].nunique() <= 1 or group_df[metric_y].nunique() <= 1:
                continue
            
            try:
                group_corr, group_p = stats.pearsonr(
                    group_df[metric_x], 
                    group_df[metric_y]
                )
                # Skip if correlation is NaN (no variance in data)
                if np.isnan(group_corr) or np.isnan(group_p):
                    continue
                    
                group_corrs[group_name] = {
                    'correlation': group_corr,
                    'p_value': group_p,
                    'n': len(group_df),
                    'significant': group_p < self.significance_level
                }
                group_directions.append(np.sign(group_corr))
            except:
                continue
        
        if len(group_corrs) < 2:
            return None
        
        # Check for paradox: overall direction differs from most groups
        # OR all groups agree but overall is opposite
        overall_sign = np.sign(overall_corr)
        group_signs = np.array(group_directions)
        
        # Classic Simpson's Paradox: all groups trend one way, overall trends opposite
        if len(group_signs) >= 2:
            # All groups have same sign (agreement)
            groups_agree = np.all(group_signs == group_signs[0])
            
            # Overall sign is opposite
            overall_opposite = overall_sign != group_signs[0]
            
            # Both correlations are significant
            overall_significant = overall_p < self.significance_level
            groups_significant = np.mean([g['significant'] for g in group_corrs.values()]) > 0.5
            
            if groups_agree and overall_opposite and overall_significant and groups_significant:
                # Calculate effect sizes
                avg_group_corr = np.mean([g['correlation'] for g in group_corrs.values()])
                reversal_magnitude = abs(overall_corr - avg_group_corr)
                
                return {
                    'type': 'simpsons_paradox',
                    'severity': 'high' if reversal_magnitude > 0.5 else 'moderate',
                    'metric_x': metric_x,
                    'metric_y': metric_y,
                    'dimension': dimension,
                    'overall_correlation': overall_corr,
                    'overall_p_value': overall_p,
                    'group_correlations': group_corrs,
                    'average_group_correlation': avg_group_corr,
                    'reversal_magnitude': reversal_magnitude,
                    'description': self._format_simpson_description(
                        metric_x, metric_y, dimension, 
                        overall_corr, avg_group_corr, group_corrs
                    )
                }
        
        # Partial Simpson's Paradox: majority of groups trend opposite to overall
        if len(group_signs) >= 3:
            majority_sign = 1 if np.sum(group_signs > 0) > len(group_signs) / 2 else -1
            
            if overall_sign != majority_sign and overall_p < self.significance_level:
                avg_group_corr = np.mean([g['correlation'] for g in group_corrs.values()])
                reversal_magnitude = abs(overall_corr - avg_group_corr)
                
                return {
                    'type': 'partial_simpsons_paradox',
                    'severity': 'moderate',
                    'metric_x': metric_x,
                    'metric_y': metric_y,
                    'dimension': dimension,
                    'overall_correlation': overall_corr,
                    'overall_p_value': overall_p,
                    'group_correlations': group_corrs,
                    'average_group_correlation': avg_group_corr,
                    'reversal_magnitude': reversal_magnitude,
                    'description': self._format_simpson_description(
                        metric_x, metric_y, dimension, 
                        overall_corr, avg_group_corr, group_corrs,
                        partial=True
                    )
                }
        
        return None
    
    def _format_simpson_description(self, 
                                    metric_x: str, 
                                    metric_y: str,
                                    dimension: str,
                                    overall_corr: float,
                                    avg_group_corr: float,
                                    group_corrs: Dict,
                                    partial: bool = False) -> str:
        """Format human-readable description of Simpson's Paradox"""
        
        overall_dir = "positive" if overall_corr > 0 else "negative"
        group_dir = "positive" if avg_group_corr > 0 else "negative"
        
        prefix = "⚠️ PARTIAL SIMPSON'S PARADOX" if partial else "🚨 SIMPSON'S PARADOX DETECTED"
        
        group_details = ", ".join([
            f"{name} (r={info['correlation']:.2f})" 
            for name, info in list(group_corrs.items())[:3]
        ])
        
        description = (
            f"{prefix}: {metric_x} vs {metric_y}\n"
            f"Overall correlation: {overall_dir} (r={overall_corr:.3f})\n"
            f"But within each {dimension} group: {group_dir} (avg r={avg_group_corr:.3f})\n"
            f"Groups: {group_details}"
        )
        
        return description
    
    def detect_confounding_variables(self) -> List[Dict]:
        """
        Detect confounding variables that hide or create false correlations
        """
        confounders = []
        
        for metric_x in self.metrics:
            for metric_y in self.metrics:
                if metric_x >= metric_y:
                    continue
                
                for dimension in self.dimensions:
                    confounder = self._check_confounding(metric_x, metric_y, dimension)
                    if confounder:
                        confounders.append(confounder)
        
        return confounders
    
    def _check_confounding(self,
                          metric_x: str,
                          metric_y: str,
                          potential_confounder: str) -> Optional[Dict]:
        """
        Check if dimension acts as confounding variable
        
        A confounder affects both variables and creates spurious correlation
        """
        clean_data = self.data[[metric_x, metric_y, potential_confounder]].dropna()
        
        if len(clean_data) < 10:
            return None
        
        # Overall correlation
        overall_corr, overall_p = stats.pearsonr(
            clean_data[metric_x], 
            clean_data[metric_y]
        )
        
        if overall_p >= self.significance_level:
            return None
        
        # Check if controlling for confounder eliminates correlation
        groups = clean_data.groupby(potential_confounder)
        within_group_corrs = []
        
        for group_name, group_df in groups:
            if len(group_df) >= 3:
                try:
                    corr, p = stats.pearsonr(group_df[metric_x], group_df[metric_y])
                    within_group_corrs.append(corr)
                except:
                    continue
        
        if len(within_group_corrs) < 2:
            return None
        
        avg_within_corr = np.mean(within_group_corrs)
        
        # Confounding detected: strong overall correlation weakens substantially within groups
        if abs(overall_corr) > 0.4 and abs(avg_within_corr) < 0.2:
            return {
                'type': 'confounding_variable',
                'metric_x': metric_x,
                'metric_y': metric_y,
                'confounder': potential_confounder,
                'overall_correlation': overall_corr,
                'within_group_correlation': avg_within_corr,
                'attenuation': abs(overall_corr - avg_within_corr),
                'description': (
                    f"🔍 CONFOUNDING DETECTED: {potential_confounder} confounds {metric_x}-{metric_y}\n"
                    f"Overall correlation: r={overall_corr:.3f}\n"
                    f"Within-group correlation: r={avg_within_corr:.3f}\n"
                    f"The strong overall correlation is largely due to {potential_confounder}"
                )
            }
        
        return None
    
    def detect_interaction_effects(self) -> List[Dict]:
        """
        Detect interaction effects: where relationship between X and Y
        depends on the level of a third variable
        """
        interactions = []
        
        for metric_x in self.metrics:
            for metric_y in self.metrics:
                if metric_x >= metric_y:
                    continue
                
                for dimension in self.dimensions:
                    interaction = self._check_interaction(metric_x, metric_y, dimension)
                    if interaction:
                        interactions.append(interaction)
        
        return interactions
    
    def _check_interaction(self,
                          metric_x: str,
                          metric_y: str,
                          moderator: str) -> Optional[Dict]:
        """
        Check if dimension moderates the relationship between metrics
        """
        clean_data = self.data[[metric_x, metric_y, moderator]].dropna()
        
        if len(clean_data) < 10:
            return None
        
        # Get group-specific correlations
        groups = clean_data.groupby(moderator)
        group_corrs = []
        group_info = {}
        
        for group_name, group_df in groups:
            if len(group_df) >= 3:
                try:
                    corr, p = stats.pearsonr(group_df[metric_x], group_df[metric_y])
                    group_corrs.append(corr)
                    group_info[group_name] = {
                        'correlation': corr,
                        'p_value': p,
                        'n': len(group_df)
                    }
                except:
                    continue
        
        if len(group_corrs) < 2:
            return None
        
        # Strong interaction: correlations vary substantially across groups
        corr_std = np.std(group_corrs)
        corr_range = max(group_corrs) - min(group_corrs)
        
        if corr_std > 0.3 or corr_range > 0.6:
            strongest_group = max(group_info.items(), key=lambda x: abs(x[1]['correlation']))
            weakest_group = min(group_info.items(), key=lambda x: abs(x[1]['correlation']))
            
            return {
                'type': 'interaction_effect',
                'metric_x': metric_x,
                'metric_y': metric_y,
                'moderator': moderator,
                'correlation_std': corr_std,
                'correlation_range': corr_range,
                'group_correlations': group_info,
                'strongest_group': strongest_group[0],
                'strongest_correlation': strongest_group[1]['correlation'],
                'weakest_group': weakest_group[0],
                'weakest_correlation': weakest_group[1]['correlation'],
                'description': (
                    f"⚡ INTERACTION EFFECT: {moderator} moderates {metric_x}-{metric_y}\n"
                    f"Correlation varies from {min(group_corrs):.3f} to {max(group_corrs):.3f}\n"
                    f"Strongest in {strongest_group[0]} (r={strongest_group[1]['correlation']:.3f})\n"
                    f"Weakest in {weakest_group[0]} (r={weakest_group[1]['correlation']:.3f})"
                )
            }
        
        return None
    
    def detect_subgroup_reversals(self) -> List[Dict]:
        """
        Detect cases where metric relationships reverse in specific subgroups
        """
        reversals = []
        
        for metric_x in self.metrics:
            for metric_y in self.metrics:
                if metric_x >= metric_y:
                    continue
                
                for dimension in self.dimensions:
                    reversal = self._check_subgroup_reversal(metric_x, metric_y, dimension)
                    if reversal:
                        reversals.append(reversal)
        
        return reversals
    
    def _check_subgroup_reversal(self,
                                 metric_x: str,
                                 metric_y: str,
                                 dimension: str) -> Optional[Dict]:
        """
        Check if any subgroup shows opposite correlation direction
        """
        clean_data = self.data[[metric_x, metric_y, dimension]].dropna()
        
        if len(clean_data) < 10:
            return None
        
        # Overall correlation
        overall_corr, overall_p = stats.pearsonr(
            clean_data[metric_x], 
            clean_data[metric_y]
        )
        
        if overall_p >= self.significance_level or abs(overall_corr) < 0.3:
            return None
        
        # Check each group
        groups = clean_data.groupby(dimension)
        reversed_groups = []
        all_groups = []
        
        for group_name, group_df in groups:
            if len(group_df) >= 5:
                # Skip if no variance in either variable
                if group_df[metric_x].nunique() <= 1 or group_df[metric_y].nunique() <= 1:
                    continue
                
                try:
                    group_corr, group_p = stats.pearsonr(
                        group_df[metric_x], 
                        group_df[metric_y]
                    )
                    
                    # Skip if NaN
                    if np.isnan(group_corr) or np.isnan(group_p):
                        continue
                    
                    group_info = {
                        'group': group_name,
                        'correlation': group_corr,
                        'p_value': group_p,
                        'n': len(group_df),
                        'is_reversed': False
                    }
                    
                    # Check for reversal
                    if (np.sign(group_corr) != np.sign(overall_corr) and 
                        group_p < self.significance_level):
                        group_info['is_reversed'] = True
                        reversed_groups.append(group_info)
                    
                    all_groups.append(group_info)
                except:
                    continue
        
        # Only report if we have at least 2 valid groups and at least 1 reversal
        if reversed_groups and len(all_groups) >= 2:
            return {
                'type': 'subgroup_reversal',
                'metric_x': metric_x,
                'metric_y': metric_y,
                'dimension': dimension,
                'overall_correlation': overall_corr,
                'overall_p_value': overall_p,
                'reversed_groups': reversed_groups,
                'all_groups': all_groups,
                'description': (
                    f"🔄 SUBGROUP REVERSAL: {metric_x} vs {metric_y}\n"
                    f"Overall: {overall_corr:.3f}, but reversed in:\n" +
                    "\n".join([
                        f"  • {g['group']}: r={g['correlation']:.3f}"
                        for g in reversed_groups
                    ])
                )
            }
        
        return None
    
    def get_all_patterns(self) -> Dict[str, List[Dict]]:
        """
        Run all pattern detection methods and return comprehensive results
        """
        print("🔍 Detecting Simpson's Paradoxes...")
        simpsons = self.detect_simpsons_paradox()
        
        print("🔍 Detecting Confounding Variables...")
        confounders = self.detect_confounding_variables()
        
        print("🔍 Detecting Interaction Effects...")
        interactions = self.detect_interaction_effects()
        
        print("🔍 Detecting Subgroup Reversals...")
        reversals = self.detect_subgroup_reversals()
        
        return {
            'simpsons_paradox': simpsons,
            'confounding': confounders,
            'interactions': interactions,
            'reversals': reversals,
            'total_patterns': len(simpsons) + len(confounders) + len(interactions) + len(reversals)
        }
    
    def get_priority_patterns(self, top_n: int = 10) -> List[Dict]:
        """
        Get top N most important patterns by severity/impact
        """
        all_patterns = []
        
        # Add all detected patterns with priority scores
        for pattern in self.paradoxes:
            priority = self._calculate_priority(pattern)
            pattern['priority_score'] = priority
            all_patterns.append(pattern)
        
        # Sort by priority
        all_patterns.sort(key=lambda x: x.get('priority_score', 0), reverse=True)
        
        return all_patterns[:top_n]
    
    def _calculate_priority(self, pattern: Dict) -> float:
        """Calculate priority score for a pattern"""
        score = 0.0
        
        # Simpson's Paradox gets highest priority
        if pattern['type'] == 'simpsons_paradox':
            score += 100
            if pattern.get('severity') == 'high':
                score += 50
        elif pattern['type'] == 'partial_simpsons_paradox':
            score += 80
        
        # Confounding variables
        elif pattern['type'] == 'confounding_variable':
            score += 70
            score += pattern.get('attenuation', 0) * 50
        
        # Interaction effects
        elif pattern['type'] == 'interaction_effect':
            score += 60
            score += pattern.get('correlation_range', 0) * 30
        
        # Subgroup reversals
        elif pattern['type'] == 'subgroup_reversal':
            score += 75
            score += len(pattern.get('reversed_groups', [])) * 10
        
        return score

#!/usr/bin/env python3
"""
SOAR Patch Validation Harness
====================================================

Purpose: Automatically extract 4 key metrics before/after patches

Patches:
  A: Orbit Guard (don't grammar cut if orbit='?')
  B: Close History (use tick-level data)
  C: Min Retries (2-3 instead of 6)

Metrics (4):
  1. GRAMMAR_CUT percentage
  2. GRAMMAR_CUT >= 240s percentage
  3. Sample count increase in 240-300s / 300-420s bins
  4. Trades "moved to" MAX_HOLD / MFE_SLOPE_CUT (from GRAMMAR)

Usage:
  python VALIDATION_HARNESS.py --baseline exit_events_baseline.jsonl
  python VALIDATION_HARNESS.py --patch-a exit_events_patch_a.jsonl
  python VALIDATION_HARNESS.py --patch-ab exit_events_patch_ab.jsonl
  python VALIDATION_HARNESS.py --patch-abc exit_events_patch_abc.jsonl
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


class ValidationHarness:
    """Analyze exit_events JSONL and extract 4 key metrics"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.events = []
        self.load_events()
    
    def load_events(self):
        """Load exit_events from JSONL"""
        try:
            with open(self.filepath) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        self.events.append(event)
                    except:
                        pass
            print(f"  Loaded {len(self.events)} events from {self.filepath}")
        except FileNotFoundError:
            print(f"  ERROR: File not found {self.filepath}")
            sys.exit(1)
    
    def metric_1_grammar_percentage(self):
        """Metric 1: What % of exits are GRAMMAR_CUT?"""
        total = len(self.events)
        grammar_count = len([e for e in self.events if e.get('exit_reason') == 'GRAMMAR_CUT'])
        
        if total == 0:
            return 0.0
        
        pct = 100 * grammar_count / total
        return pct
    
    def metric_2_grammar_240plus(self):
        """Metric 2: Of GRAMMAR_CUT, what % are >= 240s?"""
        grammar_events = [e for e in self.events if e.get('exit_reason') == 'GRAMMAR_CUT']
        
        if len(grammar_events) == 0:
            return 0.0
        
        grammar_240plus = len([e for e in grammar_events 
                              if e.get('elapsed_sec', e.get('duration_sec', 0)) >= 240])
        
        pct = 100 * grammar_240plus / len(grammar_events)
        return pct
    
    def metric_3_sample_expansion(self):
        """Metric 3: Sample count increase in 240-300 / 300-420 bins"""
        
        bin_240_300 = len([e for e in self.events 
                          if 240 <= e.get('elapsed_sec', e.get('duration_sec', 0)) < 300])
        bin_300_420 = len([e for e in self.events 
                          if 300 <= e.get('elapsed_sec', e.get('duration_sec', 0)) < 420])
        
        total_in_zone = bin_240_300 + bin_300_420
        return {
            'bin_240_300': bin_240_300,
            'bin_300_420': bin_300_420,
            'total': total_in_zone
        }
    
    def metric_4_grammar_migration(self):
        """Metric 4: How many trades migrated FROM grammar TO mfe/max_hold?"""
        
        # Count trades that would have been grammar but are now MFE/MAX_HOLD
        # (This is harder to compute exactly, so we'll use a heuristic:
        #  count MFE_SLOPE + MAX_HOLD in the 167-240s range where grammar typically fired)
        
        migrated_candidates = len([e for e in self.events 
                                  if e.get('exit_reason') in ['MFE_SLOPE_CUT', 'MAX_HOLD']
                                  and 167 <= e.get('elapsed_sec', e.get('duration_sec', 0)) < 240])
        
        return migrated_candidates
    
    def analyze(self):
        """Run all 4 metrics"""
        print(f"\n{'='*70}")
        print(f"VALIDATION ANALYSIS: {Path(self.filepath).name}")
        print(f"{'='*70}")
        
        m1 = self.metric_1_grammar_percentage()
        m2 = self.metric_2_grammar_240plus()
        m3 = self.metric_3_sample_expansion()
        m4 = self.metric_4_grammar_migration()
        
        print(f"\nMETRIC 1: GRAMMAR_CUT Percentage")
        print(f"  Total events: {len(self.events)}")
        print(f"  GRAMMAR_CUT: {len([e for e in self.events if e.get('exit_reason') == 'GRAMMAR_CUT'])} events")
        print(f"  → {m1:.2f}% of all exits")
        
        print(f"\nMETRIC 2: GRAMMAR_CUT >= 240s (Profitable Zone)")
        grammar_events = [e for e in self.events if e.get('exit_reason') == 'GRAMMAR_CUT']
        print(f"  GRAMMAR_CUT events >= 240s: {len([e for e in grammar_events if e.get('elapsed_sec', e.get('duration_sec', 0)) >= 240])}")
        print(f"  → {m2:.2f}% of GRAMMAR_CUT")
        print(f"  (Higher is better — means grammar waiting longer before cutting)")
        
        print(f"\nMETRIC 3: Sample Expansion in Expansion Zone (240-420s)")
        print(f"  240-300s bin: {m3['bin_240_300']} samples")
        print(f"  300-420s bin: {m3['bin_300_420']} samples")
        print(f"  Total in zone: {m3['total']} samples")
        print(f"  (Higher is better — means more trades entering expansion zone)")
        
        print(f"\nMETRIC 4: Grammar Migration (MFE/MAX_HOLD in 167-240s zone)")
        print(f"  Trades moved to MFE/MAX_HOLD in 167-240s: {m4}")
        print(f"  (Higher is better — trades getting out via profit/timeout, not grammar)")
        
        return {
            'grammar_percentage': m1,
            'grammar_240plus': m2,
            'sample_expansion': m3,
            'grammar_migration': m4
        }


def compare_runs(baseline, patch_a, patch_ab, patch_abc):
    """Compare metrics across all patch levels"""
    
    print(f"\n\n{'='*70}")
    print(f"PATCH PROGRESSION COMPARISON")
    print(f"{'='*70}")
    
    runs = [
        ("BASELINE", baseline),
        ("PATCH_A (Orbit Guard)", patch_a),
        ("PATCH_A+B (+ Close History)", patch_ab),
        ("PATCH_A+B+C (+ Min Retries)", patch_abc)
    ]
    
    results = {}
    for label, harness in runs:
        if harness:
            results[label] = harness.analyze()
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"METRIC PROGRESSION TABLE")
    print(f"{'='*70}")
    
    print(f"\n{'Stage':<30} {'Grammar %':<15} {'Grammar 240+':<15} {'Expansion':<15} {'Migration':<15}")
    print(f"{'-'*75}")
    
    for label, metrics in results.items():
        g_pct = metrics['grammar_percentage']
        g_240 = metrics['grammar_240plus']
        expand = metrics['sample_expansion']['total']
        migr = metrics['grammar_migration']
        
        print(f"{label:<30} {g_pct:>6.1f}%{'':<8} {g_240:>6.1f}%{'':<8} {expand:>6.0f}{'':<8} {migr:>6.0f}")
    
    # Expected direction
    print(f"\n{'='*70}")
    print(f"HYPOTHESIS VALIDATION")
    print(f"{'='*70}")
    
    print("""
✓ Expected Directions:
  1. Grammar % should DECREASE (fewer grammar exits)
  2. Grammar 240+ should INCREASE (grammars that do fire are later)
  3. Sample Expansion should INCREASE (more trades in 240-420s zone)
  4. Grammar Migration should INCREASE (grammar trades exit via other paths)

If these ALL move in expected direction → HYPOTHESIS CONFIRMED
""")


def main():
    parser = argparse.ArgumentParser(
        description="SOAR Patch Validation Harness — Automatic metric extraction"
    )
    parser.add_argument('--baseline', type=str, help='Baseline exit_events JSONL')
    parser.add_argument('--patch-a', type=str, help='Patch A exit_events JSONL')
    parser.add_argument('--patch-ab', type=str, help='Patch A+B exit_events JSONL')
    parser.add_argument('--patch-abc', type=str, help='Patch A+B+C exit_events JSONL')
    
    args = parser.parse_args()
    
    # Load all available runs
    baseline = ValidationHarness(args.baseline) if args.baseline else None
    patch_a = ValidationHarness(args.patch_a) if args.patch_a else None
    patch_ab = ValidationHarness(args.patch_ab) if args.patch_ab else None
    patch_abc = ValidationHarness(args.patch_abc) if args.patch_abc else None
    
    # If only one file provided, just analyze that one
    if args.baseline and not args.patch_a:
        baseline.analyze()
    else:
        # Compare all runs
        compare_runs(baseline, patch_a, patch_ab, patch_abc)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test Transfer Learning System

Quick test to verify the entire pipeline works.
"""

import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from tennis_simulator import generate_training_dataset
from pretrain_tennis_rules import pretrain_tennis_rules
import pandas as pd


def test_simulator():
    """Test synthetic match generation."""
    print("\n" + "="*80)
    print("TEST 1: Tennis Simulator")
    print("="*80)
    
    print("\nGenerating 100 test matches...")
    df = generate_training_dataset(
        n_matches=100,
        output_path=None,
        best_of_5=True,
        seed=42
    )
    
    print(f"\n✓ Generated {len(df)} points")
    print(f"  Average points per match: {len(df)/100:.1f}")
    print(f"  Features: {df.columns.tolist()}")
    
    # Check for critical points
    match_points = df['point_winner'].value_counts()
    print(f"\n  Point winners distribution:")
    print(f"    P1: {match_points.get(1, 0)} ({match_points.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    P2: {match_points.get(2, 0)} ({match_points.get(2, 0)/len(df)*100:.1f}%)")
    
    # Check match outcomes
    match_outcomes = df.groupby('match_id')['p1_wins_match'].first()
    print(f"\n  Match outcomes:")
    print(f"    P1 wins: {match_outcomes.sum():.0f} ({match_outcomes.mean()*100:.1f}%)")
    print(f"    P2 wins: {(1-match_outcomes).sum():.0f} ({(1-match_outcomes.mean())*100:.1f}%)")
    
    # Check tiebreaks
    tiebreak_mask = (df['p1_points'] > 3) | (df['p2_points'] > 3)
    n_tiebreak_points = tiebreak_mask.sum()
    print(f"\n  Tiebreak points: {n_tiebreak_points} ({n_tiebreak_points/len(df)*100:.1f}%)")
    
    return True


def test_feature_computation():
    """Test feature computation on synthetic data."""
    print("\n" + "="*80)
    print("TEST 2: Feature Computation")
    print("="*80)
    
    from pretrain_tennis_rules import compute_tennis_features, compute_labels
    
    print("\nGenerating 10 matches for feature test...")
    df = generate_training_dataset(n_matches=10, output_path=None, seed=42)
    
    print("Computing features...")
    X = compute_tennis_features(df)
    y_match, y_set, y_game = compute_labels(df)
    
    print(f"\n✓ Features computed successfully")
    print(f"  Shape: {X.shape}")
    print(f"  Expected: ({len(df)}, 31)")
    print(f"  Match: {X.shape == (len(df), 31)}")
    
    # Check feature ranges
    print(f"\n  Feature ranges:")
    print(f"    Min: {X.min():.3f}")
    print(f"    Max: {X.max():.3f}")
    print(f"    Mean: {X.mean():.3f}")
    
    # Check critical point features (indices 22-29)
    is_game_point_p1 = X[:, 22]
    is_set_point_p1 = X[:, 24]
    is_match_point_p1 = X[:, 26]
    
    print(f"\n  Critical points detected:")
    print(f"    Game points: {(is_game_point_p1 > 0.5).sum()}")
    print(f"    Set points: {(is_set_point_p1 > 0.5).sum()}")
    print(f"    Match points: {(is_match_point_p1 > 0.5).sum()}")
    
    # Check labels
    print(f"\n  Label distributions:")
    print(f"    Match P1: {y_match.mean():.3f}")
    print(f"    Set P1: {y_set.mean():.3f}")
    print(f"    Game P1: {y_game.mean():.3f}")
    
    return True


def test_mini_pretraining():
    """Test pre-training with minimal settings."""
    print("\n" + "="*80)
    print("TEST 3: Mini Pre-training (1000 matches, 5 epochs)")
    print("="*80)
    
    print("\nStarting mini pre-training...")
    print("(This will take ~2-3 minutes)\n")
    
    try:
        model, checkpoint = pretrain_tennis_rules(
            n_matches=1000,
            epochs=5,
            batch_size=512,
            temperature=10.0,
            output_path="models/test_pretrain_mini.pth",
            device='cpu'  # Use CPU for testing
        )
        
        print(f"\n✓ Mini pre-training successful!")
        print(f"  Model saved to: models/test_pretrain_mini.pth")
        print(f"  Training points: {checkpoint['n_training_points']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pre-training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TRANSFER LEARNING SYSTEM - TEST SUITE")
    print("="*80)
    
    tests = [
        ("Simulator", test_simulator),
        ("Features", test_feature_computation),
        ("Pre-training", test_mini_pretraining),
    ]
    
    results = {}
    
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = result
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception:")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - System ready for use!")
        print("\nNext steps:")
        print("  1. Run full pre-training:")
        print("     python tennisctl.py tennis-training --n-matches 50000 --model-out models/pretrained.pth")
        print("\n  2. Fine-tune on real data:")
        print("     python tennisctl.py complete-model --files data/*-wimbledon-points.csv \\")
        print("       --pretrained models/pretrained.pth --model-out models/final.pth")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

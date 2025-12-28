"""
Tennis Match Simulator for Pre-training Neural Networks

Generates synthetic tennis matches with perfect score tracking to teach
the neural network the fundamental rules of tennis scoring.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict


class TennisSimulator:
    """Simulates tennis matches with deterministic or probabilistic outcomes."""
    
    def __init__(self, best_of_5=True, seed=None):
        self.best_of_5 = best_of_5
        self.sets_to_win = 3 if best_of_5 else 2
        if seed is not None:
            np.random.seed(seed)
    
    def simulate_point(self, p1_skill: float, p2_skill: float, is_p1_serving: bool) -> int:
        """
        Simulate a single point.
        
        Args:
            p1_skill: P1's base skill (0.5-0.7)
            p2_skill: P2's base skill (0.5-0.7)
            is_p1_serving: True if P1 is serving
        
        Returns:
            1 if P1 wins point, 2 if P2 wins point
        """
        # Server advantage: +0.1 to win probability
        if is_p1_serving:
            p1_prob = p1_skill + 0.1
        else:
            p1_prob = p1_skill - 0.1
        
        # Normalize
        p1_prob = np.clip(p1_prob, 0.2, 0.8)
        
        return 1 if np.random.random() < p1_prob else 2
    
    def simulate_game(self, p1_skill: float, p2_skill: float, server: int) -> Tuple[int, List[Dict]]:
        """
        Simulate a single game with point-by-point tracking.
        
        Returns:
            (winner, list of point states)
        """
        points = []
        p1_points = 0
        p2_points = 0
        
        is_p1_serving = (server == 1)
        
        while True:
            # Record state before point
            state = {
                'p1_points': p1_points,
                'p2_points': p2_points,
                'server': server,
            }
            
            # Simulate point
            winner = self.simulate_point(p1_skill, p2_skill, is_p1_serving)
            
            if winner == 1:
                p1_points += 1
            else:
                p2_points += 1
            
            state['point_winner'] = winner
            points.append(state)
            
            # Check game win conditions
            if p1_points >= 4 and p1_points >= p2_points + 2:
                return 1, points
            elif p2_points >= 4 and p2_points >= p1_points + 2:
                return 2, points
    
    def simulate_tiebreak(self, p1_skill: float, p2_skill: float, server: int) -> Tuple[int, List[Dict]]:
        """Simulate a tiebreak with alternating serves."""
        points = []
        p1_points = 0
        p2_points = 0
        point_count = 0
        
        while True:
            # Determine server (alternates every 2 points, starting with 'server')
            if point_count == 0:
                current_server = server
            elif point_count % 2 == 1:
                current_server = server
            else:
                current_server = 3 - server  # Switch server
            
            is_p1_serving = (current_server == 1)
            
            state = {
                'p1_points': p1_points,
                'p2_points': p2_points,
                'server': current_server,
                'is_tiebreak': True,
            }
            
            winner = self.simulate_point(p1_skill, p2_skill, is_p1_serving)
            
            if winner == 1:
                p1_points += 1
            else:
                p2_points += 1
            
            state['point_winner'] = winner
            points.append(state)
            point_count += 1
            
            # Tiebreak win: 7+ with 2-point lead
            if p1_points >= 7 and p1_points >= p2_points + 2:
                return 1, points
            elif p2_points >= 7 and p2_points >= p1_points + 2:
                return 2, points
    
    def simulate_set(self, p1_skill: float, p2_skill: float, initial_server: int, 
                    set_number: int, p1_sets: int, p2_sets: int) -> Tuple[int, List[Dict]]:
        """Simulate a complete set."""
        games = []
        p1_games = 0
        p2_games = 0
        server = initial_server
        
        while True:
            # Check for tiebreak (6-6 for normal sets, 12-12 for final set)
            is_final_set = (set_number == 5 and self.best_of_5) or (set_number == 3 and not self.best_of_5)
            tb_threshold = 12 if is_final_set else 6
            
            if p1_games == tb_threshold and p2_games == tb_threshold:
                # Tiebreak
                game_winner, game_points = self.simulate_tiebreak(p1_skill, p2_skill, server)
                
                for point in game_points:
                    point['p1_games'] = p1_games
                    point['p2_games'] = p2_games
                    point['p1_sets'] = p1_sets
                    point['p2_sets'] = p2_sets
                    point['set_number'] = set_number
                    games.append(point)
                
                if game_winner == 1:
                    p1_games += 1
                else:
                    p2_games += 1
                
                # Tiebreak ends set
                return game_winner, games
            
            # Normal game
            game_winner, game_points = self.simulate_game(p1_skill, p2_skill, server)
            
            for point in game_points:
                point['p1_games'] = p1_games
                point['p2_games'] = p2_games
                point['p1_sets'] = p1_sets
                point['p2_sets'] = p2_sets
                point['set_number'] = set_number
                games.append(point)
            
            if game_winner == 1:
                p1_games += 1
            else:
                p2_games += 1
            
            # Alternate server
            server = 3 - server
            
            # Check set win
            if p1_games >= 6 and p1_games >= p2_games + 2:
                return 1, games
            elif p2_games >= 6 and p2_games >= p1_games + 2:
                return 2, games
    
    def simulate_match(self, p1_skill: float = 0.55, p2_skill: float = 0.55) -> pd.DataFrame:
        """
        Simulate a complete match and return point-by-point data.
        
        Args:
            p1_skill: P1's skill level (0.5-0.7)
            p2_skill: P2's skill level (0.5-0.7)
        
        Returns:
            DataFrame with each row = one point
        """
        all_points = []
        p1_sets = 0
        p2_sets = 0
        server = np.random.choice([1, 2])  # Random initial server
        set_number = 1
        
        while p1_sets < self.sets_to_win and p2_sets < self.sets_to_win:
            set_winner, set_points = self.simulate_set(
                p1_skill, p2_skill, server, set_number, p1_sets, p2_sets
            )
            
            all_points.extend(set_points)
            
            if set_winner == 1:
                p1_sets += 1
            else:
                p2_sets += 1
            
            set_number += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(all_points)
        
        # Add match outcome
        match_winner = 1 if p1_sets > p2_sets else 2
        df['p1_wins_match'] = float(match_winner == 1)
        
        # Add point number
        df['point_number'] = range(1, len(df) + 1)
        
        return df


def generate_training_dataset(n_matches: int, output_path: str = None, 
                              best_of_5: bool = True, seed: int = 42) -> pd.DataFrame:
    """
    Generate a large synthetic dataset of tennis matches.
    
    Args:
        n_matches: Number of matches to simulate
        output_path: Optional path to save CSV
        best_of_5: True for best-of-5, False for best-of-3
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with all points from all matches
    """
    simulator = TennisSimulator(best_of_5=best_of_5, seed=seed)
    
    all_matches = []
    
    print(f"Generating {n_matches} synthetic tennis matches...")
    
    for match_id in range(1, n_matches + 1):
        if match_id % 1000 == 0:
            print(f"  Generated {match_id}/{n_matches} matches...")
        
        # Vary skill levels to create diverse matches
        p1_skill = np.random.uniform(0.50, 0.65)
        p2_skill = np.random.uniform(0.50, 0.65)
        
        match_df = simulator.simulate_match(p1_skill, p2_skill)
        match_df['match_id'] = f"synthetic_{match_id:06d}"
        match_df['p1_skill'] = p1_skill
        match_df['p2_skill'] = p2_skill
        
        all_matches.append(match_df)
    
    # Combine all matches
    full_df = pd.concat(all_matches, ignore_index=True)
    
    print(f"Generated {len(full_df)} total points from {n_matches} matches")
    print(f"Average points per match: {len(full_df) / n_matches:.1f}")
    
    if output_path:
        full_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
    return full_df


if __name__ == "__main__":
    # Test the simulator
    print("Testing tennis simulator...")
    
    df = generate_training_dataset(
        n_matches=100,
        output_path="data/synthetic_tennis_matches_test.csv",
        best_of_5=True,
        seed=42
    )
    
    print("\nSample of generated data:")
    print(df.head(20))
    print("\nData summary:")
    print(df.describe())

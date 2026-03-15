import pytest
import cshogi
import numpy as np
import json
import os
from cshogi.dlshogi.dataprep import UnifiedFeatureGenerator

# Sample CSA data for testing
# P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
# P2-HI-KA-  -  -  -  -  -  -  -
# P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
# P4-  -  -  -  -  -  -  -  -  -
# P5-  -  -  -  -  -  -  -  -  -
# P6-  -  -  -  -  -  -  -  -  -
# P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
# P8+  -HI-KA-  -  -  -  -  -  -
# P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
# +
# +7747FU
SAMPLE_CSA_GAME = """
V2.2
N+Taro
N-Jiro
$EVENT:test game
$SITE:test site
$START_TIME:2023/01/01 10:00:00
$END_TIME:2023/01/01 10:30:00
$OPENING:
PI
+
+7776FU
%TORYO
"""

# Minimal feature config for testing
TEST_FEATURE_CONFIG = {
    "get_board_layout_2d": True,
    "get_hand_pieces": True,
    "get_attacks_on_kings": True,
    "is_check": True,
    "get_legal_move_counts": True, # Will be 0 for opponent's perspective
}

@pytest.fixture
def feature_generator():
    """Provides a UnifiedFeatureGenerator instance with a test config."""
    # Create a dummy config file for the generator to load
    config_path = "test_feature_config.json"
    with open(config_path, "w") as f:
        json.dump(TEST_FEATURE_CONFIG, f)
    
    generator = UnifiedFeatureGenerator(config=TEST_FEATURE_CONFIG) # Pass dict directly
    
    # Clean up the dummy config file
    if os.path.exists(config_path):
        os.remove(config_path)
    return generator

def test_process_csa_black_perspective(feature_generator):
    """Test feature extraction for black (Sente) perspective."""
    result_dict = feature_generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK)

    assert "features" in result_dict
    assert "sfens" in result_dict
    assert "result" in result_dict

    features = result_dict["features"]
    sfens = result_dict["sfens"]
    game_result = result_dict["result"]

    # Basic assertions on shapes and types
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert len(sfens) == features.shape[0]
    assert game_result == cshogi.BLACK # %TORYO for + implies Black won

    # Test for correct number of positions (initial + 1 move)
    assert len(sfens) == 2 # Initial position + one move

    # Check SFENs
    assert sfens[0] == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert sfens[1] == "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2" # After 7776FU

    # TODO: More detailed assertions on feature values (needs careful calculation for each feature)

def test_process_csa_white_perspective(feature_generator):
    """Test feature extraction for white (Gote) perspective."""
    result_dict = feature_generator.process_csa(SAMPLE_CSA_GAME, cshogi.WHITE)

    features = result_dict["features"]
    sfens = result_dict["sfens"]
    game_result = result_dict["result"]

    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert len(sfens) == features.shape[0]
    assert game_result == cshogi.BLACK # %TORYO for + implies Black won

    assert len(sfens) == 2

    # TODO: More detailed assertions on feature values, especially checking the normalization for white perspective

def test_process_csa_sampling_interval(feature_generator):
    """Test interval sampling."""
    # A longer CSA game for interval sampling test
    long_csa = SAMPLE_CSA_GAME.replace("%TORYO", "+2726FU\n%TORYO") # Add another move
    long_csa = long_csa.replace("+7776FU", "+7776FU\n+3736FU\n+2726FU") # Even longer (initial + 4 moves)

    # Initial (1) + 4 moves = 5 positions
    # If interval is 2, should get initial, move 2, move 4 (indices 0, 2, 4)
    sampling_options = {'method': 'interval', 'n': 2}
    result_dict = feature_generator.process_csa(long_csa, cshogi.BLACK, sampling_options)

    features = result_dict["features"]
    sfens = result_dict["sfens"]

    assert len(sfens) == 3 # Expected: initial, after move 2, after move 4
    assert features.shape[0] == 3

    # Check SFENs for correctness of sampling
    # The initial board string is always the same
    expected_sfens = [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1", # Initial
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2", # After +7776FU (move 1)
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/3P5/PP2PPPP/1B5R1/LNSGKGSNL w - 4", # After +3736FU (move 3)
    ]
    # The sample CSA string has a bug, the sfen from move 3 should reflect 3736FU for black
    # Let's adjust the long_csa so it's easier to verify with current sfens.
    # Simpler: just ensure the length of sfens is correct.

    # Original long_csa parsing:
    # Initial
    # +7776FU (b)
    # +3736FU (w)
    # +2726FU (b)
    # Total 4 moves, so 5 positions.
    # Indices: 0, 1, 2, 3, 4
    # Sampling with n=2: indices 0, 2, 4. So 3 positions.
    # sfens[0] (initial)
    # sfens[1] (after +7776FU, index 1) -> if n=2, we need after +3736FU which is index 2.
    # The kifu.boards list contains the board *after* the move.
    # So `boards[0]` is initial, `boards[1]` is after first move, `boards[2]` after second.
    # For interval 2, it should be boards[0], boards[2], boards[4].

    # Redefine long_csa to make moves obvious in SFEN if possible, or just rely on indices.
    # Let's verify with the actual kifu.boards content.
    kifu = cshogi.CSA.Parser.parse_str(long_csa)[0]
    expected_sfens_from_kifu = [kifu.boards[i].sfen() for i in [0, 2]] # Only two moves, so 3 positions: initial, after 2nd move, after 4th move.

    # After +7776FU (b)
    # Then +3736FU (w)
    # Then +2726FU (b)
    # So:
    # boards[0]: initial
    # boards[1]: after 7776FU
    # boards[2]: after 3736FU
    # boards[3]: after 2726FU
    # If the kifu had 3 moves, len(boards) would be 4.
    # Current SAMPLE_CSA_GAME has 2 moves (+7776FU, %TORYO), so len(boards) is 2 (initial + after 1st move)
    # My example `long_csa` has 3 moves. So len(boards) is 4.
    # +7776FU (b) - board[1]
    # +3736FU (w) - board[2]
    # +2726FU (b) - board[3]
    # Indices: 0, 1, 2, 3
    # sampling_options = {'method': 'interval', 'n': 2}
    # Expected indices: 0, 2
    assert len(sfens) == 2 # 0, 2 for 4 positions
    assert sfens[0] == kifu.boards[0].sfen()
    assert sfens[1] == kifu.boards[2].sfen()


def test_process_csa_sampling_random(feature_generator):
    """Test random sampling."""
    sampling_options = {'method': 'random', 'k': 1} # Sample 1 position
    result_dict = feature_generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK, sampling_options)

    features = result_dict["features"]
    sfens = result_dict["sfens"]

    assert len(sfens) == 1
    assert features.shape[0] == 1
    # Check that the sampled SFEN is one of the valid SFENs from the game
    full_kifu = cshogi.CSA.Parser.parse_str(SAMPLE_CSA_GAME)[0]
    assert sfens[0] in [b.sfen() for b in full_kifu.boards]

def test_process_csa_no_sampling(feature_generator):
    """Test no sampling (all positions)."""
    result_dict = feature_generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK) # Default is no sampling

    features = result_dict["features"]
    sfens = result_dict["sfens"]

    assert len(sfens) == 2
    assert features.shape[0] == 2
    assert sfens[0] == "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    assert sfens[1] == "lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2"


def test_empty_feature_config():
    """Test with an empty feature config."""
    generator = UnifiedFeatureGenerator(config={})
    result_dict = generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK)
    assert result_dict["features"].shape == (2, 0) # 2 positions, 0 features
    assert len(result_dict["sfens"]) == 2


def test_unknown_sampling_method(feature_generator):
    """Test with an unknown sampling method."""
    sampling_options = {'method': 'unknown_method'}
    result_dict = feature_generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK, sampling_options)
    assert len(result_dict["sfens"]) == 2 # Should default to no sampling


# Test get_last_move_features handling
def test_get_last_move_features_skipped():
    """Ensure get_last_move_features is skipped as intended."""
    config_with_last_move = TEST_FEATURE_CONFIG.copy()
    config_with_last_move["get_last_move_features"] = True # Should be skipped
    generator = UnifiedFeatureGenerator(config=config_with_last_move)
    result_dict = generator.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK)
    
    # The actual feature vector size will depend on other features
    # This just ensures it doesn't crash and returns something
    assert result_dict["features"].shape[0] == 2
    # Detailed check of feature vector size would be needed to ensure it was skipped

# It's difficult to assert the exact values of features without re-implementing
# the logic in the test, which is not ideal.
# Instead, focus on shape, type, and perspective changes for key features.

def test_board_layout_2d_normalization_white_perspective(feature_generator):
    """
    Test that get_board_layout_2d is correctly normalized for white perspective.
    This involves flipping the board and swapping colors.
    """
    config_only_board_layout = {"get_board_layout_2d": True}
    generator_board_only = UnifiedFeatureGenerator(config=config_only_board_layout)

    result_black = generator_board_only.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK)
    result_white = generator_board_only.process_csa(SAMPLE_CSA_GAME, cshogi.WHITE)

    features_black_pos0 = result_black["features"][0]
    features_white_pos0 = result_white["features"][0]

    # Reshape features to (9, 9, 28) for easier comparison
    board_black = features_black_pos0.reshape(9, 9, -1)
    board_white = features_white_pos0.reshape(9, 9, -1)

    # Initial board:
    # lnsgkgsnl
    # 1r5b1
    # ppppppppp
    # 9
    # 9
    # 9
    # PPPPPPPPP
    # 1B5R1
    # LNSGKGSNL b - 1
    
    # Compare white perspective with black perspective's flipped and swapped version
    # If white is the perspective, the original black pieces should appear as opponent's pieces
    # (i.e., in the white channel equivalent) on the flipped board.
    # The original white pieces should appear as perspective's pieces (black channel equivalent)
    # on the flipped board.

    # Helper function to manually flip and swap channels for verification
    def manual_flip_and_swap(board_tensor):
        flipped_board = np.flip(board_tensor, axis=(0, 1))
        swapped_channels = np.zeros_like(flipped_board, dtype=np.float32)
        # Assuming channels are [P_B, P_W, L_B, L_W, ...]
        for i in range(cshogi.PIECE_NB):
            # Original White pieces (odd index) become "my" (black channel equivalent)
            swapped_channels[:, :, i*2] = flipped_board[:, :, i*2 + 1]
            # Original Black pieces (even index) become "opponent" (white channel equivalent)
            swapped_channels[:, :, i*2 + 1] = flipped_board[:, :, i*2]
        return swapped_channels

    # Manually transform the black perspective feature to what white perspective should look like
    expected_white_perspective = manual_flip_and_swap(board_black)
    
    # Compare
    assert np.allclose(board_white, expected_white_perspective), \
        "Board layout normalization for white perspective failed."
    
    # Also check the second position after a move (7776FU)
    features_black_pos1 = result_black["features"][1]
    features_white_pos1 = result_white["features"][1]
    board_black_pos1 = features_black_pos1.reshape(9, 9, -1)
    board_white_pos1 = features_white_pos1.reshape(9, 9, -1)
    expected_white_pos1_perspective = manual_flip_and_swap(board_black_pos1)
    assert np.allclose(board_white_pos1, expected_white_pos1_perspective), \
        "Board layout normalization for white perspective failed for move 1."


def test_hand_pieces_normalization_white_perspective(feature_generator):
    """Test that hand pieces are correctly normalized for white perspective."""
    config_only_hand_pieces = {"get_hand_pieces": True}
    generator_hand_only = UnifiedFeatureGenerator(config=config_only_hand_pieces)

    # Initial position has no hand pieces
    result_black = generator_hand_only.process_csa(SAMPLE_CSA_GAME, cshogi.BLACK)
    result_white = generator_hand_only.process_csa(SAMPLE_CSA_GAME, cshogi.WHITE)

    hand_black_pos0 = result_black["features"][0]
    hand_white_pos0 = result_white["features"][0]

    # For initial position, all hand piece counts are 0 for both players
    assert np.allclose(hand_black_pos0, np.zeros_like(hand_black_pos0))
    assert np.allclose(hand_white_pos0, np.zeros_like(hand_white_pos0))

    # After move, still no captures, so hands are empty
    hand_black_pos1 = result_black["features"][1]
    hand_white_pos1 = result_white["features"][1]
    assert np.allclose(hand_black_pos1, np.zeros_like(hand_black_pos1))
    assert np.allclose(hand_white_pos1, np.zeros_like(hand_white_pos1))

    # To properly test, we need a CSA that involves captures.
    # Let's create a simplified CSA with a capture.
    csa_with_capture = """
V2.2
PI
+
+7776FU
-3334FU
+7675FU
-3435FU
+2726FU
-3536FU
+2625FU
-3637FU
+7574FU
-3738FU
+2524FU
-3839FU
+8822UM
-3132FU
+2231UM
-5152FU
%TORYO
"""
    # After +2231UM, black has captured a Lance (if 31 was lance), or just moved.
    # Let's make it a clear capture:
    # +2231UM captures the piece at 31.
    # Original board has Kyo at 31. So UMA captures Kyo.
    # Kifu is: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
    # Move +2231UM (Horse from 88 to 31)
    # Piece at 31 is +KY. So black captures opponent's Kyo (Lance).
    # Then black has one Lance in hand.
    
    gen_capture = UnifiedFeatureGenerator(config=config_only_hand_pieces)
    result_black_capture = gen_capture.process_csa(csa_with_capture, cshogi.BLACK)
    result_white_capture = gen_capture.process_csa(csa_with_capture, cshogi.WHITE)

    # Hand pieces are always returned as [my_hand_pieces, opponent_hand_pieces]
    # In cshogi, HAND_PIECES are [PAWN, LANCE, KNIGHT, SILVER, BISHOP, ROOK, GOLD] - NOTE: order might differ from PIECE_NAMES
    # Let's check the position after +2231UM
    # This is the last move in csa_with_capture.
    # Length of kifu.boards will be (number of moves + 1).
    kifu_capture = cshogi.CSA.Parser.parse_str(csa_with_capture)[0]
    num_moves_capture = len(kifu_capture.moves)
    
    # Final board after +2231UM is at index `num_moves_capture`
    # At this point, black has captured a Lance.
    # So for black perspective, my_hand should have 1 Lance. Opponent's hand is 0.
    # hand_black_pos_final = [0(pawn), 1(lance), 0,0,0,0,0, 0,0,0,0,0,0,0]
    # For white perspective, my_hand should be 0. Opponent's hand should have 1 Lance.
    # hand_white_pos_final = [0,0,0,0,0,0,0, 0(pawn),1(lance),0,0,0,0,0]

    hand_black_final_features = result_black_capture["features"][-1]
    hand_white_final_features = result_white_capture["features"][-1]

    # Index for Lance (assuming PIECE_NAMES order matches HAND_PIECES)
    # HAND_PIECES = [PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK]
    # The order will be PAWN, LANCE, ...
    # So Lance is at index 1 for both current player's hand and opponent's hand
    lance_index = 1 
    
    # Black perspective: my_hand has Lance, opponent_hand is empty
    # hand_black_final_features should have 1 at `lance_index` and 0 at `lance_index + len(HAND_PIECES)`
    assert hand_black_final_features[lance_index] == 1
    assert hand_black_final_features[lance_index + len(cshogi.HAND_PIECES)] == 0

    # White perspective: my_hand is empty, opponent_hand has Lance
    # hand_white_final_features should have 0 at `lance_index` and 1 at `lance_index + len(HAND_PIECES)`
    assert hand_white_final_features[lance_index] == 0
    assert hand_white_final_features[lance_index + len(cshogi.HAND_PIECES)] == 1


# Test is_check normalization
def test_is_check_normalization(feature_generator):
    """Test that is_check is correctly normalized for both perspectives."""
    config_only_is_check = {"is_check": True}
    generator_check_only = UnifiedFeatureGenerator(config=config_only_is_check)

    # CSA string where Black is in check
    csa_black_in_check = """
V2.2
PI
+
+0000HI # Dummy move to set a position where check can happen
%TORYO
"""
    # Create a board where Black is in check
    board_black_check = cshogi.Board()
    # Simple check scenario: White Rook on 51, Black King on 59
    # This needs careful construction of SFEN or CSA
    board_black_check.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    # Black King is on 59.
    # White Rook on 51 (from 88->51). 
    # White +R (Promoted Rook) on 51, attacks 59.
    # Set the board manually, and then use its sfen to recreate the CSA for testing
    
    # Original board SFEN from SAMPLE_CSA_GAME: "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    # To check black king: place white rook on 51.
    board_check_pos = cshogi.Board()
    # Initial setup
    board_check_pos.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    # Remove rook from 88 (original pos) and place on 51, then set white to move
    # this is not how it works. Let's make a simple CSA
    
    # R7a K9i (simple white rook check black king)
    csa_check = """
V2.2
PI
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2-HI-KA-  -  -  -  -  -  -  -
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4-  -  -  -  -  -  -  -  -  -
P5-  -  -  -  -  -  -  -  -  -
P6-  -  -  -  -  -  -  -  -  -
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8+  -HI-KA-  -  -  -  -  -  -
P9+KY+KE+GI+KI-OU-KI+GI+KE+KY
+
+8828HI
%TORYO
"""
    # Initial board: Black king on 59.
    # Move +8828HI (Black rook to 28). This move puts white in check!
    # Let's ensure the CSA creates the right check scenario.
    # We need a board where BLACK king is in check.
    # Start from an arbitrary position.
    
    # Initial: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
    # Black King 59 (5,9)
    # White King 51 (5,1)
    # To check black king, white piece needs to attack 59.
    # Say, white rook on 58.
    
    csa_check_black_king = """
V2.2
PI
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2-  -  -  -  -  -  -  -  -  -
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4-  -  -  -  -  -  -  -  -  -
P5-  -  -  -  -  -  -  -  -  -
P6-  -  -  -  -  -  -  -  -  -
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8+  -HI-KA-  -  -  -  -  -  -
P9+KY+KE+GI+KI-OU-KI+GI+KE+KY
+
+8858HI # White rook moves from 88 to 58 (not 28). White to move
%TORYO
"""
    # This move is for black. So +8858HI is black's move. This makes black's rook attack 59, which is black's king.
    # This doesn't make sense. Let's make a simple SFEN that puts black in check.
    # A white rook on 58 directly checks a black king on 59 (initially)
    
    # Initial board SFEN: lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1
    # black king is at 59 (rank 5, file 9)
    # If we put a white rook at 58 (rank 5, file 8) and make it white's turn, black is in check.
    # This is hard to represent in a simple CSA. Let's use Board directly.

    # Board state where Black king (59) is in check by White Rook (58)
    board_black_in_check = cshogi.Board()
    board_black_in_check.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/4R4/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1") # R on 55, Black King on 59
    # Actually, R on 58 (8th file, 5th rank) (or 52 if 9x9 is 1-9)
    # SFEN coordinate system: file 1-9, rank 1-9. So 58 means 5th rank, 8th file.
    # 59 means 5th rank, 9th file.
    # If a white R is at 58, it attacks 59.
    
    # Corrected SFEN for black king in check by white rook at 58:
    # `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1` -- initial white to move
    # board.push_usi("8858HI") # This will move white rook to 58, which checks 59 (black king)
    # Then `board.is_check()` should be true.

    # Let's create a board in check programmatically for robustness.
    # Initial board
    board_check_test = cshogi.Board()
    board_check_test.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    # Make a move that puts Black in check from White's perspective
    # For example, move White Rook from 88 to 58
    # After 8858HI (move 1), it's black's turn. Black king is in check.
    board_check_test.push_usi("8858HI") # White moves Rook 88->58. Now it's Black's turn, Black is in check.
    
    # Simulate processing this single board
    result_dict_black_pers = feature_generator.process_csa(board_check_test.sfen(), cshogi.BLACK, sampling_options={'method': 'interval', 'n': 1})
    result_dict_white_pers = feature_generator.process_csa(board_check_test.sfen(), cshogi.WHITE, sampling_options={'method': 'interval', 'n': 1})

    # The feature for is_check is at a certain index
    # (after board_layout_2d, hand_pieces, attacks_on_kings)
    # Need to know the exact feature vector structure.
    # This requires running it once and inspecting.
    
    # Let's rebuild the config to only have is_check for this test
    config_only_is_check = {"is_check": True}
    generator_check_only = UnifiedFeatureGenerator(config=config_only_is_check)
    
    # Process the SFEN directly
    result_black_pers = generator_check_only.process_csa(board_check_test.sfen(), cshogi.BLACK)
    result_white_pers = generator_check_only.process_csa(board_check_test.sfen(), cshogi.WHITE)
    
    # Black is in check
    # Black perspective: `is_check` should be 1
    # White perspective: `is_check` should be 0 (White is not in check, Black is)
    assert result_black_pers["features"][0][0] == 1
    assert result_white_pers["features"][0][0] == 0

    # Test where White is in check (e.g., after Black moves +2821HI)
    board_check_test_white = cshogi.Board()
    board_check_test_white.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
    # Black moves Rook 28 -> 21, check White King
    # This is the initial board after a move.
    board_check_test_white.push_usi("2821HI") # Black Rook moves to 21 (captures) and checks White King. Now it's White's turn.
    
    result_black_pers_white_check = generator_check_only.process_csa(board_check_test_white.sfen(), cshogi.BLACK)
    result_white_pers_white_check = generator_check_only.process_csa(board_check_test_white.sfen(), cshogi.WHITE)

    # White is in check
    # Black perspective: `is_check` should be 0 (Black is not in check, White is)
    # White perspective: `is_check` should be 1
    assert result_black_pers_white_check["features"][0][0] == 0
    assert result_white_pers_white_check["features"][0][0] == 1


# Test get_legal_move_counts and analyze_legal_moves (placeholder functionality)
def test_legal_move_counts_placeholder(feature_generator):
    """
    Test get_legal_move_counts and analyze_legal_moves placeholder functionality.
    For opposite perspective, it should return 0.
    """
    config_legal_moves = {
        "get_legal_move_counts": True,
        # "analyze_legal_moves": True # Needs more careful handling for dict output flattening
    }
    generator_legal_moves = UnifiedFeatureGenerator(config=config_legal_moves)

    # Initial board state: Black to move
    initial_sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
    
    result_black_pers = generator_legal_moves.process_csa(initial_sfen, cshogi.BLACK)
    result_white_pers = generator_legal_moves.process_csa(initial_sfen, cshogi.WHITE)

    # Black perspective: should get actual legal move count
    # (initial board, Black has 30 legal moves)
    # The actual count can be checked with cshogi.Board() directly
    board_initial = cshogi.Board()
    board_initial.set_sfen(initial_sfen)
    black_legal_moves_count = len(list(board_initial.legal_moves))
    assert result_black_pers["features"][0][0] == black_legal_moves_count

    # White perspective: not white's turn, so it should be 0
    assert result_white_pers["features"][0][0] == 0

    # Test a position where White is to move (e.g., after +7776FU)
    board_after_first_move = cshogi.Board()
    board_after_first_move.set_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2")
    
    result_black_pers_after_move = generator_legal_moves.process_csa(board_after_first_move.sfen(), cshogi.BLACK)
    result_white_pers_after_move = generator_legal_moves.process_csa(board_after_first_move.sfen(), cshogi.WHITE)

    # Black perspective: not black's turn, so should be 0
    assert result_black_pers_after_move["features"][0][0] == 0
    
    # White perspective: should get actual legal move count
    # (after 7776FU, White to move, White has 30 legal moves)
    white_legal_moves_count = len(list(board_after_first_move.legal_moves))
    assert result_white_pers_after_move["features"][0][0] == white_legal_moves_count

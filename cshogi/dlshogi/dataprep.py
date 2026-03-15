"""
UnifiedFeatureGenerator class for preparing shogi game data for machine learning.
"""
import cshogi
import cshogi.CSA
import numpy as np
import os

# Import necessary constants and classes from features.py
from ..features import FeatureExtractor, PIECE_NB, HAND_PIECES, PIECE_NAMES

class UnifiedFeatureGenerator:
    """
    Generates feature vectors from cshogi.Board objects for a given perspective,
    based on a configuration, and handles sampling from CSA game data.
    """
    def __init__(self, config):
        """
        UnifiedFeatureGeneratorを初期化します。

        Args:
            config (dict): 抽出する特徴量を指定する辞書。
                           キーは`cshogi.features.FeatureExtractor`のメソッド名、
                           値は真偽値（有効にする場合はTrue、無効にする場合はFalse）です。
        """
        self.config = config
        self.active_features = [name for name, active in config.items() if active]
        # Store PIECE_NAMES values for consistent hand feature order
        self.hand_piece_names_list = [PIECE_NAMES[p] for p in HAND_PIECES]

    def _parse_csa(self, csa_data):
        """
        CSAデータをパースします（ファイルパスまたは文字列）。

        Returns:
            cshogi.CSA.Kifu: パースされた棋譜オブジェクト。
        """
        if os.path.exists(csa_data):
            return cshogi.CSA.Parser.parse_file(csa_data)[0]
        else:
            # Assume csa_data is a string if it's not a file path
            return cshogi.CSA.Parser.parse_str(csa_data)[0]

    def _get_sampled_indices(self, total_moves, sampling_options):
        """
        指定されたオプションに基づいて、棋譜から局面をサンプリングするためのインデックスを生成します。

        Args:
            total_moves (int): 棋譜内の全局面数（指し手数 + 1）。
            sampling_options (dict, optional): サンプリング戦略を指定する辞書。
                - {'method': 'interval', 'n': int}: 'n'手ごとに1局面をサンプリングします。
                - {'method': 'random', 'k': int}: 棋譜からランダムに'k'局面をサンプリングします。
                Noneの場合、または指定されたメソッドが不明な場合は、全ての局面がサンプリングされます。

        Returns:
            list: 処理対象となる局面のインデックス（整数）のリスト。
        """
        if sampling_options is None:
            return list(range(total_moves)) # No sampling, use all moves

        method = sampling_options.get('method')
        if method == 'interval':
            interval = sampling_options.get('n', 1)
            return list(range(0, total_moves, interval))
        elif method == 'random':
            k = sampling_options.get('k') # Number of positions to sample
            if k is None or k >= total_moves:
                return list(range(total_moves))
            return sorted(np.random.choice(total_moves, k, replace=False))
        # Add other sampling methods here as needed
        else:
            return list(range(total_moves))

    def process_csa(self, csa_data, perspective, sampling_options=None):
        """
        Parses a CSA file (path or content) and generates a list of feature vectors,
        SFENs, and the game result, all from the specified perspective.

        Args:
            csa_data: Path to the CSA file or the CSA content as a string.
            perspective: The cshogi.BLACK or cshogi.WHITE constant for the desired perspective.
            sampling_options: Dictionary with sampling method and parameters (e.g., {'method': 'interval', 'n': 5}).

        Returns:
            A dictionary containing:
                - "features": np.ndarray of shape (num_sampled_positions, feature_dim)
                - "sfens": List of SFEN strings for sampled positions
                - "result": Game result (cshogi.BLACK, cshogi.WHITE, or cshogi.DRAW)
        """
        kifu = self._parse_csa(csa_data)
        
        boards = kifu.boards
        game_result = kifu.win # cshogi.BLACK, cshogi.WHITE, or cshogi.DRAW
        
        # Apply sampling to get indices of positions to process
        sampled_indices = self._get_sampled_indices(len(boards), sampling_options)

        feature_vectors = []
        sfens_sampled = []

        for i in sampled_indices:
            board = boards[i]
            sfen = board.sfen()
            
            vector = self.extract_from_board(board, perspective)
            feature_vectors.append(vector)
            sfens_sampled.append(sfen)
        
        # Convert list of vectors to a single NumPy array for efficiency
        if not feature_vectors: # Handle empty list if no positions are sampled
            feature_sequence = np.empty((0, 0), dtype=np.float32)
        else:
            feature_sequence = np.stack(feature_vectors, axis=0)

        return {
            "features": feature_sequence, # (num_sampled_positions, feature_dim)
            "sfens": sfens_sampled,       # list of SFEN strings for sampled positions
            "result": game_result         # BLACK, WHITE, or DRAW
        }

    def extract_from_board(self, board, perspective):
        """
        与えられた盤面状態から、指定された視点での単一の特徴量ベクトルを抽出します。

        Args:
            board (cshogi.Board): 特徴量を抽出する盤面オブジェクト。
            perspective (int): 目的の視点（cshogi.BLACK または cshogi.WHITE）。

        Returns:
            numpy.ndarray: 抽出され、指定された視点に正規化された1次元の特徴量ベクトル。
        """
        extractor = FeatureExtractor(board)
        feature_parts = []

        for feature_name in self.active_features:
            # Special handling for features requiring previous board state
            if feature_name == 'get_last_move_features':
                # This feature needs previous board and move. Not directly applicable here for raw board processing.
                # It should be handled externally or with a more complex stateful extractor.
                # For now, we will skip it or return a placeholder if configured.
                # User will need to ensure 'get_last_move_features' is not in active_features for this class.
                continue

            raw_value = getattr(extractor, feature_name)()
            normalized_value = self._normalize(feature_name, raw_value, board, perspective)
            feature_parts.append(normalized_value.flatten())

        if not feature_parts:
            # Handle case where no features were selected or processed
            return np.array([], dtype=np.float32)
        return np.concatenate(feature_parts)

    def _normalize(self, feature_name, raw_value, board, perspective):
        """
        特徴量名に基づいて、特定の正規化メソッドにディスパッチします。

        Args:
            feature_name (str): 処理する特徴量の名前。
            raw_value: `FeatureExtractor`から取得された生の特徴量データ。
            board (cshogi.Board): 現在の盤面オブジェクト。
            perspective (int): 目的の視点（cshogi.BLACK または cshogi.WHITE）。

        Returns:
            numpy.ndarray: 指定された視点に正規化された特徴量データ。
        """
        if feature_name in ['get_attacks_on_kings', 'get_king_surrounding_control', 'get_total_control']:
            return self._normalize_symmetrical_pair(raw_value, board.turn, perspective)
        elif feature_name == 'get_hand_pieces':
            return self._normalize_hand_pieces(raw_value, perspective)
        elif feature_name == 'get_board_layout_2d':
            return self._normalize_board_layout(raw_value, board.turn, perspective)
        elif feature_name == 'is_check': # is_check is specific, as it can be calculated for any king
            return self._normalize_is_check(board, perspective)
        elif feature_name in ['get_legal_move_counts', 'get_king_escape_routes', 'analyze_legal_moves']:
            # These are expensive and often only make sense for the current player's actual turn.
            # For the opponent's perspective, returning 0 or NaN might be a design choice,
            # or it requires a full temporary board manipulation (slow).
            return self._normalize_current_player_dependent_placeholder(feature_name, raw_value, board.turn, perspective)
        else:
            return self._default_normalizer(raw_value)

    def _normalize_symmetrical_pair(self, raw_value, current_turn, perspective):
        """
        (自玉への利き, 敵玉への利き)のような対称的な特徴量ペアを正規化します。
        与えられた`perspective`に基づいて、(視点側の値, 相手側の値)の順に並べ替えます。
        """
        my_val, opp_val = raw_value
        if current_turn == perspective:
            return np.array([my_val, opp_val], dtype=np.float32)
        else:
            return np.array([opp_val, my_val], dtype=np.float32)

    def _normalize_hand_pieces(self, raw_value, perspective):
        """
        持ち駒の数を正規化します。
        指定された`perspective`に基づいて、(視点側の持ち駒, 相手側の持ち駒)の順に並べて返します。
        """
        perspective_str = 'BLACK' if perspective == cshogi.BLACK else 'WHITE'
        opponent_str = 'WHITE' if perspective == cshogi.BLACK else 'BLACK'
        
        p_hand = [raw_value[perspective_str].get(p_name, 0) for p_name in self.hand_piece_names_list]
        o_hand = [raw_value[opponent_str].get(p_name, 0) for p_name in self.hand_piece_names_list]
        
        return np.array(p_hand + o_hand, dtype=np.float32)

    def _normalize_board_layout(self, raw_value, current_turn, perspective):
        """
        2D盤面レイアウトを正規化します。
        もし`perspective`が後手であれば、盤面を反転させ、駒のチャネルを交換します。
        """
        if perspective == cshogi.BLACK:
        else:
            flipped_board = np.flip(raw_value, axis=(0, 1))
            
            swapped_channels = np.zeros_like(flipped_board, dtype=np.float32)
            # Iterate through piece types
            for i in range(PIECE_NB):
                # Black pieces channels (original) become opponent's pieces
                # White pieces channels (original) become perspective's pieces
                swapped_channels[:, :, i*2] = flipped_board[:, :, i*2 + 1] # White (original) -> My (black channel equivalent)
                swapped_channels[:, :, i*2 + 1] = flipped_board[:, :, i*2] # Black (original) -> Opponent (white channel equivalent)
            return swapped_channels

    def _normalize_is_check(self, board, perspective):
        """
        指定された`perspective`の色を持つ王が王手されているかを判断します。
        """
        king_sq_of_perspective = board.king_square(perspective)
        opponent_of_perspective = 1 - perspective
        is_perspective_in_check = len(board.attackers_to(opponent_of_perspective, king_sq_of_perspective)) > 0
        return np.array([float(is_perspective_in_check)], dtype=np.float32)

    def _normalize_current_player_dependent_placeholder(self, feature_name, raw_value, current_turn, perspective):
        """
        `get_legal_move_counts`のように`current_turn`に非常に依存する特徴量のプレースホルダーです。
        相手側の視点の場合、完全な再計算を行わない限り、0を返すのが一般的な設計選択です。
        """
        if current_turn == perspective:
            # `analyze_legal_moves` returns a dict, flatten its values for now or handle specifically
            if feature_name == 'analyze_legal_moves':
                # Example: flatten the boolean and integer results from analyze_legal_moves
                # This needs careful mapping to a fixed-size vector
                analysis_results = raw_value # raw_value is the dict from analyze_legal_moves
                # Define a consistent order for its parts
                # For simplicity, returning just capture_moves for now
                return np.array([analysis_results.get('capture_moves', 0)], dtype=np.float32)
            else:
                return np.array([raw_value], dtype=np.float32)
        else:
            # If not the perspective's turn, these features are often treated as 0 or unavailable
            # For `analyze_legal_moves`, a zero vector matching its flattened size would be needed
            if feature_name == 'analyze_legal_moves':
                 # Determine the expected flattened size of analyze_legal_moves for a proper zero vector
                 # For now, a single 0 if we only output capture_moves
                 return np.array([0], dtype=np.float32)
            return np.array([0], dtype=np.float32)


    def _default_normalizer(self, raw_value):
        """
        特定の視点変換を必要としない特徴量のためのデフォルトの正規化処理です。
        """
        if isinstance(raw_value, (np.ndarray)):
            return raw_value.astype(np.float32)
        else:
            return np.array([raw_value], dtype=np.float32)

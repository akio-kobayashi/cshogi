import numpy as np
import cshogi

"""
Shogi盤面の特徴量抽出に関連する定数とクラス。

このモジュールは、cshogi.Boardオブジェクトから様々な特徴量を抽出し、
機械学習モデルの入力データとして利用することを目的としています。
"""
EMPTY = 0
PAWN = 1
LANCE = 2
KNIGHT = 3
SILVER = 4
BISHOP = 5
ROOK = 6
GOLD = 7
KING = 8
PRO_PAWN = 9
PRO_LANCE = 10
PRO_KNIGHT = 11
PRO_SILVER = 12
HORSE = 13
DRAGON = 14

# 駒の種類数
PIECE_NB = 14

# 駒の種類を人間が読みやすい名前に変換
PIECE_NAMES = {
    EMPTY: 'EMPTY',
    PAWN: 'PAWN', LANCE: 'LANCE', KNIGHT: 'KNIGHT', SILVER: 'SILVER',
    BISHOP: 'BISHOP', ROOK: 'ROOK', GOLD: 'GOLD', KING: 'KING',
    PRO_PAWN: 'PRO_PAWN', PRO_LANCE: 'PRO_LANCE', PRO_KNIGHT: 'PRO_KNIGHT',
    PRO_SILVER: 'PRO_SILVER', HORSE: 'HORSE', DRAGON: 'DRAGON'
}

# 持ち駒の種類
HAND_PIECES = [PAWN, LANCE, KNIGHT, SILVER, GOLD, BISHOP, ROOK]
HAND_PIECE_NAMES = {p: PIECE_NAMES[p] for p in HAND_PIECES}

# C6（駒価値の変化）で使うための駒の相対的な価値
PIECE_VALUES = {
    PAWN: 100, LANCE: 300, KNIGHT: 320, SILVER: 480, GOLD: 520,
    BISHOP: 850, ROOK: 950,
    PRO_PAWN: 420, PRO_LANCE: 400, PRO_KNIGHT: 420, PRO_SILVER: 500,
    HORSE: 950, DRAGON: 1150,
    KING: 15000
}


class FeatureExtractor:
    """
    cshogi.Boardオブジェクトから多様な特徴量を抽出するクラス。
    機械学習の入力データを作成する際などに利用します。
    """
    def __init__(self, board):
        """
        Initializes the FeatureExtractor.

        Args:
            board (cshogi.Board): The board object from which to extract features.
        """
        self.board = board

    def _get_board_material(self, color):
        """
        Helper to get total material value on the board for a color.
        ---
        指定された色の盤上の駒の価値の合計を計算するヘルパー関数。
        """
        value = 0
        for i in range(81):
            if self.board.colors[i] == color:
                piece_type = cshogi.PIECE_TYPE[self.board.pieces[i]]
                value += PIECE_VALUES.get(piece_type, 0)
        return value

    # --- A群: 盤面の状態に関する特徴量 ---

    def get_board_layout_2d(self):
        """
        A1: Get 2D tensor representation of the board (9x9x(14*2)).
        ---
        A1: 盤面の2Dテンソル表現を取得します (9x9x(駒種14x先後2))。
        """
        tensor = np.zeros((9, 9, PIECE_NB * 2), dtype=np.float32)
        for i in range(81):
            piece = self.board.pieces[i]
            if piece != EMPTY:
                color = self.board.colors[i]
                piece_type = cshogi.PIECE_TYPE[piece]
                ch = (piece_type - 1) * 2 + color
                y, x = divmod(i, 9)
                tensor[y, x, ch] = 1
        return tensor

    def get_king_positions(self):
        """
        A2, A3: Get king positions.
        ---
        A2, A3: 先手・後手の玉の位置を取得します。
        """
        black_king_pos = (None, None)
        white_king_pos = (None, None)
        for i in range(81):
            if self.board.pieces[i] == KING:
                y, x = divmod(i, 9)
                if self.board.colors[i] == cshogi.BLACK:
                    black_king_pos = (y, x)
                else:
                    white_king_pos = (y, x)
        return black_king_pos, white_king_pos

    def get_piece_existence(self):
        """
        A4: Binary feature for piece existence on each square.
        ---
        A4: 各マスにどの駒が存在するかをバイナリ(0/1)で表現した特徴量を取得します。
        """
        feature = np.zeros(PIECE_NB * 2 * 81, dtype=np.uint8)
        for i in range(81):
            piece = self.board.pieces[i]
            if piece != EMPTY:
                color = self.board.colors[i]
                piece_type = cshogi.PIECE_TYPE[piece]
                index = ((piece_type - 1) * 2 + color) * 81 + i
                feature[index] = 1
        return feature

    def get_hand_pieces(self):
        """
        A5-A11: Get hand piece counts.
        ---
        A5-A11: 先手・後手の持ち駒の数を取得します。
        """
        hand_pieces = {'BLACK': {}, 'WHITE': {}}
        for color in [cshogi.BLACK, cshogi.WHITE]:
            color_str = 'BLACK' if color == cshogi.BLACK else 'WHITE'
            for piece_type in HAND_PIECES:
                count = self.board.get_hand(color, piece_type)
                hand_pieces[color_str][HAND_PIECE_NAMES[piece_type]] = count
        return hand_pieces

    # --- B群: 評価値に関連する特徴量 ---

    def is_check(self):
        """
        B1: Is the current player in check?
        ---
        B1: 現在のプレイヤーが王手されているか否かを取得します。
        """
        return self.board.is_check()

    def get_attacks_on_kings(self):
        """
        B2, B3: Get the number of attackers on each king.
        ---
        B2, B3: それぞれの玉に利いている敵の駒の数を取得します。
        自玉（current player's king）への利きと、敵玉（opponent's king）への利きを返します。
        """
        # 自玉への利き
        my_king_sq = self.board.king_square(self.board.turn)
        attacks_on_my_king = len(self.board.attackers_to(1 - self.board.turn, my_king_sq))

        # 敵玉への利き
        opp_king_sq = self.board.king_square(1 - self.board.turn)
        attacks_on_opp_king = len(self.board.attackers_to(self.board.turn, opp_king_sq))
        
        return attacks_on_my_king, attacks_on_opp_king

    def get_king_surrounding_control(self):
        """
        B4, B5: Get control counts around the kings.
        ---
        B4, B5: それぞれの玉の8近傍への敵の利きの数を取得します。
        """
        my_king_sq = self.board.king_square(self.board.turn)
        opp_king_sq = self.board.king_square(1 - self.board.turn)
        
        my_king_surrounding_control = 0
        # cshogiにget_king_movesはないため、自前で生成
        y, x = divmod(my_king_sq, 9)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 9 and 0 <= nx < 9:
                    if self.board.attackers_to(1 - self.board.turn, ny * 9 + nx):
                        my_king_surrounding_control += 1

        opp_king_surrounding_control = 0
        y, x = divmod(opp_king_sq, 9)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 9 and 0 <= nx < 9:
                    if self.board.attackers_to(self.board.turn, ny * 9 + nx):
                        opp_king_surrounding_control += 1
                
        return my_king_surrounding_control, opp_king_surrounding_control

    def get_total_control(self):
        """
        B6, B7: Get total number of controlled squares.
        ---
        B6, B7: それぞれのプレイヤーが支配している（利きのある）マスの総数を取得します。
        """
        my_control_count = 0
        opp_control_count = 0
        for i in range(81):
            if self.board.attackers_to(self.board.turn, i):
                my_control_count += 1
            if self.board.attackers_to(1 - self.board.turn, i):
                opp_control_count += 1
        return my_control_count, opp_control_count

    def get_legal_move_counts(self):
        """
        B8, B9: Get the number of legal moves.
        ---
        B8, B9: 合法手の総数を取得します。王手されている場合、王手を回避する手の数になります。
        """
        return len(list(self.board.legal_moves))
    
    def get_king_escape_routes(self):
        """
        B10: Number of safe squares the king can move to.
        ---
        B10: 玉が安全に移動できるマスの数を取得します。
        """
        king_sq = self.board.king_square(self.board.turn)
        count = 0
        # 玉の移動候補を生成
        y, x = divmod(king_sq, 9)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < 9 and 0 <= nx < 9:
                    to_sq = ny * 9 + nx
                    move = cshogi.Move.from_squares(king_sq, to_sq, self.board.turn)
                    # 移動先が敵に攻撃されておらず、かつ合法手であるか
                    if not self.board.attackers_to(1 - self.board.turn, to_sq):
                        if self.board.is_legal(move):
                            count += 1
        return count

    # --- C群: 直前の指し手に関する特徴量 ---
    
    @staticmethod
    def get_last_move_features(previous_board, move):
        """
        C1-C7: Features of the last move. Requires the board state *before* the move.
        ---
        C1-C7: 直前の指し手に関する特徴量を取得します。
        この機能は、手を指す「前」の盤面状態を必要とします。

        Args:
            previous_board (cshogi.Board): 指し手が行われる前の盤面オブジェクト。
            move (cshogi.Move): 行われた指し手。

        Returns:
            dict or None: 指し手の特徴量を含む辞書、またはmoveがNoneの場合はNone。
                - 'capture' (bool): 駒取りがあったかどうか。
                - 'captured_piece_type' (int or None): 取られた駒の種類。
                - 'captured_piece_name' (str or None): 取られた駒の名前。
                - 'promotion' (bool): 成りがあったかどうか。
                - 'gave_check' (bool): 指し手によって相手に王手をかけたかどうか。
                - 'received_check' (bool): 指し手を行う前に王手されていたかどうか。
                - 'piece_value_change' (int): 指し手を行ったプレイヤーの駒価値の正味の変化。
                - 'hand_piece_change' (dict): 指し手を行ったプレイヤーの持ち駒の変化の詳細を示す辞書。
        """
        if move is None:
            return None

        current_board = previous_board.copy()
        current_board.push(move)
        
        mover_color = previous_board.turn

        features = {}
        
        # C1: 駒取りがあったか
        is_capture = previous_board.is_capture(move)
        features['capture'] = is_capture
        
        # C2: 取られた駒の種類
        captured_piece_type = None
        if is_capture:
            # 取られた駒は、移動「前」の盤面の移動先にあった駒
            captured_piece = previous_board.pieces[move.to_sq]
            captured_piece_type = cshogi.PIECE_TYPE[captured_piece]
        features['captured_piece_type'] = captured_piece_type
        features['captured_piece_name'] = PIECE_NAMES.get(captured_piece_type)
            
        # C3: 成りがあったか
        features['promotion'] = cshogi.is_promote(move)
        
        # C4: この手で相手に王手をかけたか
        features['gave_check'] = current_board.is_check()
        
        # C5: この手を指す前に王手されていたか
        features['received_check'] = previous_board.is_check()

        # C6: 駒価値の変化
        piece_value_change = 0

        # 駒取りによる変化: 取られた駒の価値は、手番のプレイヤーが獲得する
        if is_capture:
            piece_value_change += PIECE_VALUES.get(captured_piece_type, 0)

        # 成りによる変化: 成った駒の価値とその元の駒の価値の差
        if cshogi.is_promote(move):
            original_piece_type = cshogi.PIECE_TYPE[previous_board.pieces[move.from_sq]]
            promoted_piece_type = cshogi.PIECE_TYPE[current_board.pieces[move.to_sq]]
            piece_value_change += PIECE_VALUES.get(promoted_piece_type, 0) - PIECE_VALUES.get(original_piece_type, 0)
        
        # 駒打ちによる変化: 打たれた駒の価値は、手番のプレイヤーが獲得する
        # cshogi.Move object has 'is_drop' and 'drop_piece' attributes
        if move.is_drop:
            dropped_piece_type = move.drop_piece # move.drop_piece directly gives the piece type
            piece_value_change += PIECE_VALUES.get(dropped_piece_type, 0)

        features['piece_value_change'] = piece_value_change

        # C7: 持ち駒の変化
        hand_change = {}
        for piece_type in HAND_PIECES:
            prev_count = previous_board.get_hand(mover_color, piece_type)
            curr_count = current_board.get_hand(mover_color, piece_type)
            diff = curr_count - prev_count
            if diff != 0:
                hand_change[HAND_PIECE_NAMES[piece_type]] = diff
        features['hand_piece_change'] = hand_change
        
        return features

    # --- D群: 合法手に関する特徴量 ---
    
    def analyze_legal_moves(self):
        """
        D1-D5: Analyze properties of legal moves.
        ---
        D1-D5: 現在の盤面における合法手の性質を分析します。
        """
        features = {
            'capture_moves': 0,      # D1: 駒を取る合法手の数
            'check_moves': 0,        # D2: 王手となる合法手の数
            'promotion_moves': 0,    # D3: 成りを伴う合法手の数
            'is_mate': False         # D5: 1手詰みの可能性
        }
        
        try:
            legal_moves = list(self.board.legal_moves)
        except IndexError: # cshogiの潜在的なバグ（合法手がない場合）に対応
            legal_moves = []

        if not legal_moves and self.board.is_check():
            features['is_mate'] = True # 詰み
            return features

        for move in legal_moves:
            if self.board.is_capture(move):
                features['capture_moves'] += 1
            if cshogi.is_promote(move):
                features['promotion_moves'] += 1
            
            self.board.push(move)
            if self.board.is_check():
                features['check_moves'] += 1
            self.board.pop()

        # D5: is_mated()で詰みをチェック
        if self.board.is_mated():
            features['is_mate'] = True
            
        return features

    # --- E群: 対局情報に関する特徴量 ---
    
    @staticmethod
    def get_game_state_features(board, game_info):
        """
        E1-E6: Get features related to the overall game state.
        ---
        E1-E6: 対局全体に関する特徴量を取得します。
        手数、Elo、勝敗など、盤面単体からは得られない情報を使います。
        """
        features = {}
        features['ply'] = board.move_number  # E1: 現在の手数
        if 'total_moves' in game_info:
            # E2: 終局までの残り手数
            features['moves_until_end'] = game_info['total_moves'] - board.move_number
        if 'winner' in game_info:
            # E3: 勝敗
            features['outcome'] = 1 if game_info['winner'] == cshogi.BLACK else 0
        if 'black_elo' in game_info:
            features['black_elo'] = game_info['black_elo'] # E4: 先手Elo
        if 'white_elo' in game_info:
            features['white_elo'] = game_info['white_elo'] # E5: 後手Elo
        if 'black_elo' in game_info and 'white_elo' in game_info:
            # E6: Elo差
            features['elo_diff'] = game_info['black_elo'] - game_info['white_elo']
        return features


def example():
    """
    このモジュールの使い方を示すサンプル関数です。
    """
    board = cshogi.Board()
    extractor = FeatureExtractor(board)
    
    print("--- A群: 盤面の状態に関する特徴量 ---")
    print("A1: 盤面レイアウト (shape):", extractor.get_board_layout_2d().shape)
    bk, wk = extractor.get_king_positions()
    print("A2: 先手玉の位置:", bk)
    print("A3: 後手玉の位置:", wk)
    print("A4: 駒の存在 (shape):", extractor.get_piece_existence().shape)
    print("A5-A11: 持ち駒:", extractor.get_hand_pieces())
    
    print("\n--- B群: 評価値に関連する特徴量 (現手番) ---")
    print("B1: 王手されているか?:", extractor.is_check())
    my_king_attacks, opp_king_attacks = extractor.get_attacks_on_kings()
    print(f"B2: 自玉への利き ({'先手' if board.turn == 0 else '後手'}):", my_king_attacks)
    print(f"B3: 敵玉への利き:", opp_king_attacks)
    my_king_control, opp_king_control = extractor.get_king_surrounding_control()
    print("B4: 自玉周囲の敵の利き:", my_king_control)
    print("B5: 敵玉周囲の自陣の利き:", opp_king_control)
    my_total_control, opp_total_control = extractor.get_total_control()
    print("B6: 自陣の総利き数:", my_total_control)
    print("B7: 敵陣の総利き数:", opp_total_control)
    print("B8/B9: 合法手の数:", extractor.get_legal_move_counts())
    print("B10: 玉の逃げ道:", extractor.get_king_escape_routes())

    # --- C群は指し手と、その前の盤面状態が必要 ---
    print("\n--- C群: 直前の指し手に関する特徴量 ---")
    board = cshogi.Board() # 盤面をリセット
    board.push_usi('7g7f')
    
    previous_board = board.copy()
    move = cshogi.Move.from_usi('3c3d')
    board.push(move) # 指し手を進める
    print(f"解析する指し手: {move.usi()}, 適用前の盤面:\n{previous_board}")
    
    last_move_features = FeatureExtractor.get_last_move_features(previous_board, move)
    print("C1-C7:", last_move_features)
    
    # 駒取りの例
    board.set_sfen('lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1')
    move = cshogi.Move.from_usi('2h7h') # 角が香車を取る
    previous_board_capture = board.copy()
    board.push(move)
    print(f"\n駒取りの指し手を解析: {move.usi()}, 適用前の盤面:\n{previous_board_capture}")
    capture_features = FeatureExtractor.get_last_move_features(previous_board_capture, move)
    print("C1-C7 (駒取り):", capture_features)


    print("\n--- D群: 合法手に関する特徴量 (現在の盤面から) ---")
    # 少し複雑な盤面に設定
    board.set_sfen('l2g1k3/4s1g2/p1npp1p1p/1p3p3/2p1P4/1P1P1P2P/P2N1PS1P/R1G1K1B1L/LN4b1l w B 2P2Snp 43')
    extractor = FeatureExtractor(board)
    print(f"解析対象の盤面:\n{board}")
    print("D1-D5:", extractor.analyze_legal_moves())
    
    print("\n--- E群: 対局情報に関する特徴量 ---")
    game_info = {
        'total_moves': 250,
        'winner': cshogi.BLACK,
        'black_elo': 1500,
        'white_elo': 1450
    }
    print("E1-E6:", FeatureExtractor.get_game_state_features(board, game_info))

if __name__ == '__main__':
    example()

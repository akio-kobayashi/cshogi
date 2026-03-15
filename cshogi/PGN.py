from typing import List, Optional
import cshogi
from datetime import datetime

# PGN形式のマス名マッピング
PGN_SQUARE_NAMES = [
    'i9', 'i8', 'i7', 'i6', 'i5', 'i4', 'i3', 'i2', 'i1',
    'h9', 'h8', 'h7', 'h6', 'h5', 'h4', 'h3', 'h2', 'h1',
    'g9', 'g8', 'g7', 'g6', 'g5', 'g4', 'g3', 'g2', 'g1',
    'f9', 'f8', 'f7', 'f6', 'f5', 'f4', 'f3', 'f2', 'f1',
    'e9', 'e8', 'e7', 'e6', 'e5', 'e4', 'e3', 'e2', 'e1',
    'd9', 'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1',
    'c9', 'c8', 'c7', 'c6', 'c5', 'c4', 'c3', 'c2', 'c1',
    'b9', 'b8', 'b7', 'b6', 'b5', 'b4', 'b3', 'b2', 'b1',
    'a9', 'a8', 'a7', 'a6', 'a5', 'a4', 'a3', 'a2', 'a1',
]

# PGN形式の持ち駒の駒名マッピング
PGN_HAND_PIECES = [
    'P', 'L', 'N', 'S', 'G', 'B', 'R',
]

# PGN形式の駒の種類マッピング
PGN_PIECE_TYPES = [
    None,
    'P', 'L', 'N', 'S', 'B', 'R', 'G', 'K',
    '+P', '+L', '+K', '+S', '+B', '+R',
]

def move_to_san(move: int) -> str:
    """Convert a given move to Standard Algebraic Notation (SAN) for PGN.

    :param move: An integer representing the move.
    :return: A string representing the move in SAN.
    ---
    指定された指し手をPGN形式のSAN（Standard Algebraic Notation）に変換します。

    :param move: 指し手を表す整数。
    :return: SAN形式の指し手文字列。
    """
    move_to = PGN_SQUARE_NAMES[cshogi.move_to(move)]

    if cshogi.move_is_drop(move):
        # 駒打ちの場合
        return PGN_HAND_PIECES[cshogi.move_drop_hand_piece(move)] + '@' + move_to

    # 駒の移動の場合
    move_from = PGN_SQUARE_NAMES[cshogi.move_from(move)]
    promotion = '+' if cshogi.move_is_promotion(move) else ''
    return PGN_PIECE_TYPES[cshogi.move_from_piece_type(move)] + move_from + move_to + promotion

class Exporter:
    """A class to handle the exporting of a game to PGN (Portable Game Notation) format.

    :param path: Optional path to the file where the PGN formatted game will be written. If None, no file is opened initially.
    :param append: Whether to append to the existing file or create a new one. Defaults to False.
    ---
    対局をPGN（Portable Game Notation）形式でエクスポート（出力）するためのクラス。

    :param path: PGN形式の棋譜を書き込むファイルへのパス（省略可能）。Noneの場合、ファイルは初期状態では開かれません。
    :param append: 既存ファイルに追記するか、新規作成するか。デフォルトはFalse（新規作成）。
    """
    def __init__(self, path: Optional[str] = None, append: bool = False):
        if path:
            self.open(path, append)
        else:
            self.f = None

    def open(self, path: str, append: bool = False):
        """Open a file for writing the PGN formatted game.

        :param path: Path to the file.
        :param append: Whether to append to the existing file or create a new one. Defaults to False.
        ---
        PGN形式の棋譜を書き込むためにファイルを開きます。

        :param path: ファイルへのパス。
        :param append: 既存ファイルに追記するか、新規作成するか。デフォルトはFalse（新規作成）。
        """
        self.f = open(path, 'a' if append else 'w', newline='\n')

    def close(self):
        """Close the file.
        ---
        ファイルを閉じます。
        """
        self.f.close()

    def tag_pair(self, names: List[str], result: int, event: str = '?', site: str = '?', starttime: Optional[datetime] = None, round: int = 1):
        """Write the tag pairs section of the PGN format, containing metadata about the game.

        :param names: List of player names for White and Black.
        :param result: Game result code.
        :param event: Event name, defaults to "?".
        :param site: Site of the game, defaults to "?".
        :param starttime: Start time of the game, defaults to current time.
        :param round: Round number, defaults to 1.
        ---
        PGN形式のタグペアセクションを書き込みます。対局に関するメタデータを含みます。

        :param names: 先手と後手の対局者名のリスト。
        :param result: 対局結果のコード。
        :param event: イベント名。デフォルトは"?"。
        :param site: 対局場所。デフォルトは"?"。
        :param starttime: 対局開始日時。デフォルトは現在時刻。
        :param round: ラウンド番号。デフォルトは1。
        """
        if starttime is None:
            starttime = datetime.now()
        self.f.write('[Event "' + event +'"]\n')
        self.f.write('[Site "' + site + '"]\n')
        self.f.write('[Date "' + (starttime.strftime('%Y.%m.%d') if starttime else '????.??.??') + '"]\n')
        self.f.write('[Round "' + str(round) + '"]\n')
        self.f.write('[White "' + names[0] + '"]\n')
        self.f.write('[Black "' + names[1] + '"]\n')
        if result == cshogi.BLACK_WIN:
            self.result_str = '1-0'
        elif result == cshogi.WHITE_WIN:
            self.result_str = '0-1'
        elif result == cshogi.DRAW:
            self.result_str = '1/2-1/2'
        else:
            self.result_str = '*'
        self.f.write('[Result "' + self.result_str + '"]\n\n')
        self.f.flush()

    def movetext(self, moves: List[int]):
        """Write the move text section of the PGN format, containing the actual moves of the game.

        :param moves: List of moves in the game.
        ---
        PGN形式の指し手セクションを書き込みます。実際の指し手を含みます。

        :param moves: 対局の指し手のリスト。
        """
        line = ''
        for i, move in enumerate(moves):
            if i % 2 == 0:
                # 奇数番目の指し手（先手の指し手）の前に手数を追加
                part = str(i // 2 + 1) + '. '
            else:
                part = ''
            part += move_to_san(moves[i])
            if i + 1 == len(moves):
                # 最後の指し手の後に結果文字列を追加
                part += ' ' + self.result_str
            
            # 1行80文字以内に収まるように調整
            if len(line) + len(part) <= 80:
                if line != '':
                    line += ' '
                line += part
            else:
                self.f.write(line + '\n')
                line = part

        self.f.write(line + '\n\n')
        self.f.flush()

# Rules implemented by Rc0 (4-player chess, Teams)

These are the chess.com 4-player Teams rules. `rules/` implements exactly what is written here. If the code and
this page disagree, that is a bug; please open an issue.

## Board and seats

- The board is 14×14 with the four 3×3 corners removed, which leaves 160 squares. Files are `a`–`n` and ranks
  `1`–`14`. Squares are numbered 0–159 rank by rank from rank 1.
- The seats move in this order: **Red** (bottom), **Blue** (left), **Yellow** (top), **Green** (right).
- Teams are opposite seats: **Red + Yellow** and **Blue + Green**. The team to move therefore changes every ply.
- The four armies are rotations of each other. Red's back rank is `d1 … k1` = R N B Q K B N R, with pawns on
  `d2 … k2`. Blue's army is Red's rotated by (f, r) → (r, 13−f), Yellow's by (f, r) → (13−f, 13−r) and Green's by
  (f, r) → (13−r, f).

## Movement

- The pieces move as in chess. Sliders stop at the first piece they reach, and a piece may capture only
  **enemy** pieces. Your own and your partner's pieces both block, and neither can be captured.
- Pawns move "forward" relative to their seat. Red moves up the ranks, Yellow down, Blue to the right along the
  files and Green to the left.
  - A pawn may advance two squares from its starting rank if both squares are empty.
  - **En passant** is tracked for each color separately. After a double push, any enemy pawn that attacks the
    skipped square may capture en passant. The right expires when the pawn's owner moves again, and only lasts
    while that pawn is still on the square it moved to.
  - A pawn **promotes** on reaching the 11th rank counting from its own side, which is the middle of the
    opposite army's half. It can promote to a queen, rook, bishop or knight.
- **Castling** follows chess.com, rotated for each seat. For Red: kingside the king goes h1→j1 and the rook
  k1→i1; queenside the king goes h1→f1 and the rook d1→g1. The squares in between must be empty, the king must
  not be in check, and the squares the king passes over or lands on must not be attacked by either enemy.

## Check, mate and the end of the game

- A king is in **check** when either enemy can attack it.
- Moves that leave your own king in check are illegal, with one exception: **capturing an enemy king is always
  legal and ends the game at once**. The capturing team wins. This is how a king left in check by its partner's
  move, or exposed by the turn order, gets punished.
- **Checkmate** is decided when the mated player is to move and has no legal move while in check. That player's
  team loses. Because the two enemies each move before a player moves again, mates can be set up by one enemy
  and finished by the other (*turn-order mates*).
- **Stalemate** (no legal move, not in check) is a draw.
- **50-move rule**: a draw after 200 plies (50 moves by each of the four players) with no capture and no pawn move.
- **Threefold repetition**: a draw when the same position (with the same side to move, castling rights and
  en-passant state) occurs for the third time since the last capture or pawn move.

## Notation

- Moves are written `from` + `to` (for example `h2h4`), with a promotion letter if there is one (`j10j11q`).
  Castling is written as the king's two-square move (`h1j1`).
- Positions use chess.com's FEN4 format (`Position::fen4()` / `set_fen4()`).

## Verification

- `tests/test_rules.cpp` checks perft from the start position:
  20, 395, 7,800, 152,050, 3,452,310 and 77,430,383 nodes at depths 1–6. It also checks FEN4 round trips,
  make/unmake and move parsing over random games.
- During development, the rules library was compared **move for move** against a separate, independently written
  implementation. The comparison covered 30,000 random games (10,000 from the standard start and 20,000 from
  shuffled back ranks), more than 15 million positions in all. For every position it compared the full legal move
  list, the FEN4, check status, game result, 50-move counter and repetition count, and they matched everywhere.
  Random games were biased toward quiet piece moves and toward undoing the previous move, so repetitions,
  50-move draws, castling, en passant, promotion, mates and stalemates were all exercised.

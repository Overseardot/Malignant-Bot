// engine.cpp - Malingnant-Bot chess engine.
//
// Build:  g++ -O2 -std=c++17 -pthread -o engine engine.cpp
// Run:    ./engine        then type:  uci
//                                     isready
//                                     position startpos moves e2e4
//                                     go movetime 1000
// Check:  position startpos   then   go perft 5   (should print 4865609)
//         position startpos   then   go keycheck 4  (hash keys stay consistent)
//
// Move generation: legal moves incl. castling, en passant, promotion (perft-verified).
// Search: iterative deepening, aspiration windows, principal variation search,
//         transposition table, null-move pruning, late move reductions, futility
//         pruning, check extensions, killer + history move ordering, quiescence
//         search, repetition and fifty-move draw detection.
// Evaluation: tapered middlegame/endgame piece-square tables, pawn structure
//         (passed/doubled/isolated), bishop pair, rook files, mobility, king shelter.
// Protocol: UCI with time control and a Hash option.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <cmath>
#include "learning.h"   // machine_learning::score(board, side): small learned eval adjustment
using namespace std;

// ---------------------------------------------------------------- basics
enum Piece { EMPTY = 0, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
enum Color { WHITE = 0, BLACK = 1 };
enum Flag { F_EP = 1, F_CASTLE = 2, F_DOUBLE = 4, F_CAPTURE = 8 };

// A piece is (color << 3) | type. Squares are 0..63 with a1 = 0, h1 = 7, a8 = 56.
inline int colorOf(int p) { return p >> 3; }
inline int typeOf(int p) { return p & 7; }
inline int makePiece(int c, int t) { return (c << 3) | t; }
inline int fileOf(int sq) { return sq & 7; }
inline int rankOf(int sq) { return sq >> 3; }
inline bool inBoard(int f, int r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

static const int KN_DF[8] = {1, 2, 2, 1, -1, -2, -2, -1};
static const int KN_DR[8] = {2, 1, -1, -2, -2, -1, 1, 2};
static const int K_DF[8] = {1, 1, 1, 0, -1, -1, -1, 0};
static const int K_DR[8] = {1, 0, -1, -1, -1, 0, 1, 1};
// Slider directions: 0-3 are rook directions, 4-7 are bishop directions.
static const int S_DF[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int S_DR[8] = {0, 0, 1, -1, 1, -1, 1, -1};

struct Move {
    uint8_t from = 0, to = 0, promo = 0, flags = 0;
    bool operator==(const Move& o) const {
        return from == o.from && to == o.to && promo == o.promo;
    }
};

struct MoveList {
    Move m[256];
    int score[256];
    int n = 0;
};

// ---------------------------------------------------------------- zobrist hashing
static uint64_t Z_PIECE[16][64], Z_CASTLE[16], Z_EP[8], Z_SIDE;

static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static struct ZobristInit {
    ZobristInit() {
        uint64_t s = 0x4d696e6943686573ULL;
        for (auto& row : Z_PIECE) for (auto& v : row) v = splitmix(s);
        for (auto& v : Z_CASTLE) v = splitmix(s);
        for (auto& v : Z_EP) v = splitmix(s);
        Z_SIDE = splitmix(s);
    }
} zobristInit;

// ---------------------------------------------------------------- position
struct Position {
    int b[64];       // int (not uint8_t) because learning.h reads this array
    int side = WHITE;
    int castle = 0;  // bit 1 = white O-O, 2 = white O-O-O, 4 = black O-O, 8 = black O-O-O
    int ep = -1;     // en passant target square, only set when a pawn can really capture there
    int halfmove = 0;
    uint64_t key = 0;

    Position() { memset(b, 0, sizeof(b)); }

    uint64_t computeKey() const {
        uint64_t k = 0;
        for (int sq = 0; sq < 64; sq++)
            if (b[sq]) k ^= Z_PIECE[b[sq]][sq];
        k ^= Z_CASTLE[castle];
        if (ep >= 0) k ^= Z_EP[fileOf(ep)];
        if (side == BLACK) k ^= Z_SIDE;
        return k;
    }

    // Could the side to move capture en passant on square `sq`?
    bool epCapturable(int sq) const {
        int pr = rankOf(sq) + (side == WHITE ? -1 : 1);
        if (pr < 0 || pr > 7) return false;
        for (int df = -1; df <= 1; df += 2) {
            int f = fileOf(sq) + df;
            if (f >= 0 && f < 8 && b[pr * 8 + f] == makePiece(side, PAWN)) return true;
        }
        return false;
    }

    void setFEN(const string& fen) {
        istringstream ss(fen);
        string pos, stm, cas, eps;
        ss >> pos >> stm >> cas >> eps;
        if (!(ss >> halfmove)) halfmove = 0;
        memset(b, 0, sizeof(b));
        int r = 7, f = 0;
        for (char c : pos) {
            if (c == '/') { r--; f = 0; }
            else if (isdigit((unsigned char)c)) f += c - '0';
            else {
                int col = isupper((unsigned char)c) ? WHITE : BLACK, t = EMPTY;
                switch (tolower(c)) {
                    case 'p': t = PAWN; break;
                    case 'n': t = KNIGHT; break;
                    case 'b': t = BISHOP; break;
                    case 'r': t = ROOK; break;
                    case 'q': t = QUEEN; break;
                    case 'k': t = KING; break;
                }
                if (inBoard(f, r)) b[r * 8 + f] = makePiece(col, t);
                f++;
            }
        }
        side = (stm == "b") ? BLACK : WHITE;
        castle = 0;
        for (char c : cas) {
            if (c == 'K') castle |= 1;
            if (c == 'Q') castle |= 2;
            if (c == 'k') castle |= 4;
            if (c == 'q') castle |= 8;
        }
        ep = (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h')
                 ? (eps[0] - 'a') + 8 * (eps[1] - '1') : -1;
        if (ep >= 0 && !epCapturable(ep)) ep = -1;
        key = computeKey();
    }

    string toFEN() const {
        string s;
        for (int r = 7; r >= 0; r--) {
            int empty = 0;
            for (int f = 0; f < 8; f++) {
                int p = b[r * 8 + f];
                if (p == EMPTY) { empty++; continue; }
                if (empty) { s += char('0' + empty); empty = 0; }
                char c = " pnbrqk"[typeOf(p)];
                s += colorOf(p) == WHITE ? char(toupper(c)) : c;
            }
            if (empty) s += char('0' + empty);
            if (r) s += '/';
        }
        s += side == WHITE ? " w " : " b ";
        if (!castle) s += '-';
        else {
            if (castle & 1) s += 'K';
            if (castle & 2) s += 'Q';
            if (castle & 4) s += 'k';
            if (castle & 8) s += 'q';
        }
        s += ' ';
        if (ep < 0) s += '-';
        else { s += char('a' + fileOf(ep)); s += char('1' + rankOf(ep)); }
        return s + " " + to_string(halfmove) + " 1";
    }

    // Is square `sq` attacked by any piece of colour `by`?
    bool attacked(int sq, int by) const {
        int f = fileOf(sq), r = rankOf(sq);
        int pr = r + (by == WHITE ? -1 : 1);  // pawns attack "forward", so look backward
        if (pr >= 0 && pr < 8)
            for (int df = -1; df <= 1; df += 2) {
                int pf = f + df;
                if (pf >= 0 && pf < 8 && b[pr * 8 + pf] == makePiece(by, PAWN)) return true;
            }
        for (int i = 0; i < 8; i++) {
            int nf = f + KN_DF[i], nr = r + KN_DR[i];
            if (inBoard(nf, nr) && b[nr * 8 + nf] == makePiece(by, KNIGHT)) return true;
            nf = f + K_DF[i]; nr = r + K_DR[i];
            if (inBoard(nf, nr) && b[nr * 8 + nf] == makePiece(by, KING)) return true;
        }
        for (int d = 0; d < 8; d++) {
            int nf = f + S_DF[d], nr = r + S_DR[d];
            while (inBoard(nf, nr)) {
                int p = b[nr * 8 + nf];
                if (p != EMPTY) {
                    if (colorOf(p) == by) {
                        int t = typeOf(p);
                        if (t == QUEEN || (d < 4 ? t == ROOK : t == BISHOP)) return true;
                    }
                    break;
                }
                nf += S_DF[d]; nr += S_DR[d];
            }
        }
        return false;
    }

    bool inCheck(int c) const {
        for (int sq = 0; sq < 64; sq++)
            if (b[sq] == makePiece(c, KING)) return attacked(sq, c ^ 1);
        return false;
    }

    // Pseudo-legal moves (may leave own king in check; make() filters those).
    // With capsOnly, only captures and promotions are generated (for quiescence).
    void genMoves(MoveList& ml, bool capsOnly) const {
        int us = side, them = us ^ 1;
        auto add = [&](int from, int to, int promo, int flags) {
            if (capsOnly && !(flags & F_CAPTURE) && !promo) return;
            Move mv;
            mv.from = from; mv.to = to; mv.promo = promo; mv.flags = flags;
            ml.m[ml.n++] = mv;
        };
        for (int sq = 0; sq < 64; sq++) {
            int p = b[sq];
            if (p == EMPTY || colorOf(p) != us) continue;
            int t = typeOf(p), f = fileOf(sq), r = rankOf(sq);

            if (t == PAWN) {
                int dir = us == WHITE ? 1 : -1;
                int startR = us == WHITE ? 1 : 6;
                int lastR = us == WHITE ? 7 : 0;
                int nr = r + dir;
                if (nr < 0 || nr > 7) continue;
                int to = nr * 8 + f;
                if (b[to] == EMPTY) {
                    if (nr == lastR) {
                        for (int pr = QUEEN; pr >= KNIGHT; pr--) add(sq, to, pr, 0);
                    } else {
                        add(sq, to, 0, 0);
                        if (r == startR && b[to + 8 * dir] == EMPTY)
                            add(sq, to + 8 * dir, 0, F_DOUBLE);
                    }
                }
                for (int df = -1; df <= 1; df += 2) {
                    int nf = f + df;
                    if (nf < 0 || nf > 7) continue;
                    int cto = nr * 8 + nf;
                    if (b[cto] != EMPTY && colorOf(b[cto]) == them) {
                        if (nr == lastR) {
                            for (int pr = QUEEN; pr >= KNIGHT; pr--) add(sq, cto, pr, F_CAPTURE);
                        } else {
                            add(sq, cto, 0, F_CAPTURE);
                        }
                    } else if (cto == ep) {
                        add(sq, cto, 0, F_EP | F_CAPTURE);
                    }
                }
            } else if (t == KNIGHT || t == KING) {
                const int* DF = t == KNIGHT ? KN_DF : K_DF;
                const int* DR = t == KNIGHT ? KN_DR : K_DR;
                for (int i = 0; i < 8; i++) {
                    int nf = f + DF[i], nr = r + DR[i];
                    if (!inBoard(nf, nr)) continue;
                    int to = nr * 8 + nf, q = b[to];
                    if (q == EMPTY) add(sq, to, 0, 0);
                    else if (colorOf(q) == them) add(sq, to, 0, F_CAPTURE);
                }
                if (t == KING && !capsOnly) {
                    int base = us == WHITE ? 0 : 56;
                    int ks = us == WHITE ? 1 : 4, qs = us == WHITE ? 2 : 8;
                    if (sq == base + 4 && !attacked(sq, them)) {
                        if ((castle & ks) && b[base + 5] == EMPTY && b[base + 6] == EMPTY &&
                            b[base + 7] == makePiece(us, ROOK) &&
                            !attacked(base + 5, them) && !attacked(base + 6, them))
                            add(sq, base + 6, 0, F_CASTLE);
                        if ((castle & qs) && b[base + 3] == EMPTY && b[base + 2] == EMPTY &&
                            b[base + 1] == EMPTY && b[base] == makePiece(us, ROOK) &&
                            !attacked(base + 3, them) && !attacked(base + 2, them))
                            add(sq, base + 2, 0, F_CASTLE);
                    }
                }
            } else {  // bishop, rook, queen
                int d0 = (t == BISHOP) ? 4 : 0, d1 = (t == ROOK) ? 4 : 8;
                for (int d = d0; d < d1; d++) {
                    int nf = f + S_DF[d], nr = r + S_DR[d];
                    while (inBoard(nf, nr)) {
                        int to = nr * 8 + nf, q = b[to];
                        if (q == EMPTY) add(sq, to, 0, 0);
                        else {
                            if (colorOf(q) == them) add(sq, to, 0, F_CAPTURE);
                            break;
                        }
                        nf += S_DF[d]; nr += S_DR[d];
                    }
                }
            }
        }
    }

    // Plays the move on this position, updating the hash incrementally. Returns false
    // if it leaves our king in check. (Search uses copy-make: copy, then make() the copy.)
    bool make(const Move& m) {
        int p = b[m.from], us = side, captured = b[m.to];
        key ^= Z_PIECE[p][m.from];
        if (captured) key ^= Z_PIECE[captured][m.to];
        b[m.from] = EMPTY;
        if (m.flags & F_EP) {
            int cs = m.to + (us == WHITE ? -8 : 8);
            key ^= Z_PIECE[b[cs]][cs];
            b[cs] = EMPTY;
        }
        int placed = m.promo ? makePiece(us, m.promo) : p;
        b[m.to] = placed;
        key ^= Z_PIECE[placed][m.to];
        if (m.flags & F_CASTLE) {
            int rf, rt;
            if (m.to == m.from + 2) { rf = m.from + 3; rt = m.from + 1; }
            else { rf = m.from - 4; rt = m.from - 1; }
            int rook = b[rf];
            key ^= Z_PIECE[rook][rf] ^ Z_PIECE[rook][rt];
            b[rt] = rook;
            b[rf] = EMPTY;
        }
        key ^= Z_CASTLE[castle];
        auto upd = [&](int sq) {
            if (sq == 4) castle &= ~3;
            if (sq == 60) castle &= ~12;
            if (sq == 7) castle &= ~1;
            if (sq == 0) castle &= ~2;
            if (sq == 63) castle &= ~4;
            if (sq == 56) castle &= ~8;
        };
        upd(m.from);
        upd(m.to);
        key ^= Z_CASTLE[castle];
        if (ep >= 0) key ^= Z_EP[fileOf(ep)];
        ep = -1;
        halfmove = (typeOf(p) == PAWN || (m.flags & F_CAPTURE)) ? 0 : halfmove + 1;
        side ^= 1;
        key ^= Z_SIDE;
        if (m.flags & F_DOUBLE) {
            int mid = (m.from + m.to) / 2;
            if (epCapturable(mid)) { ep = mid; key ^= Z_EP[fileOf(mid)]; }
        }
        return !inCheck(us);
    }

    // "Pass" the move (used by null-move pruning).
    void makeNull() {
        if (ep >= 0) { key ^= Z_EP[fileOf(ep)]; ep = -1; }
        side ^= 1;
        key ^= Z_SIDE;
        halfmove++;
    }
};

[[maybe_unused]] static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// ---------------------------------------------------------------- move text
string moveToStr(const Move& m) {
    string s;
    s += char('a' + fileOf(m.from)); s += char('1' + rankOf(m.from));
    s += char('a' + fileOf(m.to));   s += char('1' + rankOf(m.to));
    if (m.promo) s += " pnbrqk"[m.promo];
    return s;
}

bool parseMove(const Position& pos, const string& s, Move& out) {
    if (s.size() < 4) return false;
    int from = (s[0] - 'a') + 8 * (s[1] - '1');
    int to = (s[2] - 'a') + 8 * (s[3] - '1');
    if (from < 0 || from > 63 || to < 0 || to > 63) return false;
    int promo = 0;
    if (s.size() > 4) {
        switch (tolower(s[4])) {
            case 'q': promo = QUEEN; break;
            case 'r': promo = ROOK; break;
            case 'b': promo = BISHOP; break;
            case 'n': promo = KNIGHT; break;
        }
    }
    MoveList ml;
    pos.genMoves(ml, false);
    for (int i = 0; i < ml.n; i++) {
        const Move& m = ml.m[i];
        if (m.from == from && m.to == to && m.promo == promo) {
            Position c = pos;
            if (c.make(m)) { out = m; return true; }
        }
    }
    return false;
}

// ---------------------------------------------------------------- evaluation
static const int VAL_MG[7] = {0, 100, 320, 330, 500, 900, 0};
static const int VAL_EG[7] = {0, 115, 300, 320, 530, 930, 0};

// Piece-square tables, written from White's point of view with rank 8 on top.
// Index = piece type - 1. The pawn/king rows are middlegame; the endgame versions follow.
static const int PST[6][64] = {
    {  // pawn
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0},
    {  // knight
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50},
    {  // bishop
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -20,-10,-10,-10,-10,-10,-10,-20},
    {  // rook
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0},
    {  // queen
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
         0,  0,  5,  5,  5,  5,  0, -5,
       -10,  5,  5,  5,  5,  5,  0,-10,
       -10,  0,  5,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20},
    {  // king (middlegame)
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -20,-30,-30,-40,-40,-30,-30,-20,
       -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20}};

static const int KING_EG[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50};

static const int PAWN_EG_ADV[8] = {0, 0, 2, 5, 10, 18, 30, 0};   // by relative rank
static const int PASSED_MG[8] = {0, 2, 5, 10, 20, 35, 55, 0};
static const int PASSED_EG[8] = {0, 5, 15, 30, 55, 90, 140, 0};

// Score in centipawns from the side to move's point of view.
int evaluate(const Position& pos) {
    int mg = 0, eg = 0, phase = 0;
    int pawnCnt[2][8] = {}, minR[2][8], maxR[2][8];
    for (int c = 0; c < 2; c++)
        for (int f = 0; f < 8; f++) { minR[c][f] = 8; maxR[c][f] = -1; }
    int bishops[2] = {0, 0}, kingSq[2] = {0, 0};
    int pawns = 0, minors = 0, majors = 0;

    // Pass 1: material, piece-square tables, pawn bookkeeping.
    for (int sq = 0; sq < 64; sq++) {
        int p = pos.b[sq];
        if (!p) continue;
        int t = typeOf(p), c = colorOf(p), f = fileOf(sq), r = rankOf(sq);
        int idx = (c == WHITE) ? (sq ^ 56) : sq;
        int sgn = (c == WHITE) ? 1 : -1;
        int pm = PST[t - 1][idx], pe = pm;
        if (t == KING) pe = KING_EG[idx];
        else if (t == PAWN) pe = PAWN_EG_ADV[c == WHITE ? r : 7 - r];
        mg += sgn * (VAL_MG[t] + pm);
        eg += sgn * (VAL_EG[t] + pe);
        switch (t) {
            case PAWN:
                pawns++;
                pawnCnt[c][f]++;
                minR[c][f] = min(minR[c][f], r);
                maxR[c][f] = max(maxR[c][f], r);
                break;
            case KNIGHT: minors++; phase += 1; break;
            case BISHOP: minors++; phase += 1; bishops[c]++; break;
            case ROOK: majors++; phase += 2; break;
            case QUEEN: majors++; phase += 4; break;
            case KING: kingSq[c] = sq; break;
        }
    }
    if (pawns == 0 && majors == 0 && minors <= 1) return 0;  // dead draw

    // Pass 2: pawn structure, mobility, rooks.
    for (int sq = 0; sq < 64; sq++) {
        int p = pos.b[sq];
        if (!p) continue;
        int t = typeOf(p), c = colorOf(p), f = fileOf(sq), r = rankOf(sq);
        int sgn = (c == WHITE) ? 1 : -1;
        if (t == PAWN) {
            int rel = (c == WHITE) ? r : 7 - r;
            bool passed = true;
            for (int ff = f - 1; ff <= f + 1 && passed; ff++) {
                if (ff < 0 || ff > 7) continue;
                if (c == WHITE ? maxR[BLACK][ff] > r : minR[WHITE][ff] < r) passed = false;
            }
            if (passed) { mg += sgn * PASSED_MG[rel]; eg += sgn * PASSED_EG[rel]; }
        } else if (t == KNIGHT) {
            int n = 0;
            for (int i = 0; i < 8; i++) {
                int nf = f + KN_DF[i], nr = r + KN_DR[i];
                if (!inBoard(nf, nr)) continue;
                int q = pos.b[nr * 8 + nf];
                if (q == EMPTY || colorOf(q) != c) n++;
            }
            mg += sgn * 4 * (n - 4);
            eg += sgn * 4 * (n - 4);
        } else if (t == BISHOP || t == ROOK || t == QUEEN) {
            int d0 = (t == BISHOP) ? 4 : 0, d1 = (t == ROOK) ? 4 : 8, n = 0;
            for (int d = d0; d < d1; d++) {
                int nf = f + S_DF[d], nr = r + S_DR[d];
                while (inBoard(nf, nr)) {
                    int q = pos.b[nr * 8 + nf];
                    if (q == EMPTY) n++;
                    else { if (colorOf(q) != c) n++; break; }
                    nf += S_DF[d]; nr += S_DR[d];
                }
            }
            if (t == BISHOP) { mg += sgn * 4 * (n - 6); eg += sgn * 5 * (n - 6); }
            else if (t == ROOK) {
                mg += sgn * 2 * (n - 7); eg += sgn * 4 * (n - 7);
                if (pawnCnt[c][f] == 0) {
                    int bonus = (pawnCnt[c ^ 1][f] == 0) ? 20 : 9;
                    mg += sgn * bonus; eg += sgn * bonus / 2;
                }
            } else { mg += sgn * (n - 13); eg += sgn * 2 * (n - 13); }
        }
    }

    for (int c = 0; c < 2; c++) {
        int sgn = (c == WHITE) ? 1 : -1;
        for (int f = 0; f < 8; f++) {
            int n = pawnCnt[c][f];
            if (!n) continue;
            if (n > 1) { mg -= sgn * (n - 1) * 12; eg -= sgn * (n - 1) * 20; }
            bool isolated = (f == 0 || !pawnCnt[c][f - 1]) && (f == 7 || !pawnCnt[c][f + 1]);
            if (isolated) { mg -= sgn * n * 10; eg -= sgn * n * 15; }
        }
        if (bishops[c] >= 2) { mg += sgn * 30; eg += sgn * 50; }
        // King shelter (middlegame only): friendly pawns in front of a castled king.
        int ksq = kingSq[c], kf = fileOf(ksq), kr = rankOf(ksq);
        if (c == WHITE ? kr <= 1 : kr >= 6) {
            int dir = (c == WHITE) ? 1 : -1;
            for (int ff = kf - 1; ff <= kf + 1; ff++) {
                if (ff < 0 || ff > 7) continue;
                int r1 = kr + dir, r2 = kr + 2 * dir;
                if (inBoard(ff, r1) && pos.b[r1 * 8 + ff] == makePiece(c, PAWN)) mg += sgn * 10;
                else if (inBoard(ff, r2) && pos.b[r2 * 8 + ff] == makePiece(c, PAWN)) mg += sgn * 5;
                else mg -= sgn * 10;
            }
        }
    }

    if (phase > 24) phase = 24;
    int score = (mg * phase + eg * (24 - phase)) / 24;
    // Learned adjustments stay small and additive (white's point of view, like the rest).
    score += (int)llround(machine_learning::score(pos.b, pos.side));
    return (pos.side == WHITE ? score : -score) + 10;  // small bonus for having the move
}

// ---------------------------------------------------------------- transposition table
enum { TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };
struct TTEntry {
    uint64_t key = 0;
    int16_t score = 0;
    uint16_t move = 0;
    int8_t depth = 0;
    uint8_t flag = 0;
};
static vector<TTEntry> g_tt;
static size_t g_ttMask = 0;

static void ttResize(size_t mb) {
    size_t n = 1;
    while (n * 2 * sizeof(TTEntry) <= mb * 1024 * 1024) n *= 2;
    g_tt.assign(n, TTEntry());
    g_ttMask = n - 1;
}
static void ttClear() { fill(g_tt.begin(), g_tt.end(), TTEntry()); }
static inline TTEntry* ttProbe(uint64_t key) {
    TTEntry* e = &g_tt[key & g_ttMask];
    return e->key == key ? e : nullptr;
}
static inline uint16_t encMove(const Move& m) { return m.from | (m.to << 6) | (m.promo << 12); }

// ---------------------------------------------------------------- search
static const int INF = 30000, MATE = 29000, MAXPLY = 64;
static atomic<bool> g_stop(false);

string scoreStr(int s) {
    if (abs(s) > MATE - 100) {
        int moves = (MATE - abs(s) + 1) / 2;
        return "mate " + to_string(s > 0 ? moves : -moves);
    }
    return "cp " + to_string(s);
}

static bool hasPieces(const Position& pos, int c) {
    for (int sq = 0; sq < 64; sq++) {
        int p = pos.b[sq];
        if (p && colorOf(p) == c && typeOf(p) >= KNIGHT && typeOf(p) <= QUEEN) return true;
    }
    return false;
}

struct Searcher {
    chrono::steady_clock::time_point t0;
    long long limitMs = 1000, softMs = 1LL << 60, nodes = 0;
    bool timeUp = false, hasMove = false;
    Move rootBest;
    uint64_t path[MAXPLY + 8];
    uint16_t killers[MAXPLY + 8][2];
    int history[2][64][64];
    vector<uint64_t> hist;  // keys of the game positions before the root

    long long elapsedMs() const {
        return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count();
    }
    void checkTime() {
        if (g_stop.load() || elapsedMs() >= limitMs) timeUp = true;
    }

    bool isRepeat(const Position& pos, int ply) const {
        int hl = (int)hist.size();
        for (int i = 4; i <= pos.halfmove; i += 2) {
            int idx = ply - i;
            uint64_t k;
            if (idx >= 0) k = path[idx];
            else {
                int h = hl + idx;
                if (h < 0) break;
                k = hist[h];
            }
            if (k == pos.key) return true;
        }
        return false;
    }

    // Move ordering: hash move, captures (MVV-LVA), promotions, killers, history.
    void scoreMoves(const Position& pos, MoveList& ml, int ply, uint16_t ttMove, bool root) {
        for (int i = 0; i < ml.n; i++) {
            const Move& m = ml.m[i];
            uint16_t e = encMove(m);
            int s;
            if (root && hasMove && e == encMove(rootBest)) s = 3000000;
            else if (ttMove && e == ttMove) s = 2000000;
            else if (m.flags & F_CAPTURE) {
                int victim = (m.flags & F_EP) ? PAWN : typeOf(pos.b[m.to]);
                s = 1000000 + victim * 16 - typeOf(pos.b[m.from]) + (m.promo ? 5000 : 0);
            }
            else if (m.promo) s = 900000 + m.promo;
            else if (e == killers[ply][0]) s = 800000;
            else if (e == killers[ply][1]) s = 790000;
            else s = history[pos.side][m.from][m.to];
            ml.score[i] = s;
        }
    }
    static void pick(MoveList& ml, int i) {
        int best = i;
        for (int j = i + 1; j < ml.n; j++)
            if (ml.score[j] > ml.score[best]) best = j;
        swap(ml.m[i], ml.m[best]);
        swap(ml.score[i], ml.score[best]);
    }

    int qsearch(const Position& pos, int alpha, int beta, int ply) {
        if ((++nodes & 2047) == 0) checkTime();
        if (timeUp) return 0;
        if (ply >= MAXPLY) return evaluate(pos);
        bool inChk = pos.inCheck(pos.side);
        int stand = -INF;
        if (!inChk) {
            stand = evaluate(pos);
            if (stand >= beta) return beta;
            if (stand > alpha) alpha = stand;
        }
        MoveList ml;
        pos.genMoves(ml, !inChk);
        scoreMoves(pos, ml, ply, 0, false);
        int legal = 0;
        for (int i = 0; i < ml.n; i++) {
            pick(ml, i);
            const Move& m = ml.m[i];
            if (!inChk && !m.promo) {  // delta pruning: this capture can't raise alpha
                int victim = (m.flags & F_EP) ? PAWN : typeOf(pos.b[m.to]);
                if (stand + VAL_MG[victim] + 200 < alpha) continue;
            }
            Position c = pos;
            if (!c.make(m)) continue;
            legal++;
            int score = -qsearch(c, -beta, -alpha, ply + 1);
            if (timeUp) return 0;
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        if (inChk && legal == 0) return -MATE + ply;
        return alpha;
    }

    int negamax(const Position& pos, int depth, int alpha, int beta, int ply, bool allowNull) {
        if ((++nodes & 2047) == 0) checkTime();
        if (timeUp) return 0;
        path[ply] = pos.key;
        bool pvNode = (beta - alpha > 1);
        if (ply > 0) {
            if (pos.halfmove >= 100 || isRepeat(pos, ply)) return 0;
            alpha = max(alpha, -MATE + ply);  // mate distance pruning
            beta = min(beta, MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }
        bool inChk = pos.inCheck(pos.side);
        if (inChk && ply < MAXPLY - 1) depth++;  // check extension
        if (depth <= 0) return qsearch(pos, alpha, beta, ply);
        if (ply >= MAXPLY - 1) return evaluate(pos);

        uint16_t ttMove = 0;
        if (TTEntry* e = ttProbe(pos.key)) {
            ttMove = e->move;
            if (ply > 0 && e->depth >= depth && !pvNode) {
                int s = e->score;
                if (s > MATE - 100) s -= ply;
                else if (s < -MATE + 100) s += ply;
                if (e->flag == TT_EXACT) return s;
                if (e->flag == TT_LOWER && s >= beta) return beta;
                if (e->flag == TT_UPPER && s <= alpha) return alpha;
            }
        }

        int staticEval = inChk ? -INF : evaluate(pos);
        if (!pvNode && !inChk && abs(beta) < MATE - 100) {
            // reverse futility: so far ahead that a shallow search won't change the verdict
            if (depth <= 3 && staticEval - 120 * depth >= beta) return beta;
            // null move: if passing still beats beta, a real move will too
            if (allowNull && depth >= 3 && staticEval >= beta && hasPieces(pos, pos.side)) {
                Position c = pos;
                c.makeNull();
                int R = 2 + depth / 4;
                int score = -negamax(c, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                if (timeUp) return 0;
                if (score >= beta) return beta;
            }
        }

        MoveList ml;
        pos.genMoves(ml, false);
        scoreMoves(pos, ml, ply, ttMove, ply == 0);
        int alpha0 = alpha, legal = 0;
        uint16_t bestMove = 0;
        Move quietsTried[64];
        int nQuiets = 0;
        for (int i = 0; i < ml.n; i++) {
            pick(ml, i);
            const Move m = ml.m[i];
            Position c = pos;
            if (!c.make(m)) continue;
            legal++;
            bool quiet = !(m.flags & F_CAPTURE) && !m.promo;
            bool givesCheck = false;
            if (quiet && legal > 1 && !inChk) givesCheck = c.inCheck(c.side);

            // futility pruning: quiet moves near the leaves that can't reach alpha
            if (!pvNode && !inChk && quiet && !givesCheck && legal > 1 && depth <= 2 &&
                abs(alpha) < MATE - 100 && staticEval + 150 * depth <= alpha)
                continue;

            int newDepth = depth - 1, score;
            if (legal == 1) {
                score = -negamax(c, newDepth, -beta, -alpha, ply + 1, true);
            } else {
                int red = 0;  // late move reduction
                if (depth >= 3 && quiet && !inChk && !givesCheck && legal > 3) {
                    red = 1 + (legal > 8) + (depth > 7);
                    if (red > newDepth - 1) red = max(0, newDepth - 1);
                }
                score = -negamax(c, newDepth - red, -alpha - 1, -alpha, ply + 1, true);
                if (score > alpha && red > 0)
                    score = -negamax(c, newDepth, -alpha - 1, -alpha, ply + 1, true);
                if (score > alpha && score < beta)
                    score = -negamax(c, newDepth, -beta, -alpha, ply + 1, true);
            }
            if (timeUp) return 0;

            if (score >= beta) {
                if (ply == 0) rootBest = m;
                if (quiet) {
                    uint16_t e = encMove(m);
                    if (killers[ply][0] != e) { killers[ply][1] = killers[ply][0]; killers[ply][0] = e; }
                    int bonus = depth * depth;
                    int& h = history[pos.side][m.from][m.to];
                    h = min(h + bonus, 400000);
                    for (int q = 0; q < nQuiets; q++) {   // punish quiets that failed to cut
                        int& hq = history[pos.side][quietsTried[q].from][quietsTried[q].to];
                        hq = max(hq - bonus, 0);
                    }
                }
                ttStore(pos.key, TT_LOWER, beta, depth, encMove(m), ply);
                return beta;
            }
            if (score > alpha) {
                alpha = score;
                bestMove = encMove(m);
                if (ply == 0) rootBest = m;
            }
            if (quiet && nQuiets < 64) quietsTried[nQuiets++] = m;
        }
        if (legal == 0) return inChk ? -MATE + ply : 0;  // checkmate or stalemate
        ttStore(pos.key, alpha > alpha0 ? TT_EXACT : TT_UPPER, alpha, depth, bestMove, ply);
        return alpha;
    }

    void ttStore(uint64_t key, int flag, int score, int depth, uint16_t move, int ply) {
        TTEntry* e = &g_tt[key & g_ttMask];
        if (e->key == key && e->depth > depth && flag != TT_EXACT) return;  // keep the deeper entry
        if (score > MATE - 100) score += ply;
        else if (score < -MATE + 100) score -= ply;
        e->key = key;
        e->score = (int16_t)score;
        e->depth = (int8_t)depth;
        e->flag = (uint8_t)flag;
        if (move || e->key != key) e->move = move;
    }

    // Principal variation: the root move, then whatever the hash table remembers.
    string pvString(Position pos, const Move& first, int maxLen) {
        string s = moveToStr(first);
        pos.make(first);
        vector<uint64_t> seen;
        for (int i = 1; i < maxLen; i++) {
            TTEntry* e = ttProbe(pos.key);
            if (!e || !e->move) break;
            if (find(seen.begin(), seen.end(), pos.key) != seen.end()) break;
            seen.push_back(pos.key);
            MoveList ml;
            pos.genMoves(ml, false);
            bool found = false;
            for (int j = 0; j < ml.n && !found; j++) {
                if (encMove(ml.m[j]) != e->move) continue;
                Position c = pos;
                if (!c.make(ml.m[j])) continue;
                s += " " + moveToStr(ml.m[j]);
                pos = c;
                found = true;
            }
            if (!found) break;
        }
        return s;
    }

    Move run(const Position& root, long long limit, int maxDepth, bool verbose = true,
             const vector<uint64_t>* history_ = nullptr) {
        if (g_tt.empty()) ttResize(16);
        limitMs = limit;
        softMs = limit < (1LL << 50) ? limit * 6 / 10 : (1LL << 60);
        t0 = chrono::steady_clock::now();
        nodes = 0;
        timeUp = false;
        hist = history_ ? *history_ : vector<uint64_t>();
        memset(killers, 0, sizeof(killers));
        memset(history, 0, sizeof(history));
        MoveList ml;
        root.genMoves(ml, false);
        Move best;
        bool any = false;
        for (int i = 0; i < ml.n && !any; i++) {
            Position c = root;
            if (c.make(ml.m[i])) { best = ml.m[i]; any = true; }
        }
        hasMove = false;
        if (!any) return best;
        rootBest = best;
        hasMove = true;
        int score = 0;
        for (int d = 1; d <= maxDepth; d++) {
            int alpha = -INF, beta = INF, delta = 40;
            if (d >= 5) { alpha = max(-INF, score - delta); beta = min(INF, score + delta); }
            int s;
            while (true) {   // aspiration window: widen and retry when the score falls outside
                s = negamax(root, d, alpha, beta, 0, true);
                if (timeUp) break;
                if (s <= alpha && alpha > -INF) { alpha = max(-INF, alpha - delta); delta *= 2; if (delta > 400) alpha = -INF; continue; }
                if (s >= beta && beta < INF) { beta = min(INF, beta + delta); delta *= 2; if (delta > 400) beta = INF; continue; }
                break;
            }
            if (timeUp) break;
            score = s;
            best = rootBest;
            long long ms = elapsedMs();
            if (verbose)
                cout << "info depth " << d << " score " << scoreStr(score) << " nodes " << nodes
                     << " time " << ms << " nps " << (ms > 0 ? nodes * 1000 / ms : nodes)
                     << " pv " << pvString(root, best, d) << endl;
            if (MATE - abs(score) <= d) break;  // forced mate found
            if (ms >= softMs) break;            // not enough time left for another iteration
        }
        return best;
    }
};

// ---------------------------------------------------------------- perft
uint64_t perft(const Position& pos, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    pos.genMoves(ml, false);
    uint64_t n = 0;
    for (int i = 0; i < ml.n; i++) {
        Position c = pos;
        if (!c.make(ml.m[i])) continue;
        n += (depth == 1) ? 1 : perft(c, depth - 1);
    }
    return n;
}

// Walks every line to `depth`, checking that the incrementally updated hash key
// matches a from-scratch recomputation. Returns the number of mismatches.
uint64_t keyCheck(const Position& pos, int depth) {
    if (depth == 0) return 0;
    uint64_t bad = 0;
    MoveList ml;
    pos.genMoves(ml, false);
    Position nm = pos;
    nm.makeNull();
    if (nm.key != nm.computeKey()) bad++;
    for (int i = 0; i < ml.n; i++) {
        Position c = pos;
        if (!c.make(ml.m[i])) continue;
        if (c.key != c.computeKey()) bad++;
        bad += keyCheck(c, depth - 1);
    }
    return bad;
}

// ---------------------------------------------------------------- UCI loop
// Define NO_MAIN to reuse everything above from another front end (see wasm_api.cpp).
#ifndef NO_MAIN
static thread g_thread;

static void stopSearch() {
    g_stop = true;
    if (g_thread.joinable()) g_thread.join();
}

int main() {
    Position pos;
    pos.setFEN(START_FEN);
    vector<uint64_t> gameKeys;  // hash keys of the positions played before the current one
    ttResize(16);
    string line;
    while (getline(cin, line)) {
        istringstream ss(line);
        string cmd;
        ss >> cmd;
        if (cmd == "uci") {
            cout << "id name Malingnant-Bot\nid author Overseardot\n"
                    "option name Hash type spin default 16 min 1 max 2048\nuciok" << endl;
        } else if (cmd == "isready") {
            cout << "readyok" << endl;
        } else if (cmd == "setoption") {
            stopSearch();
            string tok, name, value;
            ss >> tok;  // "name"
            while (ss >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            while (ss >> tok) value += (value.empty() ? "" : " ") + tok;
            if (name == "Hash") ttResize((size_t)max(1, min(2048, atoi(value.c_str()))));
        } else if (cmd == "ucinewgame") {
            stopSearch();
            pos.setFEN(START_FEN);
            gameKeys.clear();
            ttClear();
        } else if (cmd == "position") {
            stopSearch();
            gameKeys.clear();
            string tok;
            ss >> tok;
            if (tok == "startpos") { pos.setFEN(START_FEN); ss >> tok; }
            else if (tok == "fen") {
                string fen;
                while (ss >> tok && tok != "moves") fen += tok + " ";
                pos.setFEN(fen);
            }
            if (tok == "moves") {
                string mv;
                while (ss >> mv) {
                    Move m;
                    if (parseMove(pos, mv, m)) { gameKeys.push_back(pos.key); pos.make(m); }
                }
            }
        } else if (cmd == "go") {
            stopSearch();
            long long wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0, movestogo = 0;
            int depth = 0;
            string tok;
            while (ss >> tok) {
                if (tok == "wtime") ss >> wtime;
                else if (tok == "btime") ss >> btime;
                else if (tok == "winc") ss >> winc;
                else if (tok == "binc") ss >> binc;
                else if (tok == "movetime") ss >> movetime;
                else if (tok == "movestogo") ss >> movestogo;
                else if (tok == "depth") ss >> depth;
                else if (tok == "keycheck") {
                    int d = 1;
                    ss >> d;
                    cout << "key mismatches " << keyCheck(pos, d) << endl;
                    goto next;
                }
                else if (tok == "perft") {
                    int d = 1;
                    ss >> d;
                    uint64_t total = 0;
                    MoveList ml;
                    pos.genMoves(ml, false);
                    for (int i = 0; i < ml.n; i++) {
                        Position c = pos;
                        if (!c.make(ml.m[i])) continue;
                        uint64_t n = d > 1 ? perft(c, d - 1) : 1;
                        cout << moveToStr(ml.m[i]) << ": " << n << endl;
                        total += n;
                    }
                    cout << "nodes " << total << endl;
                    goto next;
                }
            }
            {
                long long limit;
                int maxDepth = depth > 0 ? depth : MAXPLY;
                long long mine = pos.side == WHITE ? wtime : btime;
                long long inc = pos.side == WHITE ? winc : binc;
                if (movetime > 0) limit = movetime;
                else if (mine > 0) {
                    long long mtg = movestogo > 0 ? movestogo : 30;
                    limit = mine / mtg + inc * 3 / 4;
                    limit = min(limit, mine * 8 / 10);
                    limit = max(limit, 10LL);
                } else {
                    limit = 1LL << 60;  // no clock given: run until depth is reached or "stop"
                }
                g_stop = false;
                Position copy = pos;
                vector<uint64_t> keys = gameKeys;
                g_thread = thread([copy, limit, maxDepth, keys]() {
                    static Searcher s;   // big (history tables), so keep it off the thread stack
                    Move m = s.run(copy, limit, maxDepth, true, &keys);
                    cout << "bestmove " << (s.hasMove ? moveToStr(m) : string("0000")) << endl;
                });
            }
        } else if (cmd == "stop") {
            stopSearch();
        } else if (cmd == "quit") {
            stopSearch();
            return 0;
        }
    next:;
    }
    if (g_thread.joinable()) g_thread.join();  // stdin closed: let a running search finish
    return 0;
}
#endif  // NO_MAIN
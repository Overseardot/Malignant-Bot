// engine.cpp - Malingnant-Bot chess engine.
//
// Build:  g++ -O2 -std=c++17 -pthread -o engine engine.cpp
// Run:    ./engine        then type:  uci
//                                     isready
//                                     position startpos moves e2e4
//                                     go movetime 1000
// Check:  position startpos   then   go perft 5   (should print 4865609)
//
// What it has: legal move generation (castling, en passant, promotion),
// iterative-deepening alpha-beta, quiescence search, MVV-LVA move ordering,
// material + piece-square evaluation, and a UCI interface with time control.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include "learning.h"
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

// ---------------------------------------------------------------- position
struct Position {
    int b[64];
    int side = WHITE;
    int castle = 0;  // bit 1 = white O-O, 2 = white O-O-O, 4 = black O-O, 8 = black O-O-O
    int ep = -1;     // en passant target square, or -1
    int halfmove = 0;

    Position() { for (int i = 0; i < 64; i++) b[i] = EMPTY; }

    void setFEN(const string& fen) {
        istringstream ss(fen);
        string pos, stm, cas, eps;
        ss >> pos >> stm >> cas >> eps;
        if (!(ss >> halfmove)) halfmove = 0;
        for (int i = 0; i < 64; i++) b[i] = EMPTY;
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

    // Plays the move on this position. Returns false if it leaves our king in check.
    // (Search uses copy-make: copy the position, call make() on the copy.)
    bool make(const Move& m) {
        int p = b[m.from], us = side;
        b[m.to] = p;
        b[m.from] = EMPTY;
        if (m.flags & F_EP) b[m.to + (us == WHITE ? -8 : 8)] = EMPTY;
        if (m.promo) b[m.to] = makePiece(us, m.promo);
        if (m.flags & F_CASTLE) {
            if (m.to == m.from + 2) { b[m.from + 1] = b[m.from + 3]; b[m.from + 3] = EMPTY; }
            else { b[m.from - 1] = b[m.from - 4]; b[m.from - 4] = EMPTY; }
        }
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
        ep = (m.flags & F_DOUBLE) ? (m.from + m.to) / 2 : -1;
        halfmove = (typeOf(p) == PAWN || (m.flags & F_CAPTURE)) ? 0 : halfmove + 1;
        side ^= 1;
        return !inCheck(us);
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
static const int VALUE[7] = {0, 100, 320, 330, 500, 900, 0};

// Piece-square tables, written from White's point of view with rank 8 on top.
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

// Score in centipawns from the side to move's point of view.
// This deliberately stays lightweight: the search can evaluate millions of
// positions, so every feature here is incremental-by-scan and allocation-free.
int evaluate(const Position& pos) {
    int score = 0;
    int pieceCount[2][7]{};
    int pawns[2][8]{};
    int bishops[2] = {};
    int rooks[2][8]{};
    int kingSq[2] = {-1, -1};

    for (int sq = 0; sq < 64; ++sq) {
        int p = pos.b[sq];
        if (p == EMPTY) continue;
        int t = typeOf(p), c = colorOf(p);
        pieceCount[c][t]++;
        if (t == PAWN) pawns[c][fileOf(sq)]++;
        if (t == BISHOP) bishops[c]++;
        if (t == ROOK) rooks[c][fileOf(sq)]++;
        if (t == KING) kingSq[c] = sq;
        int v = VALUE[t] + PST[t - 1][c == WHITE ? (sq ^ 56) : sq];
        score += (c == WHITE) ? v : -v;
    }

    // Bishop pair.
    if (bishops[WHITE] >= 2) score += 32;
    if (bishops[BLACK] >= 2) score -= 32;

    // Pawn structure: doubled, isolated and passed pawns.
    for (int c = WHITE; c <= BLACK; ++c) {
        int sign = c == WHITE ? 1 : -1;
        for (int f = 0; f < 8; ++f) {
            if (pawns[c][f] > 1) score += sign * -14 * (pawns[c][f] - 1);
            if (pawns[c][f] && (f == 0 || !pawns[c][f-1]) &&
                (f == 7 || !pawns[c][f+1]))
                score += sign * -10;
        }
        for (int sq = 0; sq < 64; ++sq) {
            if (typeOf(pos.b[sq]) != PAWN || colorOf(pos.b[sq]) != c) continue;
            int f = fileOf(sq), r = rankOf(sq);
            bool passed = true;
            for (int nf = max(0, f - 1); nf <= min(7, f + 1); ++nf) {
                if (c == WHITE) {
                    for (int rr = r + 1; rr < 8; ++rr)
                        if (pos.b[rr * 8 + nf] == makePiece(BLACK, PAWN)) passed = false;
                } else {
                    for (int rr = r - 1; rr >= 0; --rr)
                        if (pos.b[rr * 8 + nf] == makePiece(WHITE, PAWN)) passed = false;
                }
            }
            if (passed) {
                int advance = c == WHITE ? r - 1 : 6 - r;
                score += sign * (20 + max(0, advance) * 8);
            }
        }
    }

    // Rooks like open and semi-open files.
    for (int c = WHITE; c <= BLACK; ++c) {
        int sign = c == WHITE ? 1 : -1;
        for (int f = 0; f < 8; ++f) {
            if (!rooks[c][f]) continue;
            bool wp = pawns[WHITE][f] != 0, bp = pawns[BLACK][f] != 0;
            if (!wp && !bp) score += sign * 20 * rooks[c][f];
            else if ((c == WHITE && !wp) || (c == BLACK && !bp))
                score += sign * 10 * rooks[c][f];
        }
    }

    // Development/central control through legal move counts. This is intentionally
    // modest because mobility is expensive and should not dominate material.
    for (int c = WHITE; c <= BLACK; ++c) {
        Position q = pos;
        q.side = c;
        MoveList ml;
        q.genMoves(ml, false);
        int legal = 0;
        for (int i = 0; i < ml.n; ++i) {
            Position n = q;
            if (n.make(ml.m[i])) ++legal;
        }
        score += (c == WHITE ? 1 : -1) * min(legal, 40) * 2;
    }

    // King safety: penalize exposed kings in the middlegame; scale down as pieces
    // disappear so the engine can activate its king in endgames.
    int totalNonPawns = 0;
    for (int c = WHITE; c <= BLACK; ++c)
        for (int t = KNIGHT; t <= QUEEN; ++t) totalNonPawns += pieceCount[c][t];
    if (totalNonPawns >= 6) {
        for (int c = WHITE; c <= BLACK; ++c) {
            if (kingSq[c] < 0) continue;
            int f = fileOf(kingSq[c]), r = rankOf(kingSq[c]);
            int danger = 0;
            for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) {
                if (!df && !dr) continue;
                int nf = f + df, nr = r + dr;
                if (!inBoard(nf, nr)) continue;
                int sq = nr * 8 + nf;
                int enemy = c ^ 1;
                if (pos.attacked(sq, enemy)) danger++;
            }
            score += (c == WHITE ? 1 : -1) * (-12 * danger);
        }
    }

    // Learned adjustments are deliberately small and additive.
    score += (int)llround(machine_learning::score(pos.b, pos.side));
    return pos.side == WHITE ? score : -score;
}

// ---------------------------------------------------------------- search
static const int INF = 30000, MATE = 29000, MAXPLY = 64;
static atomic<bool> g_stop(false);

struct Zobrist {
    uint64_t piece[64][16]{};
    uint64_t side = 0;
    uint64_t castle[16]{};
    uint64_t ep[64]{};
    bool ready = false;

    static uint64_t mix(uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    void init() {
        if (ready) return;
        uint64_t seed = 0x243f6a8885a308d3ULL;
        for (int sq = 0; sq < 64; ++sq)
            for (int p = 0; p < 16; ++p)
                piece[sq][p] = mix(seed += 0x9e3779b97f4a7c15ULL);
        side = mix(seed += 0x9e3779b97f4a7c15ULL);
        for (int i = 0; i < 16; ++i) castle[i] = mix(seed += 0x9e3779b97f4a7c15ULL);
        for (int i = 0; i < 64; ++i) ep[i] = mix(seed += 0x9e3779b97f4a7c15ULL);
        ready = true;
    }

    uint64_t hash(const Position& pos) {
        init();
        uint64_t h = 0;
        for (int sq = 0; sq < 64; ++sq)
            if (pos.b[sq]) h ^= piece[sq][pos.b[sq]];
        if (pos.side == BLACK) h ^= side;
        h ^= castle[pos.castle & 15];
        if (pos.ep >= 0) h ^= ep[pos.ep];
        return h;
    }
};

static Zobrist Z;

enum TTFlag : uint8_t { TT_EXACT = 0, TT_ALPHA = 1, TT_BETA = 2 };

struct TTEntry {
    uint64_t key = 0;
    int score = 0;
    int depth = -1;
    uint8_t flag = TT_EXACT;
    Move best;
};

class TranspositionTable {
    vector<TTEntry> table;
public:
    explicit TranspositionTable(size_t mb = 32) {
        size_t bytes = mb * 1024ULL * 1024ULL;
        size_t n = max<size_t>(1, bytes / sizeof(TTEntry));
        table.resize(n);
    }

    void clear() {
        for (auto& e : table) e = TTEntry{};
    }

    TTEntry* probe(uint64_t key) {
        TTEntry& e = table[key % table.size()];
        return e.key == key ? &e : nullptr;
    }

    void store(uint64_t key, int depth, int score, uint8_t flag, const Move& best) {
        TTEntry& e = table[key % table.size()];
        if (e.key != key || depth >= e.depth) {
            e.key = key;
            e.depth = depth;
            e.score = score;
            e.flag = flag;
            e.best = best;
        }
    }
};



string scoreStr(int s) {
    if (abs(s) > MATE - 100) {
        int moves = (MATE - abs(s) + 1) / 2;
        return "mate " + to_string(s > 0 ? moves : -moves);
    }
    return "cp " + to_string(s);
}

struct Searcher {
    chrono::steady_clock::time_point t0;
    long long limitMs = 1000, nodes = 0;
    bool timeUp = false;
    Move rootBest, prevBest;
    bool havePrev = false;
    bool hasMove = false;  // false if the position has no legal move
    TranspositionTable tt;
    Move killer[MAXPLY][2]{};
    int history[2][64][64]{};

    Searcher() : tt(16) {}

    long long elapsedMs() const {
        return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count();
    }
    void checkTime() {
        if (g_stop.load() || elapsedMs() >= limitMs) timeUp = true;
    }

    // Move ordering: previous best first, then captures (MVV-LVA), then promotions.
    void scoreMoves(const Position& pos, MoveList& ml, int ply) {
        for (int i = 0; i < ml.n; i++) {
            const Move& m = ml.m[i];
            int s = 0;
            if (ply == 0 && havePrev && m == prevBest) s = 1000000;
            else if (m.flags & F_CAPTURE) {
                int victim = (m.flags & F_EP) ? PAWN : typeOf(pos.b[m.to]);
                int attacker = typeOf(pos.b[m.from]);
                s = 100000 + victim * 100 - attacker * 8;
                if (m.promo) s += 80000;
            }
            if (m.promo) s += 90000 + m.promo;
            if (!(m.flags & F_CAPTURE) && !m.promo) {
                if (ply < MAXPLY && m == killer[ply][0]) s += 70000;
                else if (ply < MAXPLY && m == killer[ply][1]) s += 60000;
                s += history[pos.side][m.from][m.to];
            }
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
        MoveList ml;
        if (inChk) {
            // In check there is no stand-pat score: every legal check evasion
            // must be searched, including quiet king moves and interpositions.
            pos.genMoves(ml, false);
        } else {
            int stand = evaluate(pos);
            if (stand >= beta) return beta;
            if (stand > alpha) alpha = stand;
            pos.genMoves(ml, true);
        }

        scoreMoves(pos, ml, ply);
        int legal = 0;
        for (int i = 0; i < ml.n; i++) {
            pick(ml, i);
            Position c = pos;
            if (!c.make(ml.m[i])) continue;
            ++legal;
            int score = -qsearch(c, -beta, -alpha, ply + 1);
            if (timeUp) return 0;
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }

        if (inChk && legal == 0) return -MATE + ply;
        return alpha;
    }

    int negamax(const Position& pos, int depth, int alpha, int beta, int ply) {
        if ((++nodes & 2047) == 0) checkTime();
        if (timeUp) return 0;
        if (ply > 0 && pos.halfmove >= 100) return 0;

        bool inChk = pos.inCheck(pos.side);
        if (inChk && ply < MAXPLY) depth++;
        if (depth <= 0) return qsearch(pos, alpha, beta, ply);

        const uint64_t key = Z.hash(pos);
        const int originalAlpha = alpha;
        const int originalBeta = beta;
        TTEntry* hit = tt.probe(key);
        if (hit && hit->depth >= depth) {
            if (hit->flag == TT_EXACT) return hit->score;
            if (hit->flag == TT_ALPHA) alpha = max(alpha, hit->score);
            else if (hit->flag == TT_BETA) beta = min(beta, hit->score);
            if (alpha >= beta) return hit->score;
        }

        // Null-move pruning: skip a turn in positions where we have enough
        // material and are not in check. Never use it near the horizon.
        if (!inChk && depth >= 3 && beta < MATE - 100) {
            int nonPawn = 0;
            for (int sq = 0; sq < 64; ++sq) {
                int p = pos.b[sq];
                if (p && colorOf(p) == pos.side && typeOf(p) >= KNIGHT && typeOf(p) <= QUEEN)
                    nonPawn += VALUE[typeOf(p)];
            }
            if (nonPawn >= 500) {
                Position n = pos;
                n.side ^= 1;
                n.ep = -1;
                n.halfmove++;
                int reduction = 2 + depth / 4;
                int score = -negamax(n, depth - 1 - reduction, -beta, -beta + 1, ply + 1);
                if (timeUp) return 0;
                if (score >= beta) return beta;
            }
        }

        MoveList ml;
        pos.genMoves(ml, false);
        scoreMoves(pos, ml, ply);
        if (hit) {
            for (int i = 0; i < ml.n; ++i) {
                if (ml.m[i] == hit->best) {
                    ml.score[i] += 800000;
                    break;
                }
            }
        }

        int legal = 0;
        int moveIndex = 0;
        Move bestMove;
        for (int i = 0; i < ml.n; ++i) {
            pick(ml, i);
            Position c = pos;
            if (!c.make(ml.m[i])) continue;
            ++legal;

            int score;
            bool quiet = !(ml.m[i].flags & F_CAPTURE) && !ml.m[i].promo;
            if (moveIndex >= 4 && depth >= 3 && quiet && !inChk) {
                int reduction = 1 + (moveIndex >= 10) + (depth >= 6);
                score = -negamax(c, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                if (score > alpha)
                    score = -negamax(c, depth - 1, -beta, -alpha, ply + 1);
            } else {
                score = -negamax(c, depth - 1, -beta, -alpha, ply + 1);
            }
            if (timeUp) return 0;
            ++moveIndex;

            if (score > alpha) {
                alpha = score;
                bestMove = ml.m[i];
                if (ply == 0) rootBest = ml.m[i];
                if (alpha >= beta) {
                    if (quiet && ply < MAXPLY) {
                        if (!(ml.m[i] == killer[ply][0])) {
                            killer[ply][1] = killer[ply][0];
                            killer[ply][0] = ml.m[i];
                        }
                        int& h = history[pos.side][ml.m[i].from][ml.m[i].to];
                        h = min(20000, h + depth * depth);
                    }
                    break;
                }
            }
        }
        if (legal == 0) return inChk ? -MATE + ply : 0;

        uint8_t flag = TT_EXACT;
        if (alpha <= originalAlpha) flag = TT_ALPHA;
        else if (alpha >= originalBeta) flag = TT_BETA;
        tt.store(key, depth, alpha, flag, bestMove);
        return alpha;
    }

    Move run(const Position& root, long long limit, int maxDepth, bool verbose = true) {
        limitMs = limit;
        t0 = chrono::steady_clock::now();
        Z.init();
        // Keep TT entries across iterative-deepening iterations so the
        // previous depth provides ordering and bounds to the next depth.
        nodes = 0;
        timeUp = false;
        MoveList ml;
        root.genMoves(ml, false);
        Move best;
        bool any = false;
        for (int i = 0; i < ml.n && !any; i++) {
            Position c = root;
            if (c.make(ml.m[i])) { best = ml.m[i]; any = true; }
        }
        if (!any) { hasMove = false; return best; }
        hasMove = true;
        prevBest = best;
        havePrev = true;
        for (int d = 1; d <= maxDepth; d++) {
            int score = negamax(root, d, -INF, INF, 0);
            if (timeUp) break;
            best = rootBest;
            prevBest = best;
            long long ms = elapsedMs();
            if (verbose) cout << "info depth " << d << " score " << scoreStr(score) << " nodes " << nodes
                 << " time " << ms << " nps " << (ms > 0 ? nodes * 1000 / ms : nodes)
                 << " pv " << moveToStr(best) << endl;
            if (MATE - abs(score) <= d) break;  // forced mate found
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
    string line;
    while (getline(cin, line)) {
        istringstream ss(line);
        string cmd;
        ss >> cmd;
        if (cmd == "uci") {
            cout << "id name Malingnant-Bot\nid author Overseardot\nuciok" << endl;
        } else if (cmd == "isready") {
            cout << "readyok" << endl;
        } else if (cmd == "ucinewgame") {
            stopSearch();
            pos.setFEN(START_FEN);
        } else if (cmd == "position") {
            stopSearch();
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
                    if (parseMove(pos, mv, m)) pos.make(m);
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
                g_thread = thread([copy, limit, maxDepth]() {
                    Searcher s;
                    Move m = s.run(copy, limit, maxDepth);
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

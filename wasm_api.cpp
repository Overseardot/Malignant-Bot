// wasm_api.cpp - exposes the engine to JavaScript so the HTML page can use it.
//
// Build for the browser (needs Emscripten: https://emscripten.org):
//   emcc wasm_api.cpp -std=c++17 -O2 -o engine.js -sMODULARIZE=1 -sEXPORT_NAME=createEngine -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sEXPORTED_RUNTIME_METHODS=ccall
//
// That produces one file, engine.js (the WebAssembly is embedded in it).
// Put it next to index.html and open the page.
//
// Every function takes the position as a FEN string, so the page keeps all the
// game state and the engine stays stateless. Strings returned from here are
// only valid until the next call (JavaScript copies them straight away).

#define NO_MAIN
#include "engine.cpp"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EXPORT extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT extern "C"  // lets you compile and test this natively too
#endif

static string g_out;

// Keep the relatively large search state off the WebAssembly call stack.
// The browser bridge is currently single-threaded, so one persistent searcher
// is sufficient and avoids repeated stack allocation for every engine call.
static Searcher g_searcher;

// Legal moves as space-separated UCI text, e.g. "e2e4 g1f3 ...".
EXPORT const char* engine_legal_moves(const char* fen) {
    Position pos;
    pos.setFEN(fen);
    MoveList ml;
    pos.genMoves(ml, false);
    g_out.clear();
    for (int i = 0; i < ml.n; i++) {
        Position c = pos;
        if (!c.make(ml.m[i])) continue;
        if (!g_out.empty()) g_out += ' ';
        g_out += moveToStr(ml.m[i]);
    }
    return g_out.c_str();
}

// The FEN after playing `move`, or "" if the move is illegal.
EXPORT const char* engine_apply_move(const char* fen, const char* move) {
    Position pos;
    pos.setFEN(fen);
    Move m;
    g_out.clear();
    if (parseMove(pos, move, m)) {
        pos.make(m);
        g_out = pos.toFEN();
    }
    return g_out.c_str();
}

// Searches for `movetime_ms` milliseconds and returns the best move, or "" if there is none.
EXPORT const char* engine_best_move(const char* fen, int movetime_ms) {
    Position pos;
    pos.setFEN(fen);
    g_stop = false;
    Move m = g_searcher.run(pos, movetime_ms, MAXPLY, false);
    g_out = g_searcher.hasMove ? moveToStr(m) : "";
    return g_out.c_str();
}

// Search from a starting FEN after the supplied UCI move history.
// This lets the searcher detect threefold repetition along the actual game path.
EXPORT const char* engine_best_move_moves(const char* fen, const char* moves, int movetime_ms) {
    Position pos;
    pos.setFEN(fen);
    vector<uint64_t> history;
    istringstream ss(moves ? moves : "");
    string tok;
    while (ss >> tok) {
        Move m;
        if (!parseMove(pos, tok, m)) break;
        history.push_back(pos.key);
        pos.make(m);
    }
    g_stop = false;
    Move m = g_searcher.run(pos, movetime_ms, MAXPLY, false, &history);
    g_out = g_searcher.hasMove ? moveToStr(m) : "";
    return g_out.c_str();
}

// Reset the shared transposition table when a new game begins.
EXPORT void engine_new_game() {
    if (g_tt.empty()) ttResize(16);
    ttClear();
}

// One of: ongoing, check, checkmate, stalemate, draw-50, draw-material.
// "checkmate" means the side to move has been mated.
EXPORT const char* engine_status(const char* fen) {
    Position pos;
    pos.setFEN(fen);
    MoveList ml;
    pos.genMoves(ml, false);
    bool anyLegal = false;
    for (int i = 0; i < ml.n && !anyLegal; i++) {
        Position c = pos;
        if (c.make(ml.m[i])) anyLegal = true;
    }
    bool chk = pos.inCheck(pos.side);
    if (!anyLegal) g_out = chk ? "checkmate" : "stalemate";
    else if (pos.halfmove >= 100) g_out = "draw-50";
    else {
        int others = 0, lastType = 0;
        for (int i = 0; i < 64; i++) {
            int t = typeOf(pos.b[i]);
            if (t != EMPTY && t != KING) { others++; lastType = t; }
        }
        if (others == 0 || (others == 1 && (lastType == KNIGHT || lastType == BISHOP)))
            g_out = "draw-material";
        else
            g_out = chk ? "check" : "ongoing";
    }
    return g_out.c_str();
}

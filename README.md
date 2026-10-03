# The-Machine

A C++ chess engine with an HTML/WebAssembly interface and a persistent self-play learning layer.

## Current architecture

HTML UI
  ↓
WASM API
  ↓
C++ chess engine
  ├─ legal move generation
  ├─ alpha-beta / negamax search
  ├─ quiescence search
  ├─ evaluation + piece-square tables
  └─ learned evaluation adjustments
          ↑
     self-play trainer
          ↓
  learned_weights.dat

## Build the engine

    g++ -O2 -std=c++17 -pthread -o engine engine.cpp

## Build the self-play learner

    g++ -O2 -std=c++17 -pthread -o self_play self_play.cpp

Run training:

    ./self_play 100 25 250 5

Arguments: games, milliseconds per move, maximum plies per game, exploration percentage.

The learner saves evaluation adjustments to learned_weights.dat. Running training again loads existing weights and continues from them.

## Browser build

Requires Emscripten:

    emcc wasm_api.cpp -std=c++17 -O2 -o engine.js -sMODULARIZE=1 -sEXPORT_NAME=createEngine -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 -sEXPORTED_RUNTIME_METHODS=ccall

Place the generated engine.js beside the HTML page.

## Learning design

The first learning layer is intentionally small and transparent. It uses only C++ and does not depend on Python, neural-network libraries, or an external chess engine.

Self-play produces positions and a game result. A bounded TD-style update adjusts a small set of material and centre-control evaluation weights. The normal hand-written evaluation remains the baseline.

This is the foundation for richer positional features, repetition-aware training, stronger self-play exploration, and more advanced learning algorithms.
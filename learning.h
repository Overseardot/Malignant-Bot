#pragma once
#include <cmath>
#include <fstream>
#include <string>

namespace machine_learning {

struct Weights {
    double material[6] = {0, 0, 0, 0, 0, 0};
    double center = 0;
    double mobility = 0;
};

inline Weights weights;
inline bool loaded = false;

inline void load(const std::string& path = "learned_weights.dat") {
#ifdef __EMSCRIPTEN__
    (void)path;
#else
    if (loaded) return;
    std::ifstream in(path);
    if (in) {
        for (double& w : weights.material) in >> w;
        in >> weights.center >> weights.mobility;
    }
#endif
    loaded = true;
}

inline void save(const std::string& path = "learned_weights.dat") {
#ifdef __EMSCRIPTEN__
    (void)path;
#else
    std::ofstream out(path);
    if (!out) return;
    out.precision(17);
    for (double w : weights.material) out << w << ' ';
    out << weights.center << ' ' << weights.mobility << '\n';
#endif
}

inline void reset() {
    weights = Weights{};
    loaded = true;
}

inline void features(const int* board, int side, double out[8]) {
    for (int i = 0; i < 8; ++i) out[i] = 0;
    for (int sq = 0; sq < 64; ++sq) {
        int p = board[sq];
        if (!p) continue;
        int c = p >> 3, t = p & 7;
        if (t < 1 || t > 6) continue;
        double sign = (c == side) ? 1.0 : -1.0;
        out[t - 1] += sign;
        int f = sq & 7, r = sq >> 3;
        if (f >= 2 && f <= 5 && r >= 2 && r <= 5) out[6] += sign;
    }
    out[7] = 0;
}

inline double score(const int* board, int side, double mobility = 0) {
    load();
    double f[8];
    features(board, side, f);
    f[7] = mobility;
    double s = weights.center * f[6] + weights.mobility * f[7];
    for (int i = 0; i < 6; ++i) s += weights.material[i] * f[i];
    return s;
}

inline void update(const int* board, int side, double outcome, double mobility = 0, double rate = 0.15) {
    load();
    double f[8];
    features(board, side, f);
    f[7] = mobility;
    double raw = score(board, side, mobility) / 1000.0;
    double prediction = std::tanh(raw);
    double error = outcome - prediction;
    for (int i = 0; i < 6; ++i) weights.material[i] += rate * error * f[i];
    weights.center += rate * error * f[6];
    weights.mobility += rate * error * f[7];
}

} // namespace machine_learning

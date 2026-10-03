// self_play.cpp - trains The-Machine's small persistent evaluation layer.
//
// Build:
//   g++ -O2 -std=c++17 -pthread -o self_play self_play.cpp
//
// Run:
//   ./self_play 100 25 250 5
//   arguments: games, milliseconds per move, max plies, exploration percent
//
// Training flow:
//   self-play -> record positions -> final game result -> bounded outcome updates -> save
// The search engine remains the same; learning changes only its evaluation.

#define NO_MAIN
#include "engine.cpp"

#include <iomanip>
#include <random>

using namespace std;

static bool terminalResult(const Position& pos, int& winner) {
    MoveList ml;
    pos.genMoves(ml, false);
    bool any = false;
    for (int i = 0; i < ml.n; ++i) {
        Position c = pos;
        if (c.make(ml.m[i])) {
            any = true;
            break;
        }
    }
    if (any) return false;
    winner = pos.inCheck(pos.side) ? (pos.side ^ 1) : -1;
    return true;
}

int main(int argc, char** argv) {
    int games = argc > 1 ? max(1, atoi(argv[1])) : 100;
    long long moveMs = argc > 2 ? max(1LL, atoll(argv[2])) : 25;
    int maxPlies = argc > 3 ? max(20, atoi(argv[3])) : 250;
    int exploration = argc > 4 ? max(0, min(100, atoi(argv[4]))) : 5;

    machine_learning::load();

    mt19937 rng((unsigned)chrono::high_resolution_clock::now().time_since_epoch().count());
    uniform_int_distribution<int> pct(1, 100);

    int whiteWins = 0, blackWins = 0, draws = 0;

    for (int game = 1; game <= games; ++game) {
        Position pos;
        pos.setFEN(START_FEN);
        vector<Position> history;
        history.reserve(maxPlies + 1);

        int winner = -1;
        for (int ply = 0; ply < maxPlies; ++ply) {
            history.push_back(pos);

            if (terminalResult(pos, winner)) break;
            if (pos.halfmove >= 100) {
                winner = -1;
                break;
            }

            Move chosen;
            bool found = false;

            // Small exploration prevents identical self-play from collapsing into
            // one deterministic opening/game. The normal path still uses search.
            if (exploration > 0 && pct(rng) <= exploration) {
                MoveList ml;
                pos.genMoves(ml, false);
                vector<Move> legal;
                for (int i = 0; i < ml.n; ++i) {
                    Position c = pos;
                    if (c.make(ml.m[i])) legal.push_back(ml.m[i]);
                }
                if (!legal.empty()) {
                    uniform_int_distribution<size_t> pick(0, legal.size() - 1);
                    chosen = legal[pick(rng)];
                    found = true;
                }
            }

            if (!found) {
                g_stop = false;
                Searcher search;
                chosen = search.run(pos, moveMs, 32, false);
                found = search.hasMove;
            }

            if (!found || !pos.make(chosen)) {
                winner = -1;
                break;
            }
        }

        if (winner == WHITE) ++whiteWins;
        else if (winner == BLACK) ++blackWins;
        else ++draws;

        for (const Position& sample : history) {
            double outcome = winner < 0 ? 0.0 : (winner == sample.side ? 1.0 : -1.0);
            machine_learning::update(sample.b, sample.side, outcome);
        }
        machine_learning::save();

        cout << "game " << game << "/" << games
             << " result " << (winner == WHITE ? "1-0" : winner == BLACK ? "0-1" : "1/2-1/2")
             << "  W " << whiteWins
             << " B " << blackWins
             << " D " << draws
             << "  weights";
        cout << fixed << setprecision(2);
        for (double w : machine_learning::weights.material) cout << " " << w;
        cout << " center " << machine_learning::weights.center << endl;
    }

    cout << "Training complete. Persistent weights saved to learned_weights.dat" << endl;
    return 0;
}

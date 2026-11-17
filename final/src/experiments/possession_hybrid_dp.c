#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/* Each dead-ball interval resets the two-state live possession decoder. */
int hybrid_live(const double *logp, const double *cost, const uint8_t *dead,
                int n, int8_t *output) {
    int8_t *back = malloc((size_t)n * 2);
    if (back == NULL) return 1;
    int cursor = 0;
    while (cursor < n) {
        if (dead[cursor]) {
            output[cursor++] = 2;
            continue;
        }
        int first = cursor;
        double home = logp[2 * cursor];
        double away = logp[2 * cursor + 1];
        ++cursor;
        while (cursor < n && !dead[cursor]) {
            double switch_home = away - cost[2 * cursor];
            double switch_away = home - cost[2 * cursor + 1];
            double next_home, next_away;
            if (home >= switch_home) {
                back[2 * cursor] = 0;
                next_home = home;
            } else {
                back[2 * cursor] = 1;
                next_home = switch_home;
            }
            if (away >= switch_away) {
                back[2 * cursor + 1] = 1;
                next_away = away;
            } else {
                back[2 * cursor + 1] = 0;
                next_away = switch_away;
            }
            home = next_home + logp[2 * cursor];
            away = next_away + logp[2 * cursor + 1];
            ++cursor;
        }
        int state = home >= away ? 0 : 1;
        output[cursor - 1] = state;
        for (int t = cursor - 1; t > first; --t) {
            state = back[2 * t + state];
            output[t - 1] = state;
        }
    }
    free(back);
    return 0;
}

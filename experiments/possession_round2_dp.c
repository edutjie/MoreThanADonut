#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static double penalty(int a, int b, double turn, double stop) {
    return a == b ? 0 : (a == 2 || b == 2 ? stop : turn);
}

void viterbi(const double *e, int n, double turn, double stop, int8_t *out) {
    double previous[3] = {e[0], e[1], e[2]};
    for (int i = 1; i < n; ++i) {
        double current[3];
        for (int s = 0; s < 3; ++s) {
            double best = -INFINITY;
            int origin = 0;
            for (int p = 0; p < 3; ++p) {
                double value = previous[p] - penalty(p, s, turn, stop);
                if (value > best || (value == best && p == s)) {
                    best = value;
                    origin = p;
                }
            }
            out[n+3*i+s] = origin;
            current[s] = best + e[3*i+s];
        }
        for (int s = 0; s < 3; ++s) previous[s] = current[s];
    }
    int state = 0;
    for (int s = 1; s < 3; ++s) if (previous[s] > previous[state]) state = s;
    out[n-1] = state;
    for (int i = n-1; i > 0; --i) {
        state = out[n+3*i+state];
        out[i-1] = state;
    }
}

/* Segment-entry score includes the transition penalty. For lengths beyond cap,
 * the log prior is linear, so one running maximum per state covers the tail. */
int hsmm(const double *e, int n, double turn, double stop, const double *prior,
         const double *slope, int cap, int8_t *out) {
    int size = 3 * (n + 1);
    double *cumulative = calloc(size, sizeof(double));
    double *entry = calloc(size, sizeof(double));
    int *start = malloc(size * sizeof(int));
    int8_t *origin = malloc(size * sizeof(int8_t));
    if (!cumulative || !entry || !start || !origin) {
        free(cumulative); free(entry); free(start); free(origin);
        return 1;
    }
    double tail[3] = {-INFINITY, -INFINITY, -INFINITY};
    int tail_start[3] = {0, 0, 0};
    double score[3] = {0, 0, 0};
    for (int t = 1; t <= n; ++t) {
        for (int s = 0; s < 3; ++s) {
            int index = 3*t+s;
            cumulative[index] = cumulative[index-3] + e[index-3];
            double best = -INFINITY;
            int boundary = 0;
            for (int duration = 1; duration <= cap && duration <= t; ++duration) {
                int j = t-duration;
                double value = entry[3*j+s] - cumulative[3*j+s] + prior[3*(duration-1)+s];
                if (value > best) { best = value; boundary = j; }
            }
            int j = t-cap-1;
            if (j >= 0) {
                double candidate = entry[3*j+s] - cumulative[3*j+s] - slope[s]*j;
                if (candidate > tail[s]) { tail[s] = candidate; tail_start[s] = j; }
                double value = tail[s] + prior[3*(cap-1)+s] + slope[s]*(t-cap);
                if (value > best) { best = value; boundary = tail_start[s]; }
            }
            score[s] = best + cumulative[index];
            start[index] = boundary;
        }
        for (int s = 0; s < 3; ++s) {
            double best = -INFINITY;
            for (int p = 0; p < 3; ++p) {
                if (p == s) continue;
                double value = score[p] - penalty(p, s, turn, stop);
                if (value > best) { best = value; origin[3*t+s] = p; }
            }
            entry[3*t+s] = best;
        }
    }
    int state = 0;
    for (int s = 1; s < 3; ++s) if (score[s] > score[state]) state = s;
    for (int t = n; t > 0;) {
        int j = start[3*t+state];
        for (int i = j; i < t; ++i) out[i] = state;
        if (j > 0) state = origin[3*j+state];
        t = j;
    }
    free(cumulative); free(entry); free(start); free(origin);
    return 0;
}

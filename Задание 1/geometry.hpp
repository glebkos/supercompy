#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace task {
constexpr double xmin = -4.0, xmax = 4.0, ymin = -6.0, ymax = 6.0;

inline bool inside(double x, double y) {
    return x * x + (std::abs(y) - 2.0) * (std::abs(y) - 2.0) < 9.0;
}

inline long double primitive(long double t) {
    t = std::clamp(t, -3.0L, 3.0L);
    return (t * std::sqrt(std::max(0.0L, 9.0L - t * t))
            + 9.0L * std::asin(t / 3.0L)) / 2.0L;
}

inline long double disk_strip(long double xl, long double xr,
                              long double yl, long double yr, long double cy) {
    yl = std::max(yl, cy - 3.0L);
    yr = std::min(yr, cy + 3.0L);
    if (yl >= yr || xr <= -3.0L || xl >= 3.0L) return 0;
    std::vector<long double> cuts{yl, yr};
    for (long double x : {xl, xr}) {
        if (std::abs(x) < 3.0L) {
            const auto dy = std::sqrt(9.0L - x * x);
            for (auto y : {cy - dy, cy + dy})
                if (yl < y && y < yr) cuts.push_back(y);
        }
    }
    std::sort(cuts.begin(), cuts.end());
    long double area = 0;
    for (std::size_t k = 1; k < cuts.size(); ++k) {
        const auto a = cuts[k - 1], b = cuts[k];
        const auto t = (a + b) / 2 - cy;
        const auto width = std::sqrt(std::max(0.0L, 9.0L - t * t));
        if (std::min(xr, width) <= std::max(xl, -width)) continue;
        const int coefficient = (xr >= width ? 1 : 0) + (xl <= -width ? 1 : 0);
        const auto constant = (xr < width ? xr : 0) - (xl > -width ? xl : 0);
        area += coefficient * (primitive(b - cy) - primitive(a - cy)) + constant * (b - a);
    }
    return area;
}

inline double intersection_area(double xl, double xr, double yl, double yr) {
    if (!(xl < xr && yl < yr)) throw std::invalid_argument("Invalid cell bounds");
    const auto area = disk_strip(xl, xr, yl, std::min(yr, 0.0), -2)
                    + disk_strip(xl, xr, std::max(yl, 0.0), yr, 2);
    return std::clamp(static_cast<double>(area), 0.0, (xr - xl) * (yr - yl));
}

struct Segment { int begin, cells; };

inline Segment segment(int cells, int parts, int coordinate) {
    const int q = cells / parts, r = cells % parts;
    return {coordinate * q + std::min(coordinate, r), q + (coordinate < r)};
}

struct ProcessGrid { int x, y; };

inline ProcessGrid process_grid(int m, int n, int processes) {
    if (m < 4 || n < 4 || processes < 1) throw std::invalid_argument("Invalid grid size");
    ProcessGrid best{0, 0};
    double score = std::numeric_limits<double>::infinity();
    for (int px = 1; px <= processes; ++px) {
        if (processes % px) continue;
        const int py = processes / px;
        if (m / px < 2 || n / py < 2) continue;
        const int min_x = m / px + 1, max_x = min_x + (m % px != 0);
        const int min_y = n / py + 1, max_y = min_y + (n % py != 0);
        if (2 * min_x < max_y || max_x > 2 * min_y) continue;
        const double candidate = std::abs(std::log((double(m) / px + 1) / (double(n) / py + 1)));
        if (candidate < score) { score = candidate; best = {px, py}; }
    }
    if (!best.x) throw std::invalid_argument("No 2D partition satisfies the domain aspect ratio [1/2,2]");
    return best;
}
}

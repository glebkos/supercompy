#include "geometry.hpp"
#include <iostream>
#include <string>

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        require(task::inside(0, 0) && task::inside(0, 4), "Interior points");
        require(!task::inside(3, 0) && !task::inside(0, 5), "Exterior and boundary points");
        require(std::abs(task::intersection_area(-.1, .1, -.1, .1) - .04) < 1e-14, "Full cell");
        require(task::intersection_area(3.1, 3.2, 1, 2) == 0, "Empty cell");
        const double pi = std::acos(-1.0);
        const double exact = 18 * pi - 18 * std::acos(2.0 / 3) + 4 * std::sqrt(5.0);
        require(std::abs(task::intersection_area(-4, 4, -6, 6) - exact) < 1e-12, "Union area");
        double total = 0;
        for (int i = 0; i < 47; ++i) for (int j = 0; j < 63; ++j) {
            const double xl = -4 + 8.0 * i / 47, xr = -4 + 8.0 * (i + 1) / 47;
            const double yl = -6 + 12.0 * j / 63, yr = -6 + 12.0 * (j + 1) / 63;
            const double area = task::intersection_area(xl, xr, yl, yr);
            require(std::abs(area - task::intersection_area(-xr, -xl, -yr, -yl)) < 1e-12, "Area symmetry");
            total += area;
        }
        require(std::abs(total - exact) < 1e-10, "Tiled union area");
        for (auto sizes : {task::ProcessGrid{80,80}, {400,600}, {800,1200}, {23,31}}) {
            for (int p : {1, 2, 4, 8, 16, 32}) {
                auto grid = task::process_grid(sizes.x, sizes.y, p);
                std::vector<int> owners((sizes.x - 1) * (sizes.y - 1), 0);
                int minx = sizes.x, maxx = 0, miny = sizes.y, maxy = 0;
                for (int x = 0; x < grid.x; ++x) for (int y = 0; y < grid.y; ++y) {
                    auto a = task::segment(sizes.x, grid.x, x);
                    auto b = task::segment(sizes.y, grid.y, y);
                    require(2 * (a.cells + 1) >= b.cells + 1 && a.cells + 1 <= 2 * (b.cells + 1), "Aspect ratio");
                    minx = std::min(minx, a.cells); maxx = std::max(maxx, a.cells);
                    miny = std::min(miny, b.cells); maxy = std::max(maxy, b.cells);
                    for (int i = std::max(1, a.begin); i < a.begin + a.cells; ++i)
                        for (int j = std::max(1, b.begin); j < b.begin + b.cells; ++j)
                            ++owners[(i - 1) * (sizes.y - 1) + j - 1];
                }
                require(maxx - minx <= 1 && maxy - miny <= 1, "Balanced domains");
                for (int count : owners) require(count == 1, "Exactly one owner per interior node");
            }
        }
        bool rejected = false;
        try { task::process_grid(4, 100, 1); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Reject impossible partition");
        std::cout << "Geometry and domain partition tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

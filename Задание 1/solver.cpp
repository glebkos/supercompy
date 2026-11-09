#include "geometry.hpp"
#include <chrono>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using Vector = std::vector<double>;
using Clock = std::chrono::steady_clock;

struct Options {
    int m = 80, n = 80, max_iterations = 0;
    double delta = 1e-8;
    bool manufactured = false, help = false;
    std::string output;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help") { o.help = true; continue; }
        if (key == "--manufactured") { o.manufactured = true; continue; }
        if (key != "--nx" && key != "--ny" && key != "--tol"
            && key != "--max-iters" && key != "--output")
            throw std::invalid_argument("Unknown option: " + key);
        if (++i == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        std::size_t used = 0;
        if (key == "--output") { o.output = value; continue; }
        if (key == "--tol") o.delta = std::stod(value, &used);
        else {
            const int n = std::stoi(value, &used);
            if (key == "--nx") o.m = n;
            else if (key == "--ny") o.n = n;
            else {
                if (n < 1) throw std::invalid_argument("--max-iters must be positive");
                o.max_iterations = n;
            }
        }
        if (used != value.size()) throw std::invalid_argument("Invalid number: " + value);
    }
    if (o.m < 4 || o.n < 4 || static_cast<long long>(o.m + 2LL) * (o.n + 2LL) > INT_MAX)
        throw std::invalid_argument("Grid must have at least 4 intervals per axis and fit MPI int counts");
    if (!std::isfinite(o.delta) || o.delta <= 0) throw std::invalid_argument("--tol must be finite and positive");
    const int unknowns = (o.m - 1) * (o.n - 1);
    if (o.max_iterations == 0) o.max_iterations = unknowns;
    if (o.max_iterations > unknowns) throw std::invalid_argument("Iteration limit exceeds number of unknowns");
    return o;
}

struct Grid {
    int rank = 0, processes = 1, px = 1, py = 1;
    int first_x, first_y, nx, ny, stride;
    double hx, hy, epsilon, ax, ay;
    Vector diagonal, rhs;
#ifdef USE_MPI
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Datatype column = MPI_DATATYPE_NULL;
    int west, east, south, north;
#endif

    explicit Grid(const Options& o) {
        int cx = 0, cy = 0;
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &processes);
#endif
        const auto dims = task::process_grid(o.m, o.n, processes);
        px = dims.x; py = dims.y;
#ifdef USE_MPI
        int shape[2]{px, py}, periodic[2]{0, 0}, coords[2];
        MPI_Cart_create(MPI_COMM_WORLD, 2, shape, periodic, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 2, coords);
        cx = coords[0]; cy = coords[1];
        MPI_Cart_shift(comm, 0, 1, &west, &east);
        MPI_Cart_shift(comm, 1, 1, &south, &north);
#endif
        const auto sx = task::segment(o.m, px, cx), sy = task::segment(o.n, py, cy);
        first_x = std::max(1, sx.begin); first_y = std::max(1, sy.begin);
        nx = sx.begin + sx.cells - first_x; ny = sy.begin + sy.cells - first_y;
        stride = ny + 2;
        hx = (task::xmax - task::xmin) / o.m; hy = (task::ymax - task::ymin) / o.n;
        epsilon = std::pow(std::max(hx, hy), 2);
        ax = 1 / (hx * hx); ay = 1 / (hy * hy);
        diagonal.resize((nx + 2) * stride, 0);
        rhs.resize(diagonal.size(), 0);
#ifdef USE_MPI
        MPI_Type_vector(nx, 1, stride, MPI_DOUBLE, &column);
        MPI_Type_commit(&column);
#endif
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) {
            const int gi = first_x + i - 1, gj = first_y + j - 1, k = index(i, j);
            const double x = task::xmin + gi * hx, y = task::ymin + gj * hy;
            const double fraction = std::clamp(task::intersection_area(x - hx / 2, x + hx / 2,
                                                       y - hy / 2, y + hy / 2) / (hx * hy), 0.0, 1.0);
            const double q = (1 - fraction) / (epsilon * epsilon);
            diagonal[k] = 2 * ax + 2 * ay + q;
            const double laplace_factor = 4 * ax * std::pow(std::sin(hx / 2), 2)
                                        + 4 * ay * std::pow(std::sin(hy / 2), 2);
            rhs[k] = (o.manufactured ? 2 * std::sin(x + y) : fraction)
                     - laplace_factor * std::sin(x + y);
        }
    }

    ~Grid() {
#ifdef USE_MPI
        if (column != MPI_DATATYPE_NULL) MPI_Type_free(&column);
        if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
    }
    int index(int i, int j) const { return i * stride + j; }
    void sum(double* values, int count) const {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_DOUBLE, MPI_SUM, comm);
#else
        (void)values; (void)count;
#endif
    }
    double maximum(double value) const {
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, comm);
#endif
        return value;
    }
    void barrier() const {
#ifdef USE_MPI
        MPI_Barrier(comm);
#endif
    }
    void exchange(Vector& v) const {
#ifdef USE_MPI
        MPI_Sendrecv(&v[index(1,1)], ny, MPI_DOUBLE, west, 10,
                     &v[index(nx+1,1)], ny, MPI_DOUBLE, east, 10, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&v[index(nx,1)], ny, MPI_DOUBLE, east, 11,
                     &v[index(0,1)], ny, MPI_DOUBLE, west, 11, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&v[index(1,1)], 1, column, south, 12,
                     &v[index(1,ny+1)], 1, column, north, 12, comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&v[index(1,ny)], 1, column, north, 13,
                     &v[index(1,0)], 1, column, south, 13, comm, MPI_STATUS_IGNORE);
#else
        (void)v;
#endif
    }
    void apply(Vector& v, Vector& result) const {
        exchange(v);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) {
            const int k = index(i, j);
            result[k] = diagonal[k] * v[k] - ax * (v[k-stride] + v[k+stride])
                                                   - ay * (v[k-1] + v[k+1]);
        }
    }
    double dot(const Vector& a, const Vector& b) const {
        double result = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:result) schedule(static)
#endif
        for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j)
            result += a[index(i,j)] * b[index(i,j)];
        sum(&result, 1);
        return hx * hy * result;
    }
};

struct Result {
    int iterations = 0, restarts = 0;
    bool converged = false;
    double step = 0, energy = 0;
};

Result solve(const Grid& g, const Options& o, Vector& w) {
    const auto size = w.size();
    Vector r(size, 0), z(size, 0), p(size, 0), ap(size, 0);
    Result result;
    double rho = 0;
    auto restart = [&]() {
        g.apply(w, ap);
        double rz = 0, energy = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rz,energy) schedule(static)
#endif
        for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j) {
            const int k = g.index(i,j);
            r[k] = g.rhs[k] - ap[k];
            z[k] = r[k] / g.diagonal[k];
            p[k] = z[k];
            rz += r[k] * z[k];
            energy -= 0.5 * (g.rhs[k] + r[k]) * w[k];
        }
        double values[2]{rz, energy};
        g.sum(values, 2);
        rho = values[0] * g.hx * g.hy;
        result.energy = values[1] * g.hx * g.hy;
    };
    restart();
    if (rho == 0) { result.converged = true; return result; }
    for (int iteration = 1; iteration <= o.max_iterations; ++iteration) {
        result.iterations = iteration;
        g.apply(p, ap);
        const double pap = g.dot(p, ap);
        if (!(pap > 0) || !std::isfinite(pap) || !(rho > 0) || !std::isfinite(rho))
            throw std::runtime_error("PCG breakdown: non-positive or non-finite scalar product");
        const double alpha = rho / pap;
        double rz = 0, pp = 0, energy = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rz,pp,energy) schedule(static)
#endif
        for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j) {
            const int k = g.index(i,j);
            w[k] += alpha * p[k];
            r[k] -= alpha * ap[k];
            z[k] = r[k] / g.diagonal[k];
            rz += r[k] * z[k];
            pp += p[k] * p[k];
            energy -= 0.5 * (g.rhs[k] + r[k]) * w[k];
        }
        double values[3]{rz, pp, energy};
        g.sum(values, 3);
        const double next_rho = values[0] * g.hx * g.hy;
        const double next_energy = values[2] * g.hx * g.hy;
        result.step = std::abs(alpha) * std::sqrt(values[1] * g.hx * g.hy);
        if (!std::isfinite(next_rho) || !std::isfinite(next_energy) || !std::isfinite(result.step))
            throw std::runtime_error("Non-finite PCG iterate");
        const double roundoff = 16 * std::numeric_limits<double>::epsilon() * (o.m - 1.0) * (o.n - 1.0)
                              * std::max({1.0, std::abs(result.energy), std::abs(next_energy)});
        if (next_energy > result.energy + roundoff) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
            for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j) {
                const int k = g.index(i,j);
                w[k] -= alpha * p[k];
            }
            if (++result.restarts > 8) throw std::runtime_error("PCG energy increased after repeated restarts");
            restart();
            if (rho == 0) { result.converged = true; break; }
            continue;
        }
        result.energy = next_energy;
        if (result.step < o.delta || next_rho == 0) { result.converged = true; break; }
        const double beta = next_rho / rho;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j) {
            const int k = g.index(i,j);
            p[k] = z[k] + beta * p[k];
        }
        rho = next_rho;
    }
    return result;
}

void save_solution(const Grid& g, const Options& o, const Vector& w) {
    Vector local;
    local.reserve(g.nx * g.ny);
    for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j)
        local.push_back(w[g.index(i,j)]);
    Vector packed;
    std::vector<int> counts(g.processes), offsets(g.processes);
    for (int rank = 0, offset = 0; rank < g.processes; ++rank) {
        const int cx = rank / g.py, cy = rank % g.py;
        const auto sx = task::segment(o.m, g.px, cx), sy = task::segment(o.n, g.py, cy);
        counts[rank] = (sx.cells - (sx.begin == 0)) * (sy.cells - (sy.begin == 0));
        offsets[rank] = offset;
        offset += counts[rank];
    }
    if (g.rank == 0) packed.resize((o.m - 1) * (o.n - 1));
#ifdef USE_MPI
    MPI_Gatherv(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                packed.data(), counts.data(), offsets.data(), MPI_DOUBLE, 0, g.comm);
#else
    packed = std::move(local);
#endif
    if (g.rank != 0) return;
    Vector full((o.m + 1) * (o.n + 1));
    for (int i = 0; i <= o.m; ++i) for (int j = 0; j <= o.n; ++j)
        full[i * (o.n + 1) + j] = std::sin(task::xmin + i * g.hx + task::ymin + j * g.hy);
    for (int rank = 0; rank < g.processes; ++rank) {
        const auto sx = task::segment(o.m, g.px, rank / g.py);
        const auto sy = task::segment(o.n, g.py, rank % g.py);
        int pos = offsets[rank];
        for (int i = std::max(1, sx.begin); i < sx.begin + sx.cells; ++i)
            for (int j = std::max(1, sy.begin); j < sy.begin + sy.cells; ++j)
                full[i * (o.n + 1) + j] += packed[pos++];
    }
    std::ofstream out(o.output);
    if (!out) throw std::runtime_error("Cannot open output file: " + o.output);
    out << "x,y,u,inside\n" << std::setprecision(17);
    for (int i = 0; i <= o.m; ++i) for (int j = 0; j <= o.n; ++j) {
        const double x = task::xmin + i * g.hx, y = task::ymin + j * g.hy;
        out << x << ',' << y << ',' << full[i * (o.n + 1) + j] << ',' << task::inside(x,y) << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("Cannot finish writing output file: " + o.output);
}

int run(const Options& o) {
    const auto setup_start = Clock::now();
    Grid g(o);
    Vector w(g.diagonal.size(), 0);
    const double setup_seconds = g.maximum(std::chrono::duration<double>(Clock::now() - setup_start).count());
    g.barrier();
    const auto start = Clock::now();
    const Result result = solve(g, o, w);
    const double seconds = g.maximum(std::chrono::duration<double>(Clock::now() - start).count());
    Vector residual(w.size(), 0);
    g.apply(w, residual);
    double error = 0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(max:error) schedule(static)
#endif
    for (int i = 1; i <= g.nx; ++i) for (int j = 1; j <= g.ny; ++j) {
        const int k = g.index(i,j);
        residual[k] = g.rhs[k] - residual[k];
        if (o.manufactured) error = std::max(error, std::abs(w[k]));
    }
    error = g.maximum(error);
    const double residual_norm = std::sqrt(g.dot(residual, residual));
    const double rhs_norm = std::sqrt(g.dot(g.rhs, g.rhs));
    int threads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
    }
#endif
    if (!o.output.empty()) save_solution(g, o, w);
    if (g.rank == 0) {
#if defined(USE_MPI) && defined(_OPENMP)
        const char* mode = "hybrid";
#elif defined(USE_MPI)
        const char* mode = "mpi";
#elif defined(_OPENMP)
        const char* mode = "omp";
#else
        const char* mode = "seq";
#endif
        std::cout << std::setprecision(17) << std::boolalpha
                  << "{\"variant\":14,\"mode\":\"" << mode << "\",\"manufactured\":" << o.manufactured
                  << ",\"nx\":" << o.m << ",\"ny\":" << o.n << ",\"processes\":" << g.processes
                  << ",\"threads\":" << threads << ",\"px\":" << g.px << ",\"py\":" << g.py
                  << ",\"epsilon\":" << g.epsilon << ",\"delta\":" << o.delta
                  << ",\"iterations\":" << result.iterations << ",\"restarts\":" << result.restarts
                  << ",\"converged\":" << result.converged << ",\"step_l2\":" << result.step
                  << ",\"residual_l2\":" << residual_norm
                  << ",\"relative_residual\":" << residual_norm / (rhs_norm > 0 ? rhs_norm : 1)
                  << ",\"energy\":" << result.energy << ",\"setup_seconds\":" << setup_seconds
                  << ",\"solve_seconds\":" << seconds;
        if (o.manufactured) std::cout << ",\"exact_max_error\":" << error;
        std::cout << "}\n";
        if (!result.converged) std::cerr << "Iteration limit reached before convergence\n";
    }
    return result.converged ? 0 : 2;
}

int main(int argc, char** argv) {
    int rank = 0;
#ifdef USE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) std::cerr << "MPI_THREAD_FUNNELED is unavailable\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    int status = 0;
    try {
        const Options o = parse(argc, argv);
        if (o.help) {
            if (rank == 0) std::cout
                << "Variant 14: -Delta u=1, u=sin(x+y), x^2+(|y|-2)^2<9\n"
                << "Options: --nx M --ny N --tol delta --max-iters K --output solution.csv\n"
                << "Defaults: M=N=80, delta=1e-8. M,N count intervals, not nodes.\n"
                << "--manufactured: verification problem with exact solution sin(x+y).\n"
                << "Set OMP_NUM_THREADS for OpenMP/hybrid. Exit 2 means no convergence.\n";
        } else status = run(o);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        status = 1;
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return status;
}

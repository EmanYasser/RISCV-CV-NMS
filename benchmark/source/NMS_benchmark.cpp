#include "benchmark.hpp"
#include "reference_cv.hpp"
#include "riscv_cv.hpp"

// Benchmark: scalar ref::NonMaxSuppression vs vec::NonMaxSuppression
// (odd-even sort, and bitonic sort when NMS_ENABLE_BITONIC is defined in NMS.hpp).

// --- tiny xorshift32 PRNG (no <random>, no heap) ---
static uint32_t rng_state = 2463534242u;
static uint32_t Xorshift32()
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static float Rand01() { return static_cast<float>(Xorshift32() & 0xFFFFu) / 65536.0f; }

constexpr int kMaxN = 1024;
static const float kIou = 0.5f;
static const float kScoreThr = 0.05f;

static float g_boxes[kMaxN * 4]; // [y1, x1, y2, x2] per box
static float g_scores[kMaxN];
static int g_n = 0;

static int64_t g_out_ref[kMaxN * 3];
static int64_t g_out_oe[kMaxN * 3];
#ifdef NMS_ENABLE_BITONIC
static int64_t g_out_bt[kMaxN * 3];
#endif

static void GenerateData(int n)
{
    g_n = n;
    for (int i = 0; i < n; ++i)
    {
        const float y = Rand01() * 600.0f;
        const float x = Rand01() * 600.0f;
        g_boxes[i * 4 + 0] = y;
        g_boxes[i * 4 + 1] = x;
        g_boxes[i * 4 + 2] = y + 20.0f + Rand01() * 60.0f;
        g_boxes[i * 4 + 3] = x + 20.0f + Rand01() * 60.0f;
        g_scores[i] = Rand01();
    }
}

// ---- the three implementations behind one signature ----
static int RunRef(int64_t max_out, int64_t* out)
{
    return ref::NonMaxSuppression(g_boxes, g_scores, 1, 1, g_n, max_out, kIou, &kScoreThr,
                                  CenterPointBox::Corners, out);
}

static int RunVecOddEven(int64_t max_out, int64_t* out)
{
    return vec::NonMaxSuppression(g_boxes, g_scores, 1, 1, g_n, max_out, kIou, &kScoreThr,
                                  CenterPointBox::Corners, out);
}

#ifdef NMS_ENABLE_BITONIC
static int RunVecBitonic(int64_t max_out, int64_t* out)
{
    return vec::NonMaxSuppression(g_boxes, g_scores, 1, 1, g_n, max_out, kIou, &kScoreThr,
                                  CenterPointBox::Corners, out, vec::NmsSort::BITONIC);
}
#endif

struct Result
{
    uint64_t cycles = 0;
    uint64_t instrs = 0;
    int rows = 0;
};

// Run `run` loops times and return the average cycles / instructions.
static Result Measure(int (*run)(int64_t, int64_t*), int64_t max_out, int64_t* out, int loops)
{
    Result r;
    for (int i = 0; i < loops; ++i)
    {
        Timer timer;
        timer.Start();
        r.rows = run(max_out, out);
        timer.Stop();
        r.cycles += timer.ElapsedCycles();
        r.instrs += timer.ElapsedInstructions();
    }
    r.cycles /= loops;
    r.instrs /= loops;
    return r;
}

static bool SameOutput(const int64_t* a, int rows_a, const int64_t* b, int rows_b)
{
    if (rows_a != rows_b) return false;
    for (int i = 0; i < rows_a * 3; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

static void PrintLine(const char* label, const Result& r)
{
    const float ipc = r.cycles ? static_cast<float>(r.instrs) / static_cast<float>(r.cycles) : 0.0f;
    printf("  %-22s: %llu cycles, %llu instructions, IPC %f\n", label,
           (unsigned long long)r.cycles, (unsigned long long)r.instrs, ipc);
}

// Local replacement for PrintTime (this repo's PrintTime has a different signature).
static void PrintCompare(const char* name, int n, const Result& vec_r, const Result& scalar_r)
{
    printf("%s: boxes: %d\n", name, n);
    PrintLine("scalar", scalar_r);
    PrintLine("vectorized", vec_r);
    if (vec_r.cycles > 0)
    {
        printf("  speedup               : %f x\n",
               static_cast<float>(scalar_r.cycles) / static_cast<float>(vec_r.cycles));
    }
}

int main()
{
#ifdef RISCV_BAREMETAL
    riscv_enable_vector();
#endif

    constexpr int loop_count = 3;
    const int sizes[] = {64, 256, 1024};
    bool all_correct = true;

    for (int s = 0; s < 3; ++s)
    {
        const int n = sizes[s];
        GenerateData(n);

        // Full run: keep up to n boxes -> filter + sort + gather + ALL IoU work.
        const Result ref_full = Measure(RunRef, n, g_out_ref, loop_count);
        const Result oe_full = Measure(RunVecOddEven, n, g_out_oe, loop_count);
        if (!SameOutput(g_out_ref, ref_full.rows, g_out_oe, oe_full.rows)) all_correct = false;

        // max_output = 1: stops after the first selected box, so this is
        // (almost) only filter + sort + gather, i.e. the sorting cost.
        int64_t one_ref[3], one_oe[3];
        const Result ref_sort = Measure(RunRef, 1, one_ref, loop_count);
        const Result oe_sort = Measure(RunVecOddEven, 1, one_oe, loop_count);
        if (!SameOutput(one_ref, ref_sort.rows, one_oe, oe_sort.rows)) all_correct = false;

        printf("=== NMS benchmark: %d boxes, %d selected ===\n", n, ref_full.rows);

        PrintCompare("NMS full: vec(odd-even) vs scalar", n, oe_full, ref_full);

        printf("filter + sort only (max_output = 1):\n");
        PrintLine("scalar", ref_sort);
        PrintLine("vec odd-even", oe_sort);

#ifdef NMS_ENABLE_BITONIC
        const Result bt_full = Measure(RunVecBitonic, n, g_out_bt, loop_count);
        if (!SameOutput(g_out_ref, ref_full.rows, g_out_bt, bt_full.rows)) all_correct = false;

        int64_t one_bt[3];
        const Result bt_sort = Measure(RunVecBitonic, 1, one_bt, loop_count);
        if (!SameOutput(one_ref, ref_sort.rows, one_bt, bt_sort.rows)) all_correct = false;

        PrintCompare("NMS full: vec(bitonic) vs scalar", n, bt_full, ref_full);
        PrintLine("vec bitonic", bt_sort);

        printf("odd-even vs bitonic, filter + sort only: %f x (odd-even cycles / bitonic cycles)\n",
               static_cast<float>(oe_sort.cycles) / static_cast<float>(bt_sort.cycles));
        printf("odd-even vs bitonic, full pipeline     : %f x\n",
               static_cast<float>(oe_full.cycles) / static_cast<float>(bt_full.cycles));
#endif
    }

    if (all_correct)
    {
        printf("Output is correct.\n");
        return 0;
    }
    printf("Output is wrong!\n");
    return 1;
}
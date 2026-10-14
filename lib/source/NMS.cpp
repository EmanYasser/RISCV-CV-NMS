#include "NMS.hpp"
#include "riscv_wrappers.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

// Vector NonMaxSuppression (RVV) for corner boxes [y1, x1, y2, x2].
// Only rvv::* wrappers are used here (no raw intrinsic calls).
//
// For every (batch, class):
//   (a) score suppression : vector compare + vcompress -> candidate scores + box indices
//   (b) sorting           : odd-even transposition sort (vector compare/merge)
//   (c) IoU suppression   : serial "pick next best" loop, vector IoU against all later boxes
//
// Step 4 (optional): define NMS_ENABLE_BITONIC to also build the bitonic sort.
// That needs four extra wrappers, listed at the bottom of this reply.
namespace
{
constexpr int kMaxBoxes = 1024; // power of two (needed by the bitonic padding)

// ---------------------------------------------------------------------
// (a) Score suppression: keep boxes with score >= threshold (all boxes when no
// threshold is set). Survivors are packed to the front of cand_score[] and
// their original box indices go to cand_idx[]. Returns the survivor count.
// ---------------------------------------------------------------------
int ScoreSuppress(const float* scores, int n, const float* score_threshold,
                  float* cand_score, int* cand_idx)
{
    const float thr = (score_threshold != nullptr)
                          ? *score_threshold
                          : -std::numeric_limits<float>::infinity();

    int out = 0;
    size_t vl = 0;
    for (int i = 0; i < n; i += static_cast<int>(vl))
    {
        vl = rvv::SetVlF32M4(static_cast<size_t>(n - i));

        const rvv::F32M4 s = rvv::LoadF32M4(&scores[i], vl);
        const rvv::MaskF32M4 keep = rvv::MaskF32GreaterEqualScalarM4(s, thr, vl);

        // box indices of this chunk: i, i+1, i+2, ...
        const rvv::U32M4 idx = rvv::AddU32M4(rvv::IotaU32M4(vl), static_cast<uint32_t>(i), vl);

        const size_t cnt = rvv::CountTrueF32M4(keep, vl);
        rvv::StoreF32M4(&cand_score[out], rvv::CompressF32M4(s, keep, vl), cnt);
        rvv::StoreI32M4(&cand_idx[out],
                        rvv::BitcastU32ToI32M4(rvv::CompressU32M4(idx, keep, vl)), cnt);
        out += static_cast<int>(cnt);
    }
    return out;
}

// ---------------------------------------------------------------------
// (b1) Odd-even transposition sort, best score first.
// m phases. Even phase: pairs (0,1),(2,3)...  Odd phase: pairs (1,2),(3,4)...
// All pairs of a phase are independent, so one vector op handles many pairs
// (strided loads: stride = 2 elements = 8 bytes).
// A pair is swapped only when the left score is STRICTLY smaller, so equal
// scores never pass each other: the sort is stable and ties keep the lower
// box index first (candidates start in ascending index order).
// ---------------------------------------------------------------------
void SortOddEven(float* score, int* idx, int m)
{
    for (int phase = 0; phase < m; ++phase)
    {
        const int start = phase & 1;
        if (m < start + 2) continue;

        float* sb = score + start;
        int* ib = idx + start;
        const size_t pairs = static_cast<size_t>((m - start) / 2);

        size_t vl = 0;
        for (size_t p = 0; p < pairs; p += vl)
        {
            vl = rvv::SetVlF32M4(pairs - p);

            const rvv::F32M4 ls = rvv::LoadStridedF32M4(sb + 2 * p, 8, vl);
            const rvv::F32M4 rs = rvv::LoadStridedF32M4(sb + 2 * p + 1, 8, vl);
            const rvv::I32M4 li = rvv::LoadStridedI32M4(ib + 2 * p, 8, vl);
            const rvv::I32M4 ri = rvv::LoadStridedI32M4(ib + 2 * p + 1, 8, vl);

            const rvv::MaskF32M4 swap = rvv::MaskF32LessThanM4(ls, rs, vl); // left < right

            rvv::StoreStridedF32M4(sb + 2 * p, 8, rvv::MergeF32M4(ls, rs, swap, vl), vl);
            rvv::StoreStridedF32M4(sb + 2 * p + 1, 8, rvv::MergeF32M4(rs, ls, swap, vl), vl);
            rvv::StoreStridedI32M4(ib + 2 * p, 8, rvv::MergeI32M4(li, ri, swap, vl), vl);
            rvv::StoreStridedI32M4(ib + 2 * p + 1, 8, rvv::MergeI32M4(ri, li, swap, vl), vl);
        }
    }
}

#ifdef NMS_ENABLE_BITONIC
// ---------------------------------------------------------------------
// (b2) Bitonic sorting network, best score first.
// Pads to a power of two with "worst possible" entries (score = -inf,
// idx = INT_MAX). In every block of 2*j elements the left half is compared
// with the right half (both contiguous). Block direction depends on bit k of
// the block start. A bitonic network is not stable, so the comparison here
// includes the tie-break (equal scores -> lower index first).
// ---------------------------------------------------------------------

// lanes where element a must come before element b
inline rvv::MaskF32M4 Before(rvv::F32M4 sa, rvv::I32M4 ia, rvv::F32M4 sb, rvv::I32M4 ib, size_t vl)
{
    return rvv::OrMaskM4(
        rvv::MaskF32LessThanM4(sb, sa, vl), // sa > sb
        rvv::AndMaskM4(rvv::MaskF32EqualM4(sa, sb, vl), rvv::MaskI32LessThanM4(ia, ib, vl), vl),
        vl);
}

void SortBitonic(float* score, int* idx, int m)
{
    if (m < 2) return;

    size_t P = 1;
    while (P < static_cast<size_t>(m)) P <<= 1;
    assert(P <= static_cast<size_t>(kMaxBoxes));

    for (size_t i = static_cast<size_t>(m); i < P; ++i)
    {
        score[i] = -std::numeric_limits<float>::infinity();
        idx[i] = std::numeric_limits<int>::max();
    }

    for (size_t k = 2; k <= P; k <<= 1)
    {
        for (size_t j = k >> 1; j > 0; j >>= 1)
        {
            for (size_t b = 0; b < P; b += 2 * j)
            {
                const bool best_first = ((b & k) == 0);

                size_t vl = 0;
                for (size_t x = 0; x < j; x += vl)
                {
                    vl = rvv::SetVlF32M4(j - x);

                    float* lsp = score + b + x;
                    float* rsp = score + b + j + x;
                    int* lip = idx + b + x;
                    int* rip = idx + b + j + x;

                    const rvv::F32M4 ls = rvv::LoadF32M4(lsp, vl);
                    const rvv::F32M4 rs = rvv::LoadF32M4(rsp, vl);
                    const rvv::I32M4 li = rvv::LoadI32M4(lip, vl);
                    const rvv::I32M4 ri = rvv::LoadI32M4(rip, vl);

                    // swap when the pair is out of order for this block's direction
                    const rvv::MaskF32M4 swap = best_first ? Before(rs, ri, ls, li, vl)
                                                           : Before(ls, li, rs, ri, vl);

                    rvv::StoreF32M4(lsp, rvv::MergeF32M4(ls, rs, swap, vl), vl);
                    rvv::StoreF32M4(rsp, rvv::MergeF32M4(rs, ls, swap, vl), vl);
                    rvv::StoreI32M4(lip, rvv::MergeI32M4(li, ri, swap, vl), vl);
                    rvv::StoreI32M4(rip, rvv::MergeI32M4(ri, li, swap, vl), vl);
                }
            }
        }
    }
}
#endif // NMS_ENABLE_BITONIC

// ---------------------------------------------------------------------
// Gather the sorted boxes into 4 separate arrays. The boxes are stored as
// [y1,x1,y2,x2] (16 bytes per box), so lane i reads from byte offset
// box_index * 16 (+4 per coordinate, done by moving the base pointer).
// ---------------------------------------------------------------------
void GatherSortedBoxes(const float* batch_boxes, const int* cand_idx, int m,
                       float* y1, float* x1, float* y2, float* x2)
{
    size_t vl = 0;
    for (int i = 0; i < m; i += static_cast<int>(vl))
    {
        vl = rvv::SetVlF32M4(static_cast<size_t>(m - i));

        const rvv::U32M4 offsets = rvv::ShiftLeftU32M4(
            rvv::BitcastI32ToU32M4(rvv::LoadI32M4(&cand_idx[i], vl)), 4, vl); // idx * 16 bytes

        rvv::StoreF32M4(&y1[i], rvv::GatherF32M4(batch_boxes + 0, offsets, vl), vl);
        rvv::StoreF32M4(&x1[i], rvv::GatherF32M4(batch_boxes + 1, offsets, vl), vl);
        rvv::StoreF32M4(&y2[i], rvv::GatherF32M4(batch_boxes + 2, offsets, vl), vl);
        rvv::StoreF32M4(&x2[i], rvv::GatherF32M4(batch_boxes + 3, offsets, vl), vl);
    }
}

// ---------------------------------------------------------------------
// (c) IoU suppression: boxes j in [first, m) get alive[j] = 0 when their IoU
// with the selected box (ay1, ax1, ay2, ax2) is greater than iou_threshold.
// Same arithmetic order as the scalar reference, so results match exactly.
// ---------------------------------------------------------------------
void IouSuppressRange(const float* y1, const float* x1, const float* y2, const float* x2,
                      uint8_t* alive, int first, int m,
                      float ay1, float ax1, float ay2, float ax2, float iou_threshold)
{
    const float aarea = (ay2 - ay1) * (ax2 - ax1);

    size_t vl = 0;
    for (int j = first; j < m; j += static_cast<int>(vl))
    {
        vl = rvv::SetVlF32M2(static_cast<size_t>(m - j));

        const rvv::F32M2 by1 = rvv::LoadF32M2(&y1[j], vl);
        const rvv::F32M2 bx1 = rvv::LoadF32M2(&x1[j], vl);
        const rvv::F32M2 by2 = rvv::LoadF32M2(&y2[j], vl);
        const rvv::F32M2 bx2 = rvv::LoadF32M2(&x2[j], vl);

        // intersection rectangle
        const rvv::F32M2 ymin = rvv::MaxF32ScalarM2(by1, ay1, vl);
        const rvv::F32M2 xmin = rvv::MaxF32ScalarM2(bx1, ax1, vl);
        const rvv::F32M2 ymax = rvv::MinF32ScalarM2(by2, ay2, vl);
        const rvv::F32M2 xmax = rvv::MinF32ScalarM2(bx2, ax2, vl);

        const rvv::F32M2 h = rvv::MaxF32ScalarM2(rvv::SubF32M2(ymax, ymin, vl), 0.0f, vl);
        const rvv::F32M2 w = rvv::MaxF32ScalarM2(rvv::SubF32M2(xmax, xmin, vl), 0.0f, vl);
        const rvv::F32M2 inter = rvv::MulF32M2(h, w, vl);

        // union = areaB + areaA - inter
        const rvv::F32M2 barea = rvv::MulF32M2(rvv::SubF32M2(by2, by1, vl),
                                               rvv::SubF32M2(bx2, bx1, vl), vl);
        const rvv::F32M2 uni = rvv::SubF32M2(rvv::AddF32ScalarM2(barea, aarea, vl), inter, vl);

        const rvv::F32M2 iou = rvv::DivF32M2(inter, uni, vl);

        // suppress where inter > 0 AND iou > threshold
        const rvv::MaskF32M2 suppress = rvv::AndMaskM2(
            rvv::MaskF32GreaterThanScalarM2(inter, 0.0f, vl),
            rvv::MaskF32GreaterThanScalarM2(iou, iou_threshold, vl), vl);

        // write 0 into alive[] only where suppress is set
        rvv::StoreU8Mf2Masked(&alive[j], rvv::SplatU8Mf2(0, vl), suppress, vl);
    }
}

// Index of the first alive box at or after 'from' (vector search), or m if none.
int NextAlive(const uint8_t* alive, int from, int m)
{
    size_t vl = 0;
    for (int i = from; i < m; i += static_cast<int>(vl))
    {
        vl = rvv::SetVlF32M2(static_cast<size_t>(m - i));
        const int first = rvv::FirstTrueM2(
            rvv::MaskU8NotEqualScalarMf2(rvv::LoadU8Mf2(&alive[i], vl), 0, vl), vl);
        if (first >= 0) return i + first;
    }
    return m;
}

int RunNms(const float* boxes, const float* scores, int num_batches, int num_classes,
           int spatial_dim, int64_t max_output_boxes_per_class, float iou_threshold,
           const float* score_threshold, CenterPointBox center_point_box,
           int64_t* selected_indices, bool use_bitonic)
{
    assert(center_point_box == CenterPointBox::Corners);
    assert(spatial_dim <= kMaxBoxes);
    (void)center_point_box;
    (void)use_bitonic;

    if (max_output_boxes_per_class <= 0 || spatial_dim <= 0) return 0;

    static float cand_score[kMaxBoxes];
    static int cand_idx[kMaxBoxes];
    static float sy1[kMaxBoxes], sx1[kMaxBoxes], sy2[kMaxBoxes], sx2[kMaxBoxes];
    static uint8_t alive[kMaxBoxes];

    int rows = 0;

    for (int b = 0; b < num_batches; ++b)
    {
        const float* batch_boxes = boxes + static_cast<size_t>(b) * spatial_dim * 4;

        for (int c = 0; c < num_classes; ++c)
        {
            const float* cls_scores =
                scores + (static_cast<size_t>(b) * num_classes + c) * spatial_dim;

            // (a) score suppression
            const int m = ScoreSuppress(cls_scores, spatial_dim, score_threshold,
                                        cand_score, cand_idx);
            if (m == 0) continue;

            // (b) sort, best score first
#ifdef NMS_ENABLE_BITONIC
            if (use_bitonic) SortBitonic(cand_score, cand_idx, m);
            else
#endif
                SortOddEven(cand_score, cand_idx, m);

            // sorted boxes -> 4 arrays, all boxes alive
            GatherSortedBoxes(batch_boxes, cand_idx, m, sy1, sx1, sy2, sx2);
            rvv::FillU8(alive, 1, static_cast<size_t>(m));

            // (c) greedy selection with vectorized IoU suppression
            int64_t count = 0;
            int i = NextAlive(alive, 0, m);
            while (i < m && count < max_output_boxes_per_class)
            {
                selected_indices[rows * 3 + 0] = b;
                selected_indices[rows * 3 + 1] = c;
                selected_indices[rows * 3 + 2] = cand_idx[i];
                ++rows;
                ++count;
                if (count >= max_output_boxes_per_class) break;

                IouSuppressRange(sy1, sx1, sy2, sx2, alive, i + 1, m,
                                 sy1[i], sx1[i], sy2[i], sx2[i], iou_threshold);
                i = NextAlive(alive, i + 1, m);
            }
        }
    }
    return rows;
}
} // namespace

// Contract entry point (odd-even sort).
int vec::NonMaxSuppression(const float* boxes,
                           const float* scores,
                           int num_batches,
                           int num_classes,
                           int spatial_dim,
                           int64_t max_output_boxes_per_class,
                           float iou_threshold,
                           const float* score_threshold,
                           CenterPointBox center_point_box,
                           int64_t* selected_indices)
{
    return RunNms(boxes, scores, num_batches, num_classes, spatial_dim,
                  max_output_boxes_per_class, iou_threshold, score_threshold,
                  center_point_box, selected_indices, false);
}

#ifdef NMS_ENABLE_BITONIC
// Step 4: same pipeline, selectable sort (needs NmsSort declared in NMS.hpp).
int vec::NonMaxSuppression(const float* boxes,
                           const float* scores,
                           int num_batches,
                           int num_classes,
                           int spatial_dim,
                           int64_t max_output_boxes_per_class,
                           float iou_threshold,
                           const float* score_threshold,
                           CenterPointBox center_point_box,
                           int64_t* selected_indices,
                           NmsSort sort_method)
{
    return RunNms(boxes, scores, num_batches, num_classes, spatial_dim,
                  max_output_boxes_per_class, iou_threshold, score_threshold,
                  center_point_box, selected_indices, sort_method == NmsSort::BITONIC);
}
#endif
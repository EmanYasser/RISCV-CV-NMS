#include "NMS.hpp"

#include <cassert>
#include <cstddef>

// Scalar NonMaxSuppression (ONNX semantics, corner boxes only).
namespace
{
// Static scratch space: no heap needed (bare-metal friendly).
constexpr int kMaxBoxes = 1024;

inline float Max(float a, float b) { return a > b ? a : b; }
inline float Min(float a, float b) { return a < b ? a : b; }

// a and b point at [y1, x1, y2, x2]. Returns true when IoU(a, b) > thresh.
inline bool SuppressByIou(const float* a, const float* b, float iou_threshold)
{
    const float ymin = Max(a[0], b[0]);
    const float xmin = Max(a[1], b[1]);
    const float ymax = Min(a[2], b[2]);
    const float xmax = Min(a[3], b[3]);

    const float h = Max(ymax - ymin, 0.0f);
    const float w = Max(xmax - xmin, 0.0f);
    const float inter = h * w;
    if (!(inter > 0.0f)) return false; 

    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float uni = (area_a + area_b) - inter;

    return (inter / uni) > iou_threshold;
}

// true when box a must come BEFORE box b in the sorted order:
// higher score first, equal scores -> lower index first.
inline bool Before(const float* s, int a, int b)
{
    return s[a] > s[b] || (s[a] == s[b] && a < b);
}
} // namespace

int ref::NonMaxSuppression(const float* boxes,
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
    assert(center_point_box == CenterPointBox::Corners);
    assert(spatial_dim <= kMaxBoxes);
    (void)center_point_box;

    if (max_output_boxes_per_class <= 0 || spatial_dim <= 0) return 0;

    static int order[kMaxBoxes];    // candidate box indices (sorted best first)
    static int selected[kMaxBoxes]; // boxes already selected for the current class

    int rows = 0;

    for (int b = 0; b < num_batches; ++b)
    {
        const float* batch_boxes = boxes + static_cast<size_t>(b) * spatial_dim * 4;

        for (int c = 0; c < num_classes; ++c)
        {
            const float* cls_scores =
                scores + (static_cast<size_t>(b) * num_classes + c) * spatial_dim;

            // 1) score suppression: drop boxes with score < score_threshold
            int m = 0;
            for (int i = 0; i < spatial_dim; ++i)
            {
                if (score_threshold == nullptr || cls_scores[i] >= *score_threshold)
                    order[m++] = i;
            }
            // 2) odd-even transposition sort (m phases)
            //    even phase: pairs (0,1),(2,3)...   odd phase: pairs (1,2),(3,4)...
            for (int phase = 0; phase < m; ++phase)
            {
                for (int i = phase & 1; i + 1 < m; i += 2)
                {
                    if (Before(cls_scores, order[i + 1], order[i]))
                    {
                        const int tmp = order[i];
                        order[i] = order[i + 1];
                        order[i + 1] = tmp;
                    }
                }
            }

            // 3) greedy selection: keep a box unless it overlaps an already
            //    selected box by more than iou_threshold
            int count = 0;
            for (int k = 0; k < m && count < max_output_boxes_per_class; ++k)
            {
                const int cand = order[k];
                const float* cand_box = batch_boxes + static_cast<size_t>(cand) * 4;

                bool keep = true;
                for (int s = 0; s < count; ++s)
                {
                    const float* sel_box = batch_boxes + static_cast<size_t>(selected[s]) * 4;
                    if (SuppressByIou(cand_box, sel_box, iou_threshold))
                    {
                        keep = false;
                        break;
                    }
                }

                if (keep)
                {
                    selected[count++] = cand;
                    selected_indices[rows * 3 + 0] = b;
                    selected_indices[rows * 3 + 1] = c;
                    selected_indices[rows * 3 + 2] = cand;
                    ++rows;
                }
            }
        }
    }
    return rows;
}
#include "src/csir.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <array>

using namespace csir;
using namespace csir_test_utils;

TEST(CSIRSubsetLikeTest, DifferenceWithTranslate1D)
{
    auto s            = make_1d(1,
                                {
                         {10, 16}
    });
    auto d            = difference(s, translate(s, +1));
    CSIR_Level_1D exp = make_1d(1,
                                {
                                    {10, 11}
    });
    ASSERT_TRUE(equal_1d(d, exp));
}

TEST(CSIRSubsetLikeTest, RectangleIntersectTranslate2D)
{
    auto A = make_rectangle(3, 0, 10, 0, 10);
    auto B = translate(A, +5, 0);
    auto I = intersection(A, B);

    // Expect rows y=0..9, interval [5,10)
    ASSERT_EQ(I.level, 3);
    ASSERT_EQ(I.y_coords.size(), 10);
    for (std::size_t i = 0; i < I.y_coords.size(); ++i)
    {
        auto y = I.y_coords[i];
        (void)y;
        auto s = I.intervals_ptr[i];
        auto e = I.intervals_ptr[i + 1];
        ASSERT_EQ(e - s, 1);
        ASSERT_EQ(I.intervals[s].start, 5);
        ASSERT_EQ(I.intervals[s].end, 10);
    }
}

TEST(CSIRSubsetLikeTest, CubeIntersectTranslate3D)
{
    auto A = make_cube(2, 0, 6, 0, 6, 0, 6);
    auto B = translate(A, 2, 1, 3);
    auto I = intersection_3d(A, B);

    // Expect z slices 3..5 present, each row y=1..5 interval [2,6)
    for (int z = 3; z < 6; ++z)
    {
        auto it = I.slices.find(z);
        ASSERT_NE(it, I.slices.end());
        const auto& S = it->second;
        // we expect 5 rows: 1..5
        ASSERT_EQ(S.y_coords.size(), 5);
        for (std::size_t i = 0; i < S.y_coords.size(); ++i)
        {
            auto s = S.intervals_ptr[i];
            auto e = S.intervals_ptr[i + 1];
            ASSERT_EQ(e - s, 1);
            ASSERT_EQ(S.intervals[s].start, 2);
            ASSERT_EQ(S.intervals[s].end, 6);
        }
    }
}

TEST(CSIRSubsetLikeTest, EmptySetsBehaviour1D)
{
    CSIR_Level_1D empty = make_1d(3, {});
    CSIR_Level_1D reg   = make_1d(3,
                                  {
                                    {5, 10}
    });
    auto u              = union_(empty, reg);
    ASSERT_TRUE(equal_1d(u, reg));
    auto i = intersection(empty, reg);
    ASSERT_TRUE(i.intervals.empty());
    auto d = difference(reg, empty);
    ASSERT_TRUE(equal_1d(d, reg));
}

TEST(CSIRSubsetLikeTest, SinglePointIntervalsTranslate1D)
{
    CSIR_Level_1D l = make_1d(3,
                              {
                                  {0, 1},
                                  {2, 3},
                                  {4, 5}
    });
    auto diff       = difference(l, translate(l, +1));
    // Should equal original
    ASSERT_EQ(diff.intervals.size(), 3);
    ASSERT_EQ(diff.intervals[0].start, 0);
    ASSERT_EQ(diff.intervals[0].end, 1);
    ASSERT_EQ(diff.intervals[1].start, 2);
    ASSERT_EQ(diff.intervals[1].end, 3);
    ASSERT_EQ(diff.intervals[2].start, 4);
    ASSERT_EQ(diff.intervals[2].end, 5);
}

TEST(CSIRSubsetLikeTest, ExtremeLevelDifferencesProjectUp1D)
{
    CSIR_Level_1D a = make_1d(0,
                              {
                                  {0, 1}
    });
    CSIR_Level_1D b = make_1d(10,
                              {
                                  {0, 1024}
    });
    auto a_up       = project_to_level(a, 10);
    auto i          = intersection(a_up, b);
    ASSERT_EQ(i.intervals.size(), 1);
    ASSERT_EQ(i.intervals[0].start, 0);
    ASSERT_EQ(i.intervals[0].end, 1024);
}

TEST(CSIRSubsetLikeTest, NegativeCoordinatesTranslate1D)
{
    CSIR_Level_1D l = make_1d(3,
                              {
                                  {-10, -5},
                                  {-2,  3 },
                                  {8,   12}
    });
    auto t          = translate(l, -15);
    // Expect [-25,-20], [-17,-12], [-7,-3]
    ASSERT_EQ(t.intervals.size(), 3);
    ASSERT_EQ(t.intervals[0].start, -25);
    ASSERT_EQ(t.intervals[0].end, -20);
    ASSERT_EQ(t.intervals[1].start, -17);
    ASSERT_EQ(t.intervals[1].end, -12);
    ASSERT_EQ(t.intervals[2].start, -7);
    ASSERT_EQ(t.intervals[2].end, -3);
}

TEST(CSIRSubsetLikeTest, SparseDistributionIntersectionEmpty2D)
{
    CSIR_Level l;
    l.level = 3;
    l.intervals_ptr.push_back(0);
    l.y_coords  = {0, 100, -50};
    l.intervals = {
        {0,   1  },
        {0,   1  },
        {200, 201}
    };
    l.intervals_ptr = {0, 1, 2, 3};
    auto i          = intersection(l, translate(l, 1, 1));
    ASSERT_TRUE(i.intervals.empty());
}

TEST(CSIRSubsetLikeTest, CheckerboardIntersectionEmpty2D)
{
    CSIR_Level l;
    l.level = 3;
    l.intervals_ptr.push_back(0);
    for (int j = 0; j < 8; j += 2)
    {
        std::vector<Interval> row;
        for (int x = j % 4; x < 8; x += 4)
        {
            row.push_back({x, x + 1});
        }
        if (!row.empty())
        {
            l.y_coords.push_back(j);
            l.intervals.insert(l.intervals.end(), row.begin(), row.end());
            l.intervals_ptr.push_back(l.intervals.size());
        }
    }
    auto i = intersection(l, translate(l, 1, 1));
    ASSERT_TRUE(i.intervals.empty());
}

TEST(CSIRSubsetLikeTest, BoundaryConditionsNonEmpty2D)
{
    CSIR_Level l;
    l.level         = 5;
    l.intervals_ptr = {0};
    // Domain with holes on y=16..18
    l.y_coords  = {16, 17, 18};
    l.intervals = {
        {0,  32},
        {0,  8 },
        {24, 32},
        {0,  32}
    };
    l.intervals_ptr = {0, 1, 3, 4};
    int dirs[4][2]  = {
        {1,  0 },
        {-1, 0 },
        {0,  1 },
        {0,  -1}
    };
    for (auto& d : dirs)
    {
        auto b = difference(l, translate(l, d[0], d[1]));
        ASSERT_FALSE(b.intervals.empty());
    }
}

TEST(CSIRSubsetLikeTest, ExtremeAspectRatiosIntersectionNonEmpty2D)
{
    CSIR_Level l;
    l.level         = 3;
    l.intervals_ptr = {0};
    // y=0 [0,1000); y=1 [0,1000)
    l.y_coords  = {0, 1};
    l.intervals = {
        {0, 1000},
        {0, 1000}
    };
    l.intervals_ptr = {0, 1, 2};
    auto i          = intersection(l, l);
    ASSERT_FALSE(i.intervals.empty());
}

TEST(CSIRSubsetLikeTest, LayeredStructureDisjointUnionNonEmpty3D)
{
    CSIR_Level_3D a;
    a.level = 3;
    for (int k = 0; k < 8; k += 2)
    {
        CSIR_Level s;
        s.level = 3;
        s.y_coords.resize(8);
        s.intervals_ptr.resize(9);
        s.intervals_ptr[0] = 0;
        for (int j = 0; j < 8; ++j)
        {
            s.y_coords[j] = j;
            s.intervals.push_back({0, 8});
            s.intervals_ptr[j + 1] = s.intervals.size();
        }
        a.slices[k] = s;
    }
    CSIR_Level_3D b;
    b.level = 3;
    for (int k = 1; k < 8; k += 2)
    {
        CSIR_Level s;
        s.level = 3;
        s.y_coords.resize(8);
        s.intervals_ptr.resize(9);
        s.intervals_ptr[0] = 0;
        for (int j = 0; j < 8; ++j)
        {
            s.y_coords[j] = j;
            s.intervals.push_back({0, 8});
            s.intervals_ptr[j + 1] = s.intervals.size();
        }
        b.slices[k] = s;
    }
    auto inter = intersection_3d(a, b);
    ASSERT_TRUE(inter.slices.empty());
    auto uni = union_3d(a, b);
    ASSERT_FALSE(uni.slices.empty());
}

TEST(CSIRSubsetLikeTest, DifferenceVsTranslate1D)
{
    CSIR_Level_1D l = make_1d(1,
                              {
                                  {0, 16}
    });
    auto d          = difference(l, translate(l, -1));
    ASSERT_EQ(d.intervals.size(), 1);
    ASSERT_EQ(d.intervals[0].start, 15);
    ASSERT_EQ(d.intervals[0].end, 16);
}

TEST(CSIRSubsetLikeTest, EmptyChecks1D)
{
    CSIR_Level_1D l1 = make_1d(1,
                               {
                                   {0, 16}
    });
    CSIR_Level_1D l2 = make_1d(1, {});
    ASSERT_FALSE(l1.empty());
    ASSERT_TRUE(l2.empty());
    ASSERT_TRUE(intersection(l1, l2).empty());
    ASSERT_FALSE(intersection(l1, l1).empty());
    ASSERT_FALSE(difference(l1, l2).empty());
    CSIR_Level_1D t = translate(l1, 16);
    ASSERT_TRUE(intersection(l1, t).empty());
}

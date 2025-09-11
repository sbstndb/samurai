#include <cassert>
#include <iostream>
#include "csir.hpp"

using namespace csir;

// Helpers
static CSIR_Level_1D make_1d(std::size_t level, std::initializer_list<Interval> iv)
{
    CSIR_Level_1D s; s.level = level; s.intervals = iv; return s;
}

static CSIR_Level make_rect(std::size_t level, int x0, int x1, int y0, int y1)
{
    CSIR_Level c; c.level = level; c.intervals_ptr.push_back(0);
    for (int y = y0; y < y1; ++y)
    {
        c.y_coords.push_back(y);
        c.intervals.push_back({x0, x1});
        c.intervals_ptr.push_back(c.intervals.size());
    }
    return c;
}

static CSIR_Level_3D make_cube(std::size_t level, int x0, int x1, int y0, int y1, int z0, int z1)
{
    CSIR_Level_3D c; c.level = level;
    auto s = make_rect(level, x0, x1, y0, y1);
    for (int z = z0; z < z1; ++z) c.slices[z] = s;
    return c;
}

int main()
{
    // 1D: difference with translate
    {
        auto s = make_1d(1, {{10,16}});
        auto d = difference(s, translate(s, +1));
        CSIR_Level_1D exp; exp.level=1; exp.intervals={{10,11}};
        assert(d.level==exp.level && d.intervals.size()==1 && d.intervals[0].start==10 && d.intervals[0].end==11);
    }

    // 2D: rectangle intersect translate
    {
        auto A = make_rect(3, 0, 10, 0, 10);
        auto B = translate(A, +5, 0);
        auto I = intersection(A, B);

        // Expect rows y=0..9, interval [5,10)
        assert(I.level==3);
        assert(I.y_coords.size()==10);
        for (std::size_t i=0;i<I.y_coords.size();++i)
        {
            auto y = I.y_coords[i]; (void)y;
            auto s = I.intervals_ptr[i]; auto e = I.intervals_ptr[i+1];
            assert(e-s==1);
            assert(I.intervals[s].start==5 && I.intervals[s].end==10);
        }
    }

    // 3D: cube intersect translate
    {
        auto A = make_cube(2, 0, 6, 0, 6, 0, 6);
        auto B = translate(A, 2, 1, 3);
        auto I = intersection_3d(A, B);

        // Expect z slices 3..5 present, each row y=1..5 interval [2,6)
        for (int z=3; z<6; ++z)
        {
            auto it = I.slices.find(z);
            assert(it!=I.slices.end());
            const auto& S = it->second;
            // we expect 5 rows: 1..5
            assert(S.y_coords.size()==5);
            for (std::size_t i=0;i<S.y_coords.size();++i)
            {
                auto s = S.intervals_ptr[i]; auto e = S.intervals_ptr[i+1];
                assert(e-s==1);
                assert(S.intervals[s].start==2 && S.intervals[s].end==6);
            }
        }
    }

    // 1D: empty sets behaviour
    {
        CSIR_Level_1D empty; empty.level=3;
        CSIR_Level_1D reg; reg.level=3; reg.intervals={{5,10}};
        auto u = union_(empty, reg);
        assert(u.level==3 && u.intervals.size()==1 && u.intervals[0].start==5 && u.intervals[0].end==10);
        auto i = intersection(empty, reg);
        assert(i.intervals.empty());
        auto d = difference(reg, empty);
        assert(d.intervals.size()==1 && d.intervals[0].start==5 && d.intervals[0].end==10);
    }

    // 1D: single point intervals & translate
    {
        CSIR_Level_1D l; l.level=3; l.intervals={{0,1},{2,3},{4,5}};
        auto diff = difference(l, translate(l, +1));
        // Should equal original
        assert(diff.intervals.size()==3);
        assert(diff.intervals[0].start==0 && diff.intervals[0].end==1);
        assert(diff.intervals[1].start==2 && diff.intervals[1].end==3);
        assert(diff.intervals[2].start==4 && diff.intervals[2].end==5);
    }

    // 1D: extreme level differences (project up)
    {
        CSIR_Level_1D a; a.level=0; a.intervals={{0,1}};
        CSIR_Level_1D b; b.level=10; b.intervals={{0,1024}};
        auto a_up = project_to_level(a, 10);
        auto i = intersection(a_up, b);
        assert(i.intervals.size()==1 && i.intervals[0].start==0 && i.intervals[0].end==1024);
    }

    // 1D: negative coordinates translate
    {
        CSIR_Level_1D l; l.level=3; l.intervals={{-10,-5},{-2,3},{8,12}};
        auto t = translate(l, -15);
        // Expect [-25,-20], [-17,-12], [-7,-3]
        assert(t.intervals.size()==3);
        assert(t.intervals[0].start==-25 && t.intervals[0].end==-20);
        assert(t.intervals[1].start==-17 && t.intervals[1].end==-12);
        assert(t.intervals[2].start==-7 && t.intervals[2].end==-3);
    }

    // 2D: sparse distribution, intersection with translate is empty
    {
        CSIR_Level l; l.level=3; l.intervals_ptr.push_back(0);
        l.y_coords = {0, 100, -50};
        l.intervals = {{0,1}, {0,1}, {200,201}};
        l.intervals_ptr = {0,1,2,3};
        auto i = intersection(l, translate(l, 1, 1));
        assert(i.intervals.empty());
    }

    // 2D: checkerboard, intersection with translate is empty
    {
        CSIR_Level l; l.level=3; l.intervals_ptr.push_back(0);
        for (int j=0;j<8;j+=2)
        {
            std::vector<Interval> row;
            for (int x = j%4; x<8; x+=4) row.push_back({x,x+1});
            if (!row.empty())
            {
                l.y_coords.push_back(j);
                l.intervals.insert(l.intervals.end(), row.begin(), row.end());
                l.intervals_ptr.push_back(l.intervals.size());
            }
        }
        auto i = intersection(l, translate(l, 1, 1));
        assert(i.intervals.empty());
    }

    // 2D: boundary conditions (non-empty for four directions)
    {
        CSIR_Level l; l.level=5; l.intervals_ptr={0};
        // Domain with holes on y=16..18
        l.y_coords = {16,17,18};
        l.intervals = {{0,32}, {0,8},{24,32}, {0,32}};
        l.intervals_ptr = {0,1,3,4};
        int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
        for (auto& d:dirs)
        {
            auto b = difference(l, translate(l, d[0], d[1]));
            assert(!b.intervals.empty());
        }
    }

    // 2D: extreme aspect ratios, intersection(self,self) non-empty
    {
        CSIR_Level l; l.level=3; l.intervals_ptr={0};
        // y=0 [0,1000); y=1 [0,1000)
        l.y_coords = {0,1};
        l.intervals = {{0,1000},{0,1000}};
        l.intervals_ptr = {0,1,2};
        auto i = intersection(l,l);
        assert(!i.intervals.empty());
    }

    // 3D: layered structure disjoint/union non-empty
    {
        CSIR_Level_3D a; a.level=3;
        for (int k=0;k<8;k+=2)
        {
            CSIR_Level s; s.level=3; s.y_coords.resize(8); s.intervals_ptr.resize(9);
            s.intervals_ptr[0]=0;
            for (int j=0;j<8;++j){ s.y_coords[j]=j; s.intervals.push_back({0,8}); s.intervals_ptr[j+1]=s.intervals.size(); }
            a.slices[k] = s;
        }
        CSIR_Level_3D b; b.level=3;
        for (int k=1;k<8;k+=2)
        {
            CSIR_Level s; s.level=3; s.y_coords.resize(8); s.intervals_ptr.resize(9);
            s.intervals_ptr[0]=0;
            for (int j=0;j<8;++j){ s.y_coords[j]=j; s.intervals.push_back({0,8}); s.intervals_ptr[j+1]=s.intervals.size(); }
            b.slices[k] = s;
        }
        auto inter = intersection_3d(a,b);
        assert(inter.slices.empty());
        auto uni = union_3d(a,b);
        assert(!uni.slices.empty());
    }

    // 1D: diff vs translate
    {
        CSIR_Level_1D l; l.level=1; l.intervals={{0,16}};
        auto d = difference(l, translate(l, -1));
        assert(d.intervals.size()==1 && d.intervals[0].start==15 && d.intervals[0].end==16);
    }

    // 1D: empty checks
    {
        CSIR_Level_1D l1; l1.level=1; CSIR_Level_1D l2; l2.level=1; l1.intervals={{0,16}};
        assert(!l1.empty());
        assert(l2.empty());
        assert(intersection(l1,l2).empty());
        assert(!intersection(l1,l1).empty());
        assert(!difference(l1,l2).empty());
        CSIR_Level_1D t = translate(l1, 16);
        assert(intersection(l1,t).empty());
    }

    std::cout << "Subset-like tests passed." << std::endl;
    return 0;
}

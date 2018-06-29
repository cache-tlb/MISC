//////////////////////////////////////////
#include <map>
#include <cstdio>
#include <random>
#include <queue>
#include <algorithm>
#include <cmath>

const double eps = 1e-12;

struct Point {
    double x, y;
    Point() : x(0), y(0) {}
    Point(double _x, double _y): x(_x), y(_y) {}
    Point operator + (const Point &that) const {
        return Point(x + that.x, y + that.y);
    }
    Point operator - (const Point &that) const {
        return Point(x - that.x, y - that.y);
    }
    Point operator * (double s) const {
        return Point(s*x, s*y);
    }
};

struct PointCmp {
    bool operator () (const std::pair<int,int> &p1, const std::pair<int,int> &p2) {
        return (p1.first < p2.first) || (p1.first == p2.first && p1.second < p2.second);
    }
};

struct Segment {
    Point p1, p2;
    int id;
    Segment() {}
    Segment(const Point &p1_, const Point &p2_) : p1(p1_), p2(p2_) {}
    double get_y(double sweepline_x) const {
        double x1 = p1.x, x2 = p2.x, y1 = p1.y, y2 = p2.y;
        if (x1 == x2) {
            if (sweepline_x == p1.x) return std::max(y1, y2);
            else {
                printf("weird! vertical segment\n");
                return -1;
            }
        } else {
            double dx = x2 - x1, dy = y2 - y1;
            double y = (sweepline_x - x1)/dx*dy + y1;
            return y;
        }
    }
};

struct Event {
    int type;       // 0: left, 1: right, 2: intersect, 3: vertical segment event
    int id1, id2;
    double x, y;
    Event() {}
    Event(double x_, double y_, int type_, int id1_, int id2_) 
        : x(x_), y(y_), type(type_), id1(id1_), id2(id2_) {}
};

struct EventCmp {
    bool operator () (const Event &a, const Event &b) {
        return (a.x > b.x) || (a.x == b.x && a.y > b.y) || (a.x == b.x && a.y == b.y && a.type > b.type)
            || (a.x == b.x && a.y == b.y && a.type == b.type && a.id1 > b.id1)
            || (a.x == b.x && a.y == b.y && a.type == b.type && a.id1 == b.id1 && a.id2 > b.id2);
    }
};

// class EventQueue {
// public:
//     void push(const Event &e) {
//         if (table.count(std::make_pair<>(e.id1, e.id2)) > 0 || table.count(std::make_pair<>(e.id2, e.id1)) > 0) return;
//         eq.push(e);
//         table[std::make_pair<>(e.id1, e.id2)] = 1;
//     }
//     void pop() {
//         eq.pop();
//     }
//     Event top() const {
//         return eq.top();
//     }
//     bool empty() const {
//         return eq.empty();
//     }
// private:
//     std::priority_queue<Event, std::vector<Event>, EventCmp> eq;
//     std::map<std::pair<int,int>,int> table;
// };
 
// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(const Point &p, const Point &q, const Point &r) {
    if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
        q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
       return true;
 
    return false;
}
 
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(const Point &p, const Point &q, const Point &r) {
    // See http://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    double val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;  // colinear
 
    return (val > 0)? 1: 2; // clock or counterclock wise
}
 
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(const Point &p1, const Point &q1, const Point &p2, const Point &q2) {
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
 
     // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
}

bool is_intersect(const Segment &s1, const Segment &s2) {
    return doIntersect(s1.p1, s1.p2, s2.p1, s2.p2);
}

Point find_intersect(const Segment &s1, const Segment &s2) {
    double dx1 = s1.p2.x - s1.p1.x;
    double dy1 = s1.p2.y - s1.p1.y;
    double dx2 = s2.p2.x - s2.p1.x;
    double dy2 = s2.p2.y - s2.p1.y;
    double tx = s2.p1.x - s1.p1.x;
    double ty = s2.p1.y - s1.p1.y;
    Point ret;
    if (dx1*dy2==dx2*dy1) {
        // colinear
        bool is_s1p1 = false, is_s1p2 = false, is_s2p1 = false, is_s2p2 = false;
        if (onSegment(s1.p1,s2.p1,s1.p2)) is_s2p1 = true;
        if (onSegment(s1.p1,s2.p2,s1.p2)) is_s2p2 = true;
        if (onSegment(s2.p1,s1.p1,s2.p2)) is_s1p1 = true;
        if (onSegment(s2.p1,s1.p2,s2.p2)) is_s1p2 = true;
        int count = 0;
        ret = Point(0,0);
        if (is_s1p1) {
            ret = ret + s1.p1; count++;
        }
        if (is_s1p2) {
            ret = ret + s1.p2; count++;
        }
        if (is_s2p1) {
            ret = ret + s2.p1; count++;
        }
        if (is_s2p2) {
            ret = ret + s2.p2; count++;
        }
        ret = ret * (1.0/count);

    } else {
        double f = dx1*dy2 - dx2*dy1;
        double s = (tx*dy1 - ty*dx1) / f;
        ret = s2.p1 + Point(dx2, dy2)*s;
    }
    return ret;
}

double random_float() {
    return double(rand()) / RAND_MAX;
    // return rand() % 100;
}

void make_random_segs(int n, std::vector<Segment> &segs) {
    segs.resize(n);
    for (int i = 0; i < n; i++) {
        double x1 = random_float(), y1 = random_float(), x2 = random_float()+1, y2 = random_float();
        segs[i].p1 = Point(x1+i*0.01, y1);
        segs[i].p2 = Point(2*x2+i*0.01, y2);
        segs[i].id = i;
    }
}

bool fuzzy_eq(const double a, const double b) {
    if (fabs(a-b) < eps) return true;
    return false;
}

struct cmp {
    static double scanline_x;
    static double eps;
    bool operator () (const Segment &a, const Segment &b) {
        double ya = a.get_y(scanline_x), yb = b.get_y(scanline_x);
        if (fuzzy_eq(ya,yb)) {
            double dy1 = a.p2.y - a.p1.y, dx1 = a.p2.x - a.p1.x, dy2 = b.p2.y - b.p1.y, dx2 = b.p2.x - b.p1.x;
            double k1 = dy1/dx1, k2 = dy2/dx2;
            if (fuzzy_eq(k1,k2)) return a.id < b.id;
            else return k1 > k2;
        } else if (ya > yb) return true;
        return false;
    }
};

double cmp::scanline_x = -1;
double cmp::eps = 1e-6;

void print_table(std::map<Segment,int,cmp> &event_queue) {
    printf("scanline list: ");
    for (std::map<Segment, int, cmp>::iterator it  = event_queue.begin(); it != event_queue.end(); it++) {
        const Segment &l = it->first;
        printf("%d ", l.id);
    }
    printf("\n");
}

inline bool has_record(std::map<std::pair<int,int>,int> &all_intersection, int id1, int id2) {
    if (all_intersection.count(std::make_pair<>(id1,id2)) > 0 || all_intersection.count(std::make_pair<>(id2,id1)) > 0) return true;
    return false;
}

inline void add_record(std::map<std::pair<int,int>,int> &all_intersection, int id1, int id2) {
    all_intersection[std::make_pair<>(id1,id2)] = 1;
}

int main(int argc, char const *argv[])
{
    srand(1);
    bool debug = false;
    std::vector<Segment> segs;
    int n_segs = 40000;
    make_random_segs(n_segs, segs);

    for (int i = 0; i < segs.size(); i++) {
        segs[i].id = i;
    }

    // brute force
    int cnt = 0;
    std::map<std::pair<int,int>,int> bf_table;
    for (int i = 0; i < segs.size(); i++) {
        for (int j = i+1; j < segs.size(); j++) {
            if (is_intersect(segs[i],segs[j])) {
                cnt++;
                bf_table[std::make_pair(segs[i].id,segs[j].id)] = 1;
            }
        }
    }
    printf("cnt1: %d\n", cnt);

    // sweepline
    std::priority_queue<Event, std::vector<Event>, EventCmp> event_queue;
    double min_x = 1e100, max_x = -1e100;
    for (int i = 0; i < segs.size(); i++) {
        Point &p1 = segs[i].p1, &p2 = segs[i].p2;
        double x1 = p1.x, y1 = p1.y;
        double x2 = p2.x, y2 = p2.y;
        if (x1 > x2 || (x1==x2 && y1 > y2)) {
            std::swap(x1, x2);
            std::swap(y1, y2);
        }
        min_x = std::min(min_x, x1);
        max_x = std::max(max_x, x2);
        event_queue.push(Event(x1, y1, 0, i, -1));
        event_queue.push(Event(x2, y2, 1, i, -1));
        if (debug) printf("seg %d: (%llf, %llf) -> (%llf, %llf)\n", i, x1,y1,x2,y2);
    }
    cmp::scanline_x = min_x - 1;
    std::map<Segment,int,cmp> scanline_list;
    std::vector<std::map<Segment,int,cmp>::iterator> it_list(n_segs);
    int cnt2 = 0;
    std::map<std::pair<int,int>,int> all_intersection;
    while (!event_queue.empty()) {
        Event event = event_queue.top();
        event_queue.pop();
        double next_x = event.x;
        if (!event_queue.empty()) {
            next_x = event_queue.top().x;
            // if (next_x == event.x) printf("same x\n");
        }
        if (debug) printf("event type: %d, x: %llf, id1: %d, id2: %d\n", event.type, event.x, event.id1, event.id2);
        if (debug) print_table(scanline_list);
        if (event.type == 0) {
            int seg_id = event.id1;
            cmp::scanline_x = event.x;// + cmp::eps;
            Segment &cur_seg = segs[seg_id];
            scanline_list[cur_seg] = seg_id;
            it_list[seg_id] = scanline_list.find(cur_seg);
            // printf("it_list[%d]: %d\n", seg_id, it_list[seg_id]->first.id);
            std::map<Segment,int,cmp>::iterator succ_it = std::next(it_list[seg_id]);
            std::map<Segment,int,cmp>::iterator pred_it = std::prev(it_list[seg_id]);
            if (succ_it != scanline_list.end()) {
                const Segment &succ_seg = succ_it->first;
                if (is_intersect(cur_seg, succ_seg)) {
                    Point intersect = find_intersect(cur_seg, succ_seg);
                    if (intersect.x >= event.x && !has_record(all_intersection, cur_seg.id, succ_seg.id)) {
                        if (debug) printf("push event succ_seg: %d %d\n", cur_seg.id, succ_seg.id);
                        event_queue.push(Event(intersect.x, intersect.y, 2, cur_seg.id, succ_seg.id));
                        add_record(all_intersection, cur_seg.id, succ_seg.id);
                    }
                }
            }
            if (pred_it != scanline_list.end() && seg_id != (pred_it->first).id) {
                const Segment &pred_seg = pred_it->first;
                if (is_intersect(cur_seg, pred_seg)) {
                    Point intersect = find_intersect(cur_seg, pred_seg);
                    if (intersect.x >= event.x && !has_record(all_intersection, pred_seg.id, cur_seg.id)) {
                        if (debug) printf("push event pred: %d %d\n", cur_seg.id, pred_seg.id);
                        event_queue.push(Event(intersect.x, intersect.y, 2, pred_seg.id, cur_seg.id));
                        add_record(all_intersection, pred_seg.id, cur_seg.id);
                    }
                }
            }
        } else if (event.type == 1) {
            int seg_id = event.id1;
            Segment &cur_seg = segs[seg_id];
            std::map<Segment,int,cmp>::iterator succ_it = std::next(it_list[seg_id]);
            std::map<Segment,int,cmp>::iterator pred_it = std::prev(it_list[seg_id]);
            if (pred_it != scanline_list.end() && (pred_it->first).id != seg_id && succ_it != scanline_list.end()) {
                const Segment &succ_seg = succ_it->first;
                const Segment &pred_seg = pred_it->first;
                if (is_intersect(pred_seg, succ_seg)) {
                    Point intersect = find_intersect(pred_seg, succ_seg);
                    if (intersect.x >= event.x && !has_record(all_intersection, pred_seg.id, succ_seg.id)) {
                        if (debug) printf("push event type=right: %d %d\n", pred_seg.id, succ_seg.id);
                        event_queue.push(Event(intersect.x, intersect.y, 2, pred_seg.id, succ_seg.id));
                        add_record(all_intersection, pred_seg.id, succ_seg.id);
                    }
                }
            }
            scanline_list.erase(it_list[seg_id]);
            it_list[seg_id] = scanline_list.end();
        } else if (event.type == 2) {
            cnt2++;
            int seg_id1 = event.id1, seg_id2 = event.id2;
            double y1 = segs[seg_id1].get_y(event.x-cmp::eps), y2 = segs[seg_id2].get_y(event.x-cmp::eps);
            if (y1 < y2) {
                printf("weird in intersection\n");
                std::swap(seg_id1, seg_id2);
            }
            // seg_id1 is higher than seg_id2
            int pred_id = -1, next_id = -1;
            std::map<Segment,int,cmp>::iterator succ_it = std::next(it_list[seg_id2]);
            std::map<Segment,int,cmp>::iterator pred_it = std::prev(it_list[seg_id1]);
            if (succ_it != scanline_list.end()) {
                next_id = (succ_it->first).id;
            }
            if (pred_it != scanline_list.end() && seg_id1 != (pred_it->first).id) {
                pred_id = (pred_it->first).id;
            }
            scanline_list.erase(it_list[seg_id1]);
            scanline_list.erase(it_list[seg_id2]);
            // move scanline
            cmp::scanline_x = event.x ;//+ cmp::eps;
            scanline_list[segs[seg_id1]] = seg_id1;
            scanline_list[segs[seg_id2]] = seg_id2;
            it_list[seg_id1] = scanline_list.find(segs[seg_id1]);
            it_list[seg_id2] = scanline_list.find(segs[seg_id2]);
            // seg_id1 is lower than seg_id2
            if (next_id >= 0) {
                const Segment &succ_seg = succ_it->first;
                if (is_intersect(segs[seg_id1], succ_seg)) {
                    Point intersect = find_intersect(segs[seg_id1], succ_seg);
                    if (intersect.x >= event.x && !has_record(all_intersection, segs[seg_id1].id, succ_seg.id)) {
                        if (debug) printf("push event type2 A: %d %d\n", segs[seg_id1].id, succ_seg.id);
                        event_queue.push(Event(intersect.x, intersect.y, 2, segs[seg_id1].id, succ_seg.id));
                        add_record(all_intersection, segs[seg_id1].id, succ_seg.id);
                    }
                }
            }
            if (pred_id >= 0) {
                const Segment &pred_seg = pred_it->first;
                if (is_intersect(segs[seg_id2], pred_seg)) {
                    Point intersect = find_intersect(segs[seg_id2], pred_seg);
                    if (intersect.x >= event.x && !has_record(all_intersection, pred_seg.id, segs[seg_id2].id)) {
                        if (debug) printf("push event type2 B: %d %d\n", pred_seg.id, segs[seg_id2].id);
                        event_queue.push(Event(intersect.x, intersect.y, 2, pred_seg.id, segs[seg_id2].id));
                        add_record(all_intersection, pred_seg.id, segs[seg_id2].id);
                    }
                }
            }
        }
        if (debug) printf("----------\n");
    }
    printf("cnt2: %d\n", cnt2);
    printf("size:%d / %d\n", bf_table.size(), all_intersection.size());
    for (std::map<std::pair<int,int>,int>::iterator it = bf_table.begin(); it != bf_table.end(); it++) {
        if (!has_record(all_intersection, it->first.first, it->first.second)) printf("%d %d\n", it->first.first, it->first.second);
    }
    return 0;
}



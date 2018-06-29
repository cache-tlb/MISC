//////////////////////////////////////////
#include <map>
#include <cstdio>
#include <random>
#include <queue>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

const double eps = 1e-10;
const double inf = 1e100;

int num_points = 200;
bool show_each_step = false;

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

struct Segment {
    Point p1, p2;
    Segment() {}
    Segment(const Point &p1_, const Point &p2_) : p1(p1_), p2(p2_) {}
};
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

struct Arc;
struct ArcCmp {
    static double scanline_y;
    bool operator () (const Arc *a, const Arc *b);
};

struct Event {
    int type;   // 0: site event; 1: circle event
    double x, y;
    int id1, id2, id3;
    Point center; // for circle event
    int uuid;
    static int instance_count;
    std::map<Arc*,int,ArcCmp>::iterator beachline_iter;
    Event() : type(-1), x(0), y(0), id1(-1), id2(-1), id3(-1), center(0,0) {uuid = instance_count++;}
    Event(double x_, double y_, int type_, int id1_, int id2_, int id3_)
        : x(x_), y(y_), type(type_), id1(id1_), id2(id2_), id3(id3_) {
            uuid = instance_count++;
    }
};

int Event::instance_count = 0;

struct EventCmp {
    bool operator () (const Event &e1, const Event &e2) {
        return (e1.y > e2.y) 
            || (e1.y == e2.y && e1.x < e2.x) 
            || (e1.y == e2.y && e1.x == e2.x && e1.type < e2.type) 
            || (e1.y == e2.y && e1.x == e2.x && e1.type == e2.type && e1.id1 < e2.id1)
            || (e1.y == e2.y && e1.x == e2.x && e1.type == e2.type && e1.id1 == e2.id1 && e1.id2 < e2.id2)
            || (e1.y == e2.y && e1.x == e2.x && e1.type == e2.type && e1.id1 == e2.id1 && e1.id2 == e2.id2 && e1.id3 < e2.id3)
            || (e1.y == e2.y && e1.x == e2.x && e1.type == e2.type && e1.id1 == e2.id1 && e1.id2 == e2.id2 && e1.id3 == e2.id3 && e1.uuid < e2.uuid);
    }
};

struct Tuple {
    int x, y;
    Tuple() {}
    Tuple(int x,int y) : x(x), y(y) {}
    bool operator < (const Tuple &that) const {
        return (x < that.x) || (x == that.x && y < that.y);
    }
};

bool fuzzy_eq(const double a, const double b) {
    if (fabs(a-b) < eps) return true;
    return false;
}

struct Arc {
    int site_id, prev_id, next_id;
    double x,y;     // of the site
    double prev_x, prev_y, next_x, next_y;
    int uuid;
    std::map<Arc*,int,ArcCmp>::iterator iter;
    static int instance_count;
    std::map<Event,int,EventCmp>::iterator event_queue_iter; // for circle event
    Arc(int id, double x_, double y_, std::map<Event,int>::iterator end_iter)
        : site_id(id), x(x_), y(y_), prev_id(-1), next_id(-1), prev_x(0), prev_y(0), next_x(0), next_y(0), event_queue_iter(end_iter) {
            uuid = instance_count++;
    }

    double left_bound(double scanline_y) const {
        if (fuzzy_eq(y,scanline_y)) return x;
        if (prev_id < 0) return -inf;
        if (fuzzy_eq(prev_y,scanline_y)) return prev_x;
        // find the parabolics, y = a*(x-b)^2+c
        double a1 = 1./(2*(y - scanline_y)), a2 = 1./(2*(prev_y - scanline_y));
        double b1 = x, b2 = prev_x;
        double c1 = (y + scanline_y)*0.5, c2 = (prev_y + scanline_y)*0.5;
        // solve: a*x^2 - 2b*x + c = 0
        double a = (a1 - a2), b = a1*b1 - a2*b2, c = a1*b1*b1 + c1 - a2*b2*b2 - c2;
        if (fuzzy_eq(y,prev_y)) return c/(2*b);
        double del_sqr = b*b-a*c;
        if (del_sqr < 0) printf("weird: no intersection of two parabolics in left bound\n");
        double delta = sqrt(del_sqr);
        double x1 = (b - delta)/a, x2 = (b + delta)/a;
        if (a < 0) {
            std::swap(x1,x2);
        }
        if (y < prev_y) return x1;
        else return x2;
    }
    double right_bound(double scanline_y) const {
        if (fuzzy_eq(y,scanline_y)) return x;
        if (next_id < 0) return inf;
        if (fuzzy_eq(next_y, scanline_y)) return next_x;
        double a1 = 1./(2*(y - scanline_y)), a2 = 1./(2*(next_y - scanline_y));
        double b1 = x, b2 = next_x;
        double c1 = (y+scanline_y)*0.5, c2 = (next_y + scanline_y)*0.5;
        double a = (a1 - a2), b = a1*b1 - a2*b2, c = a1*b1*b1 + c1 - a2*b2*b2 - c2;
        if (fuzzy_eq(y,next_y)) return c/(2*b);
        double del_sqr = b*b-a*c;
        if (del_sqr < 0) printf("weird: no intersection of two parabolics in right bound\n");
        double delta = sqrt(del_sqr);
        double x1 = (b - delta)/a, x2 = (b + delta)/a;
        if (a < 0) {
            std::swap(x1,x2);
        }
        if (y < next_y) return x2;
        else  return x1;
    }
    double get_y(double x0, double sweep_y) const {
        double xx = x, yy = y;
        if (fuzzy_eq(sweep_y, y)) {
            if (prev_id >= 0) {
                xx = prev_x;
                yy = prev_y;
            } else if (next_id >= 0) {
                xx = next_x;
                yy = next_y;
            }
        }
        double a = 1./(2*(yy - sweep_y));
        double b = xx;
        double c = (yy + sweep_y)*0.5;
        return a*(x0-b)*(x0-b) + c;
    }
};

int Arc::instance_count = 0;

bool ArcCmp::operator () (const Arc *a, const Arc *b) {
    double l1 = a->left_bound(scanline_y), l2 = b->left_bound(scanline_y);
    if (fuzzy_eq(l1,l2)) {
        double r1 = a->right_bound(scanline_y), r2 = b->right_bound(scanline_y);
        if (fuzzy_eq(r1,r2)) {
            return (a->x < b->x) 
                || (a->x == b->x && a->site_id < b->site_id)
                || (a->x == b->x && a->site_id == b->site_id && a->prev_id < b->prev_id)
                || (a->x == b->x && a->site_id == b->site_id && a->prev_id == b->prev_id && a->next_id < b->next_id);
                //|| (a->x == b->x && a->uuid < b->uuid) ;
        }
        else return r1 < r2;
    } else return l1 < l2;
}

bool circumcircle(const Point& p0, const Point& p1, const Point& p2, Point& center, double& radius){
    double dA, dB, dC, aux1, aux2, div;

    dA = p0.x * p0.x + p0.y * p0.y;
    dB = p1.x * p1.x + p1.y * p1.y;
    dC = p2.x * p2.x + p2.y * p2.y;

    aux1 = (dA*(p2.y - p1.y) + dB*(p0.y - p2.y) + dC*(p1.y - p0.y));
    aux2 = -(dA*(p2.x - p1.x) + dB*(p0.x - p2.x) + dC*(p1.x - p0.x));
    div = (2*(p0.x*(p2.y - p1.y) + p1.x*(p0.y-p2.y) + p2.x*(p1.y - p0.y)));

    if(div == 0){ 
        return false;
    }

    center.x = aux1/div;
    center.y = aux2/div;

    radius = sqrt((center.x - p0.x)*(center.x - p0.x) + (center.y - p0.y)*(center.y - p0.y));

    return true;
}

double random_float() {
    return double(rand()) / RAND_MAX;
    // return rand() % 100;
}

double ArcCmp::scanline_y = 100;

void remove_event(std::map<Event,int,EventCmp> &event_queue, std::map<Event,int,EventCmp>::iterator &it) {
    if (it != event_queue.end()) {
        // printf("remove event: %d %d %d %d\n", it->first.id1, it->first.id2, it->first.id3, it->first.uuid);
        event_queue.erase(it);
        it = event_queue.end();
    }
}

void print_arc(std::map<Arc*,int,ArcCmp> &beach_line, std::map<Event,int,EventCmp> &event_queue) {
    std::cout << "beachline: ";
    for (std::map<Arc*,int,ArcCmp>::iterator it = beach_line.begin(); it != beach_line.end(); it++) {
        std::cout << it->first->prev_id << '/' << it->first->site_id << '/' << it->first->next_id << '/';
        std::cout << '[' << it->first->left_bound(ArcCmp::scanline_y) << ',' << it->first->right_bound(ArcCmp::scanline_y) << ']' << ", ";
        if (it->first->event_queue_iter != event_queue.end())
            std::cout << 'e' << it->first->event_queue_iter->first.id1 << it->first->event_queue_iter->first.id2 << it->first->event_queue_iter->first.id3 << ';';
    }
    std::cout << std::endl;
}
void print_queue(std::map<Event,int,EventCmp> &event_queue) {
    std::cout << "events: ";
    for (std::map<Event,int,EventCmp>::iterator it = event_queue.begin(); it != event_queue.end(); it++) {
        if (it->first.type >= 2) continue;
        std::cout << "(" << it->first.x << "," << it->first.y << ")[" << it->first.type << ':' << it->first.id1 << '/' << it->first.id2 << '/' << it->first.id3 << "] ";
    }
    std::cout << std::endl;
}

void get_im_pos(double bound_x_min, double bound_x_max, double bound_y_min, double bound_y_max, int im_rows, int im_cols, double x, double y, int &i, int &j) {
    j = (x-bound_x_min)/(bound_x_max - bound_x_min)*(im_cols-1);
    i = im_rows-1-(y-bound_y_min)/(bound_y_max - bound_y_min)*(im_rows-1);
}

void ssclip(double xmin, double xmax, double ymin, double ymax, double &x1, double &y1, double &x2, double &y2) {
    Point topleft(xmin, ymax),  topright(xmax, ymax);
    Point bottomleft(xmin, ymin), bottomright(xmax, ymin);
    Segment left(topleft, bottomleft), right(topright, bottomright);
    Segment top(topleft, topright), bottom(bottomleft, bottomright);
    Point p1(x1,y1), p2(x2,y2);
    Segment s(p1,p2);
    bool ever_intersected = false;
    if (is_intersect(s,left)) {
        Point inter = find_intersect(s,left);
        if (p1.x <= xmin) p1 = inter;
        if (p2.x <= xmin) p2 = inter;
        s = Segment(p1,p2);
        ever_intersected = true;
    }
    if (is_intersect(s,right)) {
        Point inter = find_intersect(s,right);
        if (p1.x >= xmax) p1 = inter;
        if (p2.x >= xmax) p2 = inter;
        s = Segment(p1,p2);
        ever_intersected = true;
    }
    if (is_intersect(s,top)) {
        Point inter = find_intersect(s,top);
        if (p1.y >= ymax) p1 = inter;
        if (p2.y >= ymax) p2 = inter;
        s = Segment(p1,p2);
        ever_intersected = true;
    }
    if (is_intersect(s,bottom)) {
        Point inter = find_intersect(s,bottom);
        if (p1.y <= ymin) p1 = inter;
        if (p2.y <= ymin) p2 = inter;
        s = Segment(p1,p2);
        ever_intersected = true;
    }
    if (ever_intersected) {
        x1 = p1.x; y1 = p1.y;
        x2 = p2.x; y2 = p2.y;
    } else if (x1 >= xmin && x1 <= xmax && y1 >= ymin && y1 <= ymax && x2 >= xmin && x2 <= xmax && y2 >= ymin && y2 <= ymax) {
        // in range
    } else {
        x1 = x2 = xmin - 1;
        y1 = y2 = ymin - 1;
    }
}

void draw_state(int im_size, double bound_x_min, double bound_x_max, double bound_y_min, double bound_y_max, 
                std::vector<Point> &points, std::vector<cv::Vec3b> &palette, std::map<Arc*,int,ArcCmp> &beach_line, double sweep_y, 
                std::map<Tuple, Point> &start_points, std::map<Tuple, Point> &end_point, std::map<Tuple, Point> &tmp_start) {
    cv::Mat show_im = cv::Mat::zeros(im_size, im_size, CV_8UC3);
    // show sweepline
    cv::Point s1, s2;
    get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, bound_x_min, sweep_y, s1.y, s1.x);
    get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, bound_x_max, sweep_y, s2.y, s2.x);
    cv::line(show_im, s1, s2, cv::Scalar(128,128,128), 2);
    const int n = 200;
    for (std::map<Arc*,int,ArcCmp>::iterator it  = beach_line.begin(); it != beach_line.end(); it++) {
        Arc *arc = it->first;
        int sid = arc->site_id;
        double x0 = arc->x, y0 = arc->y;
        double xmin = arc->left_bound(sweep_y), xmax = arc->right_bound(sweep_y);
        xmin = std::max(xmin, bound_x_min);
        xmax = std::min(xmax, bound_x_max);
        // y = a*(x-b)^2+c
        double a = 1./(2*(y0 - sweep_y));
        double b = x0;
        double c = (y0 + sweep_y)*0.5;
        if (fuzzy_eq(y0, sweep_y)) {
            cv::Point p1, p2;
            double y1 = bound_y_max;
            if (arc->prev_id >= 0 && !fuzzy_eq(arc->prev_y,sweep_y)) {
                a = 1./(2*(arc->prev_y - sweep_y));
                b = arc->prev_x;
                c = (arc->prev_y + sweep_y)*0.5;
                y1 = a*(x0-b)*(x0-b) + c;
            } else if (arc->next_id >= 0 && !fuzzy_eq(arc->next_y, sweep_y)) {
                a = 1./(2*(arc->next_y- sweep_y));
                b = arc->next_x;
                c = (arc->next_y + sweep_y)*0.5;
                y1 = a*(x0-b)*(x0-b) + c;
            }
            get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x0, y1, p1.y, p1.x);
            get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x0, sweep_y, p2.y, p2.x);
            cv::line(show_im, p1,p2, cv::Scalar(palette[sid]), 2);
        } else {
            for (int i = 1; i <= n; i++) {
                double x1 = xmin + double(i-1)*(xmax-xmin)/n, x2 = xmin + double(i)*(xmax-xmin)/n;
                double y1 = a*(x1-b)*(x1-b) + c, y2 = a*(x2-b)*(x2-b)+c;
                if (y1 > bound_y_max || y2 > bound_y_max) continue;
                cv::Point p1, p2;
                ssclip(bound_x_min, bound_x_max, bound_y_min, bound_y_max, x1, y1, x2, y2);
                get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x1, y1, p1.y, p1.x);
                get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x2, y2, p2.y, p2.x);
                cv::line(show_im, p1, p2, cv::Scalar(palette[sid]), 2);
            }
        }
    }
    for (int i = 0; i < points.size(); i++) {
        cv::Point p;
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, points[i].x, points[i].y, p.y, p.x);
        cv::circle(show_im, p, 2, cv::Scalar(palette[i]), 2, -1);
    }

    // finished edge, 2 parts
    for (std::map<Tuple, Point>::iterator it = start_points.begin(); it != start_points.end(); it++) {
        int i = it->first.x, j = it->first.y;
        if (end_point.count(Tuple(i,j)) == 0) {
            continue;
        }
        double x1 = it->second.x, y1 = it->second.y;
        std::map<Tuple, Point>::iterator it2 = end_point.find(Tuple(i,j));
        double x2 = it2->second.x, y2 = it2->second.y;
        cv::Point p1, p2;
        ssclip(bound_x_min, bound_x_max, bound_y_min, bound_y_max, x1, y1, x2, y2);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x1, y1, p1.y, p1.x);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x2, y2, p2.y, p2.x);
        cv::line(show_im, p1, p2, cv::Scalar(64,64,64), 1);
    }
    for (std::map<Tuple, Point>::iterator it = end_point.begin(); it != end_point.end(); it++) {
        int i = it->first.x, j = it->first.y;
        if (i > j) continue;
        if (end_point.count(Tuple(j,i)) == 0) continue;
        double x1 = it->second.x, y1 = it->second.y;
        std::map<Tuple, Point>::iterator it2 = end_point.find(Tuple(j,i));
        double x2 = it2->second.x, y2 = it2->second.y;
        cv::Point p1, p2;
        ssclip(bound_x_min, bound_x_max, bound_y_min, bound_y_max, x1, y1, x2, y2);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x1, y1, p1.y, p1.x);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x2, y2, p2.y, p2.x);
        cv::line(show_im, p1, p2, cv::Scalar(64,64,64), 1);
    }

    // unfinished edge
    for (std::map<Arc*,int,ArcCmp>::iterator it  = beach_line.begin(); it != beach_line.end(); it++) {
        Arc *arc = it->first;
        int site1, site2;
        double x0;
        for (int i = 0; i < 2; i++) {
            if (i == 0 && arc->prev_id >= 0) {
                site1 = arc->prev_id;
                site2 = arc->site_id;
                x0 = arc->left_bound(sweep_y);
            }
            else if (i == 1 && arc->next_id >= 0) {
                site1 = arc->site_id;
                site2 = arc->next_id;
                x0 = arc->right_bound(sweep_y);
            } else {
                continue;
            }
            double y0 = arc->get_y(x0, sweep_y);
            if (y0 > bound_y_max) continue;
            Tuple tp(site1,site2);
            Point start;
            for (int k = 0; k < 2; k++) {
                if (k == 0 && tmp_start.count(tp) > 0) {
                    start = tmp_start[tp];
                } else if (k == 1 && start_points.count(tp) > 0) {
                    start = start_points[tp];
                } else {
                    continue;
                }
                cv::Point p1, p2;
                ssclip(bound_x_min, bound_x_max, bound_y_min, bound_y_max, start.x, start.y, x0, y0);
                get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, start.x, start.y, p1.y, p1.x);
                get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x0, y0, p2.y, p2.x);
                cv::line(show_im, p1, p2, cv::Scalar(64,64,64), 1);
            }
        }
    }
    for (std::map<Tuple, Point>::iterator it = tmp_start.begin(); it != tmp_start.end(); it++) {
        int i = it->first.x, j = it->first.y;
        if (end_point.count(Tuple(i,j)) == 0) continue;
        std::map<Tuple, Point>::iterator it2 = end_point.find(Tuple(i,j));
        double x1 = it->second.x, y1 = it->second.y;
        double x2 = it2->second.x, y2 = it2->second.y;
        cv::Point p1, p2;
        ssclip(bound_x_min, bound_x_max, bound_y_min, bound_y_max, x1, y1, x2, y2);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x1, y1, p1.y, p1.x);
        get_im_pos(bound_x_min, bound_x_max, bound_y_min, bound_y_max, im_size, im_size, x2, y2, p2.y, p2.x);
        cv::line(show_im, p1, p2, cv::Scalar(64,64,64), 1);
    }

    cv::imshow("state", show_im);
    cv::waitKey();
}

void add_point(std::map<Tuple, Point> &table, const Point &p, int i, int j) {
    if (table.count(Tuple(i,j)) > 0) printf("weird: already have point in table\n");
    table[Tuple(i,j)] = p;
}

int main(int argc, char const *argv[]) {
    srand(1);
    double bound_x_min = -2, bound_x_max = 2;
    double bound_y_min = -2, bound_y_max = 2;
    int im_width = 1001;
    std::vector<Point> points;
    for (int i = 0; i < num_points; i++) {
        double x = random_float()*3-1.5, y = random_float()*3-1.5;
        points.push_back(Point(x,y));
    }

    // points.erase(points.begin()+5, points.begin()+10);

    std::vector<cv::Vec3b> palette;
    int base_l = 64;
    for (int i = 0; i < points.size(); i++) {
        int r = base_l + rand() % (256-base_l), g = base_l + rand() % (256-base_l), b = base_l + rand() % (256-base_l);
        palette.push_back(cv::Vec3b(r,g,b));
    }
    cv::Mat show_im = cv::Mat::zeros(im_width, im_width, CV_8UC3);
    for (int i = 0; i < show_im.rows; i++) {
        double y = double(i)/(show_im.rows-1)*(bound_y_max - bound_y_min)+bound_y_min;
        for (int j = 0; j < show_im.cols; j++) {
            double x = double(j)/(show_im.cols-1)*(bound_x_max - bound_x_min)+bound_x_min;
            double min_dist = inf;
            double min_index = -1;
            for (int k = 0; k < points.size(); k++) {
                double d = hypot(x - points[k].x, y + points[k].y);
                if (d < min_dist) {
                    min_dist = d;
                    min_index = k;
                }
            }
            show_im.at<cv::Vec3b>(i,j) = palette[min_index];
        }
    }
    for (int i = 0; i < points.size(); i++) {
        cv::circle(show_im, cv::Point((points[i].x-bound_x_min)/(bound_x_max - bound_x_min)*(show_im.cols-1), 
                                      (-points[i].y-bound_y_min)/(bound_y_max - bound_y_min)*(show_im.rows-1)), 5, cv::Scalar(0,0,0), -1);
    }
    cv::imshow("im", show_im);
    cv::waitKey();

    std::map<Arc*,int,ArcCmp> beach_line;
    std::map<Event,int,EventCmp> event_queue;
    std::vector<Point> all_vertex;
    std::map<Tuple, Point> start_point, end_point;
    std::map<Tuple, Point> tmp_start;
    for (int i = 0; i < points.size(); i++) {
        Event e(points[i].x, points[i].y, 0, i, -1, -1);
        event_queue[e] = 1;
    }
    for (double y = bound_y_max/2; y >= bound_y_min; y-=0.01) {
        Event e(0,y,2,0,0,0);
        event_queue[e] = 1;
    }
    while (!event_queue.empty()) {
        std::map<Event,int,EventCmp>::iterator first_event_it = event_queue.begin();
        Event event = (first_event_it->first);
        if (event.type < 2) {
            // print_queue(event_queue);
            // printf("-------------------------------------\n");
            // printf("event: type: %d, id: %d, id2:%d, id3:%d, y: %llf, x: %llf\n", event.type, event.id1, event.id2, event.id3, event.y, event.x);
            // print_arc(beach_line,event_queue);
        }
        event_queue.erase(first_event_it);
        ArcCmp::scanline_y = event.y;
        if (event.type == 0) {
            // site event, first find pos in the beach line;
            ArcCmp::scanline_y = event.y;
            if (beach_line.empty()) {
                Arc *arc = new Arc(event.id1, event.x, event.y, event_queue.end());
                beach_line[arc] = 1;
                arc->iter = beach_line.find(arc);
            } else {
                Point center;
                double radius;
                Arc *arc = new Arc(event.id1, event.x, event.y, event_queue.end());
                std::map<Arc*,int,ArcCmp>::iterator ub_iter = beach_line.upper_bound(arc);
                delete arc;
                //if (ub_iter != beach_line.end() && ub_iter != beach_line.begin()) {
                if (ub_iter != beach_line.begin()) {
                    Arc *arc_p = NULL;
                    if (ub_iter != beach_line.end()) {
                        std::map<Arc*,int,ArcCmp>::iterator p_iter = std::prev(ub_iter, 1);
                        arc_p = p_iter->first;
                    } else {
                        std::map<Arc*,int,ArcCmp>::reverse_iterator p_iter = beach_line.rbegin();
                        arc_p = p_iter->first;
                    }
                    Arc *arc_r = new Arc(event.id1, event.x, event.y, event_queue.end());
                    arc_r->prev_id = arc_p->site_id; arc_r->prev_x = arc_p->x; arc_r->prev_y = arc_p->y;
                    arc_r->next_id = arc_p->site_id; arc_r->next_x = arc_p->x; arc_r->next_y = arc_p->y;
                    Arc *new_p = new Arc(arc_p->site_id, arc_p->x, arc_p->y, event_queue.end());
                    new_p->prev_id = arc_r->site_id; new_p->prev_x = arc_r->x; new_p->prev_y = arc_r->y;
                    new_p->next_id = arc_p->next_id; new_p->next_x = arc_p->next_x; new_p->next_y = arc_p->next_y;
                    arc_p->next_id = arc_r->site_id; arc_p->next_x = arc_r->x; arc_p->next_y = arc_r->y;
                    beach_line[arc_r] = 1;
                    beach_line[new_p] = 1;
                    arc_r->iter = beach_line.find(arc_r);
                    new_p->iter = beach_line.find(new_p);
                    double y0 = arc_p->get_y(event.x, ArcCmp::scanline_y);
                    tmp_start[Tuple(arc_r->prev_id,arc_r->site_id)] = Point(event.x, y0);
                    tmp_start[Tuple(arc_r->site_id,arc_r->next_id)] = Point(event.x, y0);
                    // remove the event
                    remove_event(event_queue, arc_p->event_queue_iter);
                    // try find new arc
                    Point prev_p(arc_p->prev_x, arc_p->prev_y), p(arc_p->x, arc_p->y), r(arc_r->x, arc_r->y);
                    if (arc_p->prev_id >= 0) {
                        bool ok = circumcircle(prev_p, p, r, center, radius);
                        if (ok && center.y - radius <= ArcCmp::scanline_y) {
                            Event e(center.x, center.y - radius, 1, arc_p->prev_id, arc_p->site_id, arc_r->site_id);
                            // printf("insert circle event A1:(%llf, %llf)[%d,%d,%d]\n", center.x, center.y - radius, arc_p->prev_id, arc_p->site_id, arc_r->site_id);
                            e.center = center;
                            e.beachline_iter = arc_p->iter;
                            event_queue[e] = 1;
                            arc_p->event_queue_iter = event_queue.find(e);
                        }
                    }
                    if (ub_iter != beach_line.end()) {
                        // rpq
                        Arc *arc_q = ub_iter->first;
                        Point q(arc_q->x, arc_q->y), next_q(arc_q->next_x, arc_q->next_y);
                        bool ok = circumcircle(r, p, q, center, radius);
                        if (ok && center.y - radius <= ArcCmp::scanline_y) {
                            Event e(center.x, center.y - radius, 1, arc_r->site_id, new_p->site_id, arc_q->site_id);
                            // printf("insert circle event A2:(%llf, %llf)[%d,%d,%d]\n", center.x, center.y - radius, arc_r->site_id, new_p->site_id, arc_q->site_id);
                            e.center = center;
                            e.beachline_iter = new_p->iter;
                            event_queue[e] = 1;
                            new_p->event_queue_iter = event_queue.find(e);
                            // print_arc(beach_line, event_queue);
                            // std::cout << new_p->event_queue_iter->first.id1 << new_p->event_queue_iter->first.id2 << new_p->event_queue_iter->first.id3<<std::endl;
                        }
                        // remove_event(event_queue, arc_q->event_queue_iter);
                    }
                } /*else if (ub_iter == beach_line.end() && ub_iter != beach_line.begin()) {
                    //  append at the end
                    std::map<Arc*,int,ArcCmp>::reverse_iterator p_iter = beach_line.rbegin();
                    Arc *arc_p = p_iter->first;
                    Arc *arc_r = new Arc(event.id1, event.x, event.y, event_queue.end());
                    arc_r->prev_id = arc_p->site_id; arc_r->prev_x = arc_p->x; arc_r->prev_y = arc_p->y;
                    arc_r->next_id = arc_p->site_id; arc_r->next_x = arc_p->x; arc_r->next_y = arc_p->y;
                    Arc *new_p = new Arc(arc_p->site_id, arc_p->x, arc_p->y, event_queue.end());
                    new_p->prev_id = arc_r->site_id; new_p->prev_x = arc_r->x; new_p->prev_y = arc_r->y;
                    new_p->next_id = arc_p->next_id; new_p->next_x = arc_p->next_x; new_p->next_y = arc_p->next_y;
                    arc_p->next_id = arc_r->site_id; arc_p->next_x = arc_r->x; arc_p->next_y = arc_r->y;
                    beach_line[arc_r] = 1;
                    beach_line[new_p] = 1;
                    remove_event(event_queue, arc_p->event_queue_iter);
                    Point prev_p(arc_p->prev_x, arc_p->prev_y), p(arc_p->x, arc_p->y), r(arc_r->x, arc_r->y);
                    if (arc_p->prev_id >= 0) {
                        bool ok = circumcircle(prev_p, p, r, center, radius);
                        if (ok && center.y - radius <= ArcCmp::scanline_y) {
                            Event e(center.x, center.y - radius, 1, arc_p->prev_id, arc_p->site_id, arc_r->site_id);
                            // printf("insert circle event B1:(%llf, %llf)[%d,%d,%d]\n", center.x, center.y - radius, arc_p->prev_id, arc_p->site_id, arc_r->site_id);
                            e.center = center;
                            e.beachline_iter = beach_line.find(arc_p);
                            event_queue[e] = 1;
                            arc_p->event_queue_iter = event_queue.find(e);
                        }
                    }
                }*/ else if (ub_iter != beach_line.end() && ub_iter == beach_line.begin()) {
                    printf("weird: get a site x < -inf\n");
                } else {
                    printf("weird: impossible happens in site event: begin == end\n");
                }
            }
        } else if (event.type == 1) {
            // circle event
            ArcCmp::scanline_y = event.y;
            std::map<Arc*,int,ArcCmp>::iterator pj_iter = event.beachline_iter;
            Arc *pj = pj_iter->first;
            pj->event_queue_iter = event_queue.end();
            if (fuzzy_eq(pj->left_bound(ArcCmp::scanline_y), pj->right_bound(ArcCmp::scanline_y)) ) {
                std::map<Arc*,int,ArcCmp>::iterator pi_iter = std::prev(pj_iter);
                std::map<Arc*,int,ArcCmp>::iterator pk_iter = std::next(pj_iter);
                if (pi_iter == beach_line.end() || pi_iter->first->uuid == pj_iter->first->uuid) {
                    printf("weird: pi not found\n");
                }
                if (pk_iter == beach_line.end()) {
                    printf("weird: pk not found\n");
                }
                if (pi_iter->first->uuid == pk_iter->first->uuid || pk_iter->first->uuid == pj_iter->first->uuid) {
                    printf("weird: arc search error\n");
                }
                Arc *pi = pi_iter->first;
                Arc *pk = pk_iter->first;
                // print_queue(event_queue);
                // printf("try remove pi:[%d,%d,%d]\n", pi->prev_id, pi->site_id, pi->next_id);
                remove_event(event_queue, pi->event_queue_iter);
                // print_queue(event_queue);
                // printf("try remove pk:[%d,%d,%d]\n", pk->prev_id, pk->site_id, pk->next_id);
                remove_event(event_queue, pk->event_queue_iter);
                // printf("end of remove\n");
                pi->next_id = pk->site_id; pi->next_x = pk->x; pi->next_y = pk->y;
                pk->prev_id = pi->site_id; pk->prev_x = pi->x; pk->prev_y = pi->y;
                Point center;
                double radius;
                Point prev_i(pi->prev_x, pi->prev_y), site_i(pi->x, pi->y), site_k(pk->x, pk->y), next_k(pk->next_x, pk->next_y);
                if (pi->prev_id >= 0) {
                    bool ok = circumcircle(prev_i, site_i, site_k, center, radius);
                    if (ok && center.y - radius <= ArcCmp::scanline_y) {
                        Event e(center.x, center.y - radius, 1, pi->prev_id, pi->site_id, pk->site_id);
                        // printf("insert circle event C1:(%llf, %llf)[%d,%d,%d]\n", center.x, center.y - radius, pi->prev_id, pi->site_id, pk->site_id);
                        e.center = center;
                        e.beachline_iter = pi_iter;
                        event_queue[e] = 1;
                        pi->event_queue_iter = event_queue.find(e);
                    }
                }
                if (pk->next_id >= 0) {
                    bool ok = circumcircle(site_i, site_k, next_k, center, radius);
                    if (ok && center.y - radius <= ArcCmp::scanline_y) {
                        Event e(center.x, center.y - radius, 1, pi->site_id, pk->site_id, pk->next_id);
                        // printf("insert circle event C2:(%llf, %llf)[%d,%d,%d]\n", center.x, center.y - radius, pi->site_id, pk->site_id, pk->next_id);
                        e.center = center;
                        e.beachline_iter = pk_iter;
                        event_queue[e] = 1;
                        pk->event_queue_iter = event_queue.find(e);
                    }
                }
                add_point(end_point, event.center, pi->site_id, pj->site_id);
                add_point(end_point, event.center, pj->site_id, pk->site_id);
                add_point(start_point, event.center, pi->site_id, pk->site_id);
                all_vertex.push_back(event.center);
                beach_line.erase(pj_iter);
                delete pj;
            } else {
                // printf("skipped\n");
            }
        } else if (event.type == 2) {
            ArcCmp::scanline_y = event.y;
            if (show_each_step) draw_state(im_width, bound_x_min, bound_x_max, bound_y_min, bound_y_max, points, palette, beach_line, ArcCmp::scanline_y, start_point, end_point, tmp_start);
        }
        if (event.type < 2) {
            if (show_each_step) draw_state(im_width, bound_x_min, bound_x_max, bound_y_min, bound_y_max, points, palette, beach_line, ArcCmp::scanline_y, start_point, end_point, tmp_start);
            // print_queue(event_queue);
            // print_arc(beach_line);
        }
    }
    /*for (int i = 0; i < all_vertex.size(); i++) {
        printf("%llf, %llf\n", all_vertex[i].x, all_vertex[i].y);
    }*/
    print_arc(beach_line,event_queue);
    draw_state(im_width, bound_x_min, bound_x_max, bound_y_min, bound_y_max, points, palette, beach_line, ArcCmp::scanline_y, start_point, end_point, tmp_start);
    return 0;
}




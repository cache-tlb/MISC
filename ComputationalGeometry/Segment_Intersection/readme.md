### Overview

This project solves the problem of segment intersection report using plane sweep method. The complexsity is O((N+I)logN), where N and I are the number of points and the number of intersections respectively. Compared to the O(N^2) brute force algorithm, it is more efficient when the number of intersections is not as large as N^2. However in the worst case, there could be (N-1)*N/2 intersections in total.

### Detail

I do not implement a balanced binary search tree to store the segments in current scan line. Instead, a STL map is used. Each element of the key takes the x-value of the scan line as parameter and the order of the items in the map is adjusted when an intersection event happens. 

The segment entrance/exit events as well as the intersection events are stored in the same priority queue. Each time the event with minimal x is popped and processed. 

1. For an entrance event, find the location in the map and insert it. Then detect potential intersection with its predecessor and successor.

2. For an exit event, find the segment in the map. remove it from the map and detect if its predecessor and successor intersect after it is removed.

3. Any detected intersection is added to the event queue. When handling an intersection event, we should first get the id of the 2 involved segments and then find the them in the map. As the scan line moves to the right of the intersection, the order of the 2 segments is changed, So we need to remove them from the map, the move the scan line(as the parameter for comparison function of the map) and insert them again. Another way of implementation is to use the `pointer` of the segments as keys for the map. We can change the order by swap the contents they point to. (When use the segment itself as the key, it is impossible to directly change their order since we have no way to change the key of an item in the map.)

The algorithm processes the events until the queue is empty.

### Degenerate cases

1. When there are vertical segments.

We can add first judge whether the vertical segments have intersetions with each other. Then we can add a new type of event associated with one vertical segment each. To handle a kind of event, we just consider the segments in current scan line. The vertical segment need not to put in the map.

2. When multiple segments intersect at the same point.

I'm not sure whether it is right. Just handle the events one by one. When there are M segments at a single point, there are totally M*(M-1)/2 intersection detected and the same number of swap is excuted. Note that swap two segments, they may not be adjacent to each other. It is OK, though. If we use the pointers as the keys for the map, it is easy to achieve.

3. Multiple intersection share the same x-coorndinate.

In fact we do not have to pay extra efforts to handle this degenerate.

4. Some segments overlap with each other.

We can cut the segments to some short segments. For example, if A and B overlaps, we can divide A into a and c, and divide B into b and c, where c is the overlapping part of A and B. Then we use a,b,c instead of A,B in the algorithm and find the intersections involved with A and B afterwards.

### Dependency

This code needs no external library.

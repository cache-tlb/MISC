#include "test_framework.h"
#include "timer.h"

void test_timer_suite() {
    // run_benchmark returns average time > 0
    {
        double avg = run_benchmark([] {
            volatile long sum = 0;
            for (int i = 0; i < 100000; ++i) {
                sum += i;
            }
        }, 3, true);
        CHECK(avg > 0.0);
    }

    // with warmup: total calls = 1 (warmup) + repetitions
    {
        int call_count = 0;
        run_benchmark([&] { ++call_count; }, 4, true);
        CHECK(call_count == 5);
    }

    // without warmup: total calls = repetitions
    {
        int call_count = 0;
        run_benchmark([&] { ++call_count; }, 4, false);
        CHECK(call_count == 4);
    }
}

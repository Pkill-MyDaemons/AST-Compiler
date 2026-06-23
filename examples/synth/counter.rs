use std::collections::HashMap;

pub struct Counter {
    pub value: i64,
    pub step: i64,
    pub label: String,
}

impl Counter {
    pub fn new(label: String, step: i64) -> Counter {
        Counter { value: 0, step, label }
    }

    pub fn increment(&mut self) -> i64 {
        self.value = self.value + self.step;
        return self.value;
    }

    pub fn decrement(&mut self) -> i64 {
        self.value = self.value - self.step;
        return self.value;
    }

    pub fn reset(&mut self) -> i64 {
        self.value = 0;
        return self.value;
    }

    pub fn is_positive(&self) -> bool {
        return self.value > 0;
    }

    pub fn scaled(&self, factor: i64) -> i64 {
        return self.value * factor;
    }
}

pub fn max_counter(counters: Vec<Counter>) -> i64 {
    let mut best: i64 = 0;
    for c in &counters {
        if c.value > best {
            best = c.value;
        }
    }
    return best;
}

pub fn sum_counters(counters: Vec<Counter>) -> i64 {
    let mut total: i64 = 0;
    for c in &counters {
        total = total + c.value;
    }
    return total;
}

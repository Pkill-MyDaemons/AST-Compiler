use std::collections::HashMap;

const MAX_RETRIES: i32 = 3;

pub struct Animal {
    pub name: String,
    pub sound: String,
}

impl Animal {
    pub fn new(name: String, sound: String) -> Self {
        Animal { name, sound }
    }

    pub fn speak(&self) -> String {
        self.sound.clone()
    }

    pub fn describe(&self) -> String {
        format!("I am {}", self.name)
    }
}

pub struct Dog {
    pub name: String,
    pub sound: String,
    pub tricks: Vec<String>,
}

impl Dog {
    pub fn new(name: String) -> Self {
        Dog {
            name,
            sound: String::from("woof"),
            tricks: vec![],
        }
    }

    pub fn learn_trick(&mut self, trick: String) {
        self.tricks.push(trick);
    }

    pub fn perform(&self, trick: &str) -> bool {
        self.tricks.contains(&trick.to_string())
    }
}

pub fn greet(name: &str) -> String {
    String::from("Hello, ") + name
}

pub fn fibonacci(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    let mut a: i64 = 0;
    let mut b: i64 = 1;
    let mut i: i64 = 2;
    while i <= n {
        let temp: i64 = a + b;
        a = b;
        b = temp;
        i = i + 1;
    }
    b
}

pub fn find_max(numbers: Vec<i64>) -> Option<i64> {
    if numbers.is_empty() {
        return None;
    }
    let mut best: i64 = numbers[0];
    for x in &numbers {
        if *x > best {
            best = *x;
        }
    }
    Some(best)
}

pub fn count_words(text: &str) -> HashMap<String, i64> {
    let mut counts: HashMap<String, i64> = HashMap::new();
    for word in text.split_whitespace() {
        let entry = counts.entry(word.to_string()).or_insert(0);
        *entry += 1;
    }
    counts
}

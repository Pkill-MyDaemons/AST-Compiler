use std::io;

fn main() {
    println!("Hello! Please enter your name:");

    let mut name = String::new();

    // Read the user's name from the console
    io::stdin()
        .read_line(&mut name)
        .expect("Failed to read line");

    // Remove trailing newline characters
    let name = name.trim();

    // Pattern match to greet known users
    match name {
        "" => println!("Hello, stranger!"),
        "Admin" => println!("Welcome back, Root User!"),
        _ => println!("Nice to meet you, {}!", name),
    }
}


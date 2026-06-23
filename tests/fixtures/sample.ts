import { readFileSync } from "fs";
import * as path from "path";

const MAX_RETRIES: number = 3;
export const DEFAULT_NAME: string = "world";

export interface Speakable {
    name: string;
    sound: string;
    speak(): string;
    describe(): string;
}

export class Animal implements Speakable {
    public name: string;
    public sound: string;

    constructor(name: string, sound: string) {
        this.name = name;
        this.sound = sound;
    }

    public speak(): string {
        return this.sound;
    }

    public describe(): string {
        return "I am " + this.name;
    }
}

export class Dog extends Animal {
    public tricks: string[];

    constructor(name: string) {
        super(name, "woof");
        this.tricks = [];
    }

    public learnTrick(trick: string): void {
        this.tricks.push(trick);
    }

    public perform(trick: string): boolean {
        if (this.tricks.includes(trick)) {
            return true;
        }
        return false;
    }
}

export function greet(name: string): string {
    return "Hello, " + name;
}

export function fibonacci(n: number): number {
    if (n <= 1) {
        return n;
    }
    let a: number = 0;
    let b: number = 1;
    let i: number = 2;
    while (i <= n) {
        const temp: number = a + b;
        a = b;
        b = temp;
        i = i + 1;
    }
    return b;
}

export function findMax(numbers: number[]): number | null {
    if (numbers.length === 0) {
        return null;
    }
    let best: number = numbers[0];
    for (const x of numbers) {
        if (x > best) {
            best = x;
        }
    }
    return best;
}

export function countWords(text: string): Map<string, number> {
    const counts: Map<string, number> = new Map();
    for (const word of text.split(" ")) {
        const current = counts.get(word) ?? 0;
        counts.set(word, current + 1);
    }
    return counts;
}

export enum Direction {
    North,
    South,
    East,
    West,
}

"""Simple bookstore management system."""
from typing import List, Optional


class Book:
    """A book in the store."""

    title: str
    author: str
    price: float
    stock: int
    genre: str

    def __init__(self, title: str, author: str, price: float, stock: int, genre: str) -> None:
        self.title = title
        self.author = author
        self.price = price
        self.stock = stock
        self.genre = genre

    def revenue(self) -> float:
        return self.price * self.stock

    def is_available(self) -> bool:
        return self.stock > 0

    def reorder(self, amount: int) -> int:
        self.stock = self.stock + amount
        return self.stock


class BookStore:
    """Manages a collection of books."""

    books: List[Book]
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.books = []

    def add_book(self, book: Book) -> None:
        self.books.append(book)

    def remove_book(self, title: str) -> bool:
        for i in range(len(self.books)):
            if self.books[i].title == title:
                self.books.pop(i)
                return True
        return False

    def find_by_author(self, author: str) -> Optional[Book]:
        for book in self.books:
            if book.author == author:
                return book
        return None

    def total_revenue(self) -> float:
        result: float = 0.0
        for book in self.books:
            result = result + book.revenue()
        return result

    def out_of_stock(self, threshold: int) -> List[Book]:
        result: List[Book] = []
        for book in self.books:
            if book.stock < threshold:
                result.append(book)
        return result

    def by_genre(self, genre: str) -> List[Book]:
        result: List[Book] = []
        for book in self.books:
            if book.genre == genre:
                result.append(book)
        return result


def most_expensive(books: List[Book]) -> Optional[Book]:
    if not books:
        return None
    best: Book = books[0]
    for b in books:
        if b.price > best.price:
            best = b
    return best


def apply_sale(books: List[Book], percent: float) -> None:
    for b in books:
        b.price = b.price * (1.0 - percent / 100.0)

"""Simple inventory management system."""
from typing import List, Optional


class Product:
    """A product in the inventory."""

    name: str
    price: float
    quantity: int
    category: str

    def __init__(self, name: str, price: float, quantity: int, category: str) -> None:
        self.name = name
        self.price = price
        self.quantity = quantity
        self.category = category

    def subtotal(self) -> float:
        return self.price * self.quantity

    def is_in_stock(self) -> bool:
        return self.quantity > 0

    def restock(self, amount: int) -> int:
        self.quantity = self.quantity + amount
        return self.quantity


class Inventory:
    """Manages a collection of products."""

    products: List[Product]

    def __init__(self) -> None:
        self.products = []

    def add_product(self, product: Product) -> None:
        self.products.append(product)

    def remove_product(self, name: str) -> bool:
        for i in range(len(self.products)):
            if self.products[i].name == name:
                self.products.pop(i)
                return True
        return False

    def find_by_name(self, name: str) -> Optional[Product]:
        for product in self.products:
            if product.name == name:
                return product
        return None

    def total_value(self) -> float:
        result: float = 0.0
        for product in self.products:
            result = result + product.subtotal()
        return result

    def low_stock(self, threshold: int) -> List[Product]:
        result: List[Product] = []
        for product in self.products:
            if product.quantity < threshold:
                result.append(product)
        return result

    def by_category(self, category: str) -> List[Product]:
        result: List[Product] = []
        for product in self.products:
            if product.category == category:
                result.append(product)
        return result


def cheapest(products: List[Product]) -> Optional[Product]:
    if not products:
        return None
    best: Product = products[0]
    for p in products:
        if p.price < best.price:
            best = p
    return best


def apply_discount(products: List[Product], percent: float) -> None:
    for p in products:
        p.price = p.price * (1.0 - percent / 100.0)

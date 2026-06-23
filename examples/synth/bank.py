"""Bank account management."""
from typing import List, Optional


class Transaction:
    """A single account transaction."""

    amount: float
    description: str
    transaction_type: str

    def __init__(self, amount: float, description: str, transaction_type: str) -> None:
        self.amount = amount
        self.description = description
        self.transaction_type = transaction_type

    def is_debit(self) -> bool:
        return self.transaction_type == "debit"

    def is_credit(self) -> bool:
        return self.transaction_type == "credit"


class BankAccount:
    """A simple bank account."""

    owner: str
    balance: float
    transactions: List[Transaction]
    account_number: str

    def __init__(self, owner: str, account_number: str, initial_balance: float) -> None:
        self.owner = owner
        self.account_number = account_number
        self.balance = initial_balance
        self.transactions = []

    def deposit(self, amount: float, description: str) -> float:
        if amount <= 0.0:
            return self.balance
        t = Transaction(amount, description, "credit")
        self.transactions.append(t)
        self.balance = self.balance + amount
        return self.balance

    def withdraw(self, amount: float, description: str) -> bool:
        if amount <= 0.0 or amount > self.balance:
            return False
        t = Transaction(amount, description, "debit")
        self.transactions.append(t)
        self.balance = self.balance - amount
        return True

    def get_history(self) -> List[Transaction]:
        return self.transactions

    def total_credits(self) -> float:
        total: float = 0.0
        for t in self.transactions:
            if t.is_credit():
                total = total + t.amount
        return total

    def total_debits(self) -> float:
        total: float = 0.0
        for t in self.transactions:
            if t.is_debit():
                total = total + t.amount
        return total


def transfer_funds(source: BankAccount, target: BankAccount, amount: float) -> bool:
    if not source.withdraw(amount, "transfer out"):
        return False
    target.deposit(amount, "transfer in")
    return True


def find_largest_transaction(account: BankAccount) -> Optional[Transaction]:
    if not account.transactions:
        return None
    best = account.transactions[0]
    for t in account.transactions:
        if t.amount > best.amount:
            best = t
    return best

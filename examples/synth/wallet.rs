pub struct Transaction {
    pub amount: f64,
    pub description: String,
    pub is_credit: bool,
}

pub struct Wallet {
    pub owner: String,
    pub balance: f64,
    pub transactions: Vec<Transaction>,
}

impl Wallet {
    pub fn new(owner: String) -> Wallet {
        Wallet { owner, balance: 0.0, transactions: Vec::new() }
    }

    pub fn deposit(&mut self, amount: f64, description: String) -> f64 {
        self.balance = self.balance + amount;
        let tx = Transaction { amount, description, is_credit: true };
        self.transactions.push(tx);
        return self.balance;
    }

    pub fn withdraw(&mut self, amount: f64, description: String) -> bool {
        if self.balance < amount {
            return false;
        }
        self.balance = self.balance - amount;
        let tx = Transaction { amount, description, is_credit: false };
        self.transactions.push(tx);
        return true;
    }

    pub fn total_credits(&self) -> f64 {
        let mut total: f64 = 0.0;
        for tx in &self.transactions {
            if tx.is_credit {
                total = total + tx.amount;
            }
        }
        return total;
    }

    pub fn total_debits(&self) -> f64 {
        let mut total: f64 = 0.0;
        for tx in &self.transactions {
            if !tx.is_credit {
                total = total + tx.amount;
            }
        }
        return total;
    }

    pub fn is_solvent(&self) -> bool {
        return self.balance > 0.0;
    }
}

pub fn richest(wallets: Vec<Wallet>) -> f64 {
    let mut best: f64 = 0.0;
    for w in &wallets {
        if w.balance > best {
            best = w.balance;
        }
    }
    return best;
}

pub fn apply_fee(wallets: Vec<Wallet>, fee: f64) -> i64 {
    let mut count: i64 = 0;
    for w in &wallets {
        if w.balance > fee {
            count = count + 1;
        }
    }
    return count;
}

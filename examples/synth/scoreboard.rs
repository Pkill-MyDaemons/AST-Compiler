pub struct Player {
    pub name: String,
    pub score: i64,
    pub level: i64,
    pub active: bool,
}

pub struct Scoreboard {
    pub players: Vec<Player>,
    pub game_name: String,
    pub max_score: i64,
}

impl Player {
    pub fn new(name: String) -> Player {
        Player { name, score: 0, level: 1, active: true }
    }

    pub fn add_points(&mut self, points: i64) -> i64 {
        self.score = self.score + points;
        return self.score;
    }

    pub fn level_up(&mut self) -> i64 {
        self.level = self.level + 1;
        return self.level;
    }

    pub fn is_winning(&self, threshold: i64) -> bool {
        return self.score > threshold;
    }
}

impl Scoreboard {
    pub fn new(game_name: String) -> Scoreboard {
        Scoreboard { game_name, players: Vec::new(), max_score: 0 }
    }

    pub fn add_player(&mut self, player: Player) {
        self.players.push(player);
    }

    pub fn remove_player(&mut self, name: String) -> bool {
        let mut i: i64 = 0;
        for p in &self.players {
            if p.name == name {
                return true;
            }
            i = i + 1;
        }
        return false;
    }

    pub fn leader(&self) -> i64 {
        let mut best: i64 = 0;
        for p in &self.players {
            if p.score > best {
                best = p.score;
            }
        }
        return best;
    }

    pub fn total_score(&self) -> i64 {
        let mut total: i64 = 0;
        for p in &self.players {
            total = total + p.score;
        }
        return total;
    }

    pub fn active_count(&self) -> i64 {
        let mut count: i64 = 0;
        for p in &self.players {
            if p.active {
                count = count + 1;
            }
        }
        return count;
    }
}

pub fn top_score(boards: Vec<Scoreboard>) -> i64 {
    let mut best: i64 = 0;
    for b in &boards {
        if b.max_score > best {
            best = b.max_score;
        }
    }
    return best;
}

pub fn scale_scores(players: Vec<Player>, factor: i64) -> i64 {
    let mut total: i64 = 0;
    for p in &players {
        total = total + p.score * factor;
    }
    return total;
}

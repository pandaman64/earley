use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
};
use typed_arena::Arena;

/// the type of interned (hash-consed) things, where equality and hash values are
/// determined by its memory address
pub struct Interned<'a, T>(&'a T);

impl<'a, T> fmt::Debug for Interned<'a, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> fmt::Display for Interned<'a, T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<'a, T> Clone for Interned<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Copy for Interned<'a, T> {}

impl<'a, T> PartialEq for Interned<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        // pointer comparison
        let this: *const T = self.0;
        let other: *const T = other.0;
        std::ptr::eq(this, other)
    }
}

impl<'a, T> Eq for Interned<'a, T> {}

impl<'a, T> PartialOrd for Interned<'a, T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T> Ord for Interned<'a, T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // pointer comparison
        let this: *const T = self.0;
        let other: *const T = other.0;
        this.cmp(&other)
    }
}

impl<'a, T> Hash for Interned<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // pointer hashing
        let this: *const T = self.0;
        this.hash(state)
    }
}

impl<'a, T> Deref for Interned<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct Interner<'ctx, T> {
    arena: Arena<T>,
    set: RefCell<HashSet<&'ctx T>>,
}

impl<'ctx, T> Default for Interner<'ctx, T> {
    fn default() -> Self {
        Self {
            arena: Default::default(),
            set: Default::default(),
        }
    }
}

impl<'ctx, T> Interner<'ctx, T>
where
    T: Eq + Hash,
{
    fn alloc(&'ctx self, v: T) -> Interned<'ctx, T> {
        let mut set = self.set.borrow_mut();
        match set.get(&v) {
            Some(r) => Interned(*r),
            None => {
                let r: &'ctx T = self.arena.alloc(v);
                assert!(set.insert(r));
                Interned(r)
            }
        }
    }
}

#[derive(Default)]
pub struct Context<'ctx> {
    nonterminals: Interner<'ctx, NonterminalData>,
    rules: Interner<'ctx, RuleData<'ctx>>,
}

impl<'ctx> Context<'ctx> {
    fn mk_nonterminal<S: Into<String>>(&'ctx self, name: S) -> Nonterminal<'ctx> {
        self.nonterminals
            .alloc(NonterminalData { name: name.into() })
    }

    fn mk_rule<V: Into<Vec<Symbol<'ctx>>>>(
        &'ctx self,
        head: Nonterminal<'ctx>,
        body: V,
    ) -> Rule<'ctx> {
        self.rules.alloc(RuleData {
            head,
            body: body.into(),
        })
    }

    fn rules(&'ctx self) -> HashMap<Nonterminal<'ctx>, HashSet<Rule<'ctx>>> {
        let mut ret = HashMap::new();
        self.rules.set.borrow().iter().for_each(|rule| {
            ret.entry(rule.head)
                .or_insert_with(HashSet::new)
                .insert(Interned(*rule));
        });
        ret
    }
}

pub type Nonterminal<'ctx> = Interned<'ctx, NonterminalData>;
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonterminalData {
    name: String,
}

impl fmt::Display for NonterminalData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Symbol<'ctx> {
    Terminal(&'ctx str),
    Nonterminal(Nonterminal<'ctx>),
}

impl<'ctx> fmt::Display for Symbol<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Symbol::*;

        match self {
            Terminal(c) => c.fmt(f),
            Nonterminal(n) => n.fmt(f),
        }
    }
}

pub type Rule<'ctx> = Interned<'ctx, RuleData<'ctx>>;
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuleData<'ctx> {
    head: Nonterminal<'ctx>,
    body: Vec<Symbol<'ctx>>,
}

impl<'ctx> fmt::Display for RuleData<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> ", self.head)?;
        for sym in self.body.iter() {
            write!(f, "{}", sym)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Item<'ctx> {
    rule: Rule<'ctx>,
    // position of the dot
    dot: usize,
    // original position in the input
    origin: usize,
}

impl<'ctx> fmt::Display for Item<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} -> ", self.origin, self.rule.head)?;
        for symbol in self.rule.body[..self.dot].iter() {
            write!(f, "{}", symbol)?;
        }
        write!(f, " . ")?;
        for symbol in self.rule.body[self.dot..].iter() {
            write!(f, "{}", symbol)?;
        }

        Ok(())
    }
}

fn recognize<'ctx>(input: &str, ctx: &'ctx Context<'ctx>, start: Nonterminal<'ctx>) -> bool {
    let rules = ctx.rules();
    let mut items = vec![HashSet::new(); input.len() + 1];
    for &rule in rules.get(&start).unwrap() {
        let item = Item {
            rule,
            dot: 0,
            origin: 0,
        };
        items[0].insert(item);
    }

    for i in 0..items.len() {
        loop {
            let mut new_items = vec![HashSet::new(); input.len() + 1];

            for &item in items[i].iter() {
                assert!(item.dot <= item.rule.body.len());

                match item.rule.body.get(item.dot) {
                    // completion
                    None => {
                        for prev_item in items[item.origin].iter().filter(|prev_item| {
                            prev_item.dot < prev_item.rule.body.len()
                                && prev_item.rule.body[prev_item.dot]
                                    == Symbol::Nonterminal(item.rule.head)
                        }) {
                            let new_item = Item {
                                rule: prev_item.rule,
                                dot: prev_item.dot + 1,
                                origin: prev_item.origin,
                            };
                            new_items[i].insert(new_item);
                        }
                    }
                    // scanning
                    Some(&Symbol::Terminal(s)) => {
                        if input[i..].starts_with(s) {
                            let new_item = Item {
                                rule: item.rule,
                                dot: item.dot + 1,
                                origin: item.origin,
                            };
                            new_items[i + s.len()].insert(new_item);
                        }
                    }
                    // prediction
                    Some(&Symbol::Nonterminal(n)) => {
                        for &rule in rules.get(&n).unwrap() {
                            let new_item = Item {
                                rule,
                                dot: 0,
                                origin: i,
                            };
                            new_items[i].insert(new_item);
                        }
                    }
                }
            }

            let mut inserted = false;
            for (old, new) in items.iter_mut().zip(new_items.into_iter()) {
                for item in new.into_iter() {
                    // if the item is not present in the old set, continue
                    inserted = old.insert(item) || inserted;
                }
            }

            if !inserted {
                break;
            }
        }
    }

    for (i, items) in items.iter().enumerate() {
        println!("{}:", i);
        for item in items.iter() {
            println!("{}", item);
        }
        println!();
    }

    items
        .last()
        .unwrap()
        .iter()
        .any(|item| item.origin == 0 && item.rule.head == start && item.dot == item.rule.body.len())
}

fn main() {
    let ctx = Context::default();

    let s = ctx.mk_nonterminal("S");
    // let r1 = ctx.mk_rule(s, [Symbol::Terminal("a")]);
    // let r2 = ctx.mk_rule(s, [Symbol::Terminal("b"), Symbol::Nonterminal(s)]);
    // println!("{}\n{}", r1, r2);

    ctx.mk_rule(s, []);
    ctx.mk_rule(
        s,
        [
            Symbol::Terminal("("),
            Symbol::Nonterminal(s),
            Symbol::Terminal(")"),
        ],
    );

    recognize("(())", &ctx, s);
}

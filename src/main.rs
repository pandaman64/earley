use either::Either;
use itertools::MultiProduct;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
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
            Terminal(c) => write!(f, "'{}'", c),
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

impl<'ctx> Item<'ctx> {
    fn nonterminal(self) -> Nonterminal<'ctx> {
        self.rule.head
    }

    fn is_completed(self) -> bool {
        self.dot == self.rule.body.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NonterminalAt<'ctx> {
    nonterminal: Nonterminal<'ctx>,
    start: usize,
    end: usize,
}

impl<'ctx> fmt::Display for NonterminalAt<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {}, {}]", self.nonterminal, self.start, self.end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeSymbol<'ctx> {
    Terminal(&'ctx str),
    Nonterminal(NonterminalAt<'ctx>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<'ctx> {
    symbol: NodeSymbol<'ctx>,
    children: Vec<Node<'ctx>>,
}

impl<'ctx> fmt::Display for Node<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt(f, 0)
    }
}

fn indent(f: &mut fmt::Formatter<'_>, level: usize) -> fmt::Result {
    for _ in 0..level {
        write!(f, "  ")?;
    }

    Ok(())
}

impl<'ctx> Node<'ctx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>, level: usize) -> fmt::Result {
        // TODO: Terminal should carry position information
        indent(f, level)?;
        match self.symbol {
            NodeSymbol::Terminal(t) => write!(f, "{}", t)?,
            NodeSymbol::Nonterminal(n) => write!(f, "{}@{}..{}", n.nonterminal, n.start, n.end)?,
        }

        for child in self.children.iter() {
            writeln!(f)?;
            indent(f, level)?;
            child.fmt(f, level + 1)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Derivations<'ctx> {
    graph: HashMap<NonterminalAt<'ctx>, HashSet<Vec<NodeSymbol<'ctx>>>>,
}

impl<'ctx> Derivations<'ctx> {
    fn iter_at(&'ctx self, nonterminal: NonterminalAt<'ctx>) -> Iter<'ctx> {
        let derivations = self.graph.get(&nonterminal).unwrap();
        Iter {
            parent: self,
            nonterminal,
            children: None,
            derivations: derivations.iter(),
        }
    }
}

/// Note that this iterator causes a stack overflow when there is a cycle in a derivation
#[derive(Clone)]
pub struct Iter<'ctx> {
    parent: &'ctx Derivations<'ctx>,
    nonterminal: NonterminalAt<'ctx>,
    // the iterator of all permutations of the children for the current derivation
    children: Option<MultiProduct<Either<std::iter::Once<Node<'ctx>>, Iter<'ctx>>>>,
    // the remaining derivations
    derivations: std::collections::hash_set::Iter<'ctx, Vec<NodeSymbol<'ctx>>>,
}

impl<'ctx> Iterator for Iter<'ctx> {
    type Item = Node<'ctx>;

    fn next(&mut self) -> Option<Self::Item> {
        use itertools::Itertools;
        loop {
            if let Some(children) = self.children.as_mut() {
                if let Some(children) = children.next() {
                    return Some(Node {
                        symbol: NodeSymbol::Nonterminal(self.nonterminal),
                        children,
                    });
                }
            }

            let current_derivation = self.derivations.next()?;
            self.children = Some(
                current_derivation
                    .iter()
                    .rev()
                    .map(|symbol| match *symbol {
                        NodeSymbol::Terminal(t) => Either::Left(std::iter::once(Node {
                            symbol: NodeSymbol::Terminal(t),
                            children: vec![],
                        })),
                        NodeSymbol::Nonterminal(n) => Either::Right(self.parent.iter_at(n)),
                    })
                    .multi_cartesian_product(),
            );
        }
    }
}

fn construct_graph<'ctx>(
    graph: &mut HashMap<NonterminalAt<'ctx>, HashSet<Vec<NodeSymbol<'ctx>>>>,
    items: &[HashSet<Item<'ctx>>],
    target: NonterminalAt<'ctx>,
    input: &str,
) -> bool {
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct State<'ctx> {
        end: usize,
        symbols: Vec<NodeSymbol<'ctx>>,
    }

    let derivations = match graph.get(&target) {
        Some(_) => return true,
        None => {
            println!("target = {}", target);
            // mark targed visited
            graph.insert(target, HashSet::new());

            let mut derivations = HashSet::new();

            for &completed in items[target.end].iter().filter(|item| {
                item.nonterminal() == target.nonterminal
                    && item.is_completed()
                    && target.start <= item.origin
            }) {
                let mut states = {
                    let mut states = HashSet::new();
                    states.insert(State {
                        end: target.end,
                        symbols: vec![],
                    });
                    states
                };

                for child in completed.rule.body.iter().rev() {
                    if states.is_empty() {
                        break;
                    }

                    match *child {
                        Symbol::Nonterminal(n) => {
                            let mut new_states = HashSet::new();
                            for state in states.into_iter() {
                                for item in items[state.end].iter().filter(|item| {
                                    item.nonterminal() == n
                                        && item.is_completed()
                                        && target.start <= item.origin
                                }) {
                                    let child_target = NonterminalAt {
                                        nonterminal: n,
                                        start: item.origin,
                                        end: state.end,
                                    };

                                    if construct_graph(graph, items, child_target, input) {
                                        let mut symbols = state.symbols.clone();
                                        symbols.push(NodeSymbol::Nonterminal(child_target));
                                        new_states.insert(State {
                                            end: item.origin,
                                            symbols,
                                        });
                                    }
                                }
                            }
                            states = new_states;
                        }
                        Symbol::Terminal(t) => {
                            states = states
                                .into_iter()
                                .filter_map(|mut state| {
                                    if input.get((state.end - t.len())..state.end) == Some(t) {
                                        state.end -= t.len();
                                        state.symbols.push(NodeSymbol::Terminal(t));
                                        Some(state)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                        }
                    }
                }

                for state in states.into_iter().filter(|state| state.end == target.start) {
                    derivations.insert(state.symbols);
                }
            }

            derivations
        }
    };

    if derivations.is_empty() {
        // we marked target visited by inserting a dummy vec
        // we remove the dummy here
        graph.remove(&target);
        false
    } else {
        graph.insert(target, derivations);
        true
    }
}

fn parse<'ctx>(
    input: &str,
    ctx: &'ctx Context<'ctx>,
    start: Nonterminal<'ctx>,
) -> Option<Derivations<'ctx>> {
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
        let mut queue: VecDeque<_> = items[i].iter().cloned().collect();

        while let Some(item) = queue.pop_front() {
            assert!(item.dot <= item.rule.body.len());

            match item.rule.body.get(item.dot) {
                // completion
                None => {
                    let mut delta = vec![];

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
                        if !items[i].contains(&new_item) {
                            // Since it's impossible to mutably borrow items[i] here,
                            // we instead add the new_item to the temporary delta vec.
                            // items[i].insert(new_item);
                            delta.push(new_item);

                            queue.push_back(new_item);
                        }
                    }

                    items[i].extend(delta);
                }
                // scanning
                Some(&Symbol::Terminal(s)) => {
                    if input[i..].starts_with(s) {
                        let new_item = Item {
                            rule: item.rule,
                            dot: item.dot + 1,
                            origin: item.origin,
                        };
                        if !items[i + s.len()].contains(&new_item) {
                            items[i + s.len()].insert(new_item);
                        }
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
                        if !items[i].contains(&new_item) {
                            items[i].insert(new_item);
                            queue.push_back(new_item);
                        }
                    }
                }
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

    if items
        .last()
        .unwrap()
        .iter()
        .any(|item| item.origin == 0 && item.rule.head == start && item.dot == item.rule.body.len())
    {
        let mut graph = HashMap::new();
        let recognize = construct_graph(
            &mut graph,
            &items,
            NonterminalAt {
                nonterminal: start,
                start: 0,
                end: input.len(),
            },
            input,
        );

        for (node, derivations) in graph.iter() {
            println!("{}:", node);
            for derivation in derivations.iter() {
                print!("derivation = ");
                let mut start = node.start;
                for symbol in derivation.iter().rev() {
                    match *symbol {
                        NodeSymbol::Nonterminal(n) => {
                            print!("{}", n);

                            assert_eq!(start, n.start);
                            start = n.end;
                        }
                        NodeSymbol::Terminal(t) => {
                            print!("['{}', {}, {}]", t, start, start + t.len());

                            start += t.len();
                        }
                    }
                }
                println!();
            }
            println!();
        }

        assert!(recognize);
        Some(Derivations { graph })
    } else {
        None
    }
}

fn main() {
    let ctx = Context::default();
    let a = ctx.mk_nonterminal("A");
    let b = ctx.mk_nonterminal("B");
    let c = ctx.mk_nonterminal("C");
    ctx.mk_rule(a, [Symbol::Nonterminal(b), Symbol::Nonterminal(c)]);
    ctx.mk_rule(b, [Symbol::Terminal("b")]);
    ctx.mk_rule(c, [Symbol::Terminal("c")]);
    let derivations = parse("bc", &ctx, a).unwrap();
    for tree in derivations.iter_at(NonterminalAt {
        nonterminal: a,
        start: 0,
        end: 2,
    }) {
        println!("{}", tree);
    }

    let ctx = Context::default();
    let a = ctx.mk_nonterminal("A");
    ctx.mk_rule(a, [Symbol::Terminal("a")]);
    ctx.mk_rule(a, [Symbol::Nonterminal(a), Symbol::Nonterminal(a)]);
    ctx.mk_rule(a, []);
    parse("aa", &ctx, a).unwrap();
}

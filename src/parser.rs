//! Data-oriented arena parser.
//!
//! Consumes lexer tokens linearly and stores tree relationships in a flat
//! `[AstNode; AST_CAP]` arena. Child linkages use `u32` indices only —
//! never boxed pointer trees.
//!
//! Arena exhaustion returns [`ParserError::ArenaOverflow`] — never panics.

use crate::lexer::{Lexer, LexerError, Token, TokenKind, MAX_TOKENS};

/// Fixed arena capacity (power-of-two friendly, fits L2 comfortably).
pub const AST_CAP: usize = 1024;

/// Sentinel index meaning "no child / no node".
pub const NIL: u32 = u32::MAX;

/// Defensive parse failure modes.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ParserError {
    /// `[AstNode; 1024]` capacity exhausted.
    ArenaOverflow = 0,
    /// Unexpected / missing token in the pipeline grammar.
    UnexpectedToken = 1,
    /// Lexer reported malformed UTF-8 / torn Tamil syllable.
    LexMalformedUtf8 = 2,
    /// Lexer token window overflow.
    LexTokenBufferFull = 3,
    /// Empty or non-pipeline input.
    EmptyInput = 4,
    /// Pipeline stage appeared before an `இருந்து` (Irundu) source was registered.
    MissingSourceContext = 5,
}

/// AST / pipeline operator kind (alias for [`NodeKind`]; includes `Join`).
pub type OpKind = NodeKind;

/// Compatibility aliases.
pub type ParseError = ParserError;
/// Historical name retained for call-site clarity.
pub const ARENA_FULL: ParserError = ParserError::ArenaOverflow;

impl ParserError {
    #[inline(always)]
    pub fn from_lexer(err: LexerError) -> Self {
        match err {
            LexerError::MalformedUtf8(_) => ParserError::LexMalformedUtf8,
            LexerError::TokenBufferFull => ParserError::LexTokenBufferFull,
        }
    }
}

/// AST node discriminant. Packed `u8` for cache-line density.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NodeKind {
    /// Root pipeline container
    Pipeline = 0,
    /// இருந்து <relation>
    From = 1,
    /// வடி <col> <op> <lit>
    Filter = 2,
    /// கணி <col> = <expr>
    Derive = 3,
    /// அடுக்கு <col>
    Sort = 4,
    /// எடு <n>
    Take = 5,
    /// தொகுப்பு <col>
    Group = 6,
    /// சுருக்கு <fn>(<col>)
    Aggregate = 7,
    /// இணை <relation> எங்கே <pred>
    Join = 8,
    /// தேடு <cols...>
    Project = 9,
    /// Column / relation identifier reference
    Ident = 10,
    /// Integer literal
    Literal = 11,
    /// Comparison / binary op node
    BinOp = 12,
    /// Column list (linked via left/right/next)
    ColumnList = 13,
    /// Empty / error
    Empty = 14,
}

/// Flat AST node — packed to exactly one half-cache-line (32 B) with
/// `align(32)` so adjacent nodes never share a false-sharing boundary.
/// Field order places `value: i64` at offset 24 to avoid internal padding.
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub struct AstNode {
    pub kind: NodeKind,
    pub op: TokenKind,
    pub _pad: [u8; 2],
    /// Source token span start (byte offset).
    pub start: u32,
    /// Source token span end.
    pub end: u32,
    /// First child / left operand index.
    pub left: u32,
    /// Second child / right operand index.
    pub right: u32,
    /// Next sibling in a list (column lists, pipeline stages).
    pub next: u32,
    /// Integer payload (limit, literal value) — last for 8-byte align at off 24.
    pub value: i64,
}

const _: () = assert!(core::mem::size_of::<AstNode>() == 32);
const _: () = assert!(core::mem::align_of::<AstNode>() == 32);

impl AstNode {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            kind: NodeKind::Empty,
            op: TokenKind::Eof,
            _pad: [0; 2],
            start: 0,
            end: 0,
            left: NIL,
            right: NIL,
            next: NIL,
            value: 0,
        }
    }
}

/// Fixed-capacity arena holding the entire query AST (64-byte aligned base).
#[repr(C, align(64))]
pub struct AstArena {
    pub nodes: [AstNode; AST_CAP],
    pub len: u32,
    pub root: u32,
}

impl AstArena {
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            nodes: [AstNode::empty(); AST_CAP],
            len: 0,
            root: NIL,
        }
    }

    /// Allocate a node. Returns [`ParserError::ArenaOverflow`] at capacity — no OOB.
    #[inline(always)]
    pub fn try_alloc(&mut self, node: AstNode) -> Result<u32, ParserError> {
        let i = self.len as usize;
        if i >= AST_CAP {
            return Err(ParserError::ArenaOverflow);
        }
        self.nodes[i] = node;
        self.len = self.len.wrapping_add(1);
        Ok(i as u32)
    }

    #[inline(always)]
    pub fn alloc(&mut self, node: AstNode) -> Option<u32> {
        self.try_alloc(node).ok()
    }

    #[inline(always)]
    pub fn get(&self, id: u32) -> Option<&AstNode> {
        let i = id as usize;
        let in_range = (id != NIL) & (i < self.len as usize);
        if in_range {
            Some(&self.nodes[i])
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, id: u32) -> Option<&mut AstNode> {
        let i = id as usize;
        let in_range = (id != NIL) & (i < self.len as usize);
        if in_range {
            Some(&mut self.nodes[i])
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        (self.len as usize) >= AST_CAP
    }
}

/// Linear recursive-descent-free parser over a fixed token window.
#[repr(C)]
pub struct Parser<'a> {
    src: &'a [u8],
    tokens: [Token; MAX_TOKENS],
    tok_len: usize,
    cursor: usize,
}

impl<'a> Parser<'a> {
    pub fn try_from_source(src: &'a [u8]) -> Result<Self, ParserError> {
        let mut lexer = Lexer::new(src);
        let mut tokens = [Token::eof(); MAX_TOKENS];
        let tok_len = lexer.tokenize_into(&mut tokens).map_err(ParserError::from_lexer)?;
        Ok(Self {
            src,
            tokens,
            tok_len,
            cursor: 0,
        })
    }

    /// Backward-compatible constructor; empty token stream on lex failure.
    pub fn from_source(src: &'a [u8]) -> Self {
        match Self::try_from_source(src) {
            Ok(p) => p,
            Err(_) => Self {
                src,
                tokens: [Token::eof(); MAX_TOKENS],
                tok_len: 1,
                cursor: 0,
            },
        }
    }

    #[inline(always)]
    fn peek(&self) -> Token {
        if self.cursor < self.tok_len {
            self.tokens[self.cursor]
        } else {
            Token::eof()
        }
    }

    #[inline(always)]
    fn bump(&mut self) -> Token {
        let t = self.peek();
        let not_eof = t.kind as u8 != TokenKind::Eof as u8;
        self.cursor += not_eof as usize;
        t
    }

    #[inline(always)]
    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParserError> {
        let t = self.peek();
        if t.kind as u8 == kind as u8 {
            Ok(self.bump())
        } else {
            Err(ParserError::UnexpectedToken)
        }
    }

    #[inline(always)]
    fn alloc_ident(&mut self, arena: &mut AstArena, tok: Token) -> Result<u32, ParserError> {
        arena.try_alloc(AstNode {
            kind: NodeKind::Ident,
            op: TokenKind::Ident,
            _pad: [0; 2],
            start: tok.start,
            end: tok.end,
            value: 0,
            left: NIL,
            right: NIL,
            next: NIL,
        })
    }

    #[inline(always)]
    fn alloc_literal(&mut self, arena: &mut AstArena, tok: Token) -> Result<u32, ParserError> {
        arena.try_alloc(AstNode {
            kind: NodeKind::Literal,
            op: TokenKind::Number,
            _pad: [0; 2],
            start: tok.start,
            end: tok.end,
            value: tok.number,
            left: NIL,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_column_list(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        let first_tok = self.expect(TokenKind::Ident)?;
        let head = self.alloc_ident(arena, first_tok)?;
        let mut tail = head;
        loop {
            if self.peek().kind != TokenKind::Comma {
                break;
            }
            self.bump();
            let tok = self.expect(TokenKind::Ident)?;
            let id = self.alloc_ident(arena, tok)?;
            if let Some(node) = arena.get_mut(tail) {
                node.next = id;
            }
            tail = id;
        }
        if let Some(node) = arena.get_mut(head) {
            node.kind = NodeKind::ColumnList;
        }
        Ok(head)
    }

    fn parse_filter(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Vadi)?;
        let col_tok = self.expect(TokenKind::Ident)?;
        let col = self.alloc_ident(arena, col_tok)?;
        let op_tok = self.bump();
        let op_ok = matches!(
            op_tok.kind,
            TokenKind::Gt | TokenKind::Lt | TokenKind::Eq
        );
        if !op_ok {
            return Err(ParserError::UnexpectedToken);
        }
        let lit_tok = self.expect(TokenKind::Number)?;
        let lit = self.alloc_literal(arena, lit_tok)?;
        let bin = arena.try_alloc(AstNode {
            kind: NodeKind::BinOp,
            op: op_tok.kind,
            _pad: [0; 2],
            start: op_tok.start,
            end: op_tok.end,
            value: 0,
            left: col,
            right: lit,
            next: NIL,
        })?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Filter,
            op: TokenKind::Vadi,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: bin,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_sort(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Adukku)?;
        let col_tok = self.expect(TokenKind::Ident)?;
        let col = self.alloc_ident(arena, col_tok)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Sort,
            op: TokenKind::Adukku,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: col,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_take(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Edu)?;
        let n_tok = self.expect(TokenKind::Number)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Take,
            op: TokenKind::Edu,
            _pad: [0; 2],
            start: n_tok.start,
            end: n_tok.end,
            value: n_tok.number,
            left: NIL,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_project(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Thedu)?;
        let cols = self.parse_column_list(arena)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Project,
            op: TokenKind::Thedu,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: cols,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_derive(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Kani)?;
        let col_tok = self.expect(TokenKind::Ident)?;
        let col = self.alloc_ident(arena, col_tok)?;
        self.expect(TokenKind::Eq)?;
        let lit_tok = self.expect(TokenKind::Number)?;
        let lit = self.alloc_literal(arena, lit_tok)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Derive,
            op: TokenKind::Kani,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: col,
            right: lit,
            next: NIL,
        })
    }

    fn parse_group(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Thoguppu)?;
        let col_tok = self.expect(TokenKind::Ident)?;
        let col = self.alloc_ident(arena, col_tok)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Group,
            op: TokenKind::Thoguppu,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: col,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_aggregate(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Surukku)?;
        let col_tok = self.expect(TokenKind::Ident)?;
        let col = self.alloc_ident(arena, col_tok)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::Aggregate,
            op: TokenKind::Surukku,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: col,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_join(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Inai)?;
        let rel_tok = self.expect(TokenKind::Ident)?;
        let rel = self.alloc_ident(arena, rel_tok)?;
        let mut pred = NIL;
        if self.peek().kind == TokenKind::Enge {
            self.bump();
            let left_tok = self.expect(TokenKind::Ident)?;
            let left = self.alloc_ident(arena, left_tok)?;
            let op_tok = self.bump();
            let right_tok = self.expect(TokenKind::Ident)?;
            let right = self.alloc_ident(arena, right_tok)?;
            pred = arena.try_alloc(AstNode {
                kind: NodeKind::BinOp,
                op: op_tok.kind,
                _pad: [0; 2],
                start: op_tok.start,
                end: op_tok.end,
                value: 0,
                left,
                right,
                next: NIL,
            })?;
        }
        arena.try_alloc(AstNode {
            kind: NodeKind::Join,
            op: TokenKind::Inai,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: rel,
            right: pred,
            next: NIL,
        })
    }

    fn parse_from(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        self.expect(TokenKind::Irundu)?;
        let rel_tok = self.expect(TokenKind::Ident)?;
        let rel = self.alloc_ident(arena, rel_tok)?;
        arena.try_alloc(AstNode {
            kind: NodeKind::From,
            op: TokenKind::Irundu,
            _pad: [0; 2],
            start: rel_tok.start,
            end: rel_tok.end,
            value: 0,
            left: rel,
            right: NIL,
            next: NIL,
        })
    }

    fn parse_stage(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        match self.peek().kind {
            TokenKind::Irundu => self.parse_from(arena),
            TokenKind::Vadi => self.parse_filter(arena),
            TokenKind::Kani => self.parse_derive(arena),
            TokenKind::Adukku => self.parse_sort(arena),
            TokenKind::Edu => self.parse_take(arena),
            TokenKind::Thoguppu => self.parse_group(arena),
            TokenKind::Surukku => self.parse_aggregate(arena),
            TokenKind::Inai => self.parse_join(arena),
            TokenKind::Thedu => self.parse_project(arena),
            TokenKind::Eof => Err(ParserError::EmptyInput),
            _ => Err(ParserError::UnexpectedToken),
        }
    }

    /// Parse a full pipeline into `arena`. Returns the root node index or a
    /// defensive [`ParserError`] (including arena overflow / missing source).
    #[inline(always)]
    pub fn parse_pipeline(&mut self, arena: &mut AstArena) -> Result<u32, ParserError> {
        // Mandate B: first stage MUST be இருந்து (Irundu) source registration.
        match self.peek().kind {
            TokenKind::Irundu => {}
            TokenKind::Eof => return Err(ParserError::EmptyInput),
            _ => return Err(ParserError::MissingSourceContext),
        }
        let first = self.parse_stage(arena)?;
        let root = arena.try_alloc(AstNode {
            kind: NodeKind::Pipeline,
            op: TokenKind::Eof,
            _pad: [0; 2],
            start: 0,
            end: 0,
            value: 0,
            left: first,
            right: NIL,
            next: NIL,
        })?;
        let mut prev = first;
        loop {
            let kind = self.peek().kind;
            if kind == TokenKind::Pipe {
                self.bump();
                let next_kind = self.peek().kind;
                if next_kind == TokenKind::Eof {
                    return Err(ParserError::UnexpectedToken);
                }
                let stage = self.parse_stage(arena)?;
                if let Some(node) = arena.get_mut(prev) {
                    node.next = stage;
                }
                prev = stage;
                continue;
            }
            if kind == TokenKind::Semi {
                self.bump();
                break;
            }
            if kind == TokenKind::Eof {
                break;
            }
            return Err(ParserError::UnexpectedToken);
        }
        arena.root = root;
        Ok(root)
    }

    #[inline(always)]
    pub fn source(&self) -> &'a [u8] {
        self.src
    }
}

/// Lex + parse a query into `arena`, surfacing arena / UTF-8 faults explicitly.
pub fn parse_query(src: &[u8], arena: &mut AstArena) -> Result<u32, ParserError> {
    let mut parser = Parser::try_from_source(src)?;
    parser.parse_pipeline(arena)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_demo_pipeline() {
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut arena = AstArena::new();
        let root = parse_query(q.as_bytes(), &mut arena).expect("parse");
        let pipe = arena.get(root).unwrap();
        assert_eq!(pipe.kind, NodeKind::Pipeline);
        let mut stage = pipe.left;
        let mut kinds = [NodeKind::Empty; 8];
        let mut n = 0usize;
        while stage != NIL && n < 8 {
            kinds[n] = arena.get(stage).unwrap().kind;
            stage = arena.get(stage).unwrap().next;
            n += 1;
        }
        assert_eq!(kinds[0], NodeKind::From);
        assert_eq!(kinds[1], NodeKind::Filter);
        assert_eq!(kinds[2], NodeKind::Sort);
        assert_eq!(kinds[3], NodeKind::Take);
        assert_eq!(kinds[4], NodeKind::Project);
        assert_eq!(n, 5);
    }

    #[test]
    fn arena_overflow_returns_defensive_error() {
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut arena = AstArena::new();
        // Saturate the flat structural boundary before parsing.
        arena.len = AST_CAP as u32;
        let err = parse_query(q.as_bytes(), &mut arena).expect_err("must overflow");
        assert_eq!(err, ParserError::ArenaOverflow);
        // No panic / no OOB — len stays at capacity.
        assert!(arena.is_full());
    }

    #[test]
    fn missing_source_context_without_irundu() {
        let q = "வடி வயது > 21 | எடு 10;";
        let mut arena = AstArena::new();
        let err = parse_query(q.as_bytes(), &mut arena).expect_err("need source");
        assert_eq!(err, ParserError::MissingSourceContext);
    }
}

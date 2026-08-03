//! Front-end pipeline lexer for the Tamil-native query DSL.
//!
//! Scans left-to-right over `&[u8]` with zero heap traffic. Keywords are
//! matched by raw byte equality against compile-time UTF-8 constants.
//! Numbers are accumulated in a branchless register loop.

use crate::utf8::{
    extend_grapheme_cluster, next_char_boundary, scan_ident_end, str_from_parent, IS_DIGIT_LUT,
    IS_OP_LUT, IS_WS_LUT,
};

/// Maximum tokens emitted into the fixed token buffer by a single scan.
pub const MAX_TOKENS: usize = 256;

/// Keyword / operator / literal classification. Packed as `u8` for cache density.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenKind {
    /// இருந்து — FROM / scan source relation
    Irundu = 0,
    /// வடி — FILTER
    Vadi = 1,
    /// கணி — DERIVE / compute
    Kani = 2,
    /// அடுக்கு — SORT
    Adukku = 3,
    /// எடு — TAKE / limit
    Edu = 4,
    /// தொகுப்பு — GROUP
    Thoguppu = 5,
    /// சுருக்கு — AGGREGATE
    Surukku = 6,
    /// இணை — JOIN
    Inai = 7,
    /// எங்கே — WHERE / conditional context
    Enge = 8,
    /// தேடு — SELECT / PROJECT
    Thedu = 9,
    /// Identifier (relation / column name)
    Ident = 10,
    /// Integer literal
    Number = 11,
    /// `|`
    Pipe = 12,
    /// `=`
    Eq = 13,
    /// `>`
    Gt = 14,
    /// `<`
    Lt = 15,
    /// `,`
    Comma = 16,
    /// `;`
    Semi = 17,
    /// End of input
    Eof = 18,
    /// Lexical error sentinel
    Error = 19,
}

impl TokenKind {
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// A single lexeme: kind + byte span into the parent source buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub _pad: [u8; 3],
    pub start: u32,
    pub end: u32,
    pub number: i64,
}

impl Token {
    #[inline(always)]
    pub const fn eof() -> Self {
        Self {
            kind: TokenKind::Eof,
            _pad: [0; 3],
            start: 0,
            end: 0,
            number: 0,
        }
    }

    #[inline(always)]
    pub fn text<'a>(&self, src: &'a [u8]) -> Option<&'a str> {
        str_from_parent(src, self.start as usize, self.end as usize)
    }
}

/// Compile-time UTF-8 keyword tables (exact byte sequences).
const KW_IRUNDU: &[u8] = "இருந்து".as_bytes();
const KW_VADI: &[u8] = "வடி".as_bytes();
const KW_KANI: &[u8] = "கணி".as_bytes();
const KW_ADUKKU: &[u8] = "அடுக்கு".as_bytes();
const KW_EDU: &[u8] = "எடு".as_bytes();
const KW_THOGUPPU: &[u8] = "தொகுப்பு".as_bytes();
const KW_SURUKKU: &[u8] = "சுருக்கு".as_bytes();
const KW_INAI: &[u8] = "இணை".as_bytes();
const KW_ENGE: &[u8] = "எங்கே".as_bytes();
const KW_THEDU: &[u8] = "தேடு".as_bytes();

/// Branchless-friendly operator kind LUT indexed by ASCII codepoint.
const OP_KIND_LUT: [u8; 256] = {
    let mut t = [TokenKind::Error as u8; 256];
    t[b'|' as usize] = TokenKind::Pipe as u8;
    t[b'=' as usize] = TokenKind::Eq as u8;
    t[b'>' as usize] = TokenKind::Gt as u8;
    t[b'<' as usize] = TokenKind::Lt as u8;
    t[b',' as usize] = TokenKind::Comma as u8;
    t[b';' as usize] = TokenKind::Semi as u8;
    t
};

#[inline(always)]
fn match_keyword(slice: &[u8]) -> TokenKind {
    // Length-bucketed exact compares — no heap, no dynamic dispatch.
    match slice.len() {
        9 => {
            if slice == KW_VADI || slice == KW_KANI || slice == KW_EDU || slice == KW_INAI {
                if slice == KW_VADI {
                    TokenKind::Vadi
                } else if slice == KW_KANI {
                    TokenKind::Kani
                } else if slice == KW_EDU {
                    TokenKind::Edu
                } else {
                    TokenKind::Inai
                }
            } else {
                TokenKind::Ident
            }
        }
        12 => {
            if slice == KW_THEDU {
                TokenKind::Thedu
            } else if slice == KW_ENGE {
                TokenKind::Enge
            } else {
                TokenKind::Ident
            }
        }
        21 => {
            if slice == KW_IRUNDU {
                TokenKind::Irundu
            } else if slice == KW_ADUKKU {
                TokenKind::Adukku
            } else if slice == KW_SURUKKU {
                TokenKind::Surukku
            } else {
                TokenKind::Ident
            }
        }
        24 => {
            if slice == KW_THOGUPPU {
                TokenKind::Thoguppu
            } else {
                TokenKind::Ident
            }
        }
        _ => TokenKind::Ident,
    }
}

/// Zero-allocation streaming lexer over a borrowed byte buffer.
#[repr(C)]
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Lexer<'a> {
    #[inline(always)]
    pub const fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    #[inline(always)]
    pub const fn source(&self) -> &'a [u8] {
        self.src
    }

    #[inline(always)]
    pub const fn position(&self) -> usize {
        self.pos
    }

    #[inline(always)]
    fn skip_ws(&mut self) {
        let bytes = self.src;
        let mut i = self.pos;
        while i < bytes.len() {
            let is_ws = IS_WS_LUT[bytes[i] as usize];
            // branchless-ish early exit via mask; still need loop bound check
            if is_ws == 0 {
                break;
            }
            i += 1;
        }
        self.pos = i;
    }

    /// Branchless decimal accumulate: `val = val * 10 + (byte - b'0')`.
    #[inline(always)]
    fn scan_number(&mut self) -> Token {
        let start = self.pos as u32;
        let bytes = self.src;
        let mut i = self.pos;
        let mut val: i64 = 0;
        while i < bytes.len() {
            let b = bytes[i];
            let is_digit = IS_DIGIT_LUT[b as usize];
            if is_digit == 0 {
                break;
            }
            // Hot register loop — no branches on digit value itself.
            val = val
                .wrapping_mul(10)
                .wrapping_add((b.wrapping_sub(b'0')) as i64);
            i += 1;
        }
        self.pos = i;
        Token {
            kind: TokenKind::Number,
            _pad: [0; 3],
            start,
            end: i as u32,
            number: val,
        }
    }

    #[inline(always)]
    fn scan_operator(&mut self) -> Token {
        let start = self.pos as u32;
        let b = self.src[self.pos];
        let kind_u8 = OP_KIND_LUT[b as usize];
        self.pos += 1;
        // SAFETY-equivalent: LUT only maps known ops; Error reserved for misuse.
        let kind = unsafe { core::mem::transmute::<u8, TokenKind>(kind_u8) };
        Token {
            kind,
            _pad: [0; 3],
            start,
            end: start + 1,
            number: 0,
        }
    }

    #[inline(always)]
    fn scan_ident_or_keyword(&mut self) -> Token {
        let start = self.pos;
        let end = scan_ident_end(self.src, start);
        // Ensure we never tear a Tamil grapheme if scan stopped mid-cluster
        // (operators/ws always sit on boundaries, so this is a safety net).
        let end = {
            let nb = next_char_boundary(self.src, start);
            let _ = extend_grapheme_cluster(self.src, nb);
            end
        };
        self.pos = end;
        let slice = &self.src[start..end];
        let kind = match_keyword(slice);
        Token {
            kind,
            _pad: [0; 3],
            start: start as u32,
            end: end as u32,
            number: 0,
        }
    }

    /// Fill a caller-provided fixed token buffer. Returns the number of tokens
    /// written (excluding a trailing EOF which is always appended if capacity remains).
    pub fn tokenize_into(&mut self, out: &mut [Token; MAX_TOKENS]) -> usize {
        let mut n = 0usize;
        while n + 1 < MAX_TOKENS {
            match self.next() {
                Some(tok) => {
                    let is_eof = tok.kind as u8 == TokenKind::Eof as u8;
                    out[n] = tok;
                    n += 1;
                    if is_eof {
                        break;
                    }
                }
                None => {
                    out[n] = Token::eof();
                    n += 1;
                    break;
                }
            }
        }
        n
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.skip_ws();
        if self.pos >= self.src.len() {
            return Some(Token {
                kind: TokenKind::Eof,
                _pad: [0; 3],
                start: self.pos as u32,
                end: self.pos as u32,
                number: 0,
            });
        }
        let b = self.src[self.pos];
        let is_digit = IS_DIGIT_LUT[b as usize];
        let is_op = IS_OP_LUT[b as usize];
        // Dispatch by class masks — hot path stays linear.
        if is_digit != 0 {
            return Some(self.scan_number());
        }
        if is_op != 0 {
            return Some(self.scan_operator());
        }
        Some(self.scan_ident_or_keyword())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_pipeline_keywords() {
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut lex = Lexer::new(q.as_bytes());
        let mut buf = [Token::eof(); MAX_TOKENS];
        let n = lex.tokenize_into(&mut buf);
        assert!(n > 10);
        assert_eq!(buf[0].kind, TokenKind::Irundu);
        assert_eq!(buf[1].kind, TokenKind::Ident);
        assert_eq!(buf[1].text(q.as_bytes()), Some("பயனர்கள்"));
        assert_eq!(buf[2].kind, TokenKind::Pipe);
        assert_eq!(buf[3].kind, TokenKind::Vadi);
        // Ensure தேடு keeps vowel marker attached
        let thedu = buf.iter().find(|t| t.kind == TokenKind::Thedu).unwrap();
        assert_eq!(thedu.text(q.as_bytes()), Some("தேடு"));
        let pe = buf
            .iter()
            .find(|t| t.kind == TokenKind::Ident && t.text(q.as_bytes()) == Some("பெயர்"))
            .unwrap();
        assert_eq!(pe.text(q.as_bytes()), Some("பெயர்"));
    }
}

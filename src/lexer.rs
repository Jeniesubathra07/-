//! Front-end pipeline lexer for the Tamil-native query DSL.
//!
//! Scans left-to-right over `&[u8]` with zero heap traffic. Keywords use
//! **maximal munch**: the longest UTF-8 identifier span is taken first, then
//! exact keyword equality is applied — so `"வடிவமைப்பு"` is `Ident`, never a
//! torn `"வடி"` keyword prefix.
//!
//! Mid-syllable / truncated UTF-8 returns
//! [`LexerError::MalformedUtf8`]`(cursor)` — never panics.

use crate::utf8::{
    checked_char_end, scan_ident_end_checked, str_from_parent, IS_DIGIT_LUT, IS_OP_LUT,
    Utf8ScanStatus, WHITESPACE_LUT,
};

/// Maximum tokens emitted into the fixed token buffer by a single scan.
/// Sized to allow stress pipelines that saturate the 1024-node AST arena.
pub const MAX_TOKENS: usize = 8192;

/// Defensive lexer failure modes (no panics on adversarial input).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub enum LexerError {
    /// Buffer ends mid-codepoint or mid-Tamil-syllable at byte `cursor`.
    MalformedUtf8(u32),
    /// Fixed token window exhausted.
    TokenBufferFull,
}

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
    /// `*` — multiply (derive math)
    Star = 20,
    /// `+` — add (derive math)
    Plus = 21,
    /// `-` — subtract (derive math)
    Minus = 22,
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
    pub const fn error(start: u32, end: u32) -> Self {
        Self {
            kind: TokenKind::Error,
            _pad: [0; 3],
            start,
            end,
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
    t[b'*' as usize] = TokenKind::Star as u8;
    t[b'+' as usize] = TokenKind::Plus as u8;
    t[b'-' as usize] = TokenKind::Minus as u8;
    t
};

/// Maximal-munch keyword classify: exact equality only after the full
/// grapheme-safe identifier span is locked. Prefixes never win.
#[inline(always)]
fn match_keyword_maximal(slice: &[u8]) -> TokenKind {
    match slice.len() {
        9 => {
            // வடி / கணி / எடு / இணை — all 9 UTF-8 bytes; exact compare only.
            let eq_vadi = slice == KW_VADI;
            let eq_kani = slice == KW_KANI;
            let eq_edu = slice == KW_EDU;
            let eq_inai = slice == KW_INAI;
            // Priority chain is exact-only; longer idents never enter this arm.
            if eq_vadi {
                TokenKind::Vadi
            } else if eq_kani {
                TokenKind::Kani
            } else if eq_edu {
                TokenKind::Edu
            } else if eq_inai {
                TokenKind::Inai
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

#[inline(always)]
fn map_utf8_err(status: Utf8ScanStatus, cursor: u32) -> LexerError {
    let _ = status;
    LexerError::MalformedUtf8(cursor)
}

/// Zero-width space UTF-8 lead detector (U+200B = E2 80 8B).
#[inline(always)]
fn zwsp_len(bytes: &[u8], i: usize) -> usize {
    let in_range = (i + 2) < bytes.len();
    let b0 = bytes.get(i).copied().unwrap_or(0);
    let b1 = bytes.get(i + 1).copied().unwrap_or(0);
    let b2 = bytes.get(i + 2).copied().unwrap_or(0);
    let hit = in_range & (b0 == 0xE2) & (b1 == 0x80) & (b2 == 0x8B);
    3usize.wrapping_mul(hit as usize)
}

/// Zero-allocation streaming lexer over a borrowed byte buffer.
#[repr(C)]
pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    /// Sticky error latched on first malformation (Iterator path).
    fault: Option<LexerError>,
}

impl<'a> Lexer<'a> {
    #[inline(always)]
    pub const fn new(src: &'a [u8]) -> Self {
        Self {
            src,
            pos: 0,
            fault: None,
        }
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
    pub const fn last_error(&self) -> Option<LexerError> {
        self.fault
    }

    /// Branchless ASCII whitespace + ZWSP strip via [`WHITESPACE_LUT`].
    #[inline(always)]
    fn skip_ws(&mut self) {
        let bytes = self.src;
        let mut i = self.pos;
        loop {
            if i >= bytes.len() {
                break;
            }
            let b = bytes[i];
            let ascii = WHITESPACE_LUT[b as usize] as usize;
            let zw = zwsp_len(bytes, i);
            // Prefer ASCII step (1) when set; else ZWSP step (3); else halt.
            let advance = if ascii != 0 {
                1usize
            } else if zw != 0 {
                zw
            } else {
                0usize
            };
            if advance == 0 {
                break;
            }
            i += advance;
        }
        self.pos = i;
    }

    /// Branchless decimal accumulate with overflow detection.
    ///
    /// On overflow, `number` is set to [`i64::MIN`] as a sentinel rejected by
    /// [`crate::parser::Parser::expect_number`]. `i64::MIN` itself is therefore
    /// not representable as a query literal (acceptable for this DSL).
    #[inline(always)]
    fn scan_number(&mut self) -> Token {
        let start = self.pos as u32;
        let bytes = self.src;
        let mut i = self.pos;
        let mut val: i64 = 0;
        let mut overflow = false;
        while i < bytes.len() {
            let b = bytes[i];
            let is_digit = IS_DIGIT_LUT[b as usize];
            if is_digit == 0 {
                break;
            }
            let digit = (b.wrapping_sub(b'0')) as i64;
            if !overflow {
                match val.checked_mul(10).and_then(|v| v.checked_add(digit)) {
                    Some(v) => val = v,
                    None => {
                        overflow = true;
                        val = i64::MIN;
                    }
                }
            }
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
        let kind = match kind_u8 {
            x if x == TokenKind::Pipe as u8 => TokenKind::Pipe,
            x if x == TokenKind::Eq as u8 => TokenKind::Eq,
            x if x == TokenKind::Gt as u8 => TokenKind::Gt,
            x if x == TokenKind::Lt as u8 => TokenKind::Lt,
            x if x == TokenKind::Comma as u8 => TokenKind::Comma,
            x if x == TokenKind::Semi as u8 => TokenKind::Semi,
            x if x == TokenKind::Star as u8 => TokenKind::Star,
            x if x == TokenKind::Plus as u8 => TokenKind::Plus,
            x if x == TokenKind::Minus as u8 => TokenKind::Minus,
            _ => TokenKind::Error,
        };
        Token {
            kind,
            _pad: [0; 3],
            start,
            end: start + 1,
            number: 0,
        }
    }

    #[inline(always)]
    fn scan_ident_or_keyword(&mut self) -> Result<Token, LexerError> {
        let start = self.pos;
        let cursor = start as u32;
        let _ = checked_char_end(self.src, start).map_err(|s| map_utf8_err(s, cursor))?;
        let end = scan_ident_end_checked(self.src, start).map_err(|s| map_utf8_err(s, cursor))?;
        self.pos = end;
        let slice = &self.src[start..end];
        if core::str::from_utf8(slice).is_err() {
            let err = LexerError::MalformedUtf8(cursor);
            self.fault = Some(err);
            return Err(err);
        }
        // Maximal munch: full span locked before keyword exact-match.
        let kind = match_keyword_maximal(slice);
        Ok(Token {
            kind,
            _pad: [0; 3],
            start: start as u32,
            end: end as u32,
            number: 0,
        })
    }

    /// Primary fallible scan step — never panics on torn syllables.
    #[inline(always)]
    pub fn next_token(&mut self) -> Result<Token, LexerError> {
        if let Some(err) = self.fault {
            return Err(err);
        }
        self.skip_ws();
        if self.pos >= self.src.len() {
            return Ok(Token {
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
        if is_digit != 0 {
            return Ok(self.scan_number());
        }
        if is_op != 0 {
            return Ok(self.scan_operator());
        }
        let cursor = self.pos as u32;
        match checked_char_end(self.src, self.pos) {
            Ok(_) => {}
            Err(status) => {
                let err = map_utf8_err(status, cursor);
                self.fault = Some(err);
                return Err(err);
            }
        }
        match self.scan_ident_or_keyword() {
            Ok(t) => Ok(t),
            Err(e) => {
                self.fault = Some(e);
                Err(e)
            }
        }
    }

    /// Fill a caller-provided fixed token buffer.
    #[inline(always)]
    pub fn tokenize_into(&mut self, out: &mut [Token; MAX_TOKENS]) -> Result<usize, LexerError> {
        let mut n = 0usize;
        while n + 1 < MAX_TOKENS {
            let tok = self.next_token()?;
            let is_eof = tok.kind as u8 == TokenKind::Eof as u8;
            out[n] = tok;
            n += 1;
            if is_eof {
                return Ok(n);
            }
        }
        Err(LexerError::TokenBufferFull)
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Ok(tok) => Some(tok),
            Err(err) => {
                let start = self.pos as u32;
                self.fault = Some(err);
                self.pos = self.pos.saturating_add(1).min(self.src.len());
                Some(Token::error(start, self.pos as u32))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_pipeline_keywords() {
        let q = "இருந்து பயனர்கள் | வடி வயது > 21 | அடுக்கு வயது | எடு 10 | தேடு பெயர், வயது;";
        let mut lex = Lexer::new(q.as_bytes());
        let mut buf = {
            use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};
            unsafe {
                let layout = Layout::new::<[Token; MAX_TOKENS]>();
                let ptr = alloc_zeroed(layout) as *mut [Token; MAX_TOKENS];
                if ptr.is_null() { handle_alloc_error(layout); }
                Box::from_raw(ptr)
            }
        };
        let n = lex.tokenize_into(&mut buf).expect("tokenize");
        assert!(n > 10);
        assert_eq!(buf[0].kind, TokenKind::Irundu);
        assert_eq!(buf[1].kind, TokenKind::Ident);
        assert_eq!(buf[1].text(q.as_bytes()), Some("பயனர்கள்"));
        assert_eq!(buf[2].kind, TokenKind::Pipe);
        assert_eq!(buf[3].kind, TokenKind::Vadi);
        let thedu = buf.iter().find(|t| t.kind == TokenKind::Thedu).unwrap();
        assert_eq!(thedu.text(q.as_bytes()), Some("தேடு"));
    }

    #[test]
    fn mid_syllable_the_cut_returns_malformed_utf8_with_cursor() {
        let full = "தே".as_bytes();
        assert_eq!(full.len(), 6);
        let truncated = &full[..4];
        let mut lex = Lexer::new(truncated);
        let err = lex.next_token().expect_err("must reject torn தே");
        assert_eq!(err, LexerError::MalformedUtf8(0));
    }

    #[test]
    fn maximal_munch_keeps_vadivamaippu_as_ident() {
        // "வடிவமைப்பு" starts with keyword bytes for வடி but must NOT tear.
        let s = "வடிவமைப்பு";
        let mut lex = Lexer::new(s.as_bytes());
        let tok = lex.next_token().expect("lex");
        assert_eq!(tok.kind, TokenKind::Ident);
        assert_eq!(tok.text(s.as_bytes()), Some("வடிவமைப்பு"));
        // Bare keyword still classifies.
        let mut lex2 = Lexer::new("வடி".as_bytes());
        assert_eq!(lex2.next_token().unwrap().kind, TokenKind::Vadi);
    }
}

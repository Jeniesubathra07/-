//! UTF-8 / Tamil grapheme-safe boundary utilities.
//!
//! All slicing pins outputs to the parent source lifetime and never tears
//! multi-byte Tamil syllables (consonant + vowel marker, e.g. `தே` = `த` + `ே`).

/// Classify a leading UTF-8 byte into its sequence length via a 256-entry LUT.
/// Values: 0 = continuation / invalid lead, 1..4 = lead sequence length.
pub const UTF8_LEN_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        t[i] = if i < 0x80 {
            1
        } else if i < 0xC0 {
            0
        } else if i < 0xE0 {
            2
        } else if i < 0xF0 {
            3
        } else if i < 0xF8 {
            4
        } else {
            0
        };
        i += 1;
    }
    t
};

/// True when the byte is an ASCII whitespace (branchless via LUT).
pub const IS_WS_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b' ' as usize] = 1;
    t[b'\t' as usize] = 1;
    t[b'\n' as usize] = 1;
    t[b'\r' as usize] = 1;
    t
};

/// True when the byte is an ASCII digit (branchless via LUT).
pub const IS_DIGIT_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut d = b'0';
    while d <= b'9' {
        t[d as usize] = 1;
        d += 1;
    }
    t
};

/// True when the byte is an ASCII operator / delimiter used by the DSL.
pub const IS_OP_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b'|' as usize] = 1;
    t[b'=' as usize] = 1;
    t[b'>' as usize] = 1;
    t[b'<' as usize] = 1;
    t[b',' as usize] = 1;
    t[b';' as usize] = 1;
    t
};

/// Advance `i` to the next UTF-8 character boundary (or end).
#[inline(always)]
pub fn next_char_boundary(bytes: &[u8], i: usize) -> usize {
    if i >= bytes.len() {
        return bytes.len();
    }
    let len = UTF8_LEN_LUT[bytes[i] as usize] as usize;
    let end = i.wrapping_add(len);
    let ok = (len != 0) & (end <= bytes.len());
    // Branchless select: if ok use end else i+1 (best-effort recover).
    let fallback = i.wrapping_add(1).min(bytes.len());
    (end & (0usize.wrapping_sub(ok as usize))) | (fallback & (0usize.wrapping_sub((!ok) as usize)))
}

/// Tamil Unicode block helpers (BMP range U+0B80..U+0BFF encoded as 3-byte UTF-8).
#[inline(always)]
pub fn is_tamil_lead(b0: u8, b1: u8) -> bool {
    // Tamil: E0 AE 80 .. E0 AF BF  =>  lead E0, second AE or AF
    (b0 == 0xE0) & ((b1 == 0xAE) | (b1 == 0xAF))
}

/// Decode a Tamil codepoint from a verified 3-byte UTF-8 sequence.
#[inline(always)]
pub fn decode_tamil_cp(b0: u8, b1: u8, b2: u8) -> u32 {
    (((b0 as u32) & 0x0F) << 12) | (((b1 as u32) & 0x3F) << 6) | ((b2 as u32) & 0x3F)
}

/// True for Tamil dependent vowel signs / virama that must stay attached to
/// the preceding consonant (prevents syllable tearing on slice boundaries).
#[inline(always)]
pub fn is_tamil_combining(cp: u32) -> bool {
    // Virama U+0BCD, vowel signs U+0BBE..U+0BCC, U+0BD7
    ((cp >= 0x0BBE) & (cp <= 0x0BCC)) | (cp == 0x0BCD) | (cp == 0x0BD7)
}

/// Extend `end` forward while the next codepoint is a Tamil combining mark.
/// Callers pass a char boundary `end` that already includes a base character.
#[inline(always)]
pub fn extend_grapheme_cluster(bytes: &[u8], mut end: usize) -> usize {
    loop {
        if end + 3 > bytes.len() {
            break;
        }
        let b0 = bytes[end];
        let b1 = bytes[end + 1];
        let b2 = bytes[end + 2];
        let tamil = is_tamil_lead(b0, b1);
        let cp = decode_tamil_cp(b0, b1, b2);
        let comb = is_tamil_combining(cp);
        // continue only when tamil && comb
        let cont = tamil & comb;
        if cont == false {
            break;
        }
        end += 3;
    }
    end
}

/// Find the end of the current identifier / keyword starting at `start`.
/// Includes full Tamil grapheme clusters; stops at whitespace or operators.
#[inline(always)]
pub fn scan_ident_end(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        let stop = (IS_WS_LUT[b as usize] | IS_OP_LUT[b as usize]) != 0;
        if stop {
            break;
        }
        let next = next_char_boundary(bytes, i);
        i = extend_grapheme_cluster(bytes, next);
    }
    i
}

/// Validate that `slice` is valid UTF-8 and return it as `&str` pinned to `bytes`.
#[inline(always)]
pub fn str_from_parent<'a>(bytes: &'a [u8], start: usize, end: usize) -> Option<&'a str> {
    let slice = bytes.get(start..end)?;
    core::str::from_utf8(slice).ok()
}

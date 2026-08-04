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
/// Includes VT (`\x0B`) and FF (`\x0C`) for adversarial layout streams.
pub const WHITESPACE_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b' ' as usize] = 1;
    t[b'\t' as usize] = 1;
    t[b'\n' as usize] = 1;
    t[b'\r' as usize] = 1;
    t[0x0B] = 1; // vertical tab
    t[0x0C] = 1; // form feed
    t
};

/// Compatibility alias for older call sites.
pub const IS_WS_LUT: [u8; 256] = WHITESPACE_LUT;

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

/// Continuation-byte LUT (0x80..0xBF).
pub const IS_CONT_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0x80u8;
    loop {
        t[i as usize] = 1;
        if i == 0xBF {
            break;
        }
        i += 1;
    }
    t
};

/// Result of a checked UTF-8 / grapheme advance.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Utf8ScanStatus {
    Ok = 0,
    /// Truncated multi-byte sequence or torn Tamil syllable at buffer end.
    Malformed = 1,
}

/// Validate a UTF-8 scalar starting at `i`. Returns the end offset on success.
#[inline(always)]
pub fn checked_char_end(bytes: &[u8], i: usize) -> Result<usize, Utf8ScanStatus> {
    if i >= bytes.len() {
        return Err(Utf8ScanStatus::Malformed);
    }
    let b0 = bytes[i];
    let len = UTF8_LEN_LUT[b0 as usize] as usize;
    if len == 0 {
        return Err(Utf8ScanStatus::Malformed);
    }
    let end = i + len;
    if end > bytes.len() {
        // Truncated multi-byte sequence (mid-codepoint / mid-syllable cut).
        return Err(Utf8ScanStatus::Malformed);
    }
    let mut k = i + 1;
    while k < end {
        if IS_CONT_LUT[bytes[k] as usize] == 0 {
            return Err(Utf8ScanStatus::Malformed);
        }
        k += 1;
    }
    Ok(end)
}

/// Advance `i` to the next UTF-8 character boundary (or end).
/// Prefer [`checked_char_end`] on hot paths that must reject torn syllables.
#[inline(always)]
pub fn next_char_boundary(bytes: &[u8], i: usize) -> usize {
    match checked_char_end(bytes, i) {
        Ok(end) => end,
        Err(_) => i.wrapping_add(1).min(bytes.len()),
    }
}

/// Tamil Unicode block helpers (BMP range U+0B80..U+0BFF encoded as 3-byte UTF-8).
#[inline(always)]
pub fn is_tamil_lead(b0: u8, b1: u8) -> bool {
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
    ((cp >= 0x0BBE) & (cp <= 0x0BCC)) | (cp == 0x0BCD) | (cp == 0x0BD7)
}

/// True for Tamil consonants that commonly host a following vowel sign.
#[inline(always)]
pub fn is_tamil_consonant(cp: u32) -> bool {
    (cp >= 0x0B95) & (cp <= 0x0BB9)
}

/// Extend `end` forward while the next codepoint is a Tamil combining mark.
/// Returns `Malformed` if a combining-mark lead is truncated at EOF (mid-syllable).
#[inline(always)]
pub fn extend_grapheme_cluster_checked(
    bytes: &[u8],
    mut end: usize,
) -> Result<usize, Utf8ScanStatus> {
    loop {
        if end >= bytes.len() {
            break;
        }
        // Partial lead leftover at EOF ⇒ torn syllable / truncated UTF-8.
        let remaining = bytes.len() - end;
        let b0 = bytes[end];
        let need = UTF8_LEN_LUT[b0 as usize] as usize;
        if need == 0 {
            return Err(Utf8ScanStatus::Malformed);
        }
        if remaining < need {
            return Err(Utf8ScanStatus::Malformed);
        }
        if need != 3 {
            break;
        }
        let b1 = bytes[end + 1];
        let b2 = bytes[end + 2];
        let tamil = is_tamil_lead(b0, b1);
        let cp = decode_tamil_cp(b0, b1, b2);
        let comb = is_tamil_combining(cp);
        if !(tamil & comb) {
            break;
        }
        // Validate continuations.
        if IS_CONT_LUT[b1 as usize] == 0 || IS_CONT_LUT[b2 as usize] == 0 {
            return Err(Utf8ScanStatus::Malformed);
        }
        end += 3;
    }
    Ok(end)
}

/// Extend `end` forward while the next codepoint is a Tamil combining mark.
#[inline(always)]
pub fn extend_grapheme_cluster(bytes: &[u8], end: usize) -> usize {
    extend_grapheme_cluster_checked(bytes, end).unwrap_or(end)
}

/// Find the end of the current identifier / keyword starting at `start`.
/// Rejects truncated UTF-8 and mid-syllable cuts at the buffer tail.
#[inline(always)]
pub fn scan_ident_end_checked(bytes: &[u8], start: usize) -> Result<usize, Utf8ScanStatus> {
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        let stop = (WHITESPACE_LUT[b as usize] | IS_OP_LUT[b as usize]) != 0;
        if stop {
            break;
        }
        let next = checked_char_end(bytes, i)?;
        i = extend_grapheme_cluster_checked(bytes, next)?;
    }
    Ok(i)
}

/// Find the end of the current identifier / keyword starting at `start`.
#[inline(always)]
pub fn scan_ident_end(bytes: &[u8], start: usize) -> usize {
    scan_ident_end_checked(bytes, start).unwrap_or(start)
}

/// Validate that `slice` is valid UTF-8 and return it as `&str` pinned to `bytes`.
#[inline(always)]
pub fn str_from_parent<'a>(bytes: &'a [u8], start: usize, end: usize) -> Option<&'a str> {
    let slice = bytes.get(start..end)?;
    core::str::from_utf8(slice).ok()
}

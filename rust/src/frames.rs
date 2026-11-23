//! MGLM replica frame codec.

const FRAME_MAGIC: &[u8; 4] = b"MGLM";
const FRAME_VERSION: u8 = 6;
const FRAME_HEADER_SIZE: usize = 10;
const FRAME_TYPE_MIN: u8 = 1;
const FRAME_TYPE_MAX: u8 = 11;
const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Encode one replica frame: magic, version, type, big-endian length, body.
pub(crate) fn encode_frame(frame_type: u8, payload_json: &str) -> Result<Vec<u8>, String> {
    if !(FRAME_TYPE_MIN..=FRAME_TYPE_MAX).contains(&frame_type) {
        return Err(format!("unknown Replica frame type {frame_type}"));
    }
    let body = payload_json.as_bytes();
    if body.len() > MAX_FRAME_BYTES {
        return Err(format!(
            "Replica frame exceeds {MAX_FRAME_BYTES} bytes"
        ));
    }
    let mut frame = Vec::with_capacity(FRAME_HEADER_SIZE + body.len());
    frame.extend_from_slice(FRAME_MAGIC);
    frame.push(FRAME_VERSION);
    frame.push(frame_type);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(body);
    Ok(frame)
}

/// Validate a received frame header and return `(type, payload length)`.
pub(crate) fn decode_frame_header(header: &[u8]) -> Result<(u8, usize), String> {
    if header.len() != FRAME_HEADER_SIZE {
        return Err("Replica frame header is truncated".into());
    }
    if &header[..4] != FRAME_MAGIC {
        return Err("Replica frame magic mismatch".into());
    }
    if header[4] != FRAME_VERSION {
        return Err(format!(
            "unsupported Replica protocol version {}",
            header[4]
        ));
    }
    if !(FRAME_TYPE_MIN..=FRAME_TYPE_MAX).contains(&header[5]) {
        return Err(format!("unknown Replica frame type {}", header[5]));
    }
    let payload_size = u32::from_be_bytes([
        header[6],
        header[7],
        header[8],
        header[9],
    ]) as usize;
    if payload_size > MAX_FRAME_BYTES {
        return Err("Replica frame length exceeds its limit".into());
    }
    Ok((header[5], payload_size))
}

#[cfg(test)]
mod frame_tests {
    use super::{decode_frame_header, encode_frame};

    #[test]
    fn frame_round_trip() {
        let frame = encode_frame(4, r#"{"request_id":"r1"}"#).unwrap();
        assert_eq!(&frame[..4], b"MGLM");
        let (frame_type, size) = decode_frame_header(&frame[..10]).unwrap();
        assert_eq!(frame_type, 4);
        assert_eq!(size, frame.len() - 10);
    }

    #[test]
    fn rejects_bad_header() {
        assert!(decode_frame_header(&[0; 9]).is_err());
        let mut frame = encode_frame(1, "{}").unwrap();
        frame[4] = 4;
        assert!(decode_frame_header(&frame[..10])
            .unwrap_err()
            .contains("version"));
    }
}


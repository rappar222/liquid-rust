use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Zero-dependency 24-bit sRGB PNG encoder backed by `flate2` zlib compression.
pub struct PngExporter;

impl PngExporter {
    /// Encode a tightly-packed `width * height * 3` RGB24 frame buffer to a `.png` file.
    pub fn save_rgb(
        path: impl AsRef<Path>,
        width: usize,
        height: usize,
        rgb: &[u8],
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("PNG dimensions must be non-zero".to_string());
        }
        let expected = width
            .checked_mul(height)
            .and_then(|px| px.checked_mul(3))
            .ok_or_else(|| "PNG dimensions overflow".to_string())?;
        if rgb.len() != expected {
            return Err(format!(
                "PNG RGB buffer length mismatch: expected {}, got {}",
                expected,
                rgb.len()
            ));
        }

        let file =
            File::create(path.as_ref()).map_err(|e| format!("Failed to create PNG file: {}", e))?;
        let mut out = BufWriter::new(file);

        // PNG 8-byte signature
        out.write_all(&[137, 80, 78, 71, 13, 10, 26, 10])
            .map_err(|e| e.to_string())?;

        // IHDR chunk (8-bit truecolor RGB, non-interlaced)
        let mut ihdr = [0u8; 13];
        ihdr[0..4].copy_from_slice(&(width as u32).to_be_bytes());
        ihdr[4..8].copy_from_slice(&(height as u32).to_be_bytes());
        ihdr[8] = 8; // bit depth = 8
        ihdr[9] = 2; // color type = 2 (RGB)
        ihdr[10] = 0; // compression = deflate
        ihdr[11] = 0; // filter = adaptive
        ihdr[12] = 0; // interlace = none
        Self::write_chunk(&mut out, b"IHDR", &ihdr)?;

        // Compress scanlines using Sub filter (filter type 1) for crisp smooth gradients
        let row_bytes = width * 3;
        let mut encoder = ZlibEncoder::new(Vec::with_capacity(expected / 3), Compression::default());
        let mut filtered_row = vec![0u8; 1 + row_bytes];
        filtered_row[0] = 1; // Sub filter
        for y in 0..height {
            let row = &rgb[y * row_bytes..(y + 1) * row_bytes];
            for i in 0..row_bytes {
                let left = if i >= 3 { row[i - 3] } else { 0 };
                filtered_row[1 + i] = row[i].wrapping_sub(left);
            }
            encoder
                .write_all(&filtered_row)
                .map_err(|e| format!("PNG zlib encode error: {}", e))?;
        }
        let idat_data = encoder
            .finish()
            .map_err(|e| format!("PNG zlib finish error: {}", e))?;
        Self::write_chunk(&mut out, b"IDAT", &idat_data)?;

        // IEND chunk
        Self::write_chunk(&mut out, b"IEND", &[])?;
        out.flush().map_err(|e| e.to_string())
    }

    fn write_chunk<W: Write>(out: &mut W, chunk_type: &[u8; 4], data: &[u8]) -> Result<(), String> {
        let len = data.len() as u32;
        out.write_all(&len.to_be_bytes()).map_err(|e| e.to_string())?;
        out.write_all(chunk_type).map_err(|e| e.to_string())?;
        out.write_all(data).map_err(|e| e.to_string())?;
        let mut crc = 0xFFFF_FFFFu32;
        crc = Self::crc32_update(crc, chunk_type);
        crc = Self::crc32_update(crc, data);
        out.write_all(&(!crc).to_be_bytes())
            .map_err(|e| e.to_string())
    }

    #[inline]
    fn crc32_update(mut crc: u32, bytes: &[u8]) -> u32 {
        for &b in bytes {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        crc
    }
}

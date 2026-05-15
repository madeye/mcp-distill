//! Benchmark storage size + write time across the four configurations:
//!   - keep_raw: on  / off
//!   - compression: none / zstd
//!
//! Synthesizes a realistic-ish session: large system prompt, many turns of
//! mixed text + tool_use + tool_result blocks, with chunks of "file content"
//! pasted into tool_result blobs (the kind of thing that bloats real traces).
//!
//! Run with:
//!     cargo run --release --example storage_bench

use std::time::Instant;

use mcp_distill::adapters;
use mcp_distill::schema::{ContentBlock, Provider, RecordKind, SessionMeta, Turn, TurnRecord};
use mcp_distill::storage::{now_rfc3339, Compression, Store, StoreOptions};
use serde_json::{json, Value};
use tempfile::TempDir;

const SESSIONS: usize = 4;
const TURNS_PER_SESSION: usize = 40;
const SYSTEM_PROMPT_KB: usize = 8;
const TOOL_RESULT_KB: usize = 6;

fn synthetic_text(kb: usize, seed: u64) -> String {
    // Realistic-ish text: code-like lines, repeated structure (compressible),
    // but not all identical.
    let mut s = String::with_capacity(kb * 1024);
    let mut x = seed;
    let words = [
        "fn", "let", "match", "Some", "None", "Result", "Ok", "Err", "self", "pub", "use", "crate",
        "tokio", "serde", "json", "async", "await", "Vec", "String", "HashMap", "Option", "impl",
        "trait", "where",
    ];
    while s.len() < kb * 1024 {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let w = words[(x as usize) % words.len()];
        s.push_str(w);
        if s.len().is_multiple_of(80) {
            s.push('\n');
        } else {
            s.push(' ');
        }
    }
    s
}

fn make_request_message(turn_idx: usize) -> Value {
    json!({
        "role": "user",
        "content": format!("Turn {}: please continue the analysis. Cite specific files.", turn_idx),
    })
}

fn make_response_with_tool(turn_idx: usize) -> Value {
    let tool_result_text = synthetic_text(TOOL_RESULT_KB, turn_idx as u64 * 9973);
    // Anthropic-shape assistant turn with text + tool_use; we then add a
    // following user turn carrying a tool_result of similar size.
    json!({
        "role": "assistant",
        "content": [
            {"type": "text", "text": format!(
                "Step {}: I'll grep for the pattern then read the matching files. \
                 Specifically I'm looking at three modules whose hot paths show \
                 redundant allocations. After reading the source I'll summarize.",
                turn_idx
            )},
            {"type": "tool_use", "id": format!("call_{turn_idx:04}"), "name": "shell",
             "input": {"command": format!("rg -n 'TODO|XXX' src --max-count {}", 5 + turn_idx % 7)}},
        ],
        "model": "claude-opus-4-7",
        "_tool_result_preview_for_next_turn": tool_result_text,
    })
}

fn make_tool_result(turn_idx: usize) -> Value {
    let text = synthetic_text(TOOL_RESULT_KB, turn_idx as u64 * 4099);
    json!({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": format!("call_{turn_idx:04}"),
             "content": text, "is_error": false},
        ],
    })
}

fn populate(store: &Store, n_sessions: usize, n_turns: usize) {
    let system_text = synthetic_text(SYSTEM_PROMPT_KB, 1);
    for s_idx in 0..n_sessions {
        let session_id = format!("bench-{s_idx:04}");
        let meta = SessionMeta {
            session_id: session_id.clone(),
            provider: Provider::Claude,
            model: Some("claude-opus-4-7".into()),
            started_at: now_rfc3339(),
            ended_at: None,
            tags: vec!["bench".into()],
            metadata: Default::default(),
        };
        store.write_meta(&meta).unwrap();

        // System turn (large, repeats across sessions — great CAS/zstd target).
        let sys_turn = Turn {
            role: mcp_distill::schema::Role::System,
            blocks: vec![ContentBlock::text(system_text.clone())],
            raw: None,
        };
        store
            .write_record(
                &session_id,
                &TurnRecord {
                    kind: RecordKind::Turn,
                    ts: now_rfc3339(),
                    session_id: session_id.clone(),
                    seq: store.next_seq(&session_id),
                    turn: Some(sys_turn),
                    meta: None,
                    usage: None,
                },
            )
            .unwrap();

        for t in 0..n_turns {
            let user_msg = make_request_message(t);
            let user_turn = adapters::to_turn(&Provider::Claude, &user_msg);
            store
                .write_record(
                    &session_id,
                    &TurnRecord {
                        kind: RecordKind::Turn,
                        ts: now_rfc3339(),
                        session_id: session_id.clone(),
                        seq: store.next_seq(&session_id),
                        turn: Some(user_turn),
                        meta: None,
                        usage: None,
                    },
                )
                .unwrap();

            let asst_msg = make_response_with_tool(t);
            let asst_turn = adapters::response_to_turn(&Provider::Claude, &asst_msg);
            store
                .write_record(
                    &session_id,
                    &TurnRecord {
                        kind: RecordKind::Turn,
                        ts: now_rfc3339(),
                        session_id: session_id.clone(),
                        seq: store.next_seq(&session_id),
                        turn: Some(asst_turn),
                        meta: None,
                        usage: None,
                    },
                )
                .unwrap();

            let tr_msg = make_tool_result(t);
            let tr_turn = adapters::to_turn(&Provider::Claude, &tr_msg);
            store
                .write_record(
                    &session_id,
                    &TurnRecord {
                        kind: RecordKind::Turn,
                        ts: now_rfc3339(),
                        session_id: session_id.clone(),
                        seq: store.next_seq(&session_id),
                        turn: Some(tr_turn),
                        meta: None,
                        usage: None,
                    },
                )
                .unwrap();
        }
    }
}

fn dir_size_bytes(p: &std::path::Path) -> u64 {
    let mut total = 0u64;
    fn walk(p: &std::path::Path, total: &mut u64) {
        let Ok(entries) = std::fs::read_dir(p) else {
            return;
        };
        for e in entries.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, total);
            } else if let Ok(meta) = path.metadata() {
                *total += meta.len();
            }
        }
    }
    walk(p, &mut total);
    total
}

fn human(n: u64) -> String {
    if n >= 1024 * 1024 {
        format!("{:.2} MB", n as f64 / (1024.0 * 1024.0))
    } else if n >= 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{n} B")
    }
}

fn run_one(label: &str, opts: StoreOptions, baseline_bytes: Option<u64>) -> u64 {
    let tmp = TempDir::new().unwrap();
    let store = Store::with_options(tmp.path().to_path_buf(), opts).unwrap();
    let started = Instant::now();
    populate(&store, SESSIONS, TURNS_PER_SESSION);
    let elapsed = started.elapsed();
    let bytes = dir_size_bytes(&tmp.path().join("sessions"));

    let ratio = baseline_bytes
        .map(|b| format!("  {:.2}x", bytes as f64 / b as f64))
        .unwrap_or_else(|| "  (baseline)".to_string());
    println!(
        "{:<28} sessions={:>4}  size={:>10}  write={:>7.0} ms{}",
        label,
        SESSIONS,
        human(bytes),
        elapsed.as_secs_f64() * 1000.0,
        ratio,
    );

    // Sanity: roundtrip the first session — must read back at least the meta + turns.
    let recs = store.iter_session("bench-0000").unwrap();
    let n_turns = recs
        .iter()
        .filter(|r| matches!(r.kind, RecordKind::Turn))
        .count();
    assert!(
        n_turns >= TURNS_PER_SESSION * 3,
        "{label}: roundtrip lost turns"
    );
    bytes
}

fn main() {
    let total_text_kb =
        SESSIONS * (SYSTEM_PROMPT_KB + TURNS_PER_SESSION * (TOOL_RESULT_KB * 2 + 1));
    println!(
        "synthetic corpus: {} sessions x {} turns; ~{} KB of message text in flight\n",
        SESSIONS, TURNS_PER_SESSION, total_text_kb
    );
    println!(
        "{:<28} {:>4}  {:>10}  {:>7}  vs baseline",
        "config", "n", "size", "write"
    );
    println!("{}", "-".repeat(72));

    let baseline = run_one(
        "raw=on   compression=none",
        StoreOptions {
            compression: Compression::None,
            zstd_level: 3,
            keep_raw: true,
        },
        None,
    );
    run_one(
        "raw=off  compression=none",
        StoreOptions {
            compression: Compression::None,
            zstd_level: 3,
            keep_raw: false,
        },
        Some(baseline),
    );
    for level in [3, 9, 19] {
        run_one(
            &format!("raw=on   compression=zstd-{level}"),
            StoreOptions {
                compression: Compression::Zstd,
                zstd_level: level,
                keep_raw: true,
            },
            Some(baseline),
        );
        run_one(
            &format!("raw=off  compression=zstd-{level}"),
            StoreOptions {
                compression: Compression::Zstd,
                zstd_level: level,
                keep_raw: false,
            },
            Some(baseline),
        );
    }
}

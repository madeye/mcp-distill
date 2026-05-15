//! JSONL storage for session traces.
//!
//! Layout:
//! ```text
//! <root>/
//!   sessions/YYYY/MM/DD/<session_id>.jsonl   # raw, lossless, append-only
//!   index.jsonl                              # one line per session (latest meta wins)
//!   exports/                                 # generated SFT datasets
//! ```

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::schema::{SessionMeta, TurnRecord};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRow {
    pub session_id: String,
    pub provider: String,
    pub model: Option<String>,
    pub started_at: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

pub struct Store {
    pub root: PathBuf,
    file_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    seq: Mutex<HashMap<String, u64>>,
}

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(root.join("sessions"))?;
        fs::create_dir_all(root.join("exports"))?;
        Ok(Self {
            root,
            file_locks: Mutex::new(HashMap::new()),
            seq: Mutex::new(HashMap::new()),
        })
    }

    pub fn default_root() -> PathBuf {
        if let Ok(env) = std::env::var("MCP_DISTILL_ROOT") {
            return PathBuf::from(shellexpand(&env));
        }
        if let Some(home) = directories::BaseDirs::new() {
            return home.home_dir().join(".mcp-distill");
        }
        PathBuf::from(".mcp-distill")
    }

    fn lock_for(&self, session_id: &str) -> Arc<Mutex<()>> {
        let mut locks = self.file_locks.lock();
        locks
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    pub fn next_seq(&self, session_id: &str) -> u64 {
        let mut s = self.seq.lock();
        let n = s.get(session_id).copied().unwrap_or(0).wrapping_add(0);
        let next = match s.get(session_id).copied() {
            Some(v) => v + 1,
            None => 0,
        };
        s.insert(session_id.to_string(), next);
        let _ = n;
        next
    }

    fn session_path(&self, meta: &SessionMeta) -> Result<PathBuf> {
        let dt: DateTime<Utc> = DateTime::parse_from_rfc3339(&meta.started_at)
            .with_context(|| format!("bad started_at {}", meta.started_at))?
            .with_timezone(&Utc);
        let dir = self
            .root
            .join("sessions")
            .join(format!("{:04}", dt.format("%Y")))
            .join(format!("{}", dt.format("%m")))
            .join(format!("{}", dt.format("%d")));
        fs::create_dir_all(&dir)?;
        Ok(dir.join(format!("{}.jsonl", meta.session_id)))
    }

    pub fn find_session_file(&self, session_id: &str) -> Option<PathBuf> {
        let want = format!("{session_id}.jsonl");
        walk(&self.root.join("sessions"), &want)
    }

    pub fn write_meta(&self, meta: &SessionMeta) -> Result<PathBuf> {
        let path = self.session_path(meta)?;
        let rec = TurnRecord {
            kind: crate::schema::RecordKind::Meta,
            ts: now_rfc3339(),
            session_id: meta.session_id.clone(),
            seq: self.next_seq(&meta.session_id),
            turn: None,
            meta: Some(meta.clone()),
            usage: None,
        };
        self.append_line(&path, &meta.session_id, &rec)?;
        self.write_index(meta)?;
        Ok(path)
    }

    pub fn write_record(&self, session_id: &str, rec: &TurnRecord) -> Result<PathBuf> {
        let path = self
            .find_session_file(session_id)
            .ok_or_else(|| anyhow!("unknown session {session_id}; call start_session first"))?;
        self.append_line(&path, session_id, rec)?;
        Ok(path)
    }

    fn append_line(&self, path: &Path, session_id: &str, rec: &TurnRecord) -> Result<()> {
        let lock = self.lock_for(session_id);
        let _g = lock.lock();
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        let line = serde_json::to_string(rec)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    fn write_index(&self, meta: &SessionMeta) -> Result<()> {
        let row = IndexRow {
            session_id: meta.session_id.clone(),
            provider: meta.provider.as_str().to_string(),
            model: meta.model.clone(),
            started_at: meta.started_at.clone(),
            tags: meta.tags.clone(),
        };
        let path = self.root.join("index.jsonl");
        let mut f = OpenOptions::new().create(true).append(true).open(path)?;
        f.write_all(serde_json::to_string(&row)?.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    pub fn list_sessions(&self) -> Result<Vec<IndexRow>> {
        let path = self.root.join("index.jsonl");
        if !path.exists() {
            return Ok(vec![]);
        }
        let f = File::open(path)?;
        let mut by_id: HashMap<String, IndexRow> = HashMap::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: IndexRow = serde_json::from_str(&line)?;
            by_id.insert(row.session_id.clone(), row);
        }
        Ok(by_id.into_values().collect())
    }

    pub fn iter_session(&self, session_id: &str) -> Result<Vec<TurnRecord>> {
        let path = self
            .find_session_file(session_id)
            .ok_or_else(|| anyhow!("unknown session {session_id}"))?;
        let f = File::open(path)?;
        let mut out = Vec::new();
        for line in BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line)?);
        }
        Ok(out)
    }
}

fn walk(dir: &Path, want: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = walk(&p, want) {
                return Some(found);
            }
        } else if p.file_name().map(|n| n == want).unwrap_or(false) {
            return Some(p);
        }
    }
    None
}

fn shellexpand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = directories::BaseDirs::new() {
            return home.home_dir().join(rest).to_string_lossy().into_owned();
        }
    }
    s.to_string()
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339()
}
